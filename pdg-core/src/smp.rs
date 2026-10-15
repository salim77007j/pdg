//! The SMP machine: one host thread per logical core, one shared bus.
//!
//! Phase 2's SMP deliverable. The PS5 CPU is an AMD Zen 2 with 8 cores and
//! 16 hardware threads; [`SmpSystem`] models the logical-processor shape
//! with `cpu.threads` host threads (default 16) that share one
//! guest-physical [`Bus`] through per-core [`Mmu`]s:
//!
//! * every core owns its page-table walker and software TLB (per-core CR3
//!   reloads are architecturally honored through [`MemoryAccess`]);
//! * RAM accesses are width-atomic on the shared bus, and LOCK-prefixed
//!   RMWs (`xchg`/`xadd`/`cmpxchg`/`lock inc`/`lock dec`) run on the host's
//!   hardware atomics, so cross-core counter protocols behave exactly like
//!   real hardware — no lost updates, ever;
//! * inter-processor interrupts go through the [`IpiDevice`] MMIO window
//!   and are mirrored onto the [`EventBus`] as [`Event::IpiSent`];
//! * each core runs on its own private stack (`stack_top - core*64 KiB`)
//!   so same-entry images (`run_all`) never collide.
//!
//! What is deliberately **not** here yet (documented, not hidden): TLB
//! shootdown IPIs (a core that rewrites shared page tables only
//! invalidates its own TLB; the Phase-5 kernel owns the coherence policy),
//! per-core APIC IDs in CPUID (kernel-phase topology enumeration), and a
//! scheduler that time-slices more vCPUs than the host has threads.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use pdg_cpu::{AccessError, CpuState, Interpreter, MemoryAccess, RunOutcome, SyscallHook};
use pdg_memory::{
    AccessKind, AccessWidth, AtomicRmw, Bus, ConsoleDevice, IpiDevice, IpiHub, Mmu, PAGE_SIZE,
};

use crate::config::PdgConfig;
use crate::events::{Event, EventBus};
use crate::system::{install_bootstrap, map_fault};

/// Per-core guest memory port: a private page-table walker (and TLB) over
/// the shared bus. This is the SMP twin of [`crate::system::GuestMemory`].
///
/// The atomics are *real*: LOCK-prefixed RMWs translate through the owning
/// MMU and then execute as hardware atomics on the shared RAM, which is
/// what makes cross-core synchronization observable to guest code.
pub struct SharedGuest {
    bus: Arc<Bus>,
    mmu: Mmu,
    /// Bumped on translation changes (CR3 reload, INVLPG, flush) so the
    /// JIT's env TLB can validate cheaply instead of flushing.
    epoch: u64,
}

impl SharedGuest {
    /// Bind a fresh walker (own TLB) to the shared bus at `cr3`.
    pub fn new(bus: Arc<Bus>, cr3: u64) -> Self {
        Self {
            bus,
            mmu: Mmu::new(cr3),
            epoch: 0,
        }
    }
}

fn width_checked(width: u8) -> Result<AccessWidth, AccessError> {
    AccessWidth::from_size(width)
        .ok_or_else(|| AccessError::Host(format!("bad access width {width}")))
}

impl MemoryAccess for SharedGuest {
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, AccessError> {
        let w = width_checked(width)?;
        self.mmu.read(&self.bus, w, addr).map_err(map_fault)
    }

    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), AccessError> {
        let w = width_checked(width)?;
        self.mmu.write(&self.bus, w, addr, value).map_err(map_fault)
    }

    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, AccessError> {
        let mut filled = 0usize;
        while filled < out.len() {
            let va = addr + filled as u64;
            let pa = match self.mmu.translate(&self.bus, va, AccessKind::Exec) {
                Ok(pa) => pa,
                // Early stop when at least one byte is available; the
                // decoder faults if the truncated window cannot hold the
                // instruction.
                Err(_) if filled > 0 => break,
                Err(e) => return Err(map_fault(e)),
            };
            let page_off = (va & (PAGE_SIZE - 1)) as usize;
            let want = (out.len() - filled).min(PAGE_SIZE as usize - page_off);
            match self.bus.read_buf(pa, &mut out[filled..filled + want]) {
                Ok(()) => filled += want,
                Err(_) if filled > 0 => break,
                Err(e) => return Err(AccessError::Host(e.to_string())),
            }
        }
        Ok(filled)
    }

    fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), AccessError> {
        let mut done = 0usize;
        while done < out.len() {
            let va = addr + done as u64;
            let pa = self
                .mmu
                .translate(&self.bus, va, AccessKind::Read)
                .map_err(map_fault)?;
            let page_off = (va & (PAGE_SIZE - 1)) as usize;
            let want = (out.len() - done).min(PAGE_SIZE as usize - page_off);
            self.bus
                .read_buf(pa, &mut out[done..done + want])
                .map_err(|e| AccessError::Host(e.to_string()))?;
            done += want;
        }
        Ok(())
    }

    fn write_buf(&mut self, addr: u64, src: &[u8]) -> Result<(), AccessError> {
        let mut done = 0usize;
        while done < src.len() {
            let va = addr + done as u64;
            let pa = self
                .mmu
                .translate(&self.bus, va, AccessKind::Write)
                .map_err(map_fault)?;
            let page_off = (va & (PAGE_SIZE - 1)) as usize;
            let want = (src.len() - done).min(PAGE_SIZE as usize - page_off);
            self.bus
                .write_buf(pa, &src[done..done + want])
                .map_err(|e| AccessError::Host(e.to_string()))?;
            done += want;
        }
        Ok(())
    }

    fn xchg(&mut self, width: u8, addr: u64, value: u64) -> Result<u64, AccessError> {
        let w = width_checked(width)?;
        let pa = self
            .mmu
            .translate(&self.bus, addr, AccessKind::Write)
            .map_err(map_fault)?;
        self.bus
            .atomic_rmw(pa, w, value, AtomicRmw::Xchg)
            .map_err(|e| AccessError::Host(e.to_string()))
    }

    fn xadd(&mut self, width: u8, addr: u64, value: u64) -> Result<u64, AccessError> {
        let w = width_checked(width)?;
        let pa = self
            .mmu
            .translate(&self.bus, addr, AccessKind::Write)
            .map_err(map_fault)?;
        self.bus
            .atomic_rmw(pa, w, value, AtomicRmw::Xadd)
            .map_err(|e| AccessError::Host(e.to_string()))
    }

    fn cmpxchg(
        &mut self,
        width: u8,
        addr: u64,
        expected: u64,
        new: u64,
    ) -> Result<u64, AccessError> {
        let w = width_checked(width)?;
        let pa = self
            .mmu
            .translate(&self.bus, addr, AccessKind::Write)
            .map_err(map_fault)?;
        self.bus
            .atomic_rmw(pa, w, new, AtomicRmw::Cmpxchg { expected })
            .map_err(|e| AccessError::Host(e.to_string()))
    }

    fn invalidate_tlb_page(&mut self, va: u64) {
        self.mmu.invlpg(va);
        self.epoch += 1;
    }

    fn invalidate_tlb_all(&mut self) {
        self.mmu.flush_tlb();
        self.epoch += 1;
    }

    fn reload_cr3(&mut self, cr3: u64) {
        self.mmu.set_cr3(cr3);
        self.epoch += 1;
    }

    fn translation_epoch(&mut self) -> u64 {
        self.epoch
    }
}

/// Which core runs where, and for how long.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreSpec {
    /// Logical core id (`0 .. cpu.threads`).
    pub core: u8,
    /// Entry RIP.
    pub entry: u64,
    /// Instruction budget for this core's run.
    pub budget: u64,
}

/// Per-core result of an [`SmpSystem::run_cores`] batch.
#[derive(Debug, Clone)]
pub struct CoreReport {
    /// The core this report belongs to.
    pub core: u8,
    /// Entry RIP the core started at.
    pub entry: u64,
    /// Guest execution outcome.
    pub outcome: RunOutcome,
}

/// Multi-core machine: `cpu.threads` host threads over one shared bus.
///
/// Build one with [`SmpSystem::new`], map images with [`SmpSystem::boot_image`]
/// (shared address space — every core sees every image), then execute with
/// [`SmpSystem::run_all`] (same entry for all cores) or
/// [`SmpSystem::run_cores`] (per-core entries and budgets).
pub struct SmpSystem {
    bus: Arc<Bus>,
    console: Arc<ConsoleDevice>,
    ipi: Arc<IpiHub>,
    events: Arc<EventBus>,
    config: PdgConfig,
    syscall_hook_factory: Option<Arc<dyn Fn() -> Box<dyn SyscallHook> + Send + Sync>>,
}

impl SmpSystem {
    /// Build the shared machine: RAM, console + IPI MMIO windows, bootstrap
    /// page tables. All cores share this address space.
    pub fn new(config: PdgConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let ram = u64::from(config.memory.ram_mib) * 1024 * 1024;
        let threads = config.cpu.threads;

        let mut bus = Bus::new(ram as usize)?;
        let console = Arc::new(ConsoleDevice::new());
        bus.attach_mmio(config.memory.console_mmio_base, 0x1000, console.clone());

        let ipi = Arc::new(IpiHub::new(usize::from(threads)));
        bus.attach_mmio(
            config.memory.ipi_mmio_base,
            0x1000,
            Arc::new(IpiDevice::new(Arc::clone(&ipi))),
        );

        install_bootstrap(&bus, &config)?;

        // Mirror guest-visible IPIs onto the host event bus.
        let events = Arc::new(EventBus::new());
        let ev = Arc::clone(&events);
        ipi.set_on_send(Box::new(move |target, vector| {
            ev.publish(Event::IpiSent { target, vector });
        }));

        Ok(Self {
            bus: Arc::new(bus),
            console,
            ipi,
            events,
            config,
            syscall_hook_factory: None,
        })
    }

    /// The shared guest-physical bus (tests, tooling, DMA setup).
    pub fn bus(&self) -> &Arc<Bus> {
        &self.bus
    }

    /// Boot console (assertions on guest output).
    pub fn console(&self) -> &Arc<ConsoleDevice> {
        &self.console
    }

    /// IPI hub (host-side interrupt injection and diagnostics).
    pub fn ipi(&self) -> &Arc<IpiHub> {
        &self.ipi
    }

    /// Event bus every component publishes to.
    pub fn events(&self) -> &Arc<EventBus> {
        &self.events
    }

    /// Active configuration snapshot.
    pub fn config(&self) -> &PdgConfig {
        &self.config
    }

    /// Give every core a fresh HLE syscall handler from `factory`.
    ///
    /// Cores run on separate host threads, so a single hook object cannot
    /// be shared — the factory pattern gives each vCPU its own instance.
    pub fn set_syscall_hook_factory(
        &mut self,
        factory: Arc<dyn Fn() -> Box<dyn SyscallHook> + Send + Sync>,
    ) {
        self.syscall_hook_factory = Some(factory);
    }

    /// Map a raw image into shared RAM (every core sees it). Returns `base`.
    pub fn boot_image(&self, image: &[u8], base: u64) -> anyhow::Result<u64> {
        let ram = u64::from(self.config.memory.ram_mib) * 1024 * 1024;
        if base % PAGE_SIZE != 0 {
            anyhow::bail!("boot image base {base:#x} must be 4 KiB-aligned");
        }
        let end = base
            .checked_add(image.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("boot image end overflows"))?;
        if end > ram {
            anyhow::bail!("boot image [{base:#x}..{end:#x}) exceeds RAM size {ram:#x}");
        }
        self.bus.ram_write_buf(base, image)?;
        Ok(base)
    }

    /// Run every logical core at `entry` (each with its own stack) and join.
    pub fn run_all(&self, entry: u64, budget: u64) -> Vec<CoreReport> {
        let specs: Vec<CoreSpec> = (0..self.config.cpu.threads)
            .map(|core| CoreSpec {
                core,
                entry,
                budget,
            })
            .collect();
        self.run_cores(&specs)
    }

    /// Run exactly the cores in `specs` (one host thread each) and join.
    /// Reports come back in the same order as `specs`; out-of-range core
    /// ids are dropped with a warning.
    pub fn run_cores(&self, specs: &[CoreSpec]) -> Vec<CoreReport> {
        if let Some(first) = specs.first() {
            self.events
                .publish(Event::BootStarted { entry: first.entry });
        }
        let threads = usize::from(self.config.cpu.threads);
        let handles: Vec<JoinHandle<CoreReport>> = specs
            .iter()
            .filter(|s| (usize::from(s.core)) < threads)
            .map(|s| self.spawn_core(*s))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("vCPU thread panicked"))
            .collect()
    }

    fn spawn_core(&self, spec: CoreSpec) -> JoinHandle<CoreReport> {
        let bus = Arc::clone(&self.bus);
        let pml4 = self.config.memory.pml4_base;
        // Each core gets a private 64 KiB stack slot below stack_top.
        let stack = self
            .config
            .boot
            .stack_top
            .saturating_sub(u64::from(spec.core) * 0x1_0000);
        let factory = self.syscall_hook_factory.clone();
        let events = Arc::clone(&self.events);
        thread::Builder::new()
            .name(format!("pdg-vcpu{}", spec.core))
            .spawn(move || {
                let mut guest = SharedGuest::new(bus, pml4);
                let mut cpu = Interpreter::new(CpuState::new(spec.entry, stack));
                if let Some(make) = &factory {
                    cpu.set_syscall_hook(make());
                }
                let outcome = cpu.run(&mut guest, spec.budget);
                match &outcome {
                    RunOutcome::Halted { instructions, .. } => {
                        events.publish(Event::Halted {
                            instructions: *instructions,
                        });
                    }
                    RunOutcome::Fault(f) => {
                        events.publish(Event::Fault {
                            reason: f.to_string(),
                        });
                    }
                    RunOutcome::MaxInstructions { .. } => {}
                }
                CoreReport {
                    core: spec.core,
                    entry: spec.entry,
                    outcome,
                }
            })
            .expect("failed to spawn vCPU thread")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_cores_boot_and_halt() {
        let sys = SmpSystem::new(PdgConfig::default()).unwrap();
        sys.boot_image(&[0xF4], 0x0010_0000).unwrap(); // hlt
        let reports = sys.run_cores(&[
            CoreSpec {
                core: 0,
                entry: 0x0010_0000,
                budget: 10,
            },
            CoreSpec {
                core: 3,
                entry: 0x0010_0000,
                budget: 10,
            },
        ]);
        assert_eq!(reports.len(), 2);
        for r in &reports {
            match r.outcome {
                RunOutcome::Halted { instructions, rip } => {
                    assert_eq!(instructions, 1);
                    assert_eq!(rip, 0x0010_0000);
                }
                ref other => panic!("core {} did not halt: {other:?}", r.core),
            }
        }
    }

    #[test]
    fn out_of_range_core_is_dropped() {
        let sys = SmpSystem::new(PdgConfig::default()).unwrap();
        sys.boot_image(&[0xF4], 0x0010_0000).unwrap();
        let reports = sys.run_cores(&[CoreSpec {
            core: 200,
            entry: 0x0010_0000,
            budget: 10,
        }]);
        assert!(reports.is_empty());
    }
}
