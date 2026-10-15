//! The `System` orchestrator: one emulated machine.
//!
//! Wires the guest-physical bus, the MMU, one CPU, the console device and
//! the DMA engine into a single machine, and owns the Phase-1 boot flow:
//!
//! 1. `System::new` builds RAM, attaches the console MMIO window and
//!    installs the bootstrap page tables (genuine x86-64 structures in
//!    guest RAM).
//! 2. `boot_image` / `load_bios` maps a payload and resets the CPU.
//! 3. `run` executes through the paged guest path and collects a report.
//!
//! Phase-1 guest memory map (simplified "flat boot"; the Orbis-accurate map
//! lands with the kernel work in Phase 5):
//!
//! ```text
//! 0x0000_0000 .. 0x0020_0000   RAM, RX   (boot image region)
//! 0x0020_0000 .. RAM end        RAM, RW NX (bootstrap page tables live here)
//! <RAM end>  .. 0xF000_0000     unmapped
//! 0xF000_0000 .. +0x1000         MMIO: boot console (identity-mapped RW)
//! 0xFF00_0000 .. 0xFFFF_FFFF     ROM window (BIOS), read-only
//!   └ 0xFFFE_0000 .. 0x1_0000_0000  ROM, executable (reset vector page)
//! reset vector: 0xFFFF_FFF0, initial RSP: boot.stack_top
//! ```

use std::sync::Arc;

use anyhow::Context;
use pdg_cpu::{AccessError, CpuState, Interpreter, MemoryAccess, RunOutcome, SyscallHook};
use pdg_memory::{
    AccessFault, AccessKind, AccessWidth, Bus, ConsoleDevice, DmaEngine, Mmu, PAGE_SIZE, PageFlags,
    PageTableBuilder,
};

use crate::clock::Clock;
use crate::config::PdgConfig;
use crate::events::{Event, EventBus};

/// Guest-physical base of the 16 MiB ROM window just below 4 GiB.
pub const ROM_BASE: u64 = 0xFF00_0000;
/// x86-64 reset vector: first instruction fetch after power-on.
pub const RESET_VECTOR: u64 = 0xFFFF_FFF0;
/// Size of the ROM window.
pub const ROM_SIZE: u64 = 0x0100_0000;

/// MMU-backed guest memory: the paged path the CPU core talks to.
pub struct GuestMemory {
    /// Guest-physical bus (RAM, ROM, MMIO).
    pub bus: Bus,
    /// Page-table walker rooted at the bootstrap CR3.
    pub mmu: Mmu,
    /// Bumped on every translation change (CR3 reload, INVLPG, flush) so
    /// translation caches (the JIT's env TLB) can validate cheaply.
    pub epoch: u64,
}

impl std::fmt::Debug for GuestMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestMemory")
            .field("bus", &self.bus)
            .field("mmu", &self.mmu)
            .finish()
    }
}

pub(crate) fn map_fault(e: AccessFault) -> AccessError {
    match e {
        AccessFault::PageFault { addr, code } => AccessError::PageFault { addr, code },
        AccessFault::NonCanonical { addr } => {
            AccessError::GeneralProtection(format!("non-canonical guest address {addr:#x}"))
        }
        AccessFault::Bus(e) => AccessError::Host(e.to_string()),
    }
}

impl MemoryAccess for GuestMemory {
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, AccessError> {
        let w = AccessWidth::from_size(width)
            .ok_or_else(|| AccessError::Host(format!("bad access width {width}")))?;
        self.mmu.read(&self.bus, w, addr).map_err(map_fault)
    }

    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), AccessError> {
        let w = AccessWidth::from_size(width)
            .ok_or_else(|| AccessError::Host(format!("bad access width {width}")))?;
        self.mmu.write(&self.bus, w, addr, value).map_err(map_fault)
    }

    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, AccessError> {
        let mut filled = 0usize;
        while filled < out.len() {
            let va = addr + filled as u64;
            let pa = match self.mmu.translate(&self.bus, va, AccessKind::Exec) {
                Ok(pa) => pa,
                // Early stop when at least one byte is available; the decoder
                // faults if the truncated window cannot hold the instruction.
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

    // Translation-change hooks: route to the MMU and bump the epoch so
    // translation caches (the JIT's env TLB) can invalidate precisely.
    // (Not overriding these was a latent bug: `MOV CR3`/`INVLPG` through
    // the single-vCPU path used to leave a stale MMU TLB behind.)
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

/// Install the shared bootstrap address space on `bus`:
///
/// * first 2 MiB — executable boot region (images load here);
/// * remaining RAM — read/write, non-executable;
/// * console + IPI MMIO windows — identity-mapped RW/NX;
/// * ROM window — read-only, final 2 MiB executable for the reset vector.
///
/// Both the single-vCPU [`System`] and the SMP machine build on this, so
/// every core of every machine boots into the same layout.
pub(crate) fn install_bootstrap(bus: &Bus, cfg: &PdgConfig) -> anyhow::Result<()> {
    let ram = u64::from(cfg.memory.ram_mib) * 1024 * 1024;
    let m2: u64 = 2 * 1024 * 1024;
    let mut pt = PageTableBuilder::new(bus, cfg.memory.pml4_base, cfg.memory.page_frame_alloc_base);
    // First 2 MiB: executable boot region (boot images live here).
    pt.map_2m(0, 0, PageFlags::PRESENT | PageFlags::RW)?;
    // Remaining RAM: read/write, non-executable.
    let mut base = m2;
    while base < ram {
        pt.map_2m(
            base,
            base,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
        )?;
        base += m2;
    }
    // Device MMIO windows (identity-mapped, RW/NX). The console and the
    // IPI device usually share one 2 MiB leaf — map it only once.
    let console_page = cfg.memory.console_mmio_base & !(m2 - 1);
    let ipi_page = cfg.memory.ipi_mmio_base & !(m2 - 1);
    pt.map_2m(
        console_page,
        console_page,
        PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
    )?;
    if ipi_page != console_page {
        pt.map_2m(
            ipi_page,
            ipi_page,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
        )?;
    }
    // ROM window: read-only; the final 2 MiB stays executable so the
    // reset vector (0xFFFF_FFF0) can run.
    let mut rom = ROM_BASE;
    while rom < ROM_BASE + ROM_SIZE - m2 {
        pt.map_2m(rom, rom, PageFlags::PRESENT | PageFlags::NX)?;
        rom += m2;
    }
    pt.map_2m(rom, rom, PageFlags::PRESENT)?;
    Ok(())
}

/// Result of a `System::run`.
#[derive(Debug, Clone)]
pub struct BootReport {
    /// Guest execution outcome.
    pub outcome: RunOutcome,
    /// Everything the guest wrote to the MMIO console.
    pub console: String,
}

/// One emulated PS5-shaped machine (Phase 1: single vCPU).
pub struct System {
    /// Paged guest memory (bus + MMU).
    pub mem: GuestMemory,
    /// The (single, for now) logical processor.
    pub cpu: Interpreter,
    /// Boot console device (also registered in the bus MMIO window).
    pub console: Arc<ConsoleDevice>,
    /// Guest-physical DMA engine.
    pub dma: DmaEngine,
    /// Inter-component event bus.
    pub events: EventBus,
    /// Virtual clock at the configured CPU frequency.
    pub clock: Clock,
    /// Active configuration snapshot.
    pub config: PdgConfig,
    entry: u64,
}

impl System {
    /// Build a machine from `config`, installing the bootstrap address space.
    pub fn new(config: PdgConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let ram = u64::from(config.memory.ram_mib) * 1024 * 1024;

        let mut bus = Bus::new(ram as usize)?;
        let console = Arc::new(ConsoleDevice::new());
        bus.attach_mmio(config.memory.console_mmio_base, 0x1000, console.clone());

        // Bootstrap page tables: genuine x86-64 structures in guest RAM
        // (shared with the SMP machine — see `install_bootstrap`).
        install_bootstrap(&bus, &config)?;

        Ok(Self {
            mem: GuestMemory {
                bus,
                mmu: Mmu::new(config.memory.pml4_base),
                epoch: 0,
            },
            cpu: Interpreter::new(CpuState::default()),
            console,
            dma: DmaEngine::new(),
            events: EventBus::new(),
            clock: Clock::new(config.cpu.freq_hz),
            config,
            entry: 0,
        })
    }

    /// Map a raw boot image at `base` and reset the CPU to it. Returns the
    /// entry point.
    pub fn boot_image(&mut self, image: &[u8], base: u64) -> anyhow::Result<u64> {
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
        self.mem.bus.ram_write_buf(base, image)?;
        self.reset_cpu(base);
        self.entry = base;
        Ok(base)
    }

    /// Load a PS5 BIOS/firmware dump (user-supplied) into the ROM window and
    /// reset the CPU to the architectural reset vector.
    ///
    /// The image is mapped to **end at the 4 GiB boundary**, so the reset
    /// vector (0xFFFF_FFF0) always lands inside the image tail — the same
    /// placement convention PC firmware uses.
    pub fn load_bios(&mut self, path: &std::path::Path) -> anyhow::Result<u64> {
        let data = std::fs::read(path).with_context(|| {
            format!(
                "failed to read BIOS/firmware image `{}` — PDG does not distribute firmware; \
                 place your own legally-dumped PS5 BIOS there",
                path.display()
            )
        })?;
        if data.is_empty() {
            anyhow::bail!("BIOS image `{}` is empty", path.display());
        }
        if data.len() as u64 > ROM_SIZE {
            anyhow::bail!(
                "BIOS image `{}` is {} bytes; the ROM window holds at most {ROM_SIZE:#x}",
                path.display(),
                data.len()
            );
        }
        let rom_base = ROM_BASE + ROM_SIZE - data.len() as u64;
        self.mem.bus.attach_rom(rom_base, data);
        self.reset_cpu(RESET_VECTOR);
        self.entry = RESET_VECTOR;
        Ok(RESET_VECTOR)
    }

    fn reset_cpu(&mut self, entry: u64) {
        let state = CpuState::new(entry, self.config.boot.stack_top);
        self.cpu = Interpreter::new(state);
    }

    /// Attach an HLE syscall handler to the CPU.
    pub fn set_syscall_hook(&mut self, hook: Box<dyn SyscallHook>) {
        self.cpu.set_syscall_hook(hook);
    }

    /// Current architectural CPU state (for debug tooling and tests).
    pub fn cpu_state(&self) -> &CpuState {
        &self.cpu.state
    }

    /// Entry point of the currently loaded image.
    pub fn entry(&self) -> u64 {
        self.entry
    }

    /// Run the machine for up to `max_instructions` guest instructions.
    pub fn run(&mut self, max_instructions: u64) -> BootReport {
        self.events
            .publish(Event::BootStarted { entry: self.entry });
        let outcome = self.cpu.run(&mut self.mem, max_instructions);
        match &outcome {
            RunOutcome::Halted { instructions, .. } => {
                self.events.publish(Event::Halted {
                    instructions: *instructions,
                });
            }
            RunOutcome::Fault(f) => {
                self.events.publish(Event::Fault {
                    reason: f.to_string(),
                });
            }
            RunOutcome::MaxInstructions { .. } => {}
        }
        BootReport {
            outcome,
            console: self.console.output(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boots_a_trivial_image() {
        let mut sys = System::new(PdgConfig::default()).unwrap();
        let entry = sys.boot_image(&[0xF4], 0x0010_0000).unwrap(); // hlt
        assert_eq!(entry, 0x0010_0000);
        let report = sys.run(10);
        match report.outcome {
            RunOutcome::Halted { instructions, rip } => {
                assert_eq!(instructions, 1);
                assert_eq!(rip, 0x0010_0000);
            }
            other => panic!("expected Halted, got {other:?}"),
        }
    }

    #[test]
    fn guest_writes_to_mmio_console_through_paging() {
        // mov rax, 'P' ; mov [0xF0000000], rax ; hlt
        // (moffs64 store: 64-bit mode sign-extends disp32, so the
        // absolute address must use the true 64-bit moffs form)
        let code = [
            0x48, 0xC7, 0xC0, 0x50, 0x00, 0x00, 0x00, // mov rax, 0x50
            0x48, 0xA3, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00,
            0x00, // mov [0xF0000000], rax
            0xF4,
        ];
        let mut sys = System::new(PdgConfig::default()).unwrap();
        sys.boot_image(&code, 0x0010_0000).unwrap();
        let report = sys.run(100);
        assert!(
            matches!(report.outcome, RunOutcome::Halted { .. }),
            "got {:?}",
            report.outcome
        );
        assert_eq!(report.console, "P");
    }

    #[test]
    fn write_to_rom_window_faults() {
        // mov rax, 0x41 ; mov [0xFF000000], rax ; hlt — the ROM pages are
        // mapped read-only, so the MMU raises an architectural #PF
        // (present | write) before the bus ever sees the store.
        let code = [
            0x48, 0xC7, 0xC0, 0x41, 0x00, 0x00, 0x00, // mov rax, 0x41
            0x48, 0xA3, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00,
            0x00, // mov [0xFF000000], rax
            0xF4,
        ];
        let mut sys = System::new(PdgConfig::default()).unwrap();
        sys.boot_image(&code, 0x0010_0000).unwrap();
        let report = sys.run(100);
        match report.outcome {
            RunOutcome::Fault(pdg_cpu::Fault::PageFault { addr, code }) => {
                assert_eq!(addr, 0xFF00_0000);
                assert_eq!(code, 0x03, "present | write");
            }
            other => panic!("expected ROM write page fault, got {other:?}"),
        }
    }

    #[test]
    fn missing_bios_reports_actionable_error() {
        let mut sys = System::new(PdgConfig::default()).unwrap();
        let err = sys
            .load_bios(std::path::Path::new("/nonexistent/bios.bin"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("legally-dumped"),
            "error should guide the user: {err}"
        );
    }
}
