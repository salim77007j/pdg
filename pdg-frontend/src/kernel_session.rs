//! The Phase-5 kernel session: a process-shaped machine.
//!
//! This is where everything meets: the session builds a guest-physical
//! bus with 512 MiB of RAM, attaches the console, the NVMe SSD (over the
//! user's image file) and the MSI-X-shaped vector table, creates a
//! process address space with an identity-mapped DMA window and MMIO
//! windows, mounts the read-only PFS, loads the demo ELF64 through the
//! loader (segments, relocations, TLS), and runs the CPU with the Orbis
//! syscall layer installed — the same doorbell/pump discipline every
//! earlier phase established: 200-instruction slices, devices and
//! interrupts serviced between them.
//!
//! The pump also delivers interrupts: the NVMe completion hook raises
//! the guest-programmed MSI-X vector, and the pump injects it through
//! the guest IDT when (and only when) the guest unmasked interrupts
//! and installed a table — exactly the contract real drivers rely on.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use pdg_cpu::state::rflags::IF;
use pdg_cpu::{CpuState, Fault, Interpreter, RunOutcome};
use pdg_io::block::HostFileBlockDevice;
use pdg_io::nvme::{NVME_FOOTPRINT, NvmeController};
use pdg_kernel::demo::{self, layout};
use pdg_kernel::loader;
use pdg_kernel::msix::MsixTable;
use pdg_kernel::orbis::{self, OrbisKernel};
use pdg_kernel::pfs::{MemoryBlockDevice, Pfs};
use pdg_kernel::vm::{ProcessMemory, VmManager};
use pdg_memory::PageFlags;
use pdg_memory::bus::Bus;
use pdg_memory::devices::ConsoleDevice;

/// Guest RAM for the session (the DMA window sits at 256 MiB).
const RAM_MIB: usize = 512;
/// Guest-physical base of the process PML4 and the frame pool above it.
const FRAME_POOL: u64 = 0x1200_0000;
/// End of the frame pool.
const FRAME_LIMIT: u64 = 0x1F00_0000;
/// Number of MSI-X entries exposed to the guest.
const MSIX_ENTRIES: usize = 4;
/// Instructions per pump slice.
const SLICE: u64 = 200;
/// Total instruction budget.
const BUDGET: u64 = 50_000_000;

/// How the process run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelOutcome {
    /// The guest called `exit(code)`.
    Exited {
        /// Process exit code.
        code: i32,
        /// Instructions executed.
        instructions: u64,
    },
    /// The CPU halted without exiting (no `exit` call).
    Halted {
        /// Instructions executed.
        instructions: u64,
    },
    /// The guest faulted.
    Fault(String),
    /// The instruction budget ran out.
    BudgetExhausted,
}

/// Everything the session observed.
#[derive(Debug, Clone)]
pub struct KernelReport {
    /// How the run ended.
    pub outcome: KernelOutcome,
    /// Console transcript.
    pub console: String,
    /// Per-stage guest statuses (1 = pass), indexed by
    /// [`layout`] stage constants.
    pub statuses: Vec<u64>,
    /// Syscalls dispatched.
    pub syscalls: u64,
    /// Whether an MSI-X interrupt was delivered through the guest IDT.
    pub interrupt_delivered: bool,
    /// Whether the guest's kevent event was triggered and drained.
    pub equeue_event_drained: bool,
    /// SSD statistics after the run.
    pub ssd_stats: pdg_io::nvme::controller::NvmeStats,
    /// Relocations the loader applied.
    pub relocations_applied: usize,
}

/// The machine plus its devices and kernel.
pub struct KernelSession {
    bus: Arc<Bus>,
    console: Arc<ConsoleDevice>,
    nvme: Arc<NvmeController<HostFileBlockDevice>>,
    msix: Arc<MsixTable>,
    kernel: OrbisKernel,
    mmu: std::sync::Arc<std::sync::Mutex<pdg_memory::Mmu>>,
    entry: u64,
    tls_vaddr: u64,
    relocations: usize,
    triggered: AtomicBool,
}

impl std::fmt::Debug for KernelSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelSession")
            .field("entry", &format_args!("{:#x}", self.entry))
            .field("tls", &format_args!("{:#x}", self.tls_vaddr))
            .finish_non_exhaustive()
    }
}

impl KernelSession {
    /// Build the session against `ssd_path` (created empty when missing —
    /// PDG never provides content, only the container the demo writes its
    /// own marker sector into).
    pub fn new(ssd_path: &Path) -> anyhow::Result<Self> {
        if !ssd_path.exists() {
            std::fs::write(ssd_path, vec![0u8; 4 * 1024 * 1024])
                .context("creating a fresh SSD image")?;
        }
        let mut bus = Bus::new(RAM_MIB * 1024 * 1024)?;
        let console = Arc::new(ConsoleDevice::new());
        bus.attach_mmio(layout::CONSOLE, 0x1000, console.clone());

        let nvme = NvmeController::new(HostFileBlockDevice::open(ssd_path)?).shared();
        bus.attach_mmio(layout::NVME, NVME_FOOTPRINT, nvme.clone());
        let msix = MsixTable::new(MSIX_ENTRIES).shared();
        bus.attach_mmio(layout::MSIX, 0x1000, msix.clone());

        // The completion hook raises MSI-X entry 0 — the vector the guest
        // programs — whenever the controller posts completions.
        {
            let msix = msix.clone();
            nvme.set_completion_hook(Arc::new(move |_posted| {
                msix.fire(0);
            }));
        }

        // Process address space.
        let bus = Arc::new(bus);
        let mut vm = VmManager::new(
            bus.clone(),
            FRAME_POOL,
            FRAME_POOL,
            FRAME_LIMIT,
            0x5000_0000_0000,
            0x5000_0010_0000,
        );
        let mmu = vm.shared_mmu();
        let rw_nx = PageFlags::PRESENT | PageFlags::RW | PageFlags::USER | PageFlags::NX;
        // DMA window (identity: devices address guest RAM directly).
        vm.map_identity_2m(layout::DMA_BASE, layout::DMA_LEN, rw_nx)
            .context("mapping the DMA window")?;
        // MMIO windows, identity-mapped.
        let m2 = 2 * 1024 * 1024;
        vm.map_identity_2m(layout::NVME, m2, rw_nx)
            .context("mapping the NVMe window")?;
        vm.map_identity_2m(layout::MSIX, m2, rw_nx)
            .context("mapping the MSI-X window")?;
        vm.map_identity_2m(layout::CONSOLE, m2, rw_nx)
            .context("mapping the console window")?;
        // Process stack.
        vm.map_anon(
            layout::STACK_TOP - layout::STACK_LEN,
            layout::STACK_LEN,
            rw_nx,
            true,
        )
        .context("mapping the stack")?;

        // Load the demo program.
        let elf =
            loader::parse_executable(&demo::kernel_demo_elf()).context("building the demo ELF")?;
        let module = loader::load(&elf, &mut vm, 0, None).context("loading the demo ELF")?;
        let tls_vaddr = module.tls.map_or(0, |t| t.vaddr);

        // The kernel: Orbis syscalls over the mounted PFS.
        let kernel = OrbisKernel::new(vm, console.clone(), layout::DEMO_PID);
        let pfs = Pfs::mount(Box::new(MemoryBlockDevice::new(demo::demo_pfs_image())))
            .context("mounting the demo PFS")?;
        kernel.mount_pfs(pfs);

        Ok(Self {
            bus,
            console,
            nvme,
            msix,
            kernel,
            mmu,
            entry: module.entry,
            tls_vaddr,
            relocations: module.relocations_applied,
            triggered: AtomicBool::new(false),
        })
    }

    /// Run the process, servicing devices, events and interrupts between
    /// 200-instruction slices.
    pub fn run(&self) -> KernelReport {
        let mut interp = Interpreter::new(CpuState::new(self.entry, layout::STACK_TOP));
        // The documented Phase-5 TLS model: FS base = TLS block base.
        interp.state.fs_base = self.tls_vaddr;
        interp.set_syscall_hook(Box::new(self.kernel.clone()));
        let mut mem = ProcessMemory::new(self.bus.clone(), self.mmu.clone());

        let mut instructions = 0u64;
        let mut interrupt_delivered = false;
        let outcome = loop {
            // 1. Pump the devices (NVMe completions fire the MSI-X hook).
            let _ = self.nvme.pump(&self.bus);

            // 2. Once the guest is waiting on its event queue, trigger the
            //    demo event (the host side of a kernel event source).
            if !self.triggered.load(Ordering::Relaxed)
                && self.kernel.kevent_call_count() >= 1
                && self.kernel.last_kqueue() != 0
            {
                let _ = self.kernel.trigger_equeue_event(
                    self.kernel.last_kqueue(),
                    layout::DEMO_IDENT,
                    orbis::ev::EVFILT_READ,
                    layout::DEMO_EVENT_DATA,
                );
                self.triggered.store(true, Ordering::Relaxed);
            }

            // 3. Deliver a pending interrupt through the guest IDT — only
            //    when the guest is ready (STI ran, LIDT installed).
            if let Some(vector) = self.msix.take_irq() {
                let interrupts_enabled = interp.state.rflags & IF != 0;
                let idt_installed = interp.state.idtr.base != 0;
                if interrupts_enabled && idt_installed {
                    match interp.inject_interrupt(vector, &mut mem) {
                        Ok(()) => interrupt_delivered = true,
                        Err(_) => self.msix.requeue(vector),
                    }
                } else {
                    self.msix.requeue(vector);
                }
            }

            // 4. Run a slice.
            let r = interp.run(&mut mem, SLICE);
            match &r {
                RunOutcome::Halted {
                    instructions: n, ..
                } => {
                    instructions += *n;
                    break KernelOutcome::Halted { instructions };
                }
                RunOutcome::Fault(Fault::ProcessExit { code }) => {
                    instructions += SLICE;
                    break KernelOutcome::Exited {
                        code: *code,
                        instructions,
                    };
                }
                RunOutcome::Fault(f) => break KernelOutcome::Fault(f.to_string()),
                RunOutcome::MaxInstructions { instructions: n } => {
                    instructions += *n;
                }
            }
            if instructions >= BUDGET {
                break KernelOutcome::BudgetExhausted;
            }
        };

        // Collect the stage statuses the guest left behind.
        let mut statuses = Vec::with_capacity(layout::ST_COUNT as usize);
        for i in 0..layout::ST_COUNT {
            let mut buf = [0u8; 8];
            let va = layout::STATUS + i * 8;
            let v = self
                .kernel
                .read_guest(va, &mut buf)
                .ok()
                .map(|()| u64::from_le_bytes(buf))
                .unwrap_or(0);
            statuses.push(v);
        }

        let equeue_drained = statuses
            .get(layout::ST_KEVENT as usize)
            .copied()
            .unwrap_or(0)
            == 1;

        KernelReport {
            outcome,
            console: self.console.output(),
            statuses,
            syscalls: self.kernel.syscall_count(),
            interrupt_delivered,
            equeue_event_drained: equeue_drained,
            ssd_stats: self.nvme.stats(),
            relocations_applied: self.relocations,
        }
    }
}
