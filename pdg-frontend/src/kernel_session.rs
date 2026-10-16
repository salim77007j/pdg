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

// ---------------------------------------------------------------------------
// The Phase-6 runtime session: the kernel machine + threads + modules
// ---------------------------------------------------------------------------

/// Instructions per scheduler round per thread.
const RT_SLICE: u64 = 200;
/// Total instruction budget.
const RT_BUDGET: u64 = 80_000_000;
/// Guest RAM for the runtime session.
const RT_RAM_MIB: usize = 512;
/// Frame pool base/limit (the Phase-5 geometry).
const RT_FRAME_POOL: u64 = 0x1200_0000;
const RT_FRAME_LIMIT: u64 = 0x1F00_0000;
/// The PRX mapping arena (high user space, far from the demo images).
const RT_MODULE_BASE: u64 = 0x6000_0000_0000;
const RT_MODULE_LIMIT: u64 = 0x6000_0100_0000;
/// The mmap arena for thread TLS blocks.
const RT_MMAP_BASE: u64 = 0x5000_0000_0000;
const RT_MMAP_LIMIT: u64 = 0x5000_1000_0000;

/// How the runtime run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeOutcome {
    /// The guest called `exit(code)`.
    Exited {
        /// Process exit code.
        code: i32,
        /// Instructions executed.
        instructions: u64,
    },
    /// The last thread exited through `thr_exit`.
    LastThreadExited {
        /// Instructions executed.
        instructions: u64,
    },
    /// The CPU halted without exiting.
    Halted {
        /// Instructions executed.
        instructions: u64,
    },
    /// The guest faulted.
    Fault(String),
    /// The instruction budget ran out.
    BudgetExhausted,
}

/// Everything the runtime session observed.
#[derive(Debug, Clone)]
pub struct RuntimeReport {
    /// How the run ended.
    pub outcome: RuntimeOutcome,
    /// Console transcript (the pre-snapshot prefix included when the run
    /// resumed from a snapshot).
    pub console: String,
    /// Per-stage guest statuses (1 = pass).
    pub statuses: Vec<u64>,
    /// Syscalls dispatched.
    pub syscalls: u64,
    /// Live threads at the end (0 once everything exited).
    pub threads: usize,
    /// Modules in the table.
    pub modules: usize,
    /// Relocations the dynlib layer applied.
    pub relocations: usize,
    /// The number of threads the scheduler materialized.
    pub threads_created: usize,
}

/// One thread's CPU state at capture time.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadSnap {
    /// Thread id.
    pub tid: u64,
    /// General registers.
    pub gpr: [u64; 16],
    /// Instruction pointer.
    pub rip: u64,
    /// Flags.
    pub rflags: u64,
    /// FS base (this thread's TLS block).
    pub fs_base: u64,
    /// The IDT base and limit (an installed table survives).
    pub idtr: (u64, u16),
}

/// A session snapshot: everything not derivable from reconstructing the
/// fresh machine — the CPU contexts, the nonzero RAM pages, the kernel
/// tables and the console transcript.
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    /// Threads, in creation order (main first).
    pub threads: Vec<ThreadSnap>,
    /// Nonzero 4 KiB guest-physical pages.
    pub ram: Vec<(u64, Vec<u8>)>,
    /// The kernel tables.
    pub kernel: pdg_kernel::orbis::KernelSnapshot,
    /// Console output so far.
    pub console: String,
}

impl SessionSnapshot {
    /// Serialize the whole machine to bytes: `"PDGS"`, version 1, the
    /// thread contexts, the sparse RAM pages, the kernel tables, the
    /// console transcript, and a SHA-256 digest over everything before
    /// it (corruption is refused, not recovered).
    pub fn to_bytes(&self) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let mut o = Vec::new();
        o.extend_from_slice(b"PDGS");
        o.extend_from_slice(&1u32.to_le_bytes());
        o.extend_from_slice(&(self.threads.len() as u32).to_le_bytes());
        for t in &self.threads {
            o.extend_from_slice(&t.tid.to_le_bytes());
            for g in &t.gpr {
                o.extend_from_slice(&g.to_le_bytes());
            }
            o.extend_from_slice(&t.rip.to_le_bytes());
            o.extend_from_slice(&t.rflags.to_le_bytes());
            o.extend_from_slice(&t.fs_base.to_le_bytes());
            o.extend_from_slice(&t.idtr.0.to_le_bytes());
            o.extend_from_slice(&t.idtr.1.to_le_bytes());
        }
        o.extend_from_slice(&(self.ram.len() as u32).to_le_bytes());
        for (pa, page) in &self.ram {
            o.extend_from_slice(&pa.to_le_bytes());
            o.extend_from_slice(&(page.len() as u32).to_le_bytes());
            o.extend_from_slice(page);
        }
        let k = self.kernel.to_bytes();
        o.extend_from_slice(&(k.len() as u32).to_le_bytes());
        o.extend_from_slice(&k);
        o.extend_from_slice(&(self.console.len() as u64).to_le_bytes());
        o.extend_from_slice(self.console.as_bytes());
        let digest = Sha256::digest(&o);
        o.extend_from_slice(&digest);
        o
    }

    /// Decode [`SessionSnapshot::to_bytes`]; `None` on any mismatch.
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        use sha2::{Digest, Sha256};
        if b.len() < 4 + 4 + 32 || &b[..4] != b"PDGS" {
            return None;
        }
        let body_len = b.len() - 32;
        let digest = Sha256::digest(&b[..body_len]);
        if digest.as_slice() != &b[body_len..] {
            return None;
        }
        let b = &b[..body_len];
        let mut p = 4usize;
        macro_rules! rd_u16 {
            () => {{
                let v = u16::from_le_bytes(b.get(p..p + 2)?.try_into().ok()?);
                p += 2;
                v
            }};
        }
        macro_rules! rd_u32 {
            () => {{
                let v = u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?);
                p += 4;
                v
            }};
        }
        macro_rules! rd_u64 {
            () => {{
                let v = u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?);
                p += 8;
                v
            }};
        }
        let version = rd_u32!();
        if version != 1 {
            return None;
        }
        let n = rd_u32!();
        let mut threads = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let tid = rd_u64!();
            let mut gpr = [0u64; 16];
            for g in &mut gpr {
                *g = rd_u64!();
            }
            let rip = rd_u64!();
            let rflags = rd_u64!();
            let fs_base = rd_u64!();
            let idtr_base = rd_u64!();
            let idtr_limit = rd_u16!();
            threads.push(ThreadSnap {
                tid,
                gpr,
                rip,
                rflags,
                fs_base,
                idtr: (idtr_base, idtr_limit),
            });
        }
        let n = rd_u32!();
        let mut ram = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let pa = rd_u64!();
            let len = rd_u32!() as usize;
            let page = b.get(p..p + len)?.to_vec();
            p += len;
            ram.push((pa, page));
        }
        let klen = rd_u32!() as usize;
        let kernel = pdg_kernel::orbis::KernelSnapshot::from_bytes(b.get(p..p + klen)?)?;
        p += klen;
        let clen = rd_u64!() as usize;
        let console = String::from_utf8(b.get(p..p + clen)?.to_vec()).ok()?;
        p += clen;
        let _ = p;
        Some(Self {
            threads,
            ram,
            kernel,
            console,
        })
    }
}

/// The Phase-6 runtime machine.
pub struct RuntimeSession {
    bus: Arc<Bus>,
    console: Arc<ConsoleDevice>,
    kernel: OrbisKernel,
    mmu: std::sync::Arc<std::sync::Mutex<pdg_memory::Mmu>>,
    entry: u64,
    tls_base: u64,
    relocations: usize,
    modules: usize,
}

impl std::fmt::Debug for RuntimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSession")
            .field("entry", &format_args!("{:#x}", self.entry))
            .field("tls", &format_args!("{:#x}", self.tls_base))
            .finish_non_exhaustive()
    }
}

/// One materialized thread: its id and its interpreter.
struct ThreadInterp {
    tid: u64,
    interp: Interpreter,
}

impl RuntimeSession {
    /// Build the machine: the Phase-5 geometry plus the dynlib-loaded main
    /// module (its `DT_NEEDED` closure with it), both thread stacks, and
    /// the demo PFS.
    pub fn new(ssd_path: &Path) -> anyhow::Result<Self> {
        use pdg_kernel::demo::layout as l5;
        use pdg_kernel::demo::layout6 as l6;
        if !ssd_path.exists() {
            std::fs::write(ssd_path, vec![0u8; 4 * 1024 * 1024])
                .context("creating a fresh SSD image")?;
        }
        let mut bus = Bus::new(RT_RAM_MIB * 1024 * 1024)?;
        let console = Arc::new(ConsoleDevice::new());
        bus.attach_mmio(l5::CONSOLE, 0x1000, console.clone());
        let nvme = NvmeController::new(HostFileBlockDevice::open(ssd_path)?).shared();
        bus.attach_mmio(l5::NVME, NVME_FOOTPRINT, nvme.clone());
        let msix = MsixTable::new(4).shared();
        bus.attach_mmio(l5::MSIX, 0x1000, msix.clone());
        {
            let msix2 = msix.clone();
            nvme.set_completion_hook(Arc::new(move |_posted| {
                msix2.fire(0);
            }));
        }

        // The PFS is created standalone so the dynlib layer can read the
        // module closure through it before it is mounted on the kernel.
        let mut pfs = Pfs::mount(Box::new(MemoryBlockDevice::new(
            demo::runtime_demo_pfs_image(),
        )))
        .context("mounting the runtime demo PFS")?;

        // The address space and the module table.
        let bus = Arc::new(bus);
        let mut vm = VmManager::new(
            bus.clone(),
            RT_FRAME_POOL,
            RT_FRAME_POOL,
            RT_FRAME_LIMIT,
            RT_MMAP_BASE,
            RT_MMAP_LIMIT,
        );
        let mut dynlib =
            pdg_kernel::dynlib::Dynlib::new(RT_MODULE_BASE, RT_MODULE_LIMIT, l6::TLS_ARENA);
        let rw_nx = PageFlags::PRESENT | PageFlags::RW | PageFlags::USER | PageFlags::NX;
        // Both thread stacks.
        vm.map_anon(l6::STACK_TOP - l6::STACK_LEN, l6::STACK_LEN, rw_nx, true)
            .context("mapping the main stack")?;
        vm.map_anon(
            l6::WORKER_STACK_TOP - l6::WORKER_STACK_LEN,
            l6::WORKER_STACK_LEN,
            rw_nx,
            true,
        )
        .context("mapping the worker stack")?;

        // Load the main module (needed closure rides along through the
        // PFS).
        let elf = loader::parse_executable(&demo::runtime_demo_elf())
            .context("building the runtime demo ELF")?;
        let mut source = pdg_kernel::dynlib::PfsModuleSource::new(&mut pfs);
        let entry = dynlib
            .load_main(elf, &mut vm, &mut source)
            .context("loading the runtime demo main module")?;
        let tls_base = dynlib.tls_base();
        let relocations: usize = dynlib
            .modules()
            .iter()
            .map(|m| m.image.relocations_applied)
            .sum();
        let modules = dynlib.modules().len();
        let mmu = vm.shared_mmu();

        let kernel = OrbisKernel::with_dynlib(vm, console.clone(), l6::DEMO_PID, dynlib);
        kernel.mount_pfs(pfs);
        Ok(Self {
            bus,
            console,
            kernel,
            mmu,
            entry,
            tls_base,
            relocations,
            modules,
        })
    }

    /// The shared console (the launcher streams it into the GUI).
    pub fn console(&self) -> Arc<ConsoleDevice> {
        self.console.clone()
    }

    /// Run the process, scheduling threads round-robin in 200-instruction
    /// slices. When `snapshot_stage` names a stage whose status turned to
    /// 1, the session captures a snapshot between rounds and continues.
    pub fn run(&self, snapshot_stage: Option<u64>) -> (RuntimeReport, Option<SessionSnapshot>) {
        use pdg_kernel::demo::layout6 as l6;
        let mut interp = Interpreter::new(CpuState::new(self.entry, l6::STACK_TOP));
        interp.state.fs_base = self.tls_base;
        interp.set_syscall_hook(Box::new(self.kernel.clone()));
        let threads = vec![ThreadInterp {
            tid: pdg_kernel::thread::MAIN_TID,
            interp,
        }];
        let (outcome, threads_created, snapshot) = self.drive(threads, 1, snapshot_stage);
        (self.report(outcome, threads_created), snapshot)
    }

    /// The scheduler: run every runnable thread one slice per round,
    /// materialize spawns, retire exits, watch the budget. `break 'sched`
    /// ends the process.
    fn drive(
        &self,
        mut threads: Vec<ThreadInterp>,
        mut threads_created: usize,
        snapshot_stage: Option<u64>,
    ) -> (RuntimeOutcome, usize, Option<SessionSnapshot>) {
        let mut mem = ProcessMemory::new(self.bus.clone(), self.mmu.clone());
        let mut instructions = 0u64;
        let mut snapshot: Option<SessionSnapshot> = None;
        let mut snap_taken = snapshot_stage.is_none();
        let outcome = 'sched: loop {
            // 1. Materialize spawned threads (thr_create's records).
            for sp in self.kernel.take_spawned_threads() {
                let mut interp = Interpreter::new(CpuState::new(sp.rip, sp.rsp));
                interp.state.fs_base = sp.tls_base;
                interp.set_syscall_hook(Box::new(self.kernel.clone()));
                interp.state.gpr[pdg_cpu::state::RDI] = sp.rdi;
                interp.state.rflags = sp.rflags;
                threads.push(ThreadInterp {
                    tid: sp.tid,
                    interp,
                });
                threads_created += 1;
            }
            // 2. Run one slice of every runnable thread.
            for tid in self.kernel.runnable_threads() {
                let Some(slot) = threads.iter_mut().find(|t| t.tid == tid) else {
                    continue;
                };
                self.kernel.set_current_thread(tid);
                match slot.interp.run(&mut mem, RT_SLICE) {
                    RunOutcome::MaxInstructions { instructions: n } => {
                        instructions += n;
                    }
                    RunOutcome::Halted {
                        instructions: n, ..
                    } => {
                        instructions += n;
                        // A guest thread halting is that thread's end
                        // (the documented PDG model).
                        self.kernel.thread_halt(tid);
                    }
                    RunOutcome::Fault(Fault::ProcessExit { code }) => {
                        instructions += RT_SLICE;
                        break 'sched RuntimeOutcome::Exited { code, instructions };
                    }
                    RunOutcome::Fault(Fault::ThreadExit { .. }) => {
                        // The table already marked it exited; the retire
                        // step below drops its interpreter.
                        instructions += RT_SLICE;
                    }
                    RunOutcome::Fault(Fault::ThreadParked { .. }) => {
                        // The syscall's return value was written before
                        // parking; the thread resumes when woken.
                    }
                    RunOutcome::Fault(f) => {
                        break 'sched RuntimeOutcome::Fault(f.to_string());
                    }
                }
            }
            // 3. Retire exited threads.
            threads.retain(|t| !self.kernel.thread_exited(t.tid));
            // 4. The process ends when its last thread is gone.
            if threads.is_empty() {
                if self.kernel.threads_all_exited() {
                    break 'sched RuntimeOutcome::LastThreadExited { instructions };
                }
                break 'sched RuntimeOutcome::Halted { instructions };
            }
            // 5. The snapshot point: after the named stage passed.
            if !snap_taken && self.stage_passed(snapshot_stage.unwrap()) {
                snapshot = Some(self.capture(&threads));
                snap_taken = true;
            }
            if instructions >= RT_BUDGET {
                break 'sched RuntimeOutcome::BudgetExhausted;
            }
        };
        (outcome, threads_created, snapshot)
    }

    /// Whether the guest set stage `stage`'s status to 1 (the snapshot
    /// trigger).
    fn stage_passed(&self, stage: u64) -> bool {
        use pdg_kernel::demo::layout6 as l6;
        let mut buf = [0u8; 8];
        self.kernel
            .read_guest(l6::STATUS + 8 * stage, &mut buf)
            .map(|()| u64::from_le_bytes(buf) == 1)
            .unwrap_or(false)
    }

    /// Capture the machine: CPU contexts, nonzero RAM pages, the kernel
    /// tables, the console transcript.
    fn capture(&self, threads: &[ThreadInterp]) -> SessionSnapshot {
        const PAGE: usize = 0x1000;
        let mut ram = Vec::new();
        for pa in (0..self.bus.ram_len()).step_by(PAGE) {
            let mut page = vec![0u8; PAGE];
            self.bus.read_buf(pa, &mut page).expect("guest RAM read");
            if page.iter().any(|b| *b != 0) {
                ram.push((pa, page));
            }
        }
        SessionSnapshot {
            threads: threads
                .iter()
                .map(|t| ThreadSnap {
                    tid: t.tid,
                    gpr: t.interp.state.gpr,
                    rip: t.interp.state.rip,
                    rflags: t.interp.state.rflags,
                    fs_base: t.interp.state.fs_base,
                    idtr: (t.interp.state.idtr.base, t.interp.state.idtr.limit),
                })
                .collect(),
            ram,
            kernel: self.kernel.snapshot_state(),
            console: self.console.output(),
        }
    }

    /// Restore a snapshot over this freshly constructed machine: RAM
    /// pages, kernel tables, console transcript. The `run` resumes from
    /// the captured CPU contexts.
    pub fn restore(&self, snap: &SessionSnapshot) {
        for (pa, page) in &snap.ram {
            let _ = self.bus.write_buf(*pa, page);
        }
        self.kernel.restore_state(&snap.kernel);
        self.console.set_output(&snap.console);
        self.mmu.lock().expect("mmu").flush_tlb();
    }

    /// Run resumed from a snapshot: the interpreters are rebuilt from the
    /// captured CPU contexts.
    pub fn run_restored(&self, snap: &SessionSnapshot) -> RuntimeReport {
        let mut threads: Vec<ThreadInterp> = Vec::new();
        for t in &snap.threads {
            let mut state = CpuState::new(t.rip, t.gpr[pdg_cpu::state::RSP]);
            state.gpr = t.gpr;
            state.rflags = t.rflags;
            state.fs_base = t.fs_base;
            state.idtr.base = t.idtr.0;
            state.idtr.limit = t.idtr.1;
            let mut interp = Interpreter::new(state);
            interp.set_syscall_hook(Box::new(self.kernel.clone()));
            threads.push(ThreadInterp { tid: t.tid, interp });
        }
        let (outcome, threads_created, _) = self.drive(threads, snap.threads.len(), None);
        self.report(outcome, threads_created)
    }

    /// Assemble the report from the final machine state.
    fn report(&self, outcome: RuntimeOutcome, threads_created: usize) -> RuntimeReport {
        use pdg_kernel::demo::layout6 as l6;
        let mut statuses = Vec::with_capacity(l6::ST_COUNT as usize);
        for i in 0..l6::ST_COUNT {
            let mut buf = [0u8; 8];
            let va = l6::STATUS + i * 8;
            let v = self
                .kernel
                .read_guest(va, &mut buf)
                .ok()
                .map(|()| u64::from_le_bytes(buf))
                .unwrap_or(0);
            statuses.push(v);
        }
        RuntimeReport {
            outcome,
            console: self.console.output(),
            statuses,
            syscalls: self.kernel.syscall_count(),
            threads: self.kernel.thread_count(),
            modules: self.modules,
            relocations: self.relocations,
            threads_created,
        }
    }
}
