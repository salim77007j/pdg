//! # pdg-kernel
//!
//! Kernel subsystem for the PDG PlayStation 5 emulator.
//!
//! The PS5 runs Orbis OS, a FreeBSD-derived system. PDG follows the
//! established HLE strategy: translate guest syscalls to host operations
//! instead of emulating the kernel binary.
//!
//! Phase 5 delivers the kernel plane:
//!
//! * [`vm`] — the process address space: guest page tables behind a shared
//!   MMU, `mmap`/`munmap`/`mprotect` over tracked regions, and the
//!   [`vm::ProcessMemory`] port the interpreter executes against.
//! * [`orbis`] — the Orbis syscall layer: verified FreeBSD-9.2-derived
//!   numbers, the honest-refusal posture, console + read-only-PFS file
//!   I/O, anonymous memory, `kqueue`/`kevent` event queues.
//! * [`loader`] — the ELF64 loader (per-segment permissions, RELATIVE and
//!   TPOFF64 relocations, TLS) and the SELF container unwrap (plain
//!   entries only; encryption/compression bits are refused, not guessed).
//! * [`pfs`] — the read-only PlayStation filesystem over a block device,
//!   plus the builder that produces images in the same layout.
//! * [`objects`] — event flags, mutexes, semaphores, equeues behind a
//!   handle-checked object table.
//! * [`msix`] — MSI-X-shaped interrupt plumbing: a guest-programmable
//!   vector table and delivery through the guest IDT at instruction
//!   boundaries.
//! * [`demo`] — the Phase-5 guest program: a hand-assembled ELF the
//!   emulated CPU runs to exercise every piece above end to end.
//! * [`exception`] / [`idt`] — architectural vector numbers and genuine
//!   x86-64 IDT gate encoding.
//! * [`syscall`] — the Phase-1 registry-based dispatcher framework
//!   (the Orbis layer implements `SyscallHook` directly; the registry
//!   remains the generic base for future tables).

pub mod demo;
pub mod exception;
pub mod idt;
pub mod loader;
pub mod msix;
pub mod objects;
pub mod orbis;
pub mod pfs;
pub mod syscall;
pub mod vm;

pub use exception::vector;
pub use idt::{ATTR_INTERRUPT_GATE, ATTR_TRAP_GATE, IdtGate};
pub use loader::{
    Dyn, DynSym, ElfError, ElfImage, LoadedImage, ProgramHeader, Rela, SelfContainer, SelfError,
    load, parse_executable,
};
pub use objects::{
    ClearMode, Equeue, EventFlag, EventFlagAttrs, KEvent, KMutex, KernelObject, ObjectError,
    ObjectKind, ObjectTable, Semaphore, WaitMode,
};
pub use orbis::OrbisKernel;
pub use pfs::{DirEntry, MemoryBlockDevice, Pfs, PfsBuilder, PfsError};
pub use syscall::{KernelError, SyscallDispatcher, SyscallNumber};
pub use vm::{ProcessMemory, VmError, VmManager};
