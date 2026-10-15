//! # pdg-kernel
//!
//! Kernel subsystem for the PDG PlayStation 5 emulator.
//!
//! The PS5 runs Orbis OS, a FreeBSD-derived system. PDG follows the
//! established HLE strategy: translate guest syscalls to host operations
//! instead of emulating the kernel binary.
//!
//! Phase 1 delivers the dispatch framework:
//!
//! * [`exception`] — x86-64 architectural vector numbers used when faults
//!   graduate from "reported" to "delivered through the guest IDT" (Phase 2).
//! * [`syscall`] — a registry-based [`SyscallDispatcher`] implementing
//!   [`pdg_cpu::SyscallHook`], with the FreeBSD-style calling convention
//!   (number in RAX, arguments in RDI/RSI/RDX/R10/R8/R9, result in RAX).
//!
//! The actual Orbis syscall numbering table is populated in Phase 5 from
//! public homebrew research; this crate deliberately contains **no guessed**
//! vendor constants.

pub mod exception;
pub mod syscall;

pub use exception::vector;
pub use syscall::{KernelError, SyscallDispatcher, SyscallNumber};
