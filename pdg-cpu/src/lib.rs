//! # pdg-cpu
//!
//! CPU subsystem for the PDG PlayStation 5 emulator, modelling the console's
//! AMD Zen 2 x86-64 cores.
//!
//! Phase 1 delivers the execution *foundations*:
//!
//! * [`state::CpuState`] — the architectural register file: 16 GPRs
//!   (all encodings: 8/16/32/64-bit and high-byte views), RIP and RFLAGS.
//! * [`fault::MemoryAccess`] — a ports-and-adapters interface between the
//!   CPU core and the platform layer, so the core has no dependency on any
//!   concrete memory implementation (see `pdg-core::system` for the guest
//!   MMU-backed adapter, and [`flat::FlatMemory`] for a direct, no-MMU mode).
//! * [`interpreter::Interpreter`] — a fetch → decode (via `iced-x86`) →
//!   execute loop covering the integer heart of x86-64: data movement,
//!   arithmetic/logic with architectural flag semantics, shifts, the stack,
//!   control flow (conditional/unconditional branches, call/ret), `syscall`
//!   hooks, and `hlt`.
//!
//! What is deliberately **not** here yet (planned Phase 2): the full
//! instruction set (string ops, MUL/DIV, BMI, SSE/AVX), the JIT recompiler,
//! SMT/multi-core scheduling, and IDT-based exception delivery. The
//! interpreter treats every unimplemented instruction as an architectural
//! `#UD` fault, never as a silent no-op.

pub mod cpuid;
pub mod fault;
pub mod flat;
pub mod hooks;
pub mod interpreter;
pub mod simd;
pub mod state;

pub use fault::{AccessError, Fault, MemoryAccess};
pub use flat::FlatMemory;
pub use hooks::SyscallHook;
pub use interpreter::{Interpreter, RunOutcome};
pub use state::CpuState;

/// Re-export of the decoder crate so dependents (kernel, tools, tests) share
/// one `iced_x86` version without adding a direct dependency.
pub use iced_x86;

/// PS5 CPU frequency in Hz (AMD Zen 2, ~3.5 GHz variable).
pub const PS5_CPU_FREQ_HZ: u64 = 3_500_000_000;
