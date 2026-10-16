//! GCN/RDNA shader recompilation.
//!
//! PS5-class games ship precompiled native GPU ISA. PDG's shader path:
//!
//! ```text
//! guest ISA ──(decode)──► Inst ──(lifter)──► SPIR-V ──(naga)──► WGSL
//! ```
//!
//! * [`decode`] — the gfx10 instruction decoder + a test encoder
//!   (bit layouts verified against LLVM's public AMDGPU backend).
//! * [`opcodes`] — verified opcode tables per instruction space.
//! * [`enc`] — encoding formats and operand semantics.
//! * [`lifter`] — GCN subset → SPIR-V (built with `rspirv`).
//! * [`naga_fe`] — SPIR-V → WGSL translation + validation through naga.
//! * [`cache`] — content-addressed on-disk shader cache.
//!
//! The lifter implements the subset it can compile *correctly* and
//! refuses everything else with a precise error naming the instruction —
//! the same no-fake-implementations discipline as the CPU tier.

pub mod cache;
pub mod decode;
pub mod enc;
pub mod lifter;
pub mod naga_fe;
pub mod opcodes;

pub use decode::{DecodeError, Encoder, Inst, SmemOffset, decode_program};
pub use lifter::{LiftError, LiftInputs, Manifest, Stage, VertexFetchDecl, VertexFormat, lift};
