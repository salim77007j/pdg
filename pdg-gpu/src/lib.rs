//! # pdg-gpu
//!
//! GPU subsystem for the PDG PlayStation 5 emulator (AMD RDNA 2, 36 CUs).
//!
//! Phase 3 delivers the translation stack end to end:
//!
//! * [`pm4`] — the PM4 command processor: packet framing, the verified
//!   opcode table, the gfx10.3 register model (offsets from the public
//!   kernel `gc_10_3_0` headers), and a CP that executes command streams
//!   into semantic [`ir::GpuCommand`]s.
//! * [`gcn`] — the shader recompilation path: a GCN/RDNA instruction
//!   decoder (encoding layouts verified against LLVM's AMDGPU backend
//!   definitions), a lifter to SPIR-V, naga-based SPIR-V→WGSL
//!   validation, and a content-addressed on-disk shader cache.
//! * [`ring`] — the GNM ring MMIO device that lets the emulated CPU
//!   submit command buffers through the memory bus, doorbell included.
//! * The [`GpuDevice`] seam — backends render [`ir::GpuCommand`] batches
//!   against a [`ir::GpuContext`]. [`NullGpu`] is the honest do-nothing
//!   backend used in tests without an adapter.
//!
//! Every table in this crate is verified against public AMD
//! documentation (Linux kernel `soc15d.h` / `gc_10_3_0_offset.h` /
//! `navi10_ip_offset.h`, LLVM AMDGPU `.td` files, shadPS4's Liverpool
//! documentation). Opcodes and shader instructions outside the verified
//! sets are refused with precise errors — never silently faked.

pub mod gcn;
pub mod ir;
pub mod pm4;
pub mod ring;
pub mod wgpu_be;

pub use wgpu_be::WgpuBackend;

use thiserror::Error;

/// Errors raised by GPU backends.
#[derive(Debug, Error)]
pub enum GpuError {
    /// The requested operation is not implemented by this backend.
    #[error("gpu backend `{backend}` does not implement: {what}")]
    Unsupported {
        /// Backend that raised the error.
        backend: &'static str,
        /// What was attempted.
        what: String,
    },
    /// The backend could not initialize (no adapter, device lost, ...).
    #[error("gpu backend `{backend}` failed to initialize: {what}")]
    Init {
        /// Backend that raised the error.
        backend: &'static str,
        /// What went wrong.
        what: String,
    },
    /// The guest programmed state the backend cannot honor yet.
    #[error("gpu backend `{backend}` refuses state: {what}")]
    BadState {
        /// Backend that raised the error.
        backend: &'static str,
        /// What was wrong.
        what: String,
    },
}

/// A GPU backend: renders [`ir::GpuCommand`] batches against the typed
/// [`ir::GpuContext`] snapshot, using `mem` for every memory-backed
/// effect (target readback into guest RAM, fence writes).
///
/// Ordering contract: a backend reports `Ok` only when the effects of
/// the batch are visible — fences are written after rendering, so a CPU
/// polling the fence address after [`GpuDevice::submit`] returns is a
/// correct synchronization pattern.
pub trait GpuDevice: Send {
    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// Submit a batch of translated commands.
    fn submit(
        &mut self,
        ctx: &ir::GpuContext,
        commands: &[ir::GpuCommand],
        mem: &mut dyn pm4::cp::CpMemPort,
    ) -> Result<(), GpuError>;
}

/// Do-nothing backend used when no host adapter exists: it executes the
/// memory-side effects that do not depend on rendering (fences, copies)
/// and skips draws and dispatches with a trace line. It never pretends
/// to draw.
#[derive(Debug, Default)]
pub struct NullGpu;

impl GpuDevice for NullGpu {
    fn name(&self) -> &'static str {
        "null-gpu"
    }

    fn submit(
        &mut self,
        _ctx: &ir::GpuContext,
        commands: &[ir::GpuCommand],
        mem: &mut dyn pm4::cp::CpMemPort,
    ) -> Result<(), GpuError> {
        for command in commands {
            match command {
                ir::GpuCommand::Fence { addr, value, wide } => {
                    if *wide {
                        mem.write_u32(*addr, *value as u32);
                        mem.write_u32(addr + 4, (*value >> 32) as u32);
                    } else {
                        mem.write_u32(*addr, *value as u32);
                    }
                }
                ir::GpuCommand::Copy { dst, src, len } => {
                    let mut bytes = Vec::with_capacity(*len as usize);
                    for w in mem.read_words(*src, (*len as usize).div_ceil(4)) {
                        bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    bytes.truncate(*len as usize);
                    mem.write_bytes(*dst, &bytes);
                }
                ir::GpuCommand::Draw { .. } | ir::GpuCommand::Dispatch { .. } => {
                    tracing::trace!(
                        backend = "null-gpu",
                        ?command,
                        "no adapter: skipping render"
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::GpuCommand;
    use crate::pm4::cp::{CpMemPort, FakeMem};

    #[test]
    fn null_gpu_writes_fences_and_copies() {
        let mut gpu = NullGpu;
        let ctx = ir::GpuContext::new();
        let mut mem = FakeMem::default();
        for (i, w) in [0x11223344u32, 0x55667788u32].iter().enumerate() {
            mem.write_u32(0x8000 + (i as u64) * 4, *w);
        }
        gpu.submit(
            &ctx,
            &[
                GpuCommand::Fence {
                    addr: 0x9000,
                    value: 0xDEAD_BEEF_CAFE_F00D,
                    wide: true,
                },
                GpuCommand::Copy {
                    dst: 0xA000,
                    src: 0x8000,
                    len: 8,
                },
                GpuCommand::Draw {
                    params: Default::default(),
                },
            ],
            &mut mem,
        )
        .unwrap();
        assert_eq!(mem.read_u32(0x9000), 0xCAFE_F00D);
        assert_eq!(mem.read_u32(0x9004), 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(0xA000), 0x11223344);
        assert_eq!(mem.read_u32(0xA000 + 4), 0x55667788);
    }
}
