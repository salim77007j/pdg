//! # pdg-gpu
//!
//! GPU subsystem for the PDG PlayStation 5 emulator (AMD RDNA 2, 36 CUs).
//!
//! Phase 1 delivers the translation *framework* only:
//!
//! * [`pm4`] — the PM4 command-processor packet framing (type-0/type-3
//!   headers). Semantic opcode tables and the wgpu backend are Phase 3
//!   work, verified against public AMD command-processor documentation.
//! * [`GpuDevice`] — the trait the rest of the machine sees; [`NullGpu`]
//!   is the honest do-nothing implementation used until the real backend
//!   exists. It never pretends to draw.
//!
//! The full pipeline (PM4 packet decoding → wgpu command buffers, RDNA ISA
//! shader recompilation → SPIR-V → WGSL via naga, hardware RT mapping,
//! async compute queues) is specified in `docs/ARCHITECTURE.md`.

pub mod pm4;

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
}

/// High-level commands the translated guest queue resolves into.
///
/// Phase 3 replaces this enum with the full PM4-derived command set
/// (draws, dispatches, barriers, blits, ...); the shape exists now so the
/// device trait is stable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuCommand {
    /// Do nothing.
    Nop,
    /// Present the current framebuffer to swapchain image `index`.
    Present {
        /// Swapchain image index.
        index: u32,
    },
}

/// A GPU backend.
pub trait GpuDevice: Send {
    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// Submit a batch of translated commands.
    fn submit(&mut self, commands: &[GpuCommand]) -> Result<(), GpuError>;
}

/// Do-nothing backend used while the real one is under construction.
#[derive(Debug, Default)]
pub struct NullGpu;

impl GpuDevice for NullGpu {
    fn name(&self) -> &'static str {
        "null-gpu"
    }

    fn submit(&mut self, commands: &[GpuCommand]) -> Result<(), GpuError> {
        for command in commands {
            tracing::trace!(?command, backend = "null-gpu", "ignoring command");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_gpu_accepts_and_discards() {
        let mut gpu = NullGpu;
        gpu.submit(&[GpuCommand::Nop, GpuCommand::Present { index: 0 }])
            .unwrap();
    }
}
