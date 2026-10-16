//! NVMe-shaped SSD model (Phase 4 storage).
//!
//! The PS5's custom SSD speaks an NVMe-shaped command protocol to the
//! host console. PDG models that contract faithfully at the queue level:
//!
//! * [`regs`] — the controller register file (BAR0-shaped window: CAP/VS/
//!   CC/CSTS/AQA/ASQ/ACQ) and the doorbell array, at offsets taken from
//!   the public NVMe 2.0 base specification.
//! * [`cmd`] — the 64-byte submission entries, 16-byte completion entries,
//!   opcodes and status codes the queue protocol carries.
//! * [`prp`] — Physical Region Page translation: the DMA model commands
//!   use to move data into and out of guest RAM (first pointer may be
//!   page-relative; PRP2 is either the second page or a pointer to a
//!   chain of page addresses).
//! * [`identify`] — the IDENTIFY Controller / Namespace structures (the
//!   subset PDG fills is documented field-by-field; unknown fields are
//!   zero, never invented).
//! * [`controller`] — the [`NvmeController`] itself: an
//!   [`MmioDevice`](pdg_memory::bus::MmioDevice) whose doorbell writes
//!   latch submissions and whose [`NvmeController::pump`] executes them
//!   against a [`BlockDevice`](crate::block::BlockDevice) with DMA into
//!   guest RAM.
//!
//! **Kraken/Oodle posture (a deliberate legal decision):** retail PS5
//! disc images and some SSD traffic are compressed with proprietary
//! codecs. PDG does not implement, embed, or link any proprietary
//! decompressor. The seam a future decompression pipeline would plug
//! into is the [`Decompressor`] trait below; users working with their
//! own legally-dumped, decrypted assets provide their own components.

use thiserror::Error;

/// Block size the model exposes (one LBA = 512 bytes, LBA format 0).
pub const LBA_SIZE: u64 = 512;

/// Namespace ID of the single exposed namespace.
pub const NAMESPACE_ID: u32 = 1;

/// Errors raised by the NVMe model.
#[derive(Debug, Error)]
pub enum NvmeError {
    /// Guest program error: invalid opcode / field / namespace / LBA.
    #[error("command refused: {0}")]
    Refused(String),
    /// Host-side I/O against the backing block device failed.
    #[error("backing device I/O: {0}")]
    Device(String),
    /// PRP walk could not cover the transfer inside guest RAM.
    #[error("prp walk: {0}")]
    Prp(String),
}

/// Decompression seam for compressed-storage pipelines.
///
/// The PS5 SSD feeds Kraken-compressed streams to the console; that
/// decompressor is proprietary and PDG will never ship one. The trait
/// exists so a *user* can wire their own legally-obtained component
/// into a future storage stack without PDG's internals changing shape.
pub trait Decompressor: Send {
    /// Decompress `input` into `output`; returns bytes written.
    fn decompress(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize, NvmeError>;
    /// Component name for diagnostics.
    fn name(&self) -> &'static str;
}

/// Pass-through "decompressor": the identity. Storage dumps that are
/// already decompressed (the only kind PDG documents supporting today)
/// need no decompression at all — this type makes that explicit.
#[derive(Debug, Default)]
pub struct IdentityDecompressor;

impl Decompressor for IdentityDecompressor {
    fn decompress(&mut self, _input: &[u8], _output: &mut [u8]) -> Result<usize, NvmeError> {
        Err(NvmeError::Refused(
            "identity decompressor: input is not compressed, do not call me".to_owned(),
        ))
    }
    fn name(&self) -> &'static str {
        "identity"
    }
}

pub mod cmd;
pub mod controller;
pub mod identify;
pub mod prp;
pub mod regs;

pub use cmd::{AdminOpcode, CompletionEntry, SubmissionEntry};
pub use controller::NvmeController;
pub use regs::{NVME_DOORBELL_BASE, NVME_FOOTPRINT};
