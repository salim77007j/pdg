//! Error type shared by the memory subsystem.

use thiserror::Error;

/// Errors produced by guest-physical memory operations.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MemoryError {
    /// No RAM/ROM/MMIO region backs this guest-physical address.
    #[error("guest physical address {addr:#018x} is not backed by any region")]
    UnmappedPhysical { addr: u64 },

    /// The access targets a read-only region (ROM).
    #[error("write to read-only region at {addr:#018x}")]
    ReadOnly { addr: u64 },

    /// The access falls outside the bounds of a region.
    #[error("physical access out of bounds: addr={addr:#018x} region_len={len:#x}")]
    OutOfBounds { addr: u64, len: u64 },

    /// Page-table manipulation failed (alignment, conflicting mapping, ...).
    #[error("page table error: {0}")]
    PageTable(String),
}
