//! Faults and the memory-access port.

use thiserror::Error;

/// Guest-visible architectural fault raised during execution.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum Fault {
    /// `#PF` — page fault. `addr` is the value that lands in CR2.
    #[error("#PF page fault at {addr:#018x} (code {code:#04x})")]
    PageFault {
        /// Faulting address.
        addr: u64,
        /// Page-fault error code.
        code: u32,
    },

    /// `#UD` — invalid opcode (or an instruction outside the implemented
    /// Phase-1 subset).
    #[error("#UD invalid or unimplemented instruction `{mnemonic}` at {rip:#018x}")]
    InvalidOpcode {
        /// Mnemonic that was rejected.
        mnemonic: String,
        /// RIP of the offending instruction.
        rip: u64,
    },

    /// `#GP` — general protection fault.
    #[error("#GP general protection fault: {reason}")]
    GeneralProtection {
        /// Human-readable cause.
        reason: String,
    },

    /// `#DE` — divide error (reserved for MUL/DIV in Phase 2).
    #[error("#DE integer divide error")]
    DivideError,

    /// `syscall` executed while no kernel hook was installed.
    #[error("syscall at {rip:#018x} with no kernel hook installed")]
    UnhandledSyscall {
        /// RIP of the `syscall` instruction.
        rip: u64,
    },

    /// The host-side emulation stack failed (emulator bug, not guest-visible).
    #[error("host error: {0}")]
    HostError(String),
}

/// Errors produced by the [`MemoryAccess`] port.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AccessError {
    /// Architectural page fault raised by the platform layer.
    #[error("page fault at {addr:#018x} (code {code:#04x})")]
    PageFault {
        /// Faulting address.
        addr: u64,
        /// Page-fault error code.
        code: u32,
    },

    /// Non-canonical address or other general protection condition.
    #[error("general protection: {0}")]
    GeneralProtection(String),

    /// Host-side memory subsystem failure (out of bounds, lock poisoned, ...).
    #[error("host memory error: {0}")]
    Host(String),
}

impl AccessError {
    /// Convert into the architectural [`Fault`] the CPU core will report.
    pub fn into_fault(self) -> Fault {
        match self {
            AccessError::PageFault { addr, code } => Fault::PageFault { addr, code },
            AccessError::GeneralProtection(reason) => Fault::GeneralProtection { reason },
            AccessError::Host(msg) => Fault::HostError(msg),
        }
    }
}

/// Width-parameterized guest memory port.
///
/// Implemented by the platform layer (`pdg-core::system::GuestMemory` for the
/// paged guest path; [`crate::flat::FlatMemory`] for direct/no-MMU mode).
/// Widths are always in bytes and must be 1, 2, 4 or 8.
pub trait MemoryAccess {
    /// Read a little-endian integer of `width` bytes at `addr`.
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, AccessError>;

    /// Write a little-endian integer of `width` bytes at `addr`.
    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), AccessError>;

    /// Fetch up to `out.len()` bytes of instruction stream starting at
    /// `addr`. Returns the number of bytes filled (at least 1, otherwise an
    /// error). Stopping early (e.g. at a page boundary) is permitted; the
    /// decoder will fault if the truncated window cannot hold the whole
    /// instruction.
    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, AccessError>;

    /// Copy exactly `out.len()` bytes out of memory.
    fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), AccessError>;

    /// Copy exactly `src.len()` bytes into memory.
    fn write_buf(&mut self, addr: u64, src: &[u8]) -> Result<(), AccessError>;
}
