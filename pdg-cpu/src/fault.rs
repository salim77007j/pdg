//! Faults and the memory-access port.

use std::ptr::NonNull;

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

    /// The guest process called `exit` (Orbis syscall 1). This is not a
    /// fault condition: the kernel HLE layer asks the core to stop cleanly
    /// with the process exit code. `run` reports it through
    /// [`RunOutcome::Fault`](crate::RunOutcome::Fault) and the session layer
    /// maps it to an "exited" outcome.
    #[error("process exit({code})")]
    ProcessExit {
        /// The exit code passed to `exit`.
        code: i32,
    },

    /// A guest thread parked on a blocking primitive (`_umtx_op` wait or
    /// mutex contention — Orbis syscall 454). Not a fault condition: the
    /// session scheduler simply stops running this thread until another
    /// thread wakes it; the syscall's return value was already written to
    /// RAX before parking.
    #[error("thread {tid} parked on umtx obj {obj:#x}")]
    ThreadParked {
        /// The parked thread's id.
        tid: u64,
        /// The guest address it is waiting on.
        obj: u64,
    },

    /// A guest thread called `thr_exit` (Orbis syscall 431). Not a fault
    /// condition: the session layer removes the thread's interpreter; when
    /// it was the last one, the process exits.
    #[error("thread {tid} exit({code})")]
    ThreadExit {
        /// The exiting thread's id.
        tid: u64,
        /// The exit code the thread reported (PDG's documented model: the
        /// raw `long` value passed as `thr_exit`'s state argument).
        code: i64,
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

    /// Translate the 4 KiB guest page containing `va` to a host pointer to
    /// that page's base, for the JIT's direct-access fast path. `Ok(None)`
    /// means the page is real but not directly host-addressable (MMIO/ROM);
    /// `Err` is an architectural fault from the translation itself.
    ///
    /// The default (no host window) forces the JIT onto its slow helper
    /// path, which keeps simple backends correct.
    fn page_ptr(&mut self, va: u64, write: bool) -> Result<Option<NonNull<u8>>, AccessError> {
        let _ = (va, write);
        Ok(None)
    }

    /// Atomic exchange: store `value`, return the old value.
    ///
    /// The default is a plain read-modify-write (correct for a backend used
    /// by a single vCPU); backends sharing RAM across vCPUs override it with
    /// real atomicity.
    fn xchg(&mut self, width: u8, addr: u64, value: u64) -> Result<u64, AccessError> {
        let old = self.read(width, addr)?;
        self.write(width, addr, value)?;
        Ok(old)
    }

    /// Atomic add: add `value`, return the old value.
    fn xadd(&mut self, width: u8, addr: u64, value: u64) -> Result<u64, AccessError> {
        let old = self.read(width, addr)?;
        self.write(width, addr, old.wrapping_add(value))?;
        Ok(old)
    }

    /// Atomic compare-and-exchange: if the current value equals `expected`,
    /// store `new`. Returns the old value either way (the caller sets ZF by
    /// comparing it against `expected`).
    fn cmpxchg(
        &mut self,
        width: u8,
        addr: u64,
        expected: u64,
        new: u64,
    ) -> Result<u64, AccessError> {
        let old = self.read(width, addr)?;
        if old == expected {
            self.write(width, addr, new)?;
        }
        Ok(old)
    }

    /// `INVLPG` hook: drop the cached translation of the 4 KiB slot
    /// containing `va` (no-op for backends without a TLB).
    fn invalidate_tlb_page(&mut self, va: u64) {
        let _ = va;
    }

    /// TLB-flush hook (e.g. `MOV CR3`): drop all cached translations.
    fn invalidate_tlb_all(&mut self) {}

    /// `MOV CR3` hook: install a new page-table root (and flush caches).
    fn reload_cr3(&mut self, cr3: u64) {
        let _ = cr3;
    }

    /// Monotonic counter bumped whenever cached translations may have
    /// changed (CR3 reload, INVLPG, full flush). Consumers that cache
    /// translations — the JIT's env TLB — compare it between uses instead
    /// of flushing unconditionally. The default (a constant) suits
    /// backends without translation state, e.g. flat memory.
    fn translation_epoch(&mut self) -> u64 {
        0
    }
}
