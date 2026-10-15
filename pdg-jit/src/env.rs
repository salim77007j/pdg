//! The execution environment shared between generated code and the Rust
//! dispatcher.
//!
//! One `JitEnv` lives at a stable address for the whole run; generated code
//! reaches it through a baked pointer constant. Everything the helpers and
//! the dispatcher need rides in this `#[repr(C)]` struct so the FFI surface
//! stays `extern "C" fn(*mut JitEnv, ...)`.
//!
//! Fault precision: every slow-path helper receives the *guest RIP of the
//! instruction it serves*. On failure it records the fault and points
//! `env.rip` at that instruction; the dispatcher then re-executes the single
//! instruction in the interpreter tier, which reproduces the exact fault
//! (or the exact side effect) — giving tier-1 precise exceptions for free.

use pdg_cpu::{AccessError, MemoryAccess};

/// Direct-mapped software-TLB entries. Power of two.
pub const TLB_ENTRIES: usize = 8192;

/// SMC bloom: 4096 bits keyed on guest code pages.
pub const SMC_WORDS: usize = 64;

/// Exit reasons returned by generated blocks.
pub mod exit {
    /// Keep going: `env.rip` holds the next block entry.
    pub const CONTINUE: i64 = 0;
    /// Guest executed `hlt`.
    pub const HALT: i64 = 1;
    /// Guest executed `syscall` (dispatcher services it, then re-enters).
    pub const SYSCALL: i64 = 2;
    /// Instruction budget exhausted (checked before any side effect).
    pub const BUDGET: i64 = 3;
    /// Unliftable instruction at `env.rip`; single-step via the interpreter.
    pub const FALLBACK: i64 = 4;
    /// Architectural fault at `env.rip`; single-step via the interpreter.
    pub const FAULT: i64 = 5;
    /// A store may have hit a code page: `env.exit_aux` = store address.
    /// The dispatcher invalidates that page's blocks, single-steps the
    /// store through the interpreter, then re-enters the JIT.
    pub const RESTART: i64 = 6;
}

/// `#[repr(C)]` execution environment.
#[repr(C)]
pub struct JitEnv {
    /// Guest GPRs in encoding order.
    pub gpr: [u64; 16],
    /// Next guest RIP (also the faulting/restart instruction when a helper
    /// bails out).
    pub rip: u64,
    /// Guest RFLAGS (authoritative between blocks).
    pub rflags: u64,
    /// Exit reason (see [`exit`]).
    pub exit_code: i64,
    /// SMC store address for [`exit::RESTART`]; fault address for diagnosis.
    pub exit_aux: u64,
    /// Remaining instruction budget for this `run()` call.
    pub budget: i64,
    /// Direct-mapped TLB tags: guest virtual page number (vaddr >> 12).
    pub tlb_tags: [u64; TLB_ENTRIES],
    /// Host page-base pointers, parallel to `tlb_tags` (0 = slow path).
    pub tlb_ptrs: [u64; TLB_ENTRIES],
    /// SMC bloom over guest code pages.
    pub smc_bloom: [u64; SMC_WORDS],
    /// Memory port (fat pointer). Only the Rust helpers dereference it,
    /// while the dispatcher exclusively owns the borrow during a block call.
    pub mem: *mut dyn MemoryAccess,
}

impl JitEnv {
    /// Fresh environment bound to `mem`.
    pub fn new(mem: *mut dyn MemoryAccess) -> Self {
        Self {
            gpr: [0; 16],
            rip: 0,
            rflags: 2,
            exit_code: exit::CONTINUE,
            exit_aux: 0,
            budget: 0,
            tlb_tags: [0; TLB_ENTRIES],
            tlb_ptrs: [0; TLB_ENTRIES],
            smc_bloom: [0; SMC_WORDS],
            mem,
        }
    }

    /// Clear the software TLB (after CR3 changes / INVLPG / fallbacks).
    pub fn clear_tlb(&mut self) {
        self.tlb_tags = [0; TLB_ENTRIES];
        self.tlb_ptrs = [0; TLB_ENTRIES];
    }

    /// Mark a guest code page in the SMC bloom.
    pub fn bloom_set(&mut self, guest_page: u64) {
        let h = bloom_hash(guest_page);
        self.smc_bloom[h / 64] |= 1u64 << (h % 64);
    }

    /// Forget all bloom bits (after invalidations).
    pub fn bloom_clear(&mut self) {
        self.smc_bloom = [0; SMC_WORDS];
    }
}

/// 12-bit bloom hash of a guest page number.
#[inline]
pub fn bloom_hash(page: u64) -> usize {
    // Fold: code pages cluster in low memory, so mix the bits first.
    let mut h = page ^ (page >> 7) ^ (page >> 13);
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 33) as usize & (SMC_WORDS * 64 - 1)
}

// ---------------------------------------------------------------------------
// FFI helpers called from generated code
// ---------------------------------------------------------------------------

/// Record an access error as the block's fault exit (at instruction `rip`).
fn record_fault(env: &mut JitEnv, rip: u64, e: &AccessError) {
    env.exit_code = exit::FAULT;
    env.rip = rip;
    match e {
        AccessError::PageFault { addr, code } => {
            env.exit_aux = *addr;
            let _ = code;
        }
        AccessError::GeneralProtection(_) | AccessError::Host(_) => {
            env.exit_aux = 0;
        }
    }
}

/// Translate a guest virtual page and fill the TLB slot.
/// Returns the host page base, or 0 when the access must use the slow path
/// (MMIO/ROM, or a translation fault — recorded against `rip`).
unsafe fn tlb_fill(env: &mut JitEnv, va: u64, write: bool, rip: u64) -> u64 {
    let page = va & !0xFFF;
    let idx = (page >> 12) as usize & (TLB_ENTRIES - 1);
    let tag = page >> 12;
    let mem: &mut dyn MemoryAccess = unsafe { &mut *env.mem };
    match mem.page_ptr(page, write) {
        Ok(Some(ptr)) => {
            let addr = ptr.as_ptr() as u64;
            env.tlb_tags[idx] = tag;
            env.tlb_ptrs[idx] = addr;
            addr
        }
        Ok(None) => {
            // Real page, not host-addressable: remember the tag with a null
            // pointer so the inline path takes the helper call.
            env.tlb_tags[idx] = tag;
            env.tlb_ptrs[idx] = 0;
            0
        }
        Err(e) => {
            record_fault(env, rip, &e);
            0
        }
    }
}

/// TLB fill for loads: returns the host page base (0 = slow path).
/// # Safety
///
/// `env` must point at a live [`JitEnv`] whose memory port is installed and
/// exclusively owned by the calling dispatcher.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pdgjit_tlb_load(env: *mut JitEnv, rip: u64, va: u64) -> u64 {
    // SAFETY: the dispatcher guarantees exclusive env access during calls.
    unsafe { tlb_fill(&mut *env, va, false, rip) }
}

/// TLB fill for stores (write-permission aware).
/// # Safety
///
/// `env` must point at a live [`JitEnv`] whose memory port is installed and
/// exclusively owned by the calling dispatcher.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pdgjit_tlb_store(env: *mut JitEnv, rip: u64, va: u64) -> u64 {
    // SAFETY: the dispatcher guarantees exclusive env access during calls.
    unsafe { tlb_fill(&mut *env, va, true, rip) }
}

macro_rules! slow_rw {
    ($read:ident, $write:ident, $w:expr) => {
        /// # Safety
        ///
        /// `env` must point at a live [`JitEnv`] whose memory port is
        /// installed and exclusively owned by the calling dispatcher.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $read(env: *mut JitEnv, rip: u64, va: u64) -> u64 {
            // SAFETY: the dispatcher guarantees exclusive env access.
            unsafe {
                let env = &mut *env;
                let mem: &mut dyn MemoryAccess = &mut *env.mem;
                match mem.read($w, va) {
                    Ok(v) => v,
                    Err(e) => {
                        record_fault(env, rip, &e);
                        0
                    }
                }
            }
        }

        /// # Safety
        ///
        /// Same contract as [`$read`]'s safety section.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $write(env: *mut JitEnv, rip: u64, va: u64, value: u64) {
            // SAFETY: the dispatcher guarantees exclusive env access.
            unsafe {
                let env = &mut *env;
                let mem: &mut dyn MemoryAccess = &mut *env.mem;
                if let Err(e) = mem.write($w, va, value) {
                    record_fault(env, rip, &e);
                }
            }
        }
    };
}

slow_rw!(pdgjit_read8, pdgjit_write8, 1);
slow_rw!(pdgjit_read16, pdgjit_write16, 2);
slow_rw!(pdgjit_read32, pdgjit_write32, 4);
slow_rw!(pdgjit_read64, pdgjit_write64, 8);

/// SMC check for a store whose page is (maybe) a code page. Called *before*
/// the store executes; exits with [`exit::RESTART`] so the dispatcher can
/// invalidate the page's blocks and single-step the store.
/// # Safety
///
/// `env` must point at a live [`JitEnv`] whose memory port is installed and
/// exclusively owned by the calling dispatcher.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pdgjit_smc_invalidate(env: *mut JitEnv, rip: u64, va: u64) {
    // SAFETY: the dispatcher guarantees exclusive env access.
    let env = unsafe { &mut *env };
    env.exit_code = exit::RESTART;
    env.rip = rip;
    env.exit_aux = va;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdg_cpu::FlatMemory;

    #[test]
    fn bloom_roundtrip() {
        // A dangling port is fine: these tests never dereference it before
        // installing a real one.
        let mut scratch = pdg_cpu::FlatMemory::new(1);
        let port: *mut dyn MemoryAccess = &mut scratch;
        let mut env = JitEnv::new(port);
        for p in [0u64, 1, 0x1234, 0xFFFF_F000 >> 12, u64::MAX] {
            env.bloom_set(p);
        }
        assert!(env.smc_bloom.iter().any(|&w| w != 0));
        env.bloom_clear();
        assert!(env.smc_bloom.iter().all(|&w| w == 0));
    }

    #[test]
    fn tlb_fill_fast_and_miss_paths() {
        let mut mem = FlatMemory::new(0x20000);
        let mut env = JitEnv::new(&mut mem as *mut dyn MemoryAccess);
        // SAFETY: `mem` outlives the calls below.
        let base = unsafe { pdgjit_tlb_load(&mut env, 0, 0x100) };
        assert_ne!(base, 0);
        let idx = (0x100 >> 12) as usize & (TLB_ENTRIES - 1);
        assert_eq!(env.tlb_tags[idx], 0x100 >> 12);
        assert_eq!(env.tlb_ptrs[idx], base);
        // Out-of-image page: Ok(None) → 0, no fault recorded.
        let bad = unsafe { pdgjit_tlb_load(&mut env, 0, 0x9000_0000) };
        assert_eq!(bad, 0);
        assert_eq!(env.exit_code, exit::CONTINUE);
        // Out-of-bounds *fetch* (same as any unmapped page at this tier):
        // FlatMemory grows on OOB writes, so build a fault through read.
        // (A read of a fresh unmapped page in FlatMemory returns an error.)
        let _ = unsafe { pdgjit_read8(&mut env, 0x40, 0x9000_0000) };
        assert_eq!(env.exit_code, exit::FAULT);
        assert_eq!(env.rip, 0x40);
    }
}
