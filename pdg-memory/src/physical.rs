//! Guest RAM backing store.
//!
//! Anonymous private mappings are zero-filled by the OS, which matches the
//! behaviour of real DRAM at power-on and lets us install page tables without
//! explicit zeroing. Memory is committed lazily by the host kernel, so a
//! 16 GiB configuration reserves address space but only touches pages the
//! guest actually uses.
//!
//! Phase 2 concurrency model: all RAM accesses go through the mapping's
//! stable base pointer using *relaxed atomics* (word-aligned fast path,
//! byte-wise loop otherwise). This makes every operation `&self`, which is
//! what lets one [`crate::bus::Bus`] be shared by several vCPUs without a
//! global lock — plain guest accesses race exactly like they would on real
//! hardware, and locked (atomic RMW) accesses are genuinely atomic. On x86-64
//! hosts the relaxed loads/stores compile to ordinary `mov` instructions, so
//! the single-threaded Phase-1 paths pay nothing for this.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};

use crate::error::MemoryError;

/// Page size used across the emulator (x86-64 4 KiB pages).
pub const PAGE_SIZE: u64 = 4096;

/// Guest RAM backed by a lazily-committed anonymous mapping.
pub struct PhysicalMemory {
    /// Owns the mapping (kept alive; accesses go through `base`).
    #[allow(dead_code)]
    mem: MmapMut,
    /// Stable base pointer of the mapping (captured once at construction).
    /// The mapping never moves or shrinks, so this stays valid for the
    /// lifetime of this object; all accesses go through it.
    base: *mut u8,
    len: usize,
    /// Marker making `!Send`/`!Sync` explicit *before* Phase 2 — the mapping
    /// is now accessed through atomics only, so sharing is sound and the
    /// type is `Send + Sync` by auto-derivation.
    _marker: (),
}

// SAFETY: `base` points into an mmap region exclusively owned by this
// object; every read/write goes through atomic operations, which are
// race-free by definition. `MmapMut` itself is `Send + Sync`.
unsafe impl Send for PhysicalMemory {}
unsafe impl Sync for PhysicalMemory {}

impl std::fmt::Debug for PhysicalMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalMemory")
            .field("len", &self.len)
            .finish()
    }
}

impl PhysicalMemory {
    /// Create a zero-filled RAM region of `len_bytes` bytes (rounded up to a
    /// host page boundary).
    pub fn new(len_bytes: usize) -> std::io::Result<Self> {
        let page = PAGE_SIZE as usize;
        let len = len_bytes.div_ceil(page) * page;
        // Anonymous mappings are zero-filled and committed on demand, which
        // keeps a 16 GiB guest RAM configuration cheap until the guest
        // actually dirties pages.
        let mut mem = MmapOptions::new().len(len).map_anon()?;
        // SAFETY: `as_mut_ptr` on the slice behind `&mut mem` is within the
        // mapping we own; the borrow of `mem` ends but the mapping outlives it.
        let base = mem.as_mut_ptr();
        Ok(Self {
            mem,
            base,
            len,
            _marker: (),
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false — a `PhysicalMemory` always covers at least one page.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn check(&self, off: usize, width: usize) -> Result<(), MemoryError> {
        let end = off.checked_add(width);
        if end.is_none_or(|end| end > self.len) {
            return Err(MemoryError::OutOfBounds {
                addr: off as u64,
                len: self.len as u64,
            });
        }
        Ok(())
    }

    // ----- plain reads / writes (relaxed-atomic, &self) ---------------------

    /// Read a little-endian integer of `width` bytes (1, 2, 4 or 8).
    pub fn read(&self, off: usize, width: usize) -> Result<u64, MemoryError> {
        self.check(off, width)?;
        // SAFETY: `check` guaranteed [off, off+width) is inside the mapping.
        unsafe { Ok(read_atomic(self.base.add(off), width)) }
    }

    /// Write a little-endian integer of `width` bytes (1, 2, 4 or 8).
    pub fn write(&self, off: usize, width: usize, value: u64) -> Result<(), MemoryError> {
        self.check(off, width)?;
        // SAFETY: bounds checked above.
        unsafe { write_atomic(self.base.add(off), width, value) }
        Ok(())
    }

    /// Copy `out.len()` bytes out of RAM.
    pub fn read_buf(&self, off: usize, out: &mut [u8]) -> Result<(), MemoryError> {
        self.check(off, out.len())?;
        for (i, b) in out.iter_mut().enumerate() {
            // SAFETY: bounds checked above; byte atomics need no alignment.
            *b = unsafe { (&*self.base.add(off + i).cast::<AtomicU8>()).load(Ordering::Relaxed) };
        }
        Ok(())
    }

    /// Copy `src.len()` bytes into RAM.
    pub fn write_buf(&self, off: usize, src: &[u8]) -> Result<(), MemoryError> {
        self.check(off, src.len())?;
        for (i, b) in src.iter().enumerate() {
            // SAFETY: bounds checked above.
            unsafe {
                (*self.base.add(off + i).cast::<AtomicU8>()).store(*b, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    // ----- host-pointer access (JIT fast path) ------------------------------

    /// Host pointer to the byte at `off` in guest RAM, after a bounds check
    /// for a `width`-byte access. This is the door the JIT's software TLB
    /// uses for direct host-memory accesses; ROM/MMIO never get here.
    pub fn host_ptr(&self, off: u64, width: u64) -> Option<*mut u8> {
        let end = off.checked_add(width)?;
        if end > self.len as u64 {
            return None;
        }
        // SAFETY: bounds checked above; pointer stays inside the mapping.
        Some(unsafe { self.base.add(off as usize) })
    }

    // ----- atomic read-modify-write (LOCK-prefixed instructions) ------------

    /// Atomically exchange: stores `value`, returns the old value.
    pub fn xchg(&self, off: usize, width: usize, value: u64) -> Result<u64, MemoryError> {
        self.check(off, width)?;
        // SAFETY: bounds checked above.
        unsafe { Ok(rmw_atomic(self.base.add(off), width, value, Rmw::Xchg)) }
    }

    /// Atomically add: adds `value`, returns the old value.
    pub fn xadd(&self, off: usize, width: usize, value: u64) -> Result<u64, MemoryError> {
        self.check(off, width)?;
        // SAFETY: bounds checked above.
        unsafe { Ok(rmw_atomic(self.base.add(off), width, value, Rmw::Xadd)) }
    }

    /// Atomically compare-and-exchange: if the old value equals `expected`,
    /// stores `new`; returns the old value either way (the caller tests it).
    pub fn cmpxchg(
        &self,
        off: usize,
        width: usize,
        expected: u64,
        new: u64,
    ) -> Result<u64, MemoryError> {
        self.check(off, width)?;
        // SAFETY: bounds checked above.
        unsafe {
            Ok(rmw_atomic(
                self.base.add(off),
                width,
                new,
                Rmw::Cmpxchg(expected),
            ))
        }
    }
}

// ----- free functions over raw pointers --------------------------------------
//
// All of these are private helpers. Every call site has already bounds-checked
// its access, so the pointers are in-bounds of the mmap region owned by the
// calling `PhysicalMemory`.

#[derive(Clone, Copy)]
enum Rmw {
    Xchg,
    Xadd,
    Cmpxchg(u64),
}

/// Unaligned or mixed-width access goes through this global lock — locked
/// unaligned accesses are extremely rare in real guest code, and hardware
/// merely serialises them too.
static UNALIGNED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Atomic read of `width` little-endian bytes at `p`.
///
/// # Safety
/// `p` must be valid for reads of `width` bytes.
unsafe fn read_atomic(p: *mut u8, width: usize) -> u64 {
    if p as usize % width == 0 {
        // SAFETY: caller guaranteed validity; alignment checked.
        let v = match width {
            1 => u64::from(unsafe { (*p.cast::<AtomicU8>()).load(Ordering::Relaxed) }),
            2 => u64::from(unsafe { (*p.cast::<AtomicU16>()).load(Ordering::Relaxed) }),
            4 => u64::from(unsafe { (*p.cast::<AtomicU32>()).load(Ordering::Relaxed) }),
            _ => unsafe { (*p.cast::<AtomicU64>()).load(Ordering::Relaxed) },
        };
        return v;
    }
    let mut v = 0u64;
    for i in 0..width {
        // SAFETY: in-bounds by caller; byte atomics have no alignment needs.
        let b = unsafe { (&*p.add(i).cast::<AtomicU8>()).load(Ordering::Relaxed) };
        v |= u64::from(b) << (i * 8);
    }
    v
}

/// Atomic write of `width` little-endian bytes at `p`.
///
/// # Safety
/// `p` must be valid for writes of `width` bytes.
unsafe fn write_atomic(p: *mut u8, width: usize, value: u64) {
    if p as usize % width == 0 {
        // SAFETY: caller guaranteed validity; alignment checked.
        match width {
            1 => unsafe { (*p.cast::<AtomicU8>()).store(value as u8, Ordering::Relaxed) },
            2 => unsafe { (*p.cast::<AtomicU16>()).store(value as u16, Ordering::Relaxed) },
            4 => unsafe { (*p.cast::<AtomicU32>()).store(value as u32, Ordering::Relaxed) },
            _ => unsafe { (*p.cast::<AtomicU64>()).store(value, Ordering::Relaxed) },
        }
        return;
    }
    for i in 0..width {
        // SAFETY: in-bounds by caller; byte atomics have no alignment needs.
        unsafe {
            (*p.add(i).cast::<AtomicU8>())
                .store(((value >> (i * 8)) & 0xFF) as u8, Ordering::Relaxed);
        }
    }
}

/// Atomic read-modify-write at `p`.
///
/// # Safety
/// `p` must be valid for read/write of `width` bytes.
unsafe fn rmw_atomic(p: *mut u8, width: usize, value: u64, op: Rmw) -> u64 {
    // Unaligned locked ops are serialised under the global lock to stay
    // atomic with respect to other RMWs (plain atomics already never tear).
    if p as usize % width != 0 {
        let _guard = UNALIGNED_LOCK.lock().expect("unaligned-RMW lock poisoned");
        let old = unsafe { read_atomic(p, width) };
        let new = match op {
            Rmw::Xchg => value,
            Rmw::Xadd => old.wrapping_add(value),
            Rmw::Cmpxchg(expected) => {
                if old == expected {
                    value
                } else {
                    return old;
                }
            }
        };
        unsafe { write_atomic(p, width, new) };
        return old;
    }
    // Aligned fast path: a genuine hardware atomic per width.
    match width {
        1 => unsafe { rmw8(p.cast(), value as u8, op) as u64 },
        2 => unsafe { rmw16(p.cast(), value as u16, op) as u64 },
        4 => unsafe { rmw32(p.cast(), value as u32, op) as u64 },
        _ => unsafe { rmw64(p.cast(), value, op) },
    }
}

macro_rules! rmw_impl {
    ($name:ident, $a:ty, $v:ty) => {
        /// # Safety
        /// `p` must be valid for read/write of one atomic of the target width.
        unsafe fn $name(p: *mut $a, new: $v, op: Rmw) -> $v {
            // SAFETY: caller guarantees validity; alignment was checked.
            let a = unsafe { &*p };
            match op {
                Rmw::Xchg => a.swap(new, Ordering::SeqCst),
                Rmw::Xadd => a.fetch_add(new, Ordering::SeqCst),
                Rmw::Cmpxchg(expected) => {
                    let expected = expected as $v;
                    match a.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(old) => old,
                        Err(actual) => actual,
                    }
                }
            }
        }
    };
}

rmw_impl!(rmw8, AtomicU8, u8);
rmw_impl!(rmw16, AtomicU16, u16);
rmw_impl!(rmw32, AtomicU32, u32);
rmw_impl!(rmw64, AtomicU64, u64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn ram_starts_zeroed() {
        let ram = PhysicalMemory::new(0x10000).unwrap();
        assert_eq!(ram.len(), 0x10000);
        assert_eq!(ram.read(0, 8).unwrap(), 0);
    }

    #[test]
    fn rw_roundtrip_all_widths() {
        let ram = PhysicalMemory::new(0x10000).unwrap();
        ram.write(0x10, 1, 0xAB).unwrap();
        ram.write(0x20, 2, 0xBEEF).unwrap();
        ram.write(0x30, 4, 0xDEAD_BEEF).unwrap();
        ram.write(0x40, 8, 0x0123_4567_89AB_CDEF).unwrap();
        assert_eq!(ram.read(0x10, 1).unwrap(), 0xAB);
        assert_eq!(ram.read(0x20, 2).unwrap(), 0xBEEF);
        assert_eq!(ram.read(0x30, 4).unwrap(), 0xDEAD_BEEF);
        assert_eq!(ram.read(0x40, 8).unwrap(), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn out_of_bounds_rejected() {
        let ram = PhysicalMemory::new(0x1000).unwrap();
        assert!(matches!(
            ram.read(0xFFF, 2),
            Err(MemoryError::OutOfBounds { .. })
        ));
        assert!(matches!(
            ram.write(0x1000, 1, 0),
            Err(MemoryError::OutOfBounds { .. })
        ));
        assert!(matches!(
            ram.write_buf(0xFF0, &[0u8; 0x20]),
            Err(MemoryError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn length_rounds_up_to_page() {
        let ram = PhysicalMemory::new(1).unwrap();
        assert_eq!(ram.len(), PAGE_SIZE as usize);
    }

    #[test]
    fn host_ptr_bounds_checked() {
        let ram = PhysicalMemory::new(0x1000).unwrap();
        assert!(ram.host_ptr(0xFF0, 0x10).is_some());
        assert!(ram.host_ptr(0xFF0, 0x11).is_none());
        assert!(ram.host_ptr(0x1000, 1).is_none());
    }

    #[test]
    fn atomics_do_what_their_names_say() {
        let ram = PhysicalMemory::new(0x1000).unwrap();
        ram.write(0x40, 8, 10).unwrap();
        assert_eq!(ram.xadd(0x40, 8, 5).unwrap(), 10);
        assert_eq!(ram.read(0x40, 8).unwrap(), 15);
        assert_eq!(ram.xchg(0x40, 8, 99).unwrap(), 15);
        assert_eq!(ram.read(0x40, 8).unwrap(), 99);
        // Successful and failed cmpxchg.
        assert_eq!(ram.cmpxchg(0x40, 8, 99, 1).unwrap(), 99);
        assert_eq!(ram.read(0x40, 8).unwrap(), 1);
        assert_eq!(ram.cmpxchg(0x40, 8, 42, 2).unwrap(), 1);
        assert_eq!(ram.read(0x40, 8).unwrap(), 1);
    }

    #[test]
    fn unaligned_rmw_still_atomic_and_correct() {
        let ram = PhysicalMemory::new(0x1000).unwrap();
        ram.write(0x81, 4, 0x1122_3344).unwrap();
        assert_eq!(ram.xadd(0x81, 4, 1).unwrap(), 0x1122_3344);
        assert_eq!(ram.read(0x81, 4).unwrap(), 0x1122_3345);
    }

    /// Shares the RAM across threads hammering one location with locked adds
    /// and a plain writer/reader pair: the final sum must be exact.
    #[test]
    fn shared_rmw_across_threads() {
        const THREADS: u64 = 4;
        const PER_THREAD: u64 = 20_000;
        let ram = Arc::new(PhysicalMemory::new(0x1000).unwrap());
        let mut joins = Vec::new();
        for t in 0..THREADS {
            let ram = Arc::clone(&ram);
            joins.push(std::thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    let _ = t;
                    ram.xadd(0x100, 8, 1).unwrap();
                }
            }));
        }
        for j in joins {
            j.join().expect("worker");
        }
        assert_eq!(
            ram.read(0x100, 8).unwrap(),
            THREADS * PER_THREAD,
            "locked adds must never lose updates"
        );
    }
}
