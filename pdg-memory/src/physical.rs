//! Guest RAM backing store.
//!
//! Anonymous private mappings are zero-filled by the OS, which matches the
//! behaviour of real DRAM at power-on and lets us install page tables without
//! explicit zeroing. Memory is committed lazily by the host kernel, so a
//! 16 GiB configuration reserves address space but only touches pages the
//! guest actually uses.

use memmap2::{MmapMut, MmapOptions};

use crate::error::MemoryError;

/// Page size used across the emulator (x86-64 4 KiB pages).
pub const PAGE_SIZE: u64 = 4096;

/// Guest RAM backed by a lazily-committed anonymous mapping.
pub struct PhysicalMemory {
    mem: MmapMut,
    len: usize,
}

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
        let mem = MmapOptions::new().len(len).map_anon()?;
        Ok(Self { mem, len })
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

    /// Read a little-endian integer of `width` bytes (1, 2, 4 or 8).
    pub fn read(&self, off: usize, width: usize) -> Result<u64, MemoryError> {
        self.check(off, width)?;
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(&self.mem[off..off + width]);
        Ok(u64::from_le_bytes(buf))
    }

    /// Write a little-endian integer of `width` bytes (1, 2, 4 or 8).
    pub fn write(&mut self, off: usize, width: usize, value: u64) -> Result<(), MemoryError> {
        self.check(off, width)?;
        let bytes = value.to_le_bytes();
        self.mem[off..off + width].copy_from_slice(&bytes[..width]);
        Ok(())
    }

    /// Copy `out.len()` bytes out of RAM.
    pub fn read_buf(&self, off: usize, out: &mut [u8]) -> Result<(), MemoryError> {
        let end = off.checked_add(out.len());
        if end.is_none_or(|end| end > self.len) {
            return Err(MemoryError::OutOfBounds {
                addr: off as u64,
                len: self.len as u64,
            });
        }
        out.copy_from_slice(&self.mem[off..off + out.len()]);
        Ok(())
    }

    /// Copy `src.len()` bytes into RAM.
    pub fn write_buf(&mut self, off: usize, src: &[u8]) -> Result<(), MemoryError> {
        let end = off.checked_add(src.len());
        if end.is_none_or(|end| end > self.len) {
            return Err(MemoryError::OutOfBounds {
                addr: off as u64,
                len: self.len as u64,
            });
        }
        self.mem[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_starts_zeroed() {
        let ram = PhysicalMemory::new(0x10000).unwrap();
        assert_eq!(ram.len(), 0x10000);
        assert_eq!(ram.read(0, 8).unwrap(), 0);
    }

    #[test]
    fn rw_roundtrip_all_widths() {
        let mut ram = PhysicalMemory::new(0x10000).unwrap();
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
        let mut ram = PhysicalMemory::new(0x1000).unwrap();
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
}
