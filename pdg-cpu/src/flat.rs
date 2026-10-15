//! Direct (no-MMU) memory used by tools, benches and early bring-up.
//!
//! `FlatMemory` maps guest addresses 1:1 onto a byte array. It is *not* the
//! paged guest path (that is `pdg-core::system::GuestMemory`); it exists so
//! that CPU tests, fuzzers and the disassembler can run against raw code
//! without building page tables.

use crate::fault::{AccessError, MemoryAccess};

/// Flat memory image: guest virtual address `a` is byte `a` of the image.
pub struct FlatMemory {
    data: Vec<u8>,
}

impl std::fmt::Debug for FlatMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatMemory")
            .field("len", &self.data.len())
            .finish()
    }
}

impl FlatMemory {
    /// Create a zero-filled image of `len` bytes.
    pub fn new(len: usize) -> Self {
        Self { data: vec![0; len] }
    }

    /// Copy code/data into the image at `offset`.
    pub fn load(&mut self, offset: usize, bytes: &[u8]) {
        let end = offset + bytes.len();
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[offset..end].copy_from_slice(bytes);
    }

    /// Raw image access.
    pub fn image(&self) -> &[u8] {
        &self.data
    }
}

impl MemoryAccess for FlatMemory {
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, AccessError> {
        let off = addr as usize;
        let end = off.checked_add(width as usize);
        if end.is_none_or(|end| end > self.data.len()) {
            return Err(AccessError::Host(format!(
                "flat memory read out of bounds: {addr:#x}"
            )));
        }
        let mut buf = [0u8; 8];
        buf[..width as usize].copy_from_slice(&self.data[off..off + width as usize]);
        Ok(u64::from_le_bytes(buf))
    }

    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), AccessError> {
        let off = addr as usize;
        let end = off.checked_add(width as usize);
        if end.is_none_or(|end| end > self.data.len()) {
            return Err(AccessError::Host(format!(
                "flat memory write out of bounds: {addr:#x}"
            )));
        }
        let bytes = value.to_le_bytes();
        self.data[off..off + width as usize].copy_from_slice(&bytes[..width as usize]);
        Ok(())
    }

    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, AccessError> {
        let off = addr as usize;
        if off >= self.data.len() {
            return Err(AccessError::Host(format!(
                "flat memory fetch out of bounds: {addr:#x}"
            )));
        }
        let n = out.len().min(self.data.len() - off);
        out[..n].copy_from_slice(&self.data[off..off + n]);
        Ok(n)
    }

    fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), AccessError> {
        let off = addr as usize;
        let end = off.checked_add(out.len());
        if end.is_none_or(|end| end > self.data.len()) {
            return Err(AccessError::Host(format!(
                "flat memory read out of bounds: {addr:#x}"
            )));
        }
        out.copy_from_slice(&self.data[off..off + out.len()]);
        Ok(())
    }

    fn write_buf(&mut self, addr: u64, src: &[u8]) -> Result<(), AccessError> {
        let off = addr as usize;
        let end = off.checked_add(src.len());
        if end.is_none_or(|end| end > self.data.len()) {
            return Err(AccessError::Host(format!(
                "flat memory write out of bounds: {addr:#x}"
            )));
        }
        self.data[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_read_back() {
        let mut mem = FlatMemory::new(0x1000);
        mem.load(0x100, &[0xDE, 0xAD]);
        assert_eq!(mem.read(2, 0x100).unwrap(), 0xADDE);
    }

    #[test]
    fn fetch_stops_at_image_end() {
        let mut mem = FlatMemory::new(0x10);
        mem.load(0xE, &[0x90; 2]);
        let mut out = [0u8; 16];
        let n = mem.fetch(0xE, &mut out).unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn out_of_bounds_reports_host_error() {
        let mut mem = FlatMemory::new(0x10);
        assert!(matches!(mem.read(1, 0x10), Err(AccessError::Host(_))));
    }
}
