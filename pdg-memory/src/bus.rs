//! Guest-physical address bus: RAM, ROM regions and MMIO devices.

use std::{ops::Range, sync::Arc};

use crate::{error::MemoryError, physical::PhysicalMemory};

/// Width of a bus access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessWidth {
    /// 1 byte.
    W8,
    /// 2 bytes.
    W16,
    /// 4 bytes.
    W32,
    /// 8 bytes.
    W64,
}

impl AccessWidth {
    /// Size in bytes.
    pub const fn size(self) -> usize {
        match self {
            AccessWidth::W8 => 1,
            AccessWidth::W16 => 2,
            AccessWidth::W32 => 4,
            AccessWidth::W64 => 8,
        }
    }

    /// Parse from a byte width.
    pub fn from_size(width: u8) -> Option<Self> {
        match width {
            1 => Some(AccessWidth::W8),
            2 => Some(AccessWidth::W16),
            4 => Some(AccessWidth::W32),
            8 => Some(AccessWidth::W64),
            _ => None,
        }
    }
}

/// A memory-mapped I/O device. Reads have no side effects beyond what the
/// device models; `offset` is relative to the registered base address.
///
/// Devices are expected to be `Send + Sync` with internal mutability because
/// the bus hands out shared references.
pub trait MmioDevice: Send + Sync {
    /// Stable identifier shown in traces and error messages.
    fn name(&self) -> &'static str;

    /// MMIO read. Default: returns 0.
    fn read(&self, offset: u64, width: AccessWidth) -> u64 {
        let _ = (offset, width);
        0
    }

    /// MMIO write. Default: ignores the write.
    fn write(&self, offset: u64, width: AccessWidth, value: u64) {
        let _ = (offset, width, value);
    }
}

/// A read-only region (e.g. the BIOS window at the top of the 4 GiB space).
#[derive(Debug)]
pub struct RomRegion {
    /// Guest-physical base address.
    pub base: u64,
    /// Region contents.
    pub data: Vec<u8>,
}

/// Guest-physical address bus.
///
/// Decodes accesses in this order: RAM (starting at physical 0), ROM regions,
/// then MMIO regions. Anything else is [`MemoryError::UnmappedPhysical`].
pub struct Bus {
    ram: PhysicalMemory,
    roms: Vec<RomRegion>,
    mmio: Vec<(Range<u64>, Arc<dyn MmioDevice>)>,
}

impl std::fmt::Debug for Bus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bus")
            .field("ram_len", &self.ram.len())
            .field(
                "roms",
                &self
                    .roms
                    .iter()
                    .map(|r| (r.base, r.data.len()))
                    .collect::<Vec<_>>(),
            )
            .field(
                "mmio",
                &self
                    .mmio
                    .iter()
                    .map(|(r, d)| (r.clone(), d.name()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Bus {
    /// Create a bus with `ram_len` bytes of guest RAM.
    pub fn new(ram_len: usize) -> std::io::Result<Self> {
        Ok(Self {
            ram: PhysicalMemory::new(ram_len)?,
            roms: Vec::new(),
            mmio: Vec::new(),
        })
    }

    /// Guest RAM length in bytes.
    pub fn ram_len(&self) -> u64 {
        self.ram.len() as u64
    }

    /// Attach a read-only region at `base`.
    pub fn attach_rom(&mut self, base: u64, data: Vec<u8>) {
        tracing::debug!(base, len = data.len(), "attach ROM region");
        self.roms.push(RomRegion { base, data });
    }

    /// Attach an MMIO device covering `[base, base + len)`.
    pub fn attach_mmio(&mut self, base: u64, len: u64, device: Arc<dyn MmioDevice>) {
        tracing::debug!(base, len, name = device.name(), "attach MMIO device");
        self.mmio.push((base..base + len, device));
    }

    fn rom_containing(&self, addr: u64, width: usize) -> Option<&RomRegion> {
        self.roms
            .iter()
            .find(|r| addr >= r.base && addr + width as u64 <= r.base + r.data.len() as u64)
    }

    fn mmio_containing(&self, addr: u64) -> Option<&(Range<u64>, Arc<dyn MmioDevice>)> {
        // MMIO registers are small; require the whole access to fit.
        self.mmio
            .iter()
            .find(|(range, _)| range.contains(&addr) && range.end > addr)
    }

    /// Read an integer of `width` bytes at guest-physical `addr`.
    pub fn read(&self, width: AccessWidth, addr: u64) -> Result<u64, MemoryError> {
        let w = width.size();
        if addr < self.ram_len() {
            return self.ram.read(addr as usize, w);
        }
        if let Some(rom) = self.rom_containing(addr, w) {
            let mut buf = [0u8; 8];
            buf[..w].copy_from_slice(&rom.data[(addr - rom.base) as usize..][..w]);
            return Ok(u64::from_le_bytes(buf));
        }
        if let Some((range, dev)) = self.mmio_containing(addr) {
            // Fall back to a byte-wise split when the access straddles the
            // region end; single-register accesses take the fast path above.
            let mut value = 0u64;
            for i in 0..w as u64 {
                let a = addr + i;
                if a >= range.end {
                    return Err(MemoryError::UnmappedPhysical { addr });
                }
                value |= dev.read(a - range.start, AccessWidth::W8) << (i * 8);
            }
            return Ok(value);
        }
        Err(MemoryError::UnmappedPhysical { addr })
    }

    /// Write an integer of `width` bytes at guest-physical `addr`.
    pub fn write(&mut self, width: AccessWidth, addr: u64, value: u64) -> Result<(), MemoryError> {
        let w = width.size();
        if addr < self.ram_len() {
            return self.ram.write(addr as usize, w, value);
        }
        if self.rom_containing(addr, w).is_some()
            || self
                .roms
                .iter()
                .any(|r| r.base <= addr && addr < r.base + r.data.len() as u64)
        {
            return Err(MemoryError::ReadOnly { addr });
        }
        if let Some((range, dev)) = self.mmio_containing(addr) {
            for i in 0..w as u64 {
                let a = addr + i;
                if a >= range.end {
                    return Err(MemoryError::UnmappedPhysical { addr });
                }
                dev.write(a - range.start, AccessWidth::W8, (value >> (i * 8)) & 0xFF);
            }
            return Ok(());
        }
        Err(MemoryError::UnmappedPhysical { addr })
    }

    /// Copy `out.len()` bytes out of the bus. The whole access must fit
    /// inside one contiguous region (RAM or a single ROM); MMIO falls back
    /// to byte-wise reads.
    pub fn read_buf(&self, addr: u64, out: &mut [u8]) -> Result<(), MemoryError> {
        let end = addr
            .checked_add(out.len() as u64)
            .ok_or(MemoryError::OutOfBounds { addr, len: 0 })?;
        if addr < self.ram_len() && end <= self.ram_len() {
            self.ram.read_buf(addr as usize, out)?;
            return Ok(());
        }
        if let Some(rom) = self
            .roms
            .iter()
            .find(|r| addr >= r.base && end <= r.base + r.data.len() as u64)
        {
            out.copy_from_slice(&rom.data[(addr - rom.base) as usize..][..out.len()]);
            return Ok(());
        }
        // Generic fallback (MMIO or straddling): byte-wise.
        for (i, b) in out.iter_mut().enumerate() {
            *b = self.read(AccessWidth::W8, addr + i as u64)? as u8;
        }
        Ok(())
    }

    /// Copy `src.len()` bytes into the bus. The whole access must fit inside
    /// one contiguous writable region (RAM); ROM refuses; MMIO goes through
    /// byte-wise writes.
    pub fn write_buf(&mut self, addr: u64, src: &[u8]) -> Result<(), MemoryError> {
        let end = addr
            .checked_add(src.len() as u64)
            .ok_or(MemoryError::OutOfBounds { addr, len: 0 })?;
        if addr < self.ram_len() && end <= self.ram_len() {
            self.ram.write_buf(addr as usize, src)?;
            return Ok(());
        }
        if self
            .roms
            .iter()
            .any(|r| addr >= r.base && addr < r.base + r.data.len() as u64)
        {
            return Err(MemoryError::ReadOnly { addr });
        }
        for (i, b) in src.iter().enumerate() {
            self.write(AccessWidth::W8, addr + i as u64, *b as u64)?;
        }
        Ok(())
    }

    /// End (exclusive) of the contiguous region containing `addr`, used by
    /// the DMA engine to chunk transfers that cross region boundaries.
    pub fn region_end(&self, addr: u64) -> Option<u64> {
        if addr < self.ram_len() {
            return Some(self.ram_len());
        }
        if let Some(rom) = self
            .roms
            .iter()
            .find(|r| addr >= r.base && addr < r.base + r.data.len() as u64)
        {
            return Some(rom.base + rom.data.len() as u64);
        }
        if let Some((range, _)) = self.mmio.iter().find(|(range, _)| range.contains(&addr)) {
            return Some(range.end);
        }
        None
    }

    /// Direct physical RAM write used by loaders before paging is active.
    pub fn ram_write_buf(&mut self, off: u64, src: &[u8]) -> Result<(), MemoryError> {
        self.ram.write_buf(off as usize, src)
    }

    /// Direct physical RAM read used by debug tooling.
    pub fn ram_read_buf(&self, off: u64, out: &mut [u8]) -> Result<(), MemoryError> {
        self.ram.read_buf(off as usize, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CounterDevice {
        writes: AtomicU64,
    }

    impl MmioDevice for CounterDevice {
        fn name(&self) -> &'static str {
            "test-counter"
        }
        fn read(&self, offset: u64, _width: AccessWidth) -> u64 {
            offset
        }
        fn write(&self, _offset: u64, _width: AccessWidth, _value: u64) {
            self.writes.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn ram_rw_via_bus() {
        let mut bus = Bus::new(0x10000).unwrap();
        bus.write(AccessWidth::W64, 0x100, 0x0123_4567_89AB_CDEF)
            .unwrap();
        assert_eq!(
            bus.read(AccessWidth::W64, 0x100).unwrap(),
            0x0123_4567_89AB_CDEF
        );
    }

    #[test]
    fn rom_is_readable_but_not_writable() {
        let mut bus = Bus::new(0x1000).unwrap();
        bus.attach_rom(0xF000, vec![1, 2, 3, 4]);
        assert_eq!(bus.read(AccessWidth::W32, 0xF000).unwrap(), 0x0403_0201);
        assert!(matches!(
            bus.write(AccessWidth::W8, 0xF000, 0),
            Err(MemoryError::ReadOnly { .. })
        ));
    }

    #[test]
    fn mmio_device_receives_writes() {
        let mut bus = Bus::new(0x1000).unwrap();
        let dev = Arc::new(CounterDevice {
            writes: AtomicU64::new(0),
        });
        bus.attach_mmio(0x2000, 0x100, dev.clone());
        bus.write(AccessWidth::W32, 0x2010, 0x1122_3344).unwrap();
        assert_eq!(dev.writes.load(Ordering::SeqCst), 4);
        // The device echoes its per-byte offsets; a byte read returns the
        // offset directly.
        assert_eq!(bus.read(AccessWidth::W8, 0x2018).unwrap(), 0x18);
    }

    #[test]
    fn unmapped_access_errors() {
        let mut bus = Bus::new(0x1000).unwrap();
        assert!(matches!(
            bus.read(AccessWidth::W8, 0x9000),
            Err(MemoryError::UnmappedPhysical { .. })
        ));
        assert!(matches!(
            bus.write(AccessWidth::W8, 0x9000, 1),
            Err(MemoryError::UnmappedPhysical { .. })
        ));
    }

    #[test]
    fn read_write_buf_roundtrip() {
        let mut bus = Bus::new(0x10000).unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        bus.write_buf(0x500, &data).unwrap();
        let mut out = [0u8; 64];
        bus.read_buf(0x500, &mut out).unwrap();
        assert_eq!(&out, &data[..]);
    }

    #[test]
    fn region_end_reports_boundaries() {
        let mut bus = Bus::new(0x1000).unwrap();
        bus.attach_rom(0xF000, vec![0; 0x100]);
        bus.attach_mmio(
            0x2000,
            0x10,
            Arc::new(CounterDevice {
                writes: AtomicU64::new(0),
            }),
        );
        assert_eq!(bus.region_end(0x800), Some(0x1000));
        assert_eq!(bus.region_end(0xF080), Some(0xF100));
        assert_eq!(bus.region_end(0x2008), Some(0x2010));
        assert_eq!(bus.region_end(0x9000), None);
    }
}
