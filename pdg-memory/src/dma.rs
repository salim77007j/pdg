//! DMA engine.
//!
//! Phase 1 implements the synchronous completion model: `submit` copies the
//! transfer through the guest-physical bus (crossing RAM, ROM and MMIO region
//! boundaries correctly) and records it. The async queueing that models the
//! PS5's custom SSD DMA path lands in Phase 4 together with the storage
//! stack — see `docs/ARCHITECTURE.md`.

use crate::{bus::Bus, error::MemoryError};

/// A completed DMA transfer record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaRecord {
    /// Monotonic transfer id.
    pub id: u64,
    /// Source guest-physical address.
    pub src: u64,
    /// Destination guest-physical address.
    pub dst: u64,
    /// Transfer length in bytes.
    pub len: u64,
}

/// Guest-physical DMA engine.
#[derive(Debug, Default)]
pub struct DmaEngine {
    next_id: u64,
    history: Vec<DmaRecord>,
}

impl DmaEngine {
    /// Create an idle engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a transfer; copies immediately through the bus and returns the
    /// completion record.
    pub fn submit(
        &mut self,
        bus: &Bus,
        src: u64,
        dst: u64,
        len: u64,
    ) -> Result<DmaRecord, MemoryError> {
        let mut buf = [0u8; 4096];
        let (mut s, mut d, mut remaining) = (src, dst, len);
        while remaining > 0 {
            let src_end = bus
                .region_end(s)
                .ok_or(MemoryError::UnmappedPhysical { addr: s })?;
            let dst_end = bus
                .region_end(d)
                .ok_or(MemoryError::UnmappedPhysical { addr: d })?;
            let chunk = remaining
                .min(src_end - s)
                .min(dst_end - d)
                .min(buf.len() as u64);
            bus.read_buf(s, &mut buf[..chunk as usize])?;
            bus.write_buf(d, &buf[..chunk as usize])?;
            s += chunk;
            d += chunk;
            remaining -= chunk;
        }
        let record = DmaRecord {
            id: self.next_id,
            src,
            dst,
            len,
        };
        self.next_id += 1;
        self.history.push(record);
        tracing::trace!(id = record.id, src, dst, len, "dma transfer complete");
        Ok(record)
    }

    /// Completed transfers, oldest first.
    pub fn history(&self) -> &[DmaRecord] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bus::Bus, devices::ConsoleDevice};
    use std::sync::Arc;

    #[test]
    fn dma_copies_within_ram() {
        let bus = Bus::new(0x10_0000).unwrap();
        bus.write_buf(0x1000, b"hello, dma world!").unwrap();
        let mut dma = DmaEngine::new();
        let rec = dma.submit(&bus, 0x1000, 0x2000, 17).unwrap();
        let mut out = [0u8; 17];
        bus.read_buf(0x2000, &mut out).unwrap();
        assert_eq!(&out, b"hello, dma world!");
        assert_eq!(rec.len, 17);
        assert_eq!(dma.history().len(), 1);
    }

    #[test]
    fn dma_crosses_region_boundaries() {
        // RAM 0..0x100, MMIO (console) 0x100..0x120: a transfer starting
        // near the end of RAM runs across the region boundary, and the
        // chunking loop must split it correctly. MMIO reads return 0.
        let mut bus = Bus::new(0x100).unwrap();
        bus.attach_mmio(0x100, 0x20, Arc::new(ConsoleDevice::new()));
        bus.write_buf(0xF8, b"01234567").unwrap();
        let mut dma = DmaEngine::new();
        dma.submit(&bus, 0xF8, 0x40, 16).unwrap();
        let mut out = [0u8; 16];
        bus.read_buf(0x40, &mut out).unwrap();
        assert_eq!(&out[..8], b"01234567"); // from RAM
        assert_eq!(&out[8..], &[0u8; 8]); // from MMIO (reads as zero)
    }

    #[test]
    fn dma_rejects_unmapped_source() {
        let bus = Bus::new(0x1000).unwrap();
        let mut dma = DmaEngine::new();
        assert!(matches!(
            dma.submit(&bus, 0x9000, 0x100, 8),
            Err(MemoryError::UnmappedPhysical { .. })
        ));
    }
}
