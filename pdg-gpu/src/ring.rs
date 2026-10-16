//! GNM-style ring MMIO device.
//!
//! The console's graphics driver submits work by placing PM4 packets in
//! a command ring in guest RAM and ringing a doorbell. PDG models that
//! interface on the memory bus:
//!
//! * The guest (emulated CPU) programs the ring base/size/write-pointer
//!   through MMIO, fills the ring, then writes the doorbell register.
//! * The doorbell latches a submission request; the host **pumps** the
//!   ring ([`GnmRingDevice::pump`]) — reading the new ring words from
//!   guest RAM, executing them through the
//!   [`CommandProcessor`](crate::pm4::cp::CommandProcessor), and
//!   rendering the produced commands on the bound
//!   [`GpuDevice`](crate::GpuDevice) with memory-backed effects (fences,
//!   target readback) landing in guest RAM.
//!
//! The split mirrors real hardware: GPUs are asynchronous, the driver
//! rings and later polls fences, and the host services the ring from its
//! own scheduling point (the frontend's render loop in Phase 6, the
//! test harness here). A doorbell write that arrives while the device is
//! busy just raises the generation counter.

use crate::GpuDevice;
use pdg_memory::bus::{AccessWidth, Bus, MmioDevice};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Ring MMIO layout (dword offsets from the device base).
pub mod regs {
    /// Ring buffer base address (guest physical).
    pub const RING_BASE: u64 = 0x00;
    /// Ring buffer size in dwords.
    pub const RING_SIZE: u64 = 0x04;
    /// Number of dwords written by the CPU (the write pointer).
    pub const RING_WPTR: u64 = 0x08;
    /// Doorbell: writing any value latches a submission request.
    pub const RING_DOORBELL: u64 = 0x0C;
    /// Total register footprint.
    pub const FOOTPRINT: u64 = 0x10;
}

/// Ring device errors.
#[derive(Debug, Error)]
pub enum RingError {
    /// The CP refused a packet.
    #[error("command processor: {0}")]
    Cp(#[from] crate::pm4::cp::CpError),
    /// The backend refused the batch.
    #[error("gpu device: {0}")]
    Gpu(#[from] crate::GpuError),
    /// The device was re-locked from within itself (a bug, not guest-possible).
    #[error("ring device busy")]
    Busy,
}

/// The GNM ring device.
pub struct GnmRingDevice {
    inner: Mutex<Inner>,
    /// Doorbell generation the host has serviced.
    serviced: AtomicU64,
    /// Doorbell generation requested by the guest.
    requested: AtomicU64,
}

struct Inner {
    /// Command processor (persistent register state).
    cp: crate::pm4::cp::CommandProcessor,
    /// Rendering backend.
    gpu: Box<dyn GpuDevice>,
    /// Ring base (guest physical).
    base: u64,
    /// Ring size in dwords.
    size: u32,
    /// CPU write pointer in dwords.
    wptr: u32,
}

impl GnmRingDevice {
    /// Create a ring device over a command processor and backend.
    #[must_use]
    pub fn new(cp: crate::pm4::cp::CommandProcessor, gpu: Box<dyn GpuDevice>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                cp,
                gpu,
                base: 0,
                size: 0,
                wptr: 0,
            }),
            serviced: AtomicU64::new(0),
            requested: AtomicU64::new(0),
        }
    }

    /// Wrap in `Arc` for bus registration.
    #[must_use]
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Service pending doorbells: execute and render everything up to
    /// the current write pointer.
    ///
    /// Returns how many doorbell generations were serviced.
    pub fn pump(&self, bus: &Bus) -> Result<u64, RingError> {
        let mut done = 0u64;
        loop {
            let requested = self.requested.load(Ordering::Acquire);
            let serviced = self.serviced.load(Ordering::Acquire);
            if requested == serviced {
                return Ok(done);
            }
            self.doorbell_once(bus)?;
            self.serviced.store(requested, Ordering::Release);
            done += 1;
        }
    }

    fn doorbell_once(&self, bus: &Bus) -> Result<(), RingError> {
        let mut inner = self.inner.lock().map_err(|_| RingError::Busy)?;
        let count = inner.wptr.min(inner.size);
        if inner.base == 0 || count == 0 {
            return Ok(());
        }
        let mut words = Vec::with_capacity(count as usize);
        for i in 0..count {
            let addr = inner.base + u64::from(i) * 4;
            words.push(bus.read(AccessWidth::W32, addr).unwrap_or(0) as u32);
        }
        let mut port = BusPort { bus };
        inner.cp.submit(&words, &mut port).map_err(RingError::Cp)?;
        let commands = inner.cp.take_commands();
        let ctx = inner.cp.context().clone_state();
        inner
            .gpu
            .submit(&ctx, &commands, &mut port)
            .map_err(RingError::Gpu)?;
        Ok(())
    }

    /// Whether a doorbell is pending service.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.requested.load(Ordering::Acquire) > self.serviced.load(Ordering::Acquire)
    }
}

impl MmioDevice for GnmRingDevice {
    fn name(&self) -> &'static str {
        "gnm-ring"
    }

    fn read(&self, offset: u64, width: AccessWidth) -> u64 {
        if width != AccessWidth::W32 {
            return 0;
        }
        if offset == regs::RING_DOORBELL {
            return self.serviced.load(Ordering::Acquire);
        }
        let Ok(inner) = self.inner.lock() else {
            return 0;
        };
        match offset {
            regs::RING_BASE => inner.base,
            regs::RING_SIZE => u64::from(inner.size),
            regs::RING_WPTR => u64::from(inner.wptr),
            _ => 0,
        }
    }

    fn write(&self, offset: u64, _width: AccessWidth, value: u64) {
        if offset == regs::RING_DOORBELL {
            // Latch the request; the host services it at its next pump.
            self.requested.fetch_add(1, Ordering::Release);
            return;
        }
        if let Ok(mut inner) = self.inner.lock() {
            match offset {
                regs::RING_BASE => inner.base = value,
                regs::RING_SIZE => inner.size = value as u32,
                regs::RING_WPTR => inner.wptr = value as u32,
                _ => {}
            }
        }
    }
}

/// A [`CpMemPort`](crate::pm4::cp::CpMemPort) over the shared bus.
pub struct BusPort<'a> {
    bus: &'a Bus,
}

impl crate::pm4::cp::CpMemPort for BusPort<'_> {
    fn read_u32(&mut self, addr: u64) -> u32 {
        self.bus.read(AccessWidth::W32, addr).unwrap_or(0) as u32
    }

    fn write_u32(&mut self, addr: u64, value: u32) {
        let _ = self.bus.write(AccessWidth::W32, addr, u64::from(value));
    }

    fn read_words(&mut self, addr: u64, len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| {
                self.bus
                    .read(AccessWidth::W32, addr + (i as u64) * 4)
                    .unwrap_or(0) as u32
            })
            .collect()
    }

    fn write_bytes(&mut self, addr: u64, data: &[u8]) {
        for (i, b) in data.iter().enumerate() {
            let _ = self
                .bus
                .write(AccessWidth::W8, addr + i as u64, u64::from(*b));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pm4::cp::CpMemPort;

    #[test]
    fn bus_port_roundtrips_words() {
        let bus = Bus::new(1 << 20).expect("bus alloc");
        let mut port = BusPort { bus: &bus };
        port.write_u32(0x1000, 0xAABB_CCDD);
        assert_eq!(port.read_u32(0x1000), 0xAABB_CCDD);
        let w = port.read_words(0x1000, 1);
        assert_eq!(w, vec![0xAABB_CCDD]);
        port.write_bytes(0x2000, &[1, 2, 3, 4]);
        assert_eq!(port.read_u32(0x2000), 0x0403_0201);
    }

    #[test]
    fn doorbell_latches_and_pump_executes() {
        use crate::ir::GpuCommand;
        use crate::pm4::Header;

        let bus = Bus::new(1 << 20).expect("bus alloc");
        // Ring at 0x1000: one DRAW_INDEX_AUTO packet + RELEASE_MEM fence.
        let words: Vec<u32> = vec![
            // SET_CONTEXT_REG: VGT_PRIMITIVE_TYPE (offset 0x2242) = TRILIST.
            Header::type3(0x69, 2).raw(),
            0x2242,
            4,
            Header::type3(0x2D, 2).raw(),
            3, // count
            0, // initiator
            Header::type3(0x49, 6).raw(),
            0,       // event
            2 << 29, // data_sel = 64-bit immediate
            0x2000,  // addr lo
            0,       // addr hi
            0xDEAD,  // data lo
            0xBEEF,  // data hi
        ];
        for (i, w) in words.iter().enumerate() {
            let _ = bus.write(AccessWidth::W32, 0x1000 + (i as u64) * 4, u64::from(*w));
        }
        let ring = GnmRingDevice::new(
            crate::pm4::cp::CommandProcessor::new(),
            Box::new(crate::NullGpu),
        )
        .shared();
        // Program the ring through MMIO.
        ring.write(regs::RING_BASE, AccessWidth::W32, 0x1000);
        ring.write(regs::RING_SIZE, AccessWidth::W32, words.len() as u64);
        ring.write(regs::RING_WPTR, AccessWidth::W32, words.len() as u64);
        ring.write(regs::RING_DOORBELL, AccessWidth::W32, 1);
        assert!(ring.has_pending());
        let serviced = ring.pump(&bus).unwrap();
        assert_eq!(serviced, 1);
        assert!(!ring.has_pending());
        assert_eq!(bus.read(AccessWidth::W32, 0x2000).unwrap(), 0xDEAD);
        assert_eq!(bus.read(AccessWidth::W32, 0x2004).unwrap(), 0xBEEF);
        let _ = GpuCommand::Nop;
    }
}
