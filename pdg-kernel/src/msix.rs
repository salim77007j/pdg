//! MSI-X-shaped interrupt plumbing.
//!
//! Devices need to signal the cores. The honest Phase-5 shape is the
//! public MSI-X layout: a guest-programmable table of entries, each
//! `{ message address u64, message data u32, vector control u32 }`, where
//! the x86 message-address magic is `0xFEE0_0000`-based and the low byte
//! of the message data is the IDT vector — both public architectural
//! facts, not guesses.
//!
//! Delivery is the same discipline every other device follows: **latch
//! now, deliver at the next pump**. A device fires its entry; the vector
//! latches in the table; the session pump — between 200-instruction
//! slices, never inside one — hands it to
//! [`pdg_cpu::Interpreter::inject_interrupt`], which pushes the
//! architectural frame through the guest IDT exactly like `#PF`
//! delivery. The guest handler runs, `iretq`s, and the interrupted code
//! continues.
//!
//! The pending-bit array (PBA) of the full spec is not guest-visible in
//! Phase 5, and masked vectors latch but are not delivered — both are
//! documented gaps, not silent ones.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use pdg_memory::bus::{AccessWidth, MmioDevice};

/// The x86 local-APIC MSI message-address base (public architectural
/// constant: destination `0xFF`, the local APIC, redirection hint off).
pub const MSI_ADDR_MAGIC: u32 = 0xFEE0_0000;

/// The `vector control` register's mask bit (public MSI-X bit).
pub const CTRL_MASK_BIT: u32 = 1;

/// A guest-programmable MSI-X-shaped vector table.
///
/// MMIO layout (16 bytes per entry, entry `i` at `i * 0x10`):
///
/// | offset | access | meaning |
/// |---|---|---|
/// | `+0x0` | RW | message address low |
/// | `+0x4` | RW | message address high |
/// | `+0x8` | RW | message data (low 8 bits = IDT vector) |
/// | `+0xC` | RW | vector control (bit 0 = masked) |
pub struct MsixTable {
    /// Entry count.
    n: usize,
    /// Flat RW image of the table (what the guest sees).
    image: std::sync::Mutex<Vec<u32>>,
    /// A vector latched by `fire`, awaiting the pump.
    irq: AtomicU8,
    /// Whether `irq` holds a value.
    irq_valid: AtomicBool,
}

impl MsixTable {
    /// A table with `n` entries, all masked and zeroed.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            image: std::sync::Mutex::new(vec![0u32; n * 4]),
            irq: AtomicU8::new(0),
            irq_valid: AtomicBool::new(false),
        }
    }

    /// Wrap in `Arc` for bus registration.
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when there are no entries.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Fire `entry`: when unmasked and its message data names a vector,
    /// latch that vector for the pump. Returns the vector when the fire
    /// will be delivered.
    pub fn fire(&self, entry: usize) -> Option<u8> {
        let image = self.image.lock().expect("msix");
        if entry >= self.n {
            return None;
        }
        let ctrl = image[entry * 4 + 3];
        if ctrl & CTRL_MASK_BIT != 0 {
            // Masked: latched pending in real hardware; Phase 5 records
            // the gap honestly instead of inventing PBA semantics.
            return None;
        }
        let data = image[entry * 4 + 2];
        let vector = (data & 0xFF) as u8;
        if vector == 0 {
            return None;
        }
        drop(image);
        self.irq.store(vector, Ordering::SeqCst);
        self.irq_valid.store(true, Ordering::SeqCst);
        Some(vector)
    }

    /// Take the latched vector for delivery (the pump calls this once per
    /// slice boundary); `requeue` puts it back when the core cannot take
    /// it yet (IF clear or no IDT).
    pub fn take_irq(&self) -> Option<u8> {
        if !self.irq_valid.swap(false, Ordering::SeqCst) {
            return None;
        }
        Some(self.irq.load(Ordering::SeqCst))
    }

    /// Put a vector back after a refused delivery.
    pub fn requeue(&self, vector: u8) {
        self.irq.store(vector, Ordering::SeqCst);
        self.irq_valid.store(true, Ordering::SeqCst);
    }

    /// Whether a vector is latched.
    pub fn irq_pending(&self) -> bool {
        self.irq_valid.load(Ordering::SeqCst)
    }
}

impl MmioDevice for MsixTable {
    fn name(&self) -> &'static str {
        "msix-table"
    }

    fn read(&self, offset: u64, width: AccessWidth) -> u64 {
        if offset % 4 != 0 {
            return 0;
        }
        let idx = (offset / 4) as usize;
        let image = self.image.lock().expect("msix");
        match width {
            AccessWidth::W32 => u64::from(image.get(idx).copied().unwrap_or(0)),
            AccessWidth::W64 => {
                let lo = u64::from(image.get(idx).copied().unwrap_or(0));
                let hi = u64::from(image.get(idx + 1).copied().unwrap_or(0));
                lo | (hi << 32)
            }
            _ => 0,
        }
    }

    fn write(&self, offset: u64, width: AccessWidth, value: u64) {
        if offset % 4 != 0 {
            return;
        }
        let idx = (offset / 4) as usize;
        let mut image = self.image.lock().expect("msix");
        match width {
            AccessWidth::W32 => {
                if let Some(slot) = image.get_mut(idx) {
                    *slot = value as u32;
                }
            }
            AccessWidth::W64 => {
                if let Some(slot) = image.get_mut(idx) {
                    *slot = value as u32;
                }
                if let Some(slot) = image.get_mut(idx + 1) {
                    *slot = (value >> 32) as u32;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_and_fires_a_vector() {
        let t = MsixTable::new(2);
        t.write(0x0, AccessWidth::W32, u64::from(MSI_ADDR_MAGIC));
        t.write(0x8, AccessWidth::W32, 0x40); // vector 0x40
        t.write(0xC, AccessWidth::W32, 0); // unmasked
        assert_eq!(t.read(0x8, AccessWidth::W32), 0x40);
        assert_eq!(t.fire(0), Some(0x40));
        assert!(t.irq_pending());
        assert_eq!(t.take_irq(), Some(0x40));
        assert!(!t.irq_pending());
        assert_eq!(t.take_irq(), None);
    }

    #[test]
    fn masked_entries_do_not_deliver() {
        let t = MsixTable::new(1);
        t.write(0x8, AccessWidth::W32, 0x41);
        t.write(0xC, AccessWidth::W32, u64::from(CTRL_MASK_BIT));
        assert_eq!(t.fire(0), None);
        assert!(!t.irq_pending());
    }

    #[test]
    fn out_of_range_fires_are_dropped() {
        let t = MsixTable::new(1);
        assert_eq!(t.fire(7), None);
        t.write(0x18, AccessWidth::W32, 0xFFFF); // entry 1 register 3 — absent
        assert_eq!(t.read(0x18, AccessWidth::W32), 0);
    }

    #[test]
    fn requeue_roundtrip() {
        let t = MsixTable::new(1);
        t.write(0x8, AccessWidth::W32, 0x55);
        t.fire(0);
        let v = t.take_irq().unwrap();
        t.requeue(v);
        assert!(t.irq_pending());
        assert_eq!(t.take_irq(), Some(0x55));
    }
}
