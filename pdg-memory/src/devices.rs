//! Built-in MMIO devices.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::bus::{AccessWidth, MmioDevice};

/// Memory-mapped text console for early-boot diagnostics.
///
/// Phase 1 guest payloads (and later, early firmware traces) write bytes to
/// the console MMIO window; the host side collects them for display. This is
/// the PDG equivalent of a serial port — it keeps boot tests observable
/// without depending on any guest OS driver.
pub struct ConsoleDevice {
    out: Mutex<String>,
}

impl ConsoleDevice {
    /// Create an empty console.
    pub fn new() -> Self {
        Self {
            out: Mutex::new(String::new()),
        }
    }

    /// Snapshot everything written so far.
    pub fn output(&self) -> String {
        self.out.lock().expect("console lock poisoned").clone()
    }

    /// Discard collected output (used between test runs).
    pub fn reset(&self) {
        self.out.lock().expect("console lock poisoned").clear();
    }

    /// Restore a transcript (session snapshot resume: the pre-snapshot
    /// output stays part of the observed console).
    pub fn set_output(&self, s: &str) {
        *self.out.lock().expect("console lock poisoned") = s.to_owned();
    }

    fn emit(&self, value: u64, width: AccessWidth) {
        let mut out = self.out.lock().expect("console lock poisoned");
        for i in 0..width.size() {
            let byte = ((value >> (i * 8)) & 0xFF) as u8;
            if byte != 0 {
                out.push(byte as char);
            }
        }
    }
}

impl Default for ConsoleDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for ConsoleDevice {
    fn name(&self) -> &'static str {
        "mmio-console"
    }

    fn read(&self, _offset: u64, _width: AccessWidth) -> u64 {
        0
    }

    fn write(&self, _offset: u64, width: AccessWidth, value: u64) {
        self.emit(value, width);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_bytes_in_order() {
        let console = ConsoleDevice::new();
        console.write(0, AccessWidth::W8, b'H' as u64);
        console.write(0, AccessWidth::W8, b'i' as u64);
        assert_eq!(console.output(), "Hi");
    }

    #[test]
    fn wide_writes_emit_little_endian_bytes() {
        let console = ConsoleDevice::new();
        console.write(0, AccessWidth::W32, u64::from(u32::from_le_bytes(*b"PDG!")));
        assert_eq!(console.output(), "PDG!");
    }

    #[test]
    fn reset_clears_output() {
        let console = ConsoleDevice::new();
        console.write(0, AccessWidth::W8, b'x' as u64);
        console.reset();
        assert_eq!(console.output(), "");
    }
}

/// Set when a mailbox holds a pending vector (bit 8 of the encoded value).
const IPI_PENDING: u64 = 1 << 8;

/// Inter-processor-interrupt plumbing for the Phase-2 SMP machine.
///
/// An [`IpiHub`] holds one mailbox per logical core. Cores (or the host)
/// deliver vectors with [`IpiHub::send`]; a core consumes its mailbox with
/// [`IpiHub::poll`]. The [`IpiDevice`] MMIO front-end exposes the same
/// operations to guest code, so cross-core signaling is testable end to end
/// without any host-side API.
///
/// MMIO window layout (base = `memory.ipi_mmio_base`, length 0x1000):
///
/// | offset | access | meaning |
/// |---|---|---|
/// | `0x000` | read | logical core count |
/// | `0x000` | write | deliver an IPI: `value = (vector << 8) \| target` |
/// | `0x100 + core*8` | read | poll `core`'s mailbox: `0x100 \| vector` while pending (and clear it); `0` when empty |
/// Mirror callback invoked on every delivered IPI as `(target, vector)`.
type IpiMirror = Box<dyn Fn(u8, u8) + Send>;

pub struct IpiHub {
    mailboxes: Vec<AtomicU64>,
    on_send: Mutex<Option<IpiMirror>>,
}

impl IpiHub {
    /// Create `cores` empty mailboxes.
    pub fn new(cores: usize) -> Self {
        Self {
            mailboxes: (0..cores).map(|_| AtomicU64::new(0)).collect(),
            on_send: Mutex::new(None),
        }
    }

    /// Number of cores served by this hub.
    pub fn core_count(&self) -> usize {
        self.mailboxes.len()
    }

    /// Install a mirror callback invoked on every delivered IPI as
    /// `(target, vector)` — used to publish onto the host event bus.
    pub fn set_on_send(&self, cb: IpiMirror) {
        *self.on_send.lock().expect("ipi hub lock poisoned") = Some(cb);
    }

    /// Deliver `vector` to `target`. Returns `false` for out-of-range cores
    /// (the drop is reported; IPIs are best-effort by nature).
    pub fn send(&self, target: u8, vector: u8) -> bool {
        if let Some(m) = self.mailboxes.get(usize::from(target)) {
            m.store(IPI_PENDING | u64::from(vector), Ordering::Release);
            if let Some(cb) = self.on_send.lock().expect("ipi hub lock poisoned").as_ref() {
                cb(target, vector);
            }
            true
        } else {
            tracing::warn!(target, vector, "IPI to out-of-range core dropped");
            false
        }
    }

    /// Poll (and clear) `core`'s mailbox. `None` when empty.
    pub fn poll(&self, core: usize) -> Option<u8> {
        let m = self.mailboxes.get(core)?;
        let v = m.swap(0, Ordering::AcqRel);
        if v & IPI_PENDING != 0 {
            Some((v & 0xFF) as u8)
        } else {
            None
        }
    }

    /// Whether `core` has an undelivered IPI (diagnostics).
    pub fn pending(&self, core: usize) -> bool {
        self.mailboxes
            .get(core)
            .is_some_and(|m| m.load(Ordering::Relaxed) & IPI_PENDING != 0)
    }
}

/// MMIO front-end over an [`IpiHub`] (see the hub docs for the layout).
pub struct IpiDevice {
    hub: Arc<IpiHub>,
}

impl IpiDevice {
    /// Expose `hub` to the guest.
    pub fn new(hub: Arc<IpiHub>) -> Self {
        Self { hub }
    }

    /// The hub behind this device.
    pub fn hub(&self) -> &Arc<IpiHub> {
        &self.hub
    }
}

impl MmioDevice for IpiDevice {
    fn name(&self) -> &'static str {
        "mmio-ipi"
    }

    fn read(&self, offset: u64, _width: AccessWidth) -> u64 {
        if offset == 0 {
            return self.hub.core_count() as u64;
        }
        if offset >= 0x100 && offset % 8 == 0 {
            let core = (offset - 0x100) / 8;
            if core < self.hub.core_count() as u64 {
                return match self.hub.poll(core as usize) {
                    Some(vector) => IPI_PENDING | u64::from(vector),
                    None => 0,
                };
            }
        }
        0
    }

    fn write(&self, offset: u64, _width: AccessWidth, value: u64) {
        if offset == 0 {
            let target = (value & 0xFF) as u8;
            let vector = ((value >> 8) & 0xFF) as u8;
            self.hub.send(target, vector);
        }
        // Mailbox registers are read-only; unknown writes drop.
    }
}

#[cfg(test)]
mod ipi_tests {
    use super::*;

    #[test]
    fn ipi_hub_roundtrip() {
        let hub = IpiHub::new(4);
        assert_eq!(hub.core_count(), 4);
        assert!(!hub.pending(1));
        assert!(hub.send(1, 0x42));
        assert!(hub.pending(1));
        assert_eq!(hub.poll(1), Some(0x42));
        assert!(!hub.pending(1));
        assert_eq!(hub.poll(1), None);
        assert!(!hub.send(9, 1), "out-of-range core is dropped");
    }

    #[test]
    fn ipi_hub_mirror_callback() {
        let hub = IpiHub::new(2);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        hub.set_on_send(Box::new(move |t, v| {
            sink.lock().expect("sink poisoned").push((t, v));
        }));
        hub.send(0, 7);
        assert_eq!(*seen.lock().expect("seen poisoned"), vec![(0u8, 7u8)]);
    }

    #[test]
    fn ipi_device_mmio_layout() {
        let hub = Arc::new(IpiHub::new(2));
        let dev = IpiDevice::new(Arc::clone(&hub));
        assert_eq!(dev.read(0, AccessWidth::W32), 2);
        dev.write(0, AccessWidth::W32, (0x42 << 8) | 1);
        assert_eq!(dev.read(0x108, AccessWidth::W64), IPI_PENDING | 0x42);
        assert_eq!(dev.read(0x108, AccessWidth::W64), 0, "poll consumed it");
        assert_eq!(dev.read(0x100, AccessWidth::W64), 0, "empty mailbox");
    }
}
