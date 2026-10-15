//! Built-in MMIO devices.

use std::sync::Mutex;

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
