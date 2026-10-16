//! IDENTIFY Controller / Namespace structures.
//!
//! PDG fills only the fields it can fill *honestly* (identifying
//! itself as a PDG virtual drive); every other byte stays zero, and
//! the filled offsets are listed here so any diff against a capture
//! is meaningful:
//!
//! * Controller (CNS 0): VID/SSVID `0x5044` ("PD"), SN at 4 ("PDG-SSD-
//!   000001"), MN at 24 ("PDG Virtual NVMe SSD"), FR at 64 ("0.4.0"),
//!   NN (number of namespaces) at 536 = 1.
//! * Namespace (CNS 1): NSZE/NCAP/NUSE in the first 24 bytes (drive
//!   size in 512-byte LBAs), NLBAF at 25 = 0 (one format), FLBAS at
//!   26 = 0, LBAF[0] at 128 with LBADS = 9 (512-byte blocks, no
//!   metadata).

/// Serial number bytes (20-byte field at offset 4).
pub const SERIAL: &[u8; 16] = b"PDG-SSD-000001\0\0";
/// Model number bytes (40-byte field at offset 24).
pub const MODEL: &[u8; 34] = b"PDG Virtual NVMe SSD (no BIOS)\0\0\0\0";
/// Firmware revision (8-byte field at offset 64).
pub const FIRMWARE: &[u8; 8] = b"0.4.0\0\0\0";
/// PDG's fake PCI vendor word: ASCII "PD".
pub const VENDOR: u16 = 0x5044;

/// Build the 4096-byte IDENTIFY CONTROLLER structure for `lba_count`.
#[must_use]
pub fn identify_controller(lba_count: u64) -> [u8; 4096] {
    let mut out = [0u8; 4096];
    // VID (0), SSVID (2)
    out[0..2].copy_from_slice(&VENDOR.to_le_bytes());
    out[2..4].copy_from_slice(&VENDOR.to_le_bytes());
    // SN (4..24), MN (24..64), FR (64..72)
    out[4..20].copy_from_slice(&SERIAL[..16]);
    out[24..58].copy_from_slice(&MODEL[..34]);
    out[64..72].copy_from_slice(&FIRMWARE[..8]);
    // NN — number of namespaces (offset 536).
    out[536..540].copy_from_slice(&1u32.to_le_bytes());
    let _ = lba_count; // controller identify carries no size fields PDG fills
    out
}

/// Build the 4096-byte IDENTIFY NAMESPACE structure for `lba_count`
/// 512-byte blocks.
#[must_use]
pub fn identify_namespace(lba_count: u64) -> [u8; 4096] {
    let mut out = [0u8; 4096];
    // NSZE (0), NCAP (8), NUSE (16)
    for (off, v) in [(0usize, lba_count), (8, lba_count), (16, lba_count)] {
        out[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    // NLBAF (25): number of LBA formats minus one = 0.
    out[25] = 0;
    // FLBAS (26): format 0 in use.
    out[26] = 0;
    // LBAF[0] (128): MS = 0, LBADS = 9 (512 bytes).
    let lbaf0: u32 = 9 << 16;
    out[128..132].copy_from_slice(&lbaf0.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_identify_fields_land_at_documented_offsets() {
        let data = identify_controller(4096);
        assert_eq!(u16::from_le_bytes([data[0], data[1]]), VENDOR);
        // SN is NUL-padded ASCII starting at 4.
        let sn_end = data[4..24].iter().position(|&b| b == 0).unwrap() + 4;
        assert_eq!(&data[4..sn_end], b"PDG-SSD-000001");
        let mn_end = data[24..64].iter().position(|&b| b == 0).unwrap() + 24;
        assert_eq!(&data[24..mn_end], b"PDG Virtual NVMe SSD (no BIOS)");
        // FR at 64.
        assert_eq!(&data[64..67], b"0.4");
        // NN at 536.
        assert_eq!(u32::from_le_bytes(data[536..540].try_into().unwrap()), 1);
    }

    #[test]
    fn namespace_identify_reports_geometry() {
        let data = identify_namespace(1024);
        let nsze = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(nsze, 1024);
        assert_eq!(data[25], 0, "one LBA format");
        let lbaf0 = u32::from_le_bytes(data[128..132].try_into().unwrap());
        assert_eq!(lbaf0 >> 16, 9, "LBADS=9 → 512-byte blocks");
        assert_eq!(lbaf0 & 0xFFFF, 0, "no metadata");
    }
}
