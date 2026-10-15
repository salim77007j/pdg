//! PM4 command-processor packet framing.
//!
//! The PS5's graphics command processor consumes PM4 packets, the packet
//! language documented publicly by AMD for GCN/RDNA parts. This module
//! implements the *framing layer* only — header encode/decode and packet
//! type dispatch. The semantic opcode table (register writes, draws,
//! dispatches) is populated in Phase 3 after cross-checking the public AMD
//! command-processor documentation; nothing here guesses opcode meanings.

/// Type value of a type-3 (opcode) packet in the two high bits.
pub const PKT_TYPE_3: u32 = 3;

/// A PM4 packet header word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header(u32);

impl Header {
    /// Build a type-3 header for `opcode` carrying `count` payload words.
    ///
    /// # Panics
    /// Panics if `count` exceeds 13 bits or `opcode` exceeds 7 bits.
    pub fn type3(opcode: u8, count: u16) -> Self {
        assert!(count < 0x4000, "PM4 payload count exceeds 13 bits");
        assert!(opcode < 0x80, "PM4 opcode exceeds 7 bits");
        Self(0xC000_0000 | (u32::from(opcode) << 16) | u32::from(count))
    }

    /// Wrap a raw header word.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw header word.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Packet type (bits 31..30).
    pub const fn packet_type(self) -> u32 {
        self.0 >> 30
    }

    /// Opcode (bits 22..16). Only meaningful for type-3 packets.
    pub const fn opcode(self) -> u8 {
        ((self.0 >> 16) & 0x7F) as u8
    }

    /// Payload word count (bits 13..0).
    pub const fn count(self) -> u16 {
        (self.0 & 0x3FFF) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type3_roundtrip() {
        let h = Header::type3(0x61, 12);
        assert_eq!(h.packet_type(), PKT_TYPE_3);
        assert_eq!(h.opcode(), 0x61);
        assert_eq!(h.count(), 12);
        assert_eq!(Header::from_raw(h.raw()), h);
    }

    #[test]
    fn zero_count_opcode_zero() {
        let h = Header::type3(0, 0);
        assert_eq!(h.raw(), 0xC000_0000);
        assert_eq!(h.count(), 0);
        assert_eq!(h.opcode(), 0);
    }

    #[test]
    #[should_panic(expected = "13 bits")]
    fn oversize_count_rejected() {
        Header::type3(0, 0x4000);
    }

    #[test]
    #[should_panic(expected = "7 bits")]
    fn oversize_opcode_rejected() {
        Header::type3(0x80, 0);
    }
}
