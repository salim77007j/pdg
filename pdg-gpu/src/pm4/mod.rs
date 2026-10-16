//! PM4 command-processor packet handling.
//!
//! The PS5's graphics command processor consumes PM4 packets, the packet
//! language documented publicly by AMD for GCN/RDNA parts. This module
//! tree implements it in three layers:
//!
//! * [`Header`] — packet framing: type-0 (raw register writes), type-2
//!   (padding) and type-3 (opcode) headers, with validated encode/decode.
//! * [`opcode`] — the verified type-3 opcode table.
//! * [`decode`] — the packet-stream decoder, including indirect-buffer
//!   (IB) chain handling.
//! * [`regs`] — the gfx10.3 register model (offsets verified against the
//!   public kernel `gc_10_3_0` headers).
//! * [`cp`] — the command processor: executes packets against a register
//!   context and emits semantic [`crate::ir::GpuCommand`]s for a backend.
//!
//! Nothing in this module guesses semantics: every opcode either has a
//! verified implementation or is rejected with a precise error naming it.

pub mod cp;
pub mod decode;
pub mod opcode;
pub mod regs;

pub use decode::PacketDecoder;

/// Type value of a type-0 (raw register write) packet in the two high bits.
pub const PKT_TYPE_0: u32 = 0;
/// Type value of a type-2 (padding) packet in the two high bits.
pub const PKT_TYPE_2: u32 = 2;
/// Type value of a type-3 (opcode) packet in the two high bits.
pub const PKT_TYPE_3: u32 = 3;

/// A PM4 packet header word.
///
/// Layout (AMD public documentation; identical across GCN→RDNA):
///
/// ```text
/// type-0: [ 0 t=0 | count(13) | base_reg(16) ]
/// type-2: [ 0x00000000 ]
/// type-3: [ 1 t=3 | predicate | opcode(7) | count(13) ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header(u32);

impl Header {
    /// Build a type-0 header writing `count+1` words to `base_reg`.
    ///
    /// Type-0 layout: type bits `0b00`, base register in bits
    /// [29:16], count in [13:0].
    ///
    /// # Panics
    /// Panics if `base_reg` exceeds 14 bits or `count` exceeds 13 bits.
    pub fn type0(base_reg: u16, count: u16) -> Self {
        assert!(count < 0x4000, "PM4 payload count exceeds 13 bits");
        assert!(
            base_reg < 0x4000,
            "PM4 type-0 base register exceeds 14 bits"
        );
        Self((u32::from(count) & 0x3FFF) | (u32::from(base_reg) << 16))
    }

    /// Build a type-2 (padding) header: type bits `0b10`, rest zero.
    pub fn type2() -> Self {
        Self(0x8000_0000)
    }

    /// Build a type-3 header for `opcode` carrying `count` payload words.
    ///
    /// gfx9+ widened the opcode field to 8 bits ([23:16]).
    ///
    /// # Panics
    /// Panics if `count` exceeds 13 bits or `opcode` exceeds 8 bits.
    pub fn type3(opcode: u8, count: u16) -> Self {
        assert!(count < 0x4000, "PM4 payload count exceeds 13 bits");
        Self(0xC000_0000 | (u32::from(opcode) << 16) | (u32::from(count) & 0x3FFF))
    }

    /// Wrap a raw header word.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw header word.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Packet type (bits 31..30).
    #[must_use]
    pub const fn packet_type(self) -> u32 {
        self.0 >> 30
    }

    /// Predicate bit (bit 28, type-3 only).
    #[must_use]
    pub const fn predicate(self) -> bool {
        (self.0 >> 28) & 1 == 1
    }

    /// Opcode (bits 23..16, 8 bits on gfx9+). Type-3 only.
    #[must_use]
    pub const fn opcode(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    /// Payload word count (bits 13..0).
    ///
    /// For type-0 packets this is `n-1` (a count of 0 means one word),
    /// matching the hardware; type-3 counts are exact.
    #[must_use]
    pub const fn count(self) -> u16 {
        (self.0 & 0x3FFF) as u16
    }

    /// Type-0 base register offset (bits 29..16).
    #[must_use]
    pub const fn base_reg(self) -> u16 {
        ((self.0 >> 16) & 0x3FFF) as u16
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
        assert!(!h.predicate());
        assert_eq!(Header::from_raw(h.raw()), h);
    }

    #[test]
    fn type3_predicate_bit() {
        let raw = 0xC000_0000 | (1 << 28) | (0x10 << 16) | 3;
        let h = Header::from_raw(raw);
        assert!(h.predicate());
        assert_eq!(h.opcode(), 0x10);
        assert_eq!(h.count(), 3);
    }

    #[test]
    fn type0_roundtrip() {
        let h = Header::type0(0x2000, 4);
        assert_eq!(h.packet_type(), PKT_TYPE_0);
        assert_eq!(h.base_reg(), 0x2000);
        assert_eq!(h.count(), 4);
        assert_eq!(h.raw(), 0x2000_0004);
    }

    #[test]
    fn type2_is_padding() {
        let h = Header::type2();
        assert_eq!(h.packet_type(), PKT_TYPE_2);
        assert_eq!(h.raw(), 0x8000_0000);
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
    fn wide_gfx9_opcodes_encode() {
        // KFD packets (0xA0+) use the widened 8-bit opcode field.
        let h = Header::type3(0xA2, 4);
        assert_eq!(h.opcode(), 0xA2);
        assert_eq!(h.raw(), 0xC0A2_0004);
    }
}
