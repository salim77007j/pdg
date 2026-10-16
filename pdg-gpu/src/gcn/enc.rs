//! GCN/RDNA instruction encoding formats and operand semantics.
//!
//! Every bit layout in this module is verified against LLVM's public
//! AMDGPU backend definitions (`SOPInstructions.td`, `VOP1/2/3Instructions.td`,
//! `SIInstrFormats.td`, `SMInstructions.td`, `DSInstructions.td`,
//! `BUFInstructions.td`) and the operand-selector tables against
//! `SIDefines.h` (`EncValues`) + `AMDGPUDisassembler.cpp`
//! (`decodeSrcOp`/`decodeSpecialReg32`).
//!
//! Layout summary (bits are inclusive, high:low):
//!
//! | format | size | fields |
//! |--------|------|--------|
//! | SOP1   | 32   | src0{7-0}, op{15-8}, sdst{22-16}, enc{31-23}=0x17D |
//! | SOP2   | 32   | src0{7-0}, src1{15-8}, sdst{22-16}, op{29-23}, enc{31-30}=0b10 |
//! | SOPC   | 32   | src0{7-0}, src1{15-8}, op{22-16}, enc{31-23}=0x17E |
//! | SOPK   | 32   | simm16{15-0}, sdst{22-16}, op{27-23}, enc{31-28}=0xB |
//! | SOPP   | 32   | simm16{15-0}, op{22-16}, enc{31-23}=0x17F |
//! | VOP1   | 32   | src0{8-0}, op{16-9}, vdst{24-17}, enc{31-25}=0x3F |
//! | VOP2   | 32   | src0{8-0}, src1{16-9}, vdst{24-17}, op{30-25}, enc{31}=0 |
//! | VOP3   | 64   | vdst{7-0}, clamp{15}, op{25-16}, enc{31-26}=0x35 (gfx10), src0{40-32}, src1{49-41}, src2{58-50}, omod{60-59}, neg{63-61} |
//! | SMEM   | 64   | sbase{5-0}(×2), sdst{12-6}, GLC{16}, imm{17}, op{25-18}, enc{31-26}=0x30, off{52-32}, soe{63-57} |
//! | DS     | 64   | offset0{7-0}, offset1{15-8}, gds{17}, op{25-18}, enc{31-26}=0x36, addr{39-32}, data0{47-40}, data1{55-48}, vdst{63-56} |
//! | MUBUF  | 64   | offset{11-0}, offen{12}, idxen{13}, glc{14}, sccb{15}, lds{16}, slc{17}, op{24-18}, enc{31-26}=0x38, vaddr{39-32}, vdata{47-40}, srsrc{52-48}(×4), soffset{63-56} |
//! | EXP    | 64   | en{3-0}, tgt{9-4}, compr{10}, done{11}, vm{12}, enc{31-26}=0x3E, src0-3{39-32..63-56} |
//!
//! gfx10 notes baked in: VOP3 uses the 0x35 encoding with a 10-bit
//! opcode; the classic 32-bit VINTRP (0x32) was removed in RDNA —
//! interpolation is the VOP3 form (op 0x200-0x202) and is decoded as VOP3.

/// 32-bit format discriminators (bits 31..30 where applicable).
pub mod sop {
    /// SOP2: bits 31..30 = 0b10.
    pub const SOP2: u32 = 0b10;
    /// SOP1: bits 31..23 = 0x17D.
    pub const SOP1: u32 = 0x17D;
    /// SOPC: bits 31..23 = 0x17E.
    pub const SOPC: u32 = 0x17E;
    /// SOPP: bits 31..23 = 0x17F.
    pub const SOPP: u32 = 0x17F;
    /// SOPK: bits 31..28 = 0xB.
    pub const SOPK: u32 = 0xB;
}

/// 6-bit high encodings (bits 31..26).
pub mod enc6 {
    /// VOP3 (gfx10).
    pub const VOP3: u32 = 0x35;
    /// SMEM (gfx8+).
    pub const SMEM: u32 = 0x30;
    /// DS.
    pub const DS: u32 = 0x36;
    /// MUBUF.
    pub const MUBUF: u32 = 0x38;
    /// EXP.
    pub const EXP: u32 = 0x3E;
    /// VINTRP (gfx6-9 only; RDNA removed it — the decoder reports the
    /// format for diagnosis but the gfx10 opcode table has no entries).
    pub const VINTRP: u32 = 0x32;
    /// MIMG.
    pub const MIMG: u32 = 0x3C;
}

/// Extract `width` bits at `shift` from `word`.
#[must_use]
pub const fn bits(word: u32, shift: u32, width: u32) -> u32 {
    if width == 0 {
        return 0;
    }
    if width >= 32 {
        return word >> shift;
    }
    (word >> shift) & ((1 << width) - 1)
}

/// Inline constant decoded from a selector (LLVM `EncValues`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InlineConst {
    /// 32-bit integer value (selectors 0x80..0xD0 = 0..-16).
    Int(i32),
    /// 32-bit float value (selectors 0xF0..0xF8).
    F32(f32),
    /// 64-bit float value (0.5, 1.0, ... widened).
    F64(f64),
}

/// Inline-constant selector ranges (LLVM `SIDefines.h::EncValues`).
pub mod inline_sel {
    /// First integer selector (value 0).
    pub const INT_MIN: u32 = 128;
    /// Last positive integer selector (value 64).
    pub const INT_POS_MAX: u32 = 192;
    /// Last negative integer selector (value -16).
    pub const INT_MAX: u32 = 208;
    /// First float selector (0.5).
    pub const FLOAT_MIN: u32 = 240;
    /// Last float selector (1/(2π)).
    pub const FLOAT_MAX: u32 = 248;
    /// Literal-constant sentinel: the next dword holds the value.
    pub const LITERAL: u32 = 255;
}

/// Decode an inline-constant selector, if it is one.
#[must_use]
pub fn inline_const(selector: u32) -> Option<InlineConst> {
    use InlineConst::{F32, F64, Int};
    match selector {
        s if (inline_sel::INT_MIN..=inline_sel::INT_POS_MAX).contains(&s) => {
            Some(Int((s - inline_sel::INT_MIN) as i32))
        }
        s if (inline_sel::INT_POS_MAX + 1..=inline_sel::INT_MAX).contains(&s) => {
            Some(InlineConst::Int(-((s - inline_sel::INT_POS_MAX) as i32)))
        }
        inline_sel::FLOAT_MIN => Some(F32(0.5)),
        241 => Some(F32(-0.5)),
        242 => Some(F32(1.0)),
        243 => Some(F32(-1.0)),
        244 => Some(F32(2.0)),
        245 => Some(F32(-2.0)),
        246 => Some(F32(4.0)),
        247 => Some(F32(-4.0)),
        inline_sel::FLOAT_MAX => Some(F64(1.0 / (2.0 * std::f64::consts::PI))),
        _ => None,
    }
}

/// The inline-constant selector for `c`, if it is encodable.
#[must_use]
pub fn inline_selector(c: InlineConst) -> Option<u32> {
    match c {
        InlineConst::Int(v) if (0..=64).contains(&v) => Some(128 + v as u32),
        InlineConst::Int(v) if (-16..=-1).contains(&v) => Some(192 + (-v) as u32),
        InlineConst::F32(0.5) => Some(240),
        InlineConst::F32(-0.5) => Some(241),
        InlineConst::F32(1.0) => Some(242),
        InlineConst::F32(-1.0) => Some(243),
        InlineConst::F32(2.0) => Some(244),
        InlineConst::F32(-2.0) => Some(245),
        InlineConst::F32(4.0) => Some(246),
        InlineConst::F32(-4.0) => Some(247),
        InlineConst::F64(v) if (v - 1.0 / (2.0 * std::f64::consts::PI)).abs() < f64::EPSILON => {
            Some(248)
        }
        _ => None,
    }
}

/// A decoded source operand.
///
/// VGPR indices and SGPR indices are raw register numbers (an SGPR pair
/// such as `s[4:5]` decodes to [`Source::Sgpr(4)`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    /// Vector register `vN`.
    Vgpr(u8),
    /// Scalar register `sN` (first of the pair/tuple for wide operands).
    Sgpr(u8),
    /// An inline constant.
    Const(InlineConst),
    /// A literal constant (value from the trailing dword).
    Literal(u32),
    /// A special scalar (`M0`, `EXEC`, `VCC`, ...), by LLVM's encoding.
    Special(u32),
    /// Trap-handler registers.
    Ttmp(u8),
}

impl Source {
    /// Decode a 9-bit VOP source field.
    #[must_use]
    pub fn decode9(value: u32) -> Self {
        match value {
            0x100..=0x1FF => Self::Vgpr((value & 0xFF) as u8),
            0..=105 => Self::Sgpr(value as u8),
            108..=123 => Self::Ttmp((value - 108) as u8),
            v if inline_const(v).is_some() => Self::Const(inline_const(v).unwrap()),
            255 => Self::Literal(0), // filled by the decoder
            v => Self::Special(v),
        }
    }

    /// Decode an 8-bit SOP source field.
    #[must_use]
    pub fn decode8(value: u32) -> Self {
        match value {
            0..=105 => Self::Sgpr(value as u8),
            108..=123 => Self::Ttmp((value - 108) as u8),
            v if inline_const(v).is_some() => Self::Const(inline_const(v).unwrap()),
            255 => Self::Literal(0),
            v => Self::Special(v),
        }
    }

    /// Encode a 9-bit source.
    #[must_use]
    pub fn encode9(self) -> Option<u32> {
        match self {
            Self::Vgpr(v) => Some(0x100 + u32::from(v)),
            Self::Sgpr(v) if v <= 105 => Some(u32::from(v)),
            Self::Ttmp(v) => Some(108 + u32::from(v)),
            Self::Literal(_) => Some(255),
            Self::Const(c) => inline_selector(c),
            _ => None,
        }
    }

    /// Encode an 8-bit source.
    #[must_use]
    pub fn encode8(self) -> Option<u32> {
        match self {
            Self::Sgpr(v) if v <= 105 => Some(u32::from(v)),
            Self::Ttmp(v) => Some(108 + u32::from(v)),
            Self::Literal(_) => Some(255),
            Self::Const(c) => inline_selector(c),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_extraction() {
        assert_eq!(bits(0b1011_0000, 5, 3), 0b101);
        assert_eq!(bits(0xFFFF_FFFF, 31, 1), 1);
        assert_eq!(bits(0, 0, 32), 0);
        assert_eq!(bits(0x8000_0000, 31, 1), 1);
    }

    #[test]
    fn inline_integers() {
        assert_eq!(inline_const(0x80), Some(InlineConst::Int(0)));
        assert_eq!(inline_const(0x80 + 40), Some(InlineConst::Int(40)));
        assert_eq!(inline_const(0xC0), Some(InlineConst::Int(64)));
        assert_eq!(inline_const(0xC1), Some(InlineConst::Int(-1)));
        assert_eq!(inline_const(0xD0), Some(InlineConst::Int(-16)));
        assert_eq!(inline_const(0xD1), None, "gap below floats");
        assert_eq!(inline_const(0xEF), None, "gap below floats");
    }

    #[test]
    fn inline_floats() {
        assert_eq!(inline_const(0xF0), Some(InlineConst::F32(0.5)));
        assert_eq!(inline_const(0xF2), Some(InlineConst::F32(1.0)));
        assert_eq!(
            inline_const(0xF8),
            Some(InlineConst::F64(1.0 / (2.0 * std::f64::consts::PI)))
        );
        assert_eq!(inline_const(0xF9), None);
    }

    #[test]
    fn source_decode9() {
        assert_eq!(Source::decode9(0x105), Source::Vgpr(5));
        assert_eq!(Source::decode9(0x100), Source::Vgpr(0));
        assert_eq!(Source::decode9(7), Source::Sgpr(7));
        assert_eq!(Source::decode9(105), Source::Sgpr(105));
        assert_eq!(Source::decode9(0xF2), Source::Const(InlineConst::F32(1.0)));
        assert_eq!(Source::decode9(255), Source::Literal(0));
        assert_eq!(Source::decode9(126), Source::Special(126)); // EXEC_LO
        assert_eq!(Source::decode9(108), Source::Ttmp(0));
    }

    #[test]
    fn source_roundtrips() {
        for v in [
            Source::Vgpr(255),
            Source::Vgpr(0),
            Source::Sgpr(100),
            Source::Ttmp(15),
        ] {
            assert_eq!(Source::decode9(v.encode9().unwrap()), v, "9-bit");
        }
        for v in [Source::Sgpr(101), Source::Sgpr(0), Source::Ttmp(7)] {
            assert_eq!(Source::decode8(v.encode8().unwrap()), v, "8-bit");
        }
    }
}
