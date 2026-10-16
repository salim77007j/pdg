//! GCN (gfx10 / RDNA) instruction decoder and test encoder.
//!
//! The decoder walks a program image (`&[u32]`) and produces [`Inst`]s.
//! Literal-constant sources consume the trailing dword, exactly like the
//! hardware. The [`Encoder`] builds programs from the same structures —
//! its outputs feed the roundtrip tests, the lifter tests, and the
//! end-to-end GPU integration suite.
//!
//! Field semantics verified against LLVM's public AMDGPU backend and
//! its own test encodings (e.g. `v_add_f32_e32 v1, v2, v3` → dword
//! `0x06020702`: src0 = `0x102` = VGPR2 through the 9-bit enum, src1 =
//! `0x03` = VGPR3 through the plain 8-bit index, vdst = `0x01`).
//!
//! Branch targets in SOPP are *relative* (dword offset from the next
//! instruction), matching the ISA; the decoder keeps them raw.

#[cfg(test)]
use super::enc::InlineConst;
use super::enc::{self, Source};
use thiserror::Error;

/// Extract `width` bits at `shift` from a 64-bit word.
#[must_use]
pub const fn bits64(word: u64, shift: u32, width: u32) -> u64 {
    if width == 0 {
        return 0;
    }
    (word >> shift) & ((1u64 << width) - 1)
}

/// How an SMEM offset is expressed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmemOffset {
    /// Immediate offset, in dwords (gfx9+ scaling).
    Imm(u32),
    /// Offset held in an SGPR (the SOE field).
    Sgpr(u8),
}

/// A decode failure.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// The image ended mid-instruction.
    #[error("truncated instruction at word {at}")]
    Truncated {
        /// Word offset.
        at: usize,
    },
    /// A literal-constant source had no literal dword behind it.
    #[error("literal source without a literal dword at word {at}")]
    MissingLiteral {
        /// Word offset.
        at: usize,
    },
}

/// A decoded instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
    /// SOP1: one scalar source, one scalar destination.
    Sop1 {
        /// Opcode number.
        op: u8,
        /// Destination SGPR (or special).
        sdst: Source,
        /// Source operand.
        src0: Source,
    },
    /// SOP2: two scalar sources, one scalar destination.
    Sop2 {
        /// Opcode number.
        op: u8,
        /// Destination SGPR (or special).
        sdst: Source,
        /// First source.
        src0: Source,
        /// Second source.
        src1: Source,
    },
    /// SOPC: scalar compare (writes SCC).
    Sopc {
        /// Opcode number.
        op: u8,
        /// First source.
        src0: Source,
        /// Second source.
        src1: Source,
    },
    /// SOPK: scalar immediate op.
    Sopk {
        /// Opcode number.
        op: u8,
        /// Destination SGPR.
        sdst: Source,
        /// Signed 16-bit immediate.
        simm16: i16,
    },
    /// SOPP: scalar control-flow / synchronization.
    Sopp {
        /// Opcode number.
        op: u8,
        /// Raw 16-bit payload (branch offset in dwords for branches).
        simm16: u16,
    },
    /// SMEM: scalar memory load.
    Smem {
        /// Opcode number.
        op: u8,
        /// Destination SGPR base (tuple extends upward).
        sdst: Source,
        /// SGPR-pair base (field×2 = first SGPR).
        sbase: Source,
        /// Offset source.
        offset: SmemOffset,
        /// GLC cache bit.
        glc: bool,
    },
    /// VOP1: one vector source.
    Vop1 {
        /// Opcode number.
        op: u8,
        /// Destination VGPR.
        vdst: u8,
        /// Source operand.
        src0: Source,
    },
    /// VOP2: two vector sources (e64 forms decode as [`Inst::Vop3`]).
    ///
    /// The 8-bit `src1` field encodes a plain VGPR index (verified
    /// against LLVM's own test encodings: `v_add_f32_e32 v1, v2, v3`
    /// encodes src1 = 3 for `v3`), with 0xFF as the literal sentinel;
    /// SGPR sources in src1 position require the e64 form.
    Vop2 {
        /// Opcode number.
        op: u8,
        /// Destination VGPR.
        vdst: u8,
        /// First source (9-bit enum).
        src0: Source,
        /// Second source (8-bit plain VGPR / literal).
        src1: Source,
    },
    /// VOP3: three sources, modifiers, 10-bit opcode.
    Vop3 {
        /// Opcode number.
        op: u16,
        /// Destination VGPR.
        vdst: u8,
        /// First source.
        src0: Source,
        /// Second source.
        src1: Source,
        /// Third source.
        src2: Source,
        /// Negation modifiers (src0, src1, src2).
        neg: [bool; 3],
        /// Absolute-value modifiers (src0, src1, src2).
        abs: [bool; 3],
        /// Output modifier (0=none, 1=mul2, 2=mul4, 3=div2).
        omod: u8,
        /// Clamp modifier.
        clamp: bool,
    },
    /// DS: LDS read/write.
    Ds {
        /// Opcode number.
        op: u8,
        /// Address VGPR.
        addr: u8,
        /// Data VGPR 0.
        data0: u8,
        /// Data VGPR 1.
        data1: u8,
        /// Destination VGPR.
        vdst: u8,
        /// Byte offset 0.
        offset0: u16,
        /// Byte offset 1.
        offset1: u16,
        /// GDS (vs LDS) selector.
        gds: bool,
    },
    /// MUBUF: buffer load/store through a descriptor SGPR quad.
    Mubuf {
        /// Opcode number.
        op: u8,
        /// Data VGPR (base).
        vdata: u8,
        /// Address VGPR (base).
        vaddr: u8,
        /// Resource-descriptor SGPR base (field×4).
        srsrc: Source,
        /// SGPR offset base.
        soffset: Source,
        /// Unsigned byte offset.
        offset: u16,
        /// Offsets from `vaddr`.
        offen: bool,
        /// Index from `vaddr` (descriptor stride).
        idxen: bool,
        /// GLC cache bit.
        glc: bool,
    },
    /// EXP: param/pixel export.
    Exp {
        /// Export target (0-7 params, 8-63 MRTZ).
        tgt: u8,
        /// Which source lanes are enabled.
        en: [bool; 4],
        /// Compressed (packed) export.
        compr: bool,
        /// End-of-program marker.
        done: bool,
        /// Valid mask.
        vm: bool,
        /// Source VGPRs.
        src: [u8; 4],
    },
}

impl Inst {
    /// Size in words (4 or 8 bytes worth, plus a literal dword).
    #[must_use]
    pub fn size_words(&self) -> usize {
        let has_literal = match self {
            Self::Sop1 { src0, .. } | Self::Vop1 { src0, .. } => is_literal(src0),
            Self::Sop2 { src0, src1, .. }
            | Self::Sopc { src0, src1, .. }
            | Self::Vop2 { src0, src1, .. } => is_literal(src0) || is_literal(src1),
            Self::Vop3 {
                src0, src1, src2, ..
            } => is_literal(src0) || is_literal(src1) || is_literal(src2),
            _ => false,
        };
        let body = match self {
            Self::Sop1 { .. }
            | Self::Sop2 { .. }
            | Self::Sopc { .. }
            | Self::Sopk { .. }
            | Self::Sopp { .. }
            | Self::Vop1 { .. }
            | Self::Vop2 { .. } => 1,
            _ => 2,
        };
        body + usize::from(has_literal)
    }
}

fn is_literal(s: &Source) -> bool {
    matches!(s, Source::Literal(_))
}

/// Decode a whole program image.
///
/// Unknown opcodes still decode (the opcode number is kept); the lifter
/// decides what it can compile.
pub fn decode_program(image: &[u32]) -> Result<Vec<(usize, Inst)>, DecodeError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < image.len() {
        let (inst, consumed) = decode_at(image, at)?;
        out.push((at, inst));
        at += consumed;
    }
    Ok(out)
}

/// Decode one instruction at `at`, returning it and words consumed.
fn decode_at(image: &[u32], at: usize) -> Result<(Inst, usize), DecodeError> {
    let w = image.get(at).ok_or(DecodeError::Truncated { at })?;
    let w = *w;
    match w >> 30 {
        0b01 if enc::bits(w, 25, 7) == 0x3F => {
            // VOP1: src0{8-0}, op{16-9}, vdst{24-17}, enc{31-25}=0x3F.
            // (VOP1 is the VOP2 opcode-space marker 0b111111.)
            let src0 = finish_literal(Source::decode9(enc::bits(w, 0, 9)), image, at)?;
            let op = enc::bits(w, 9, 8) as u8;
            let vdst = enc::bits(w, 17, 8) as u8;
            Ok((Inst::Vop1 { op, vdst, src0 }, consumed_words(&[src0])))
        }
        0b01 | 0b00 => {
            // VOP2 (opcodes < 0x20 live under 0b00, >= 0x20 under 0b01;
            // the 0x3F slot is the VOP1 marker handled above).
            let src0 = finish_literal(Source::decode9(enc::bits(w, 0, 9)), image, at)?;
            let src1 = finish_literal(decode_vop2_src1(enc::bits(w, 9, 8)), image, at)?;
            let vdst = enc::bits(w, 17, 8) as u8;
            let op = enc::bits(w, 25, 6) as u8;
            Ok((
                Inst::Vop2 {
                    op,
                    vdst,
                    src0,
                    src1,
                },
                consumed_words(&[src0, src1]),
            ))
        }
        0b10 => {
            // SOP1/SOPC/SOPP/SOPK/SOP2 — disambiguated by the high
            // encoding fields. The three fixed-encoding SOP formats must
            // be tested before SOPK/SOP2, whose prefixes cover them.
            if enc::bits(w, 23, 9) == enc::sop::SOP1 {
                let src0 = finish_literal(Source::decode8(enc::bits(w, 0, 8)), image, at)?;
                let op = enc::bits(w, 8, 8) as u8;
                let sdst = Source::decode8(enc::bits(w, 16, 7));
                Ok((Inst::Sop1 { op, sdst, src0 }, consumed_words(&[src0])))
            } else if enc::bits(w, 23, 9) == enc::sop::SOPC {
                let src0 = finish_literal(Source::decode8(enc::bits(w, 0, 8)), image, at)?;
                let src1 = finish_literal(Source::decode8(enc::bits(w, 8, 8)), image, at)?;
                let op = enc::bits(w, 16, 7) as u8;
                Ok((Inst::Sopc { op, src0, src1 }, consumed_words(&[src0, src1])))
            } else if enc::bits(w, 23, 9) == enc::sop::SOPP {
                let simm16 = enc::bits(w, 0, 16) as u16;
                let op = enc::bits(w, 16, 7) as u8;
                Ok((Inst::Sopp { op, simm16 }, 1))
            } else if enc::bits(w, 28, 4) == enc::sop::SOPK {
                let simm16 = enc::bits(w, 0, 16) as u16;
                let sdst = Source::decode8(enc::bits(w, 16, 7));
                let op = enc::bits(w, 23, 5) as u8;
                Ok((
                    Inst::Sopk {
                        op,
                        sdst,
                        simm16: simm16 as i16,
                    },
                    1,
                ))
            } else {
                // SOP2: src0{7-0}, src1{15-8}, sdst{22-16}, op{29-23}.
                let src0 = finish_literal(Source::decode8(enc::bits(w, 0, 8)), image, at)?;
                let src1 = finish_literal(Source::decode8(enc::bits(w, 8, 8)), image, at)?;
                let sdst = Source::decode8(enc::bits(w, 16, 7));
                let op = enc::bits(w, 23, 7) as u8;
                Ok((
                    Inst::Sop2 {
                        op,
                        sdst,
                        src0,
                        src1,
                    },
                    consumed_words(&[src0, src1]),
                ))
            }
        }
        0b11 => {
            // 64-bit formats, {31-26} ∈ {VOP3=0x35, SMEM=0x30, DS=0x36,
            // MUBUF=0x38, EXP=0x3E, ...}. First word is the low dword.
            let hi6 = enc::bits(w, 26, 6);
            let second = u64::from(*image.get(at + 1).ok_or(DecodeError::Truncated { at })?);
            let q = (second << 32) | u64::from(w);
            match hi6 {
                x if x == enc::enc6::VOP3 => {
                    let op = enc::bits(w, 16, 10) as u16;
                    let vdst = enc::bits(w, 0, 8) as u8;
                    let src0 = finish_literal(Source::decode9(bits64(q, 32, 9) as u32), image, at)?;
                    let src1 = finish_literal(Source::decode9(bits64(q, 41, 9) as u32), image, at)?;
                    let src2 = finish_literal(Source::decode9(bits64(q, 50, 9) as u32), image, at)?;
                    Ok((
                        Inst::Vop3 {
                            op,
                            vdst,
                            src0,
                            src1,
                            src2,
                            neg: [
                                bits64(q, 61, 1) == 1,
                                bits64(q, 62, 1) == 1,
                                bits64(q, 63, 1) == 1,
                            ],
                            abs: [
                                enc::bits(w, 8, 1) == 1,
                                enc::bits(w, 9, 1) == 1,
                                enc::bits(w, 10, 1) == 1,
                            ],
                            omod: bits64(q, 59, 2) as u8,
                            clamp: enc::bits(w, 15, 1) == 1,
                        },
                        2 + literal_words(&[src0, src1, src2]),
                    ))
                }
                x if x == enc::enc6::SMEM => {
                    // sbase{5-0} (pair), sdst{12-6}, glc{16}, imm{17},
                    // op{25-18}, offset{52-32} in dwords, soe{63-57}.
                    let op = enc::bits(w, 18, 8) as u8;
                    let sdst = Source::decode8(enc::bits(w, 6, 7));
                    let sbase = Source::Sgpr((enc::bits(w, 0, 6) * 2) as u8);
                    let imm = enc::bits(w, 17, 1) == 1;
                    let soe = bits64(q, 57, 7) as u32;
                    let offset = if imm {
                        SmemOffset::Imm(bits64(q, 32, 21) as u32)
                    } else if soe != 0 {
                        SmemOffset::Sgpr(soe as u8)
                    } else {
                        SmemOffset::Imm(0)
                    };
                    Ok((
                        Inst::Smem {
                            op,
                            sdst,
                            sbase,
                            offset,
                            glc: enc::bits(w, 16, 1) == 1,
                        },
                        2,
                    ))
                }
                x if x == enc::enc6::DS => {
                    let op = enc::bits(w, 18, 8) as u8;
                    Ok((
                        Inst::Ds {
                            op,
                            addr: bits64(q, 32, 8) as u8,
                            data0: bits64(q, 40, 8) as u8,
                            data1: bits64(q, 48, 8) as u8,
                            vdst: bits64(q, 56, 8) as u8,
                            offset0: enc::bits(w, 0, 8) as u16,
                            offset1: enc::bits(w, 8, 8) as u16,
                            gds: enc::bits(w, 17, 1) == 1,
                        },
                        2,
                    ))
                }
                x if x == enc::enc6::MUBUF => {
                    let op = enc::bits(w, 18, 7) as u8;
                    Ok((
                        Inst::Mubuf {
                            op,
                            vaddr: bits64(q, 32, 8) as u8,
                            vdata: bits64(q, 40, 8) as u8,
                            srsrc: Source::Sgpr((bits64(q, 48, 5) * 4) as u8),
                            soffset: Source::decode8(bits64(q, 56, 8) as u32),
                            offset: enc::bits(w, 0, 12) as u16,
                            offen: enc::bits(w, 12, 1) == 1,
                            idxen: enc::bits(w, 13, 1) == 1,
                            glc: enc::bits(w, 14, 1) == 1,
                        },
                        2,
                    ))
                }
                x if x == enc::enc6::EXP => Ok((
                    Inst::Exp {
                        tgt: enc::bits(w, 4, 6) as u8,
                        en: [
                            enc::bits(w, 0, 1) == 1,
                            enc::bits(w, 1, 1) == 1,
                            enc::bits(w, 2, 1) == 1,
                            enc::bits(w, 3, 1) == 1,
                        ],
                        compr: enc::bits(w, 10, 1) == 1,
                        done: enc::bits(w, 11, 1) == 1,
                        vm: enc::bits(w, 12, 1) == 1,
                        src: [
                            bits64(q, 32, 8) as u8,
                            bits64(q, 40, 8) as u8,
                            bits64(q, 48, 8) as u8,
                            bits64(q, 56, 8) as u8,
                        ],
                    },
                    2,
                )),
                _ => Err(DecodeError::Truncated { at }),
            }
        }
        _ => Err(DecodeError::Truncated { at }),
    }
}

/// Decode the VOP2 8-bit src1 field: plain VGPR index, 0xFF literal.
fn decode_vop2_src1(v: u32) -> Source {
    if v == 0xFF {
        Source::Literal(0)
    } else {
        Source::Vgpr(v as u8)
    }
}

/// If the source is the literal sentinel, attach the trailing dword.
fn finish_literal(src: Source, image: &[u32], at: usize) -> Result<Source, DecodeError> {
    if matches!(src, Source::Literal(_)) {
        let lit = *image
            .get(at + 1)
            .ok_or(DecodeError::MissingLiteral { at })?;
        Ok(Source::Literal(lit))
    } else {
        Ok(src)
    }
}

/// Words consumed given possible literal sources.
fn consumed_words(srcs: &[Source]) -> usize {
    1 + literal_words(srcs)
}

fn literal_words(srcs: &[Source]) -> usize {
    usize::from(srcs.iter().any(is_literal))
}

/// A test-side encoder for the same subset (mirrors the decode tables;
/// its purpose is roundtrip tests and building hand-written shaders).
pub struct Encoder {
    /// Output words.
    pub words: Vec<u32>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// New empty program.
    #[must_use]
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Current length in dwords.
    #[must_use]
    pub fn len_words(&self) -> u32 {
        self.words.len() as u32
    }

    /// Whether the program is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    fn push_literal(&mut self, srcs: &[Source]) {
        for s in srcs {
            if let Source::Literal(v) = s {
                self.words.push(*v);
                return;
            }
        }
    }

    /// Append a SOP1 instruction.
    pub fn sop1(&mut self, op: u8, sdst: u8, src0: Source) -> &mut Self {
        let src0e = src0.encode8().unwrap_or(0xFF);
        let w = (enc::sop::SOP1 << 23) | (u32::from(sdst) << 16) | (u32::from(op) << 8) | src0e;
        self.words.push(w);
        self.push_literal(&[src0]);
        self
    }

    /// Append a SOP2 instruction.
    pub fn sop2(&mut self, op: u8, sdst: u8, src0: Source, src1: Source) -> &mut Self {
        let src0e = src0.encode8().unwrap_or(0xFF);
        let src1e = src1.encode8().unwrap_or(0xFF);
        let w = (enc::sop::SOP2 << 30)
            | (u32::from(op) << 23)
            | (u32::from(sdst) << 16)
            | (src1e << 8)
            | src0e;
        self.words.push(w);
        self.push_literal(&[src0, src1]);
        self
    }

    /// Append a SOPC compare.
    pub fn sopc(&mut self, op: u8, src0: Source, src1: Source) -> &mut Self {
        let src0e = src0.encode8().unwrap_or(0xFF);
        let src1e = src1.encode8().unwrap_or(0xFF);
        let w = (enc::sop::SOPC << 23) | (u32::from(op) << 16) | (src1e << 8) | src0e;
        self.words.push(w);
        self.push_literal(&[src0, src1]);
        self
    }

    /// Append a SOPK instruction.
    pub fn sopk(&mut self, op: u8, sdst: u8, simm16: u16) -> &mut Self {
        let w = (enc::sop::SOPK << 28)
            | (u32::from(op) << 23)
            | (u32::from(sdst) << 16)
            | u32::from(simm16);
        self.words.push(w);
        self
    }

    /// Append a SOPP instruction (`simm16` = raw payload).
    pub fn sopp(&mut self, op: u8, simm16: u16) -> &mut Self {
        let w = (enc::sop::SOPP << 23) | (u32::from(op) << 16) | u32::from(simm16);
        self.words.push(w);
        self
    }

    /// Append an SMEM load (`sbase`/`sdst` are SGPR base indices).
    pub fn smem_load(&mut self, op: u8, sdst: u8, sbase: u8, offset_dwords: u32) -> &mut Self {
        let w0 = (enc::enc6::SMEM << 26)
            | (u32::from(op) << 18)
            | (1 << 17) // imm offset present
            | (u32::from(sdst) << 6)
            | u32::from(sbase >> 1);
        let w1 = offset_dwords & 0x1F_FFFF;
        self.words.push(w0);
        self.words.push(w1);
        self
    }

    /// Append a VOP1 instruction.
    pub fn vop1(&mut self, op: u8, vdst: u8, src0: Source) -> &mut Self {
        let src0e = src0.encode9().unwrap_or(0xFF);
        let w = (0x3F << 25) | (u32::from(vdst) << 17) | (u32::from(op) << 9) | src0e;
        self.words.push(w);
        self.push_literal(&[src0]);
        self
    }

    /// Append a VOP2 instruction.
    pub fn vop2(&mut self, op: u8, vdst: u8, src0: Source, src1: Source) -> &mut Self {
        let src0e = src0.encode9().unwrap_or(0xFF);
        let src1e = match src1 {
            Source::Vgpr(v) => u32::from(v),
            Source::Literal(_) => 0xFF,
            _ => 0xFF,
        };
        let w = (u32::from(op) << 25) | (u32::from(vdst) << 17) | (src1e << 9) | src0e;
        self.words.push(w);
        self.push_literal(&[src0, src1]);
        self
    }

    /// Append a VOP3 instruction.
    pub fn vop3(
        &mut self,
        op: u16,
        vdst: u8,
        src0: Source,
        src1: Source,
        src2: Source,
    ) -> &mut Self {
        let src0e = u64::from(src0.encode9().unwrap_or(0xFF));
        let src1e = u64::from(src1.encode9().unwrap_or(0xFF));
        let src2e = u64::from(src2.encode9().unwrap_or(0xFF));
        let v: u64 = (u64::from(enc::enc6::VOP3) << 26)
            | (u64::from(op) << 16)
            | u64::from(vdst)
            | (src0e << 32)
            | (src1e << 41)
            | (src2e << 50);
        self.words.push((v & 0xFFFF_FFFF) as u32);
        self.words.push((v >> 32) as u32);
        self.push_literal(&[src0, src1, src2]);
        self
    }

    /// Append a DS instruction.
    #[allow(clippy::too_many_arguments)]
    pub fn ds(
        &mut self,
        op: u8,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
        offset0: u16,
        offset1: u16,
    ) -> &mut Self {
        let w0 = (enc::enc6::DS << 26)
            | (u32::from(op) << 18)
            | u32::from(offset0)
            | (u32::from(offset1) << 8);
        let w1 = u32::from(addr)
            | (u32::from(data0) << 8)
            | (u32::from(data1) << 16)
            | (u32::from(vdst) << 24);
        self.words.push(w0);
        self.words.push(w1);
        self
    }

    /// Append an MUBUF instruction.
    #[allow(clippy::too_many_arguments)]
    pub fn mubuf(
        &mut self,
        op: u8,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
    ) -> &mut Self {
        let w0 = (enc::enc6::MUBUF << 26)
            | (u32::from(op) << 18)
            | u32::from(offset)
            | (u32::from(offen) << 12)
            | (u32::from(idxen) << 13);
        let w1 = u32::from(vaddr)
            | (u32::from(vdata) << 8)
            | ((u32::from(srsrc >> 2) & 0x1F) << 16)
            | (u32::from(soffset) << 24);
        self.words.push(w0);
        self.words.push(w1);
        self
    }

    /// Append an EXP instruction.
    pub fn exp(&mut self, tgt: u8, en: [bool; 4], done: bool, src: [u8; 4]) -> &mut Self {
        let mut w0 = (enc::enc6::EXP << 26) | (u32::from(tgt) << 4);
        for (i, e) in en.iter().enumerate() {
            w0 |= u32::from(*e) << i;
        }
        if done {
            w0 |= 1 << 11;
        }
        let w1 = u32::from(src[0])
            | (u32::from(src[1]) << 8)
            | (u32::from(src[2]) << 16)
            | (u32::from(src[3]) << 24);
        self.words.push(w0);
        self.words.push(w1);
        self
    }

    /// Finish and return the program words.
    #[must_use]
    pub fn build(&self) -> Vec<u32> {
        self.words.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::opcodes as ops;
    use super::*;

    fn lit(v: u32) -> Source {
        Source::Literal(v)
    }

    #[test]
    fn llvm_reference_encoding_decodes() {
        // LLVM vop2.s: `v_add_f32_e32 v1, v2, v3` (SI bytes
        // [0x02,0x07,0x02,0x06] little-endian dword):
        let w = 0x0602_0702u32;
        let decoded = decode_program(&[w]).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Vop2 {
                op: ops::vop2::V_ADD_F32,
                vdst: 1,
                src0: Source::Vgpr(2),
                src1: Source::Vgpr(3),
            }
        );
        // And `v_add_f32 v1, 1.0, v3` → [0xf2,0x06,0x02,0x06]:
        let w = 0x0602_06F2u32;
        let decoded = decode_program(&[w]).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Vop2 {
                op: ops::vop2::V_ADD_F32,
                vdst: 1,
                src0: Source::Const(InlineConst::F32(1.0)),
                src1: Source::Vgpr(3),
            }
        );
    }

    #[test]
    fn vop1_roundtrip() {
        let mut e = Encoder::new();
        e.vop1(ops::vop1::V_MOV_B32, 5, Source::Sgpr(4));
        let prog = e.build();
        let decoded = decode_program(&prog).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded[0].1,
            Inst::Vop1 {
                op: ops::vop1::V_MOV_B32,
                vdst: 5,
                src0: Source::Sgpr(4),
            }
        );
    }

    #[test]
    fn vop2_with_inline_and_literal() {
        let mut e = Encoder::new();
        e.vop2(
            ops::vop2::V_ADD_F32,
            1,
            Source::Const(InlineConst::F32(1.0)),
            lit(0x40490FDB),
        );
        let prog = e.build();
        assert_eq!(prog.len(), 2, "literal consumes a dword");
        let decoded = decode_program(&prog).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Vop2 {
                op: ops::vop2::V_ADD_F32,
                vdst: 1,
                src0: Source::Const(InlineConst::F32(1.0)),
                src1: lit(0x40490FDB),
            }
        );
        assert_eq!(decoded[0].0, 0);
    }

    #[test]
    fn sop2_roundtrip() {
        let mut e = Encoder::new();
        e.sop2(ops::sop2::S_ADD_U32, 0, Source::Sgpr(2), Source::Sgpr(3));
        let decoded = decode_program(&e.build()).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Sop2 {
                op: ops::sop2::S_ADD_U32,
                sdst: Source::Sgpr(0),
                src0: Source::Sgpr(2),
                src1: Source::Sgpr(3),
            }
        );
    }

    #[test]
    fn sopp_and_sopk_roundtrip() {
        let mut e = Encoder::new();
        e.sopk(ops::sopk::S_MOVK_I32, 5, 0x1234);
        e.sopp(ops::sopp::S_ENDPGM, 0);
        let decoded = decode_program(&e.build()).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Sopk {
                op: ops::sopk::S_MOVK_I32,
                sdst: Source::Sgpr(5),
                simm16: 0x1234,
            }
        );
        assert_eq!(
            decoded[1].1,
            Inst::Sopp {
                op: ops::sopp::S_ENDPGM,
                simm16: 0
            }
        );
    }

    #[test]
    fn smem_roundtrip() {
        let mut e = Encoder::new();
        e.smem_load(ops::smem::S_LOAD_DWORDX4, 4, 8, 0x40);
        let decoded = decode_program(&e.build()).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Smem {
                op: ops::smem::S_LOAD_DWORDX4,
                sdst: Source::Sgpr(4),
                sbase: Source::Sgpr(8),
                offset: SmemOffset::Imm(0x40),
                glc: false,
            }
        );
    }

    #[test]
    fn vop3_roundtrip() {
        let mut e = Encoder::new();
        e.vop3(
            ops::vop3::V_MAD_F32,
            7,
            Source::Vgpr(1),
            Source::Vgpr(2),
            Source::Const(InlineConst::F32(0.5)),
        );
        let decoded = decode_program(&e.build()).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Vop3 {
                op: ops::vop3::V_MAD_F32,
                vdst: 7,
                src0: Source::Vgpr(1),
                src1: Source::Vgpr(2),
                src2: Source::Const(InlineConst::F32(0.5)),
                neg: [false; 3],
                abs: [false; 3],
                omod: 0,
                clamp: false,
            }
        );
    }

    #[test]
    fn ds_and_exp_roundtrip() {
        let mut e = Encoder::new();
        e.ds(ops::ds::DS_WRITE_B32, 3, 4, 0, 0, 0x10, 0);
        e.exp(0, [true; 4], true, [1, 2, 3, 4]);
        let decoded = decode_program(&e.build()).unwrap();
        assert_eq!(
            decoded[0].1,
            Inst::Ds {
                op: ops::ds::DS_WRITE_B32,
                addr: 3,
                data0: 4,
                data1: 0,
                vdst: 0,
                offset0: 0x10,
                offset1: 0,
                gds: false,
            }
        );
        assert_eq!(
            decoded[1].1,
            Inst::Exp {
                tgt: 0,
                en: [true; 4],
                compr: false,
                done: true,
                vm: false,
                src: [1, 2, 3, 4],
            }
        );
    }

    #[test]
    fn mubuf_roundtrip() {
        let mut e = Encoder::new();
        e.mubuf(
            ops::mubuf::BUFFER_LOAD_DWORDX4,
            1,
            0,
            4,
            0xFF,
            0,
            true,
            false,
        );
        let decoded = decode_program(&e.build()).unwrap();
        match &decoded[0].1 {
            Inst::Mubuf {
                op,
                vdata,
                vaddr,
                srsrc,
                soffset,
                offset,
                offen,
                idxen,
                glc,
            } => {
                assert_eq!(*op, ops::mubuf::BUFFER_LOAD_DWORDX4);
                assert_eq!(*vdata, 1);
                assert_eq!(*vaddr, 0);
                assert_eq!(*srsrc, Source::Sgpr(4));
                assert_eq!(*soffset, Source::Literal(0));
                assert_eq!(*offset, 0);
                assert!(offen);
                assert!(!idxen);
                assert!(!glc);
            }
            other => panic!("expected Mubuf, got {other:?}"),
        }
    }

    #[test]
    fn truncated_program_rejected() {
        let mut e = Encoder::new();
        e.vop3(
            ops::vop3::V_MAD_F32,
            7,
            Source::Vgpr(1),
            Source::Vgpr(2),
            Source::Vgpr(3),
        );
        let mut prog = e.build();
        prog.pop(); // drop the second half
        assert!(matches!(
            decode_program(&prog),
            Err(DecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn literal_without_dword_rejected() {
        // Hand-build: v_mov_b32 v0, <literal> with no trailing dword.
        let w = (0x3F_u32 << 25) | (u32::from(ops::vop1::V_MOV_B32) << 9) | 0xFF;
        assert!(matches!(
            decode_program(&[w]),
            Err(DecodeError::MissingLiteral { .. })
        ));
    }

    #[test]
    fn program_stream_advances_over_literals() {
        let mut e = Encoder::new();
        e.sop2(ops::sop2::S_ADD_U32, 0, lit(0xAAAA), Source::Sgpr(1));
        e.vop1(ops::vop1::V_MOV_B32, 2, Source::Vgpr(3));
        let prog = e.build();
        let decoded = decode_program(&prog).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(
            decoded[1].0, 2,
            "second instruction starts after the literal"
        );
    }
}
