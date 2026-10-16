//! PM4 packet-stream decoder.
//!
//! Walks a ring-buffer slice (or an indirect-buffer body) and yields
//! [`Packet`]s with their raw payloads. The semantic execution of the
//! packets lives in [`super::cp`]; this layer only frames.
//!
//! Payload layouts in this crate follow the public AMD command-processor
//! documentation as mirrored by the Linux kernel (`soc15d.h`) and the
//! shadPS4 Liverpool documentation (`pm4_cmds.h`).

use super::opcode::ItOpcode;
use super::{Header, PKT_TYPE_0, PKT_TYPE_2, PKT_TYPE_3};
use thiserror::Error;

/// A framing error found while walking a stream.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// Not enough words left for the announced payload.
    #[error("truncated packet at word {at}: need {need} payload words, have {have}")]
    Truncated {
        /// Stream offset of the header word.
        at: usize,
        /// Words the header announced.
        need: usize,
        /// Words actually available.
        have: usize,
    },
    /// Packet type 1 does not exist on GCN/RDMA parts.
    #[error("invalid packet type 1 at word {at}")]
    InvalidType1 {
        /// Stream offset of the header word.
        at: usize,
    },
}

/// One decoded PM4 packet and its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Packet<'a> {
    /// Type-0: write `data` words to consecutive registers starting at
    /// `base_reg` (in the space implied by the ring — config).
    Type0 {
        /// First register offset.
        base_reg: u16,
        /// Words to write.
        data: &'a [u32],
    },
    /// Type-2: padding word(s).
    Type2,
    /// Type-3: opcode packet. `raw_opcode` is kept for unknown opcodes;
    /// `opcode` is `Some` when PDG knows the packet.
    Type3 {
        /// Decoded opcode, when in PDG's table.
        opcode: Option<ItOpcode>,
        /// The raw 7-bit opcode number.
        raw_opcode: u8,
        /// Predicate bit.
        predicate: bool,
        /// Payload words after the header.
        data: &'a [u32],
    },
}

impl Packet<'_> {
    /// The decoded opcode, if any.
    #[must_use]
    pub fn opcode(&self) -> Option<ItOpcode> {
        match self {
            Self::Type3 { opcode, .. } => *opcode,
            _ => None,
        }
    }
}

/// Decoding statistics, surfaced for tests and tracing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeStats {
    /// Number of type-0 packets.
    pub type0: u32,
    /// Number of type-2 padding packets.
    pub type2: u32,
    /// Number of type-3 packets.
    pub type3: u32,
    /// Number of type-3 packets with an opcode outside PDG's table.
    pub unknown_opcodes: u32,
}

/// A cursor over a PM4 word stream.
#[derive(Debug)]
pub struct PacketDecoder<'a> {
    words: std::slice::Iter<'a, u32>,
    offset: usize,
    stats: DecodeStats,
}

impl<'a> PacketDecoder<'a> {
    /// Start decoding at the beginning of `words`.
    #[must_use]
    pub fn new(words: &'a [u32]) -> Self {
        Self {
            words: words.iter(),
            offset: 0,
            stats: DecodeStats::default(),
        }
    }

    /// Current stream offset (in words).
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Stats so far.
    #[must_use]
    pub fn stats(&self) -> DecodeStats {
        self.stats
    }

    /// Whether the stream is exhausted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.len() == 0
    }
}

impl<'a> Iterator for PacketDecoder<'a> {
    type Item = Result<Packet<'a>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let at = self.offset;
        let raw = *self.words.next()?;
        let header = Header::from_raw(raw);
        self.offset += 1;
        match header.packet_type() {
            PKT_TYPE_3 => {
                let count = usize::from(header.count());
                let remaining = self.words.len();
                if remaining < count {
                    return Some(Err(DecodeError::Truncated {
                        at,
                        need: count,
                        have: remaining,
                    }));
                }
                let data: &[u32] = &self.words.as_slice()[..count];
                for _ in 0..count {
                    self.words.next();
                }
                self.offset += count;
                self.stats.type3 += 1;
                let raw_opcode = header.opcode();
                let opcode = ItOpcode::from_raw(raw_opcode);
                if opcode.is_none() {
                    self.stats.unknown_opcodes += 1;
                }
                Some(Ok(Packet::Type3 {
                    opcode,
                    raw_opcode,
                    predicate: header.predicate(),
                    data,
                }))
            }
            PKT_TYPE_0 => {
                // Type-0 count is n-1 encoded: a count of 0 writes one word.
                let count = usize::from(header.count()) + 1;
                let remaining = self.words.len();
                if remaining < count {
                    return Some(Err(DecodeError::Truncated {
                        at,
                        need: count,
                        have: remaining,
                    }));
                }
                let data: &[u32] = &self.words.as_slice()[..count];
                for _ in 0..count {
                    self.words.next();
                }
                self.offset += count;
                self.stats.type0 += 1;
                Some(Ok(Packet::Type0 {
                    base_reg: header.base_reg(),
                    data,
                }))
            }
            PKT_TYPE_2 => {
                self.stats.type2 += 1;
                Some(Ok(Packet::Type2))
            }
            // Packet type 1 does not exist.
            _ => Some(Err(DecodeError::InvalidType1 { at })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pm4::opcode::ItOpcode;

    fn hdr3(opcode: u8, count: u16) -> u32 {
        Header::type3(opcode, count).raw()
    }

    #[test]
    fn decodes_type3_with_payload() {
        let stream = [hdr3(0x10, 2), 0xAA, 0xBB];
        let pkts: Vec<_> = PacketDecoder::new(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(pkts.len(), 1);
        match &pkts[0] {
            Packet::Type3 {
                opcode,
                raw_opcode,
                predicate,
                data,
            } => {
                assert_eq!(*opcode, Some(ItOpcode::Nop));
                assert_eq!(*raw_opcode, 0x10);
                assert!(!*predicate);
                assert_eq!(*data, [0xAA, 0xBB]);
            }
            other => panic!("expected type3, got {other:?}"),
        }
        let mut dec = PacketDecoder::new(&stream);
        for p in dec.by_ref() {
            let _ = p.unwrap();
        }
        assert_eq!(dec.stats().type3, 1);
    }

    #[test]
    fn decodes_type0_with_n_minus_one_count() {
        let stream = [Header::type0(0x2000, 1).raw(), 0x11, 0x22];
        let pkts: Vec<_> = PacketDecoder::new(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        match &pkts[0] {
            Packet::Type0 { base_reg, data } => {
                assert_eq!(*base_reg, 0x2000);
                // count field 1 means two words
                assert_eq!(*data, [0x11, 0x22]);
            }
            other => panic!("expected type0, got {other:?}"),
        }
    }

    #[test]
    fn decodes_type2_padding() {
        let stream = [Header::type2().raw(); 3];
        let pkts: Vec<_> = PacketDecoder::new(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(pkts.len(), 3);
        assert!(pkts.iter().all(|p| matches!(p, Packet::Type2)));
    }

    #[test]
    fn truncated_type3_is_reported() {
        let stream = [hdr3(0x10, 5), 1, 2];
        let err = PacketDecoder::new(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap_err();
        assert_eq!(
            err,
            DecodeError::Truncated {
                at: 0,
                need: 5,
                have: 2
            }
        );
    }

    #[test]
    fn type1_is_rejected() {
        let stream = [0x4000_0000];
        let err = PacketDecoder::new(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap_err();
        assert_eq!(err, DecodeError::InvalidType1 { at: 0 });
    }

    #[test]
    fn unknown_opcode_still_framed() {
        let stream = [hdr3(0x14, 1), 0x1];
        let pkts: Vec<_> = PacketDecoder::new(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        match &pkts[0] {
            Packet::Type3 {
                opcode,
                raw_opcode,
                data,
                ..
            } => {
                assert_eq!(*opcode, None);
                assert_eq!(*raw_opcode, 0x14);
                assert_eq!(*data, [0x1]);
            }
            other => panic!("expected type3, got {other:?}"),
        }
        let mut dec = PacketDecoder::new(&stream);
        for p in dec.by_ref() {
            let _ = p.unwrap();
        }
        assert_eq!(dec.stats().unknown_opcodes, 1);
    }

    #[test]
    fn mixed_stream_offsets() {
        let stream = [
            Header::type2().raw(),
            hdr3(0x69, 3),
            0,
            1,
            2,
            Header::type0(0x2000, 0).raw(),
            7,
        ];
        let mut dec = PacketDecoder::new(&stream);
        assert!(matches!(dec.next().unwrap().unwrap(), Packet::Type2));
        let p = dec.next().unwrap().unwrap();
        assert!(p.opcode() == Some(ItOpcode::SetContextReg));
        assert_eq!(dec.offset(), 5);
        let p = dec.next().unwrap().unwrap();
        assert!(matches!(p, Packet::Type0 { .. }));
        assert_eq!(dec.offset(), 7);
        assert!(dec.is_empty());
        assert!(dec.next().is_none());
    }
}
