//! Queue entries: the 64-byte submission command and the 16-byte
//! completion entry, plus the opcode and status number spaces.
//!
//! Field layouts follow the public NVMe 2.0 specification. Only the
//! commands PDG models have [`TryFrom`]/encode helpers; other opcodes
//! decode (so they can be *named* in refusals) but are refused at
//! execution, never guessed.

/// Admin-queue opcodes PDG understands (value = the opcode byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdminOpcode {
    /// Delete I/O Submission Queue.
    DeleteIoSq = 0x00,
    /// Create I/O Submission Queue.
    CreateIoSq = 0x01,
    /// Get Log Page.
    GetLogPage = 0x02,
    /// Delete I/O Completion Queue.
    DeleteIoCq = 0x04,
    /// Create I/O Completion Queue.
    CreateIoCq = 0x05,
    /// Identify (controller / namespace).
    Identify = 0x06,
    /// Abort — no command is ever in flight across a doorbell in this
    /// model, so aborting zero commands is the correct behavior.
    Abort = 0x08,
    /// Set Features.
    SetFeatures = 0x09,
    /// Get Features (PDG answers the Number-of-Queues feature only).
    GetFeatures = 0x0A,
    /// An opcode outside the modeled set.
    Unknown(u8),
}

impl AdminOpcode {
    /// The opcode byte (for building CDW0; `as`-casts are illegal on
    /// enums with data variants).
    #[must_use]
    pub fn byte(self) -> u8 {
        match self {
            Self::DeleteIoSq => 0x00,
            Self::CreateIoSq => 0x01,
            Self::GetLogPage => 0x02,
            Self::DeleteIoCq => 0x04,
            Self::CreateIoCq => 0x05,
            Self::Identify => 0x06,
            Self::Abort => 0x08,
            Self::SetFeatures => 0x09,
            Self::GetFeatures => 0x0A,
            Self::Unknown(v) => v,
        }
    }
}

impl From<u8> for AdminOpcode {
    fn from(v: u8) -> Self {
        match v {
            0x00 => Self::DeleteIoSq,
            0x01 => Self::CreateIoSq,
            0x02 => Self::GetLogPage,
            0x04 => Self::DeleteIoCq,
            0x05 => Self::CreateIoCq,
            0x06 => Self::Identify,
            0x08 => Self::Abort,
            0x09 => Self::SetFeatures,
            0x0A => Self::GetFeatures,
            other => Self::Unknown(other),
        }
    }
}

/// NVM I/O-queue opcodes PDG understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IoOpcode {
    /// Flush — the backing file writes are synchronous; a no-op is honest.
    Flush = 0x00,
    /// Write.
    Write = 0x01,
    /// Read.
    Read = 0x02,
    /// An opcode outside the modeled set.
    Unknown(u8),
}

impl IoOpcode {
    /// The opcode byte (for building CDW0).
    #[must_use]
    pub fn byte(self) -> u8 {
        match self {
            Self::Flush => 0x00,
            Self::Write => 0x01,
            Self::Read => 0x02,
            Self::Unknown(v) => v,
        }
    }
}

impl From<u8> for IoOpcode {
    fn from(v: u8) -> Self {
        match v {
            0x00 => Self::Flush,
            0x01 => Self::Write,
            0x02 => Self::Read,
            other => Self::Unknown(other),
        }
    }
}

/// Generic (status-code-type 0) status values PDG reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Successful completion.
    Success = 0x00,
    /// Invalid Command Opcode.
    InvalidOpcode = 0x01,
    /// Invalid Field in Command.
    InvalidField = 0x02,
    /// Data Transfer Error.
    DataTransferError = 0x04,
    /// Command Abort Requested / aborted nothing — see [`AdminOpcode::Abort`].
    Aborted = 0x07,
    /// Invalid Namespace or Format.
    InvalidNamespaceOrFormat = 0x0B,
    /// Command-specific: LBA Out Of Range.
    LbaOutOfRange = 0x80,
    /// Command-specific: Capacity Exceeded.
    CapacityExceeded = 0x81,
    /// Command-specific: Namespace Not Ready.
    NamespaceNotReady = 0x82,
}

impl Status {
    /// SC + SCT(0) packed into the CQE dword-3 status field (bits 15:1).
    #[must_use]
    pub fn packed(self) -> u32 {
        (self as u32) << 1
    }
}

/// Size of one submission queue entry.
pub const SQE_SIZE: usize = 64;
/// Size of one completion queue entry.
pub const CQE_SIZE: usize = 16;

/// A decoded 64-byte submission entry.
///
/// Only the union members the modeled commands use are named; the raw
/// dwords are kept so command-specific fields are reachable without
/// inventing meaning for undefined bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmissionEntry {
    /// CDW0 raw: opcode(7:0), FUSE(9:8), PSDT(11:10).
    pub cdw0: u32,
    /// Namespace identifier (CDW1).
    pub nsid: u32,
    /// PRP entry 1 (64-bit, guest-physical).
    pub prp1: u64,
    /// PRP entry 2 (page pointer or PRP-list pointer).
    pub prp2: u64,
    /// Command-specific dwords CDW10..CDW15.
    pub cdw10: [u32; 6],
}

impl SubmissionEntry {
    /// Opcode byte.
    #[must_use]
    pub fn opcode(&self) -> u8 {
        (self.cdw0 & 0xFF) as u8
    }

    /// FUSE field (first/second fused) — PDG refuses nonzero.
    #[must_use]
    pub fn fuse(&self) -> u8 {
        ((self.cdw0 >> 8) & 0x3) as u8
    }

    /// PSDT field (data pointer type): `0b00` = PRP, `0b10` = SGL.
    /// PDG supports PRPs only and refuses SGL commands.
    #[must_use]
    pub fn psdt(&self) -> u8 {
        ((self.cdw0 >> 10) & 0x3) as u8
    }

    /// Decode a little-endian 64-byte SQE.
    #[must_use]
    pub fn from_bytes(raw: &[u8; SQE_SIZE]) -> Self {
        let dw = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().expect("dword"));
        let qw = |i: usize| u64::from_le_bytes(raw[i * 4..i * 4 + 8].try_into().expect("qword"));
        Self {
            cdw0: dw(0),
            nsid: dw(1),
            prp1: qw(4),
            prp2: qw(6),
            cdw10: [dw(10), dw(11), dw(12), dw(13), dw(14), dw(15)],
        }
    }

    /// Encode into a little-endian 64-byte SQE (used by tests and the
    /// in-tree demo payload builder).
    #[must_use]
    pub fn to_bytes(self) -> [u8; SQE_SIZE] {
        let mut out = [0u8; SQE_SIZE];
        out[0..4].copy_from_slice(&self.cdw0.to_le_bytes());
        out[4..8].copy_from_slice(&self.nsid.to_le_bytes());
        out[16..24].copy_from_slice(&self.prp1.to_le_bytes());
        out[24..32].copy_from_slice(&self.prp2.to_le_bytes());
        for (i, v) in self.cdw10.iter().enumerate() {
            let off = (10 + i) * 4;
            out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Build an IDENTIFY (CNS in CDW10) pointing at `buffer`.
    #[must_use]
    pub fn identify(cns: u32, buffer: u64) -> Self {
        Self {
            cdw0: u32::from(AdminOpcode::Identify.byte()),
            nsid: if cns == 1 { super::NAMESPACE_ID } else { 0 },
            prp1: buffer,
            prp2: 0,
            cdw10: [cns, 0, 0, 0, 0, 0],
        }
    }

    /// Build a CREATE I/O COMPLETION QUEUE for `qid` at `base`, `entries`
    /// (1-based), interrupt-vector `iv`. CDW10 packs size+id; CDW11 (here
    /// `cdw10[1]`) holds the physically-contiguous + interrupt-enable bits
    /// and the vector.
    #[must_use]
    pub fn create_io_cq(qid: u16, base: u64, entries: u32, iv: u8) -> Self {
        Self {
            cdw0: u32::from(AdminOpcode::CreateIoCq.byte()),
            nsid: 0,
            prp1: base,
            prp2: 0,
            cdw10: [
                (entries - 1) | (u32::from(qid) << 16),
                1 | u32::from(iv),
                0,
                0,
                0,
                0,
            ],
        }
    }

    /// Build a CREATE I/O SUBMISSION QUEUE for `qid` at `base`, `entries`
    /// (1-based), completing into `cqid`. CDW11 (here `cdw10[1]`) packs the
    /// physically-contiguous bit and the completion-queue id.
    #[must_use]
    pub fn create_io_sq(qid: u16, base: u64, entries: u32, cqid: u16) -> Self {
        Self {
            cdw0: u32::from(AdminOpcode::CreateIoSq.byte()),
            nsid: 0,
            prp1: base,
            prp2: 0,
            cdw10: [
                (entries - 1) | (u32::from(qid) << 16),
                1 | (u32::from(cqid) << 16),
                0,
                0,
                0,
                0,
            ],
        }
    }

    /// Build a READ/WRITE of `blocks` LBAs starting at `slba`, data at
    /// `buffer` (PRP1; PRP2 supplied by the caller for multi-page moves).
    #[must_use]
    pub fn read_write(
        write: bool,
        nsid: u32,
        slba: u64,
        blocks: u32,
        buffer: u64,
        prp2: u64,
    ) -> Self {
        let cdw12 = (blocks - 1) & 0xFFFF; // NLB-1 in the low half
        Self {
            cdw0: if write {
                u32::from(IoOpcode::Write.byte())
            } else {
                u32::from(IoOpcode::Read.byte())
            },
            nsid,
            prp1: buffer,
            prp2,
            cdw10: [
                (slba & 0xFFFF_FFFF) as u32,
                (slba >> 32) as u32,
                cdw12,
                0,
                0,
                0,
            ],
        }
    }
}

/// A 16-byte completion queue entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionEntry {
    /// DW0: command result.
    pub result: u32,
    /// DW2 bits 15:0 — SQ head pointer at posting time.
    pub sqhd: u16,
    /// DW2 bits 31:16 — submission queue id.
    pub sqid: u16,
    /// DW3 bits 31:16 — command id.
    pub cid: u16,
    /// DW3 bits 15:1 — status (SCT=0, SC in bits 15:8... packed form).
    pub status_packed: u32,
    /// DW3 bit 0 — phase tag.
    pub phase: bool,
}

impl CompletionEntry {
    /// Encode into 16 little-endian bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; CQE_SIZE] {
        let dw3 = (u32::from(self.cid) << 16) | self.status_packed | u32::from(self.phase);
        let dw2 = u32::from(self.sqhd) | (u32::from(self.sqid) << 16);
        let mut out = [0u8; CQE_SIZE];
        out[0..4].copy_from_slice(&self.result.to_le_bytes());
        out[8..12].copy_from_slice(&dw2.to_le_bytes());
        out[12..16].copy_from_slice(&dw3.to_le_bytes());
        out
    }

    /// Decode from 16 little-endian bytes.
    #[must_use]
    pub fn from_bytes(raw: &[u8; CQE_SIZE]) -> Self {
        let dw = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().expect("dword"));
        let dw2 = dw(2);
        let dw3 = dw(3);
        Self {
            result: dw(0),
            sqhd: (dw2 & 0xFFFF) as u16,
            sqid: (dw2 >> 16) as u16,
            cid: (dw3 >> 16) as u16,
            status_packed: dw3 & 0xFFFE,
            phase: dw3 & 1 != 0,
        }
    }

    /// The generic status code (type 0), or `None` for other status types.
    #[must_use]
    pub fn status(&self) -> Option<Status> {
        // SCT occupies bits 9:8 in the packed field; PDG only posts SCT 0.
        if self.status_packed >> 9 & 0x3 != 0 {
            return None;
        }
        Some(match (self.status_packed >> 1) & 0xFF {
            0x00 => Status::Success,
            0x01 => Status::InvalidOpcode,
            0x02 => Status::InvalidField,
            0x04 => Status::DataTransferError,
            0x07 => Status::Aborted,
            0x0B => Status::InvalidNamespaceOrFormat,
            0x80 => Status::LbaOutOfRange,
            0x81 => Status::CapacityExceeded,
            0x82 => Status::NamespaceNotReady,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqe_roundtrips_through_bytes() {
        let cmd = SubmissionEntry::read_write(true, 1, 0x1234_5678_9ABC, 8, 0x400_000, 0x401_000);
        let raw = cmd.to_bytes();
        let back = SubmissionEntry::from_bytes(&raw);
        assert_eq!(cmd, back);
        assert_eq!(back.opcode(), IoOpcode::Write.byte());
        assert_eq!(back.nsid, 1);
        // SLBA reassembled from CDW10/11.
        let slba = u64::from(back.cdw10[0]) | (u64::from(back.cdw10[1]) << 32);
        assert_eq!(slba, 0x1234_5678_9ABC);
    }

    #[test]
    fn cqe_roundtrips_and_decodes_status() {
        let cqe = CompletionEntry {
            result: 0xCAFE,
            sqhd: 4,
            sqid: 1,
            cid: 0x22,
            status_packed: Status::Success.packed(),
            phase: true,
        };
        let raw = cqe.to_bytes();
        let back = CompletionEntry::from_bytes(&raw);
        assert_eq!(back.result, 0xCAFE);
        assert_eq!(back.sqid, 1);
        assert_eq!(back.cid, 0x22);
        assert!(back.phase);
        assert_eq!(back.status(), Some(Status::Success));
    }

    #[test]
    fn opcode_tables_name_everything() {
        assert_eq!(AdminOpcode::from(0x06), AdminOpcode::Identify);
        assert_eq!(AdminOpcode::from(0x1C), AdminOpcode::Unknown(0x1C));
        assert_eq!(IoOpcode::from(0x02), IoOpcode::Read);
        assert_eq!(IoOpcode::from(0x0D), IoOpcode::Unknown(0x0D));
    }
}
