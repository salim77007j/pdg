//! Controller register file — the BAR0-shaped MMIO window.
//!
//! Offsets follow the public NVMe 2.0 base specification ("Controller
//! Registers" chapter): capability/registers in the first page, doorbells
//! in the second (`0x1000`), one 4-byte doorbell per queue direction
//! (`DSTRD = 0`).

/// Capability register (CAP, 64-bit, offset `0x00`).
///
/// PDG reports: `MQES = 4095` (4096-entry queues max), `TO = 1` (500 ms
/// timeout), doorbell stride 0 (4 bytes), everything else zero.
pub const REG_CAP: u64 = 0x00;
/// Version register (VS, offset `0x08`) — NVMe 2.0 (`0x0002_0000`).
pub const REG_VS: u64 = 0x08;
/// Interrupt Mask Set register (INTMS, offset `0x0C`) — present but inert.
pub const REG_INTMS: u64 = 0x0C;
/// Interrupt Mask Clear register (INTMC, offset `0x10`).
pub const REG_INTMC: u64 = 0x10;
/// Controller Configuration (CC, offset `0x14`).
pub const REG_CC: u64 = 0x14;
/// Controller Status (CSTS, offset `0x1C`).
pub const REG_CSTS: u64 = 0x1C;
/// Admin Queue Attributes (AQA, offset `0x24`).
pub const REG_AQA: u64 = 0x24;
/// Admin Submission Queue base (ASQ, 64-bit, offset `0x28`).
pub const REG_ASQ: u64 = 0x28;
/// Admin Completion Queue base (ACQ, 64-bit, offset `0x2C`).
pub const REG_ACQ: u64 = 0x2C;
/// Reserved region ends here; doorbells start at the next page.
pub const REG_END: u64 = 0x30;

/// Doorbell array base (offset `0x1000`): two 4-byte doorbells per queue —
/// SQ tail then CQ head (`0x1000 + 8*queue + 0/4`).
pub const NVME_DOORBELL_BASE: u64 = 0x1000;

/// Total MMIO footprint: registers page + doorbell page.
pub const NVME_FOOTPRINT: u64 = 0x2000;

/// CC bit 0: enable. 0→1 resets and enables the controller.
pub const CC_EN: u32 = 1 << 0;
/// CC bits 4:1: I/O command set selected (0 = NVM).
pub const CC_CSS_SHIFT: u32 = 1;
/// CC bits 7:4: memory page size log2 minus 12 (PDG: 4 KiB pages only).
pub const CC_MPS_SHIFT: u32 = 4;
/// CC bits 15:14: shutdown notification (0 = normal, `11b` = abrupt).
pub const CC_SHN_SHIFT: u32 = 14;
/// CC bits 18:17 / 22:21: completion/submission queue entry size log2.
pub const CC_IOCQES_SHIFT: u32 = 17;
pub const CC_IOSQES_SHIFT: u32 = 21;

/// CSTS bit 0: ready. Set once CC.EN went 0→1 with queues configured.
pub const CSTS_RDY: u32 = 1 << 0;
/// CSTS bit 1: controller fatal status.
pub const CSTS_CFS: u32 = 1 << 1;
/// CSTS bits 3:2: shutdown status (1 = normal-complete).
pub const CSTS_SHST_SHIFT: u32 = 2;
/// CSTS bit 4: namespace list changed (never raised in Phase 4).
pub const CSTS_NSSRO: u32 = 1 << 4;

/// The CAP value PDG reports (MQES 4095, TO 1).
pub const CAP_VALUE: u64 = 0x0000_0FFF | (1 << 24);
/// The version PDG implements and reports in VS: NVMe 2.0.
pub const VS_VALUE: u32 = 0x0002_0000;

/// Build an AQA value from queue sizes in entries.
///
/// `asqes`/`acqes` are *0-based* (a value of `n` means `n + 1` entries),
/// matching the spec's `ASQES`/`ACQES` fields.
#[must_use]
pub fn aqa_from_sizes(acqes_minus1: u8, asqes_minus1: u8) -> u32 {
    (u32::from(acqes_minus1) << 24) | (u32::from(asqes_minus1) << 16)
}

/// Decoded CC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CtrlConfig {
    /// CC.EN — controller enable.
    pub enable: bool,
    /// CC.MPS field (memory page size = 2^(12+mps)); PDG supports 0 only.
    pub mps: u8,
    /// CC.CSS field (I/O command set); PDG supports 0 (NVM) only.
    pub css: u8,
    /// CC.SHN field (shutdown notification).
    pub shn: u8,
    /// Raw register value (for MMIO reads).
    pub raw: u32,
}

impl CtrlConfig {
    /// Decode the CC register value.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        Self {
            enable: raw & CC_EN != 0,
            mps: ((raw >> CC_MPS_SHIFT) & 0xF) as u8,
            css: ((raw >> CC_CSS_SHIFT) & 0x7) as u8,
            shn: ((raw >> CC_SHN_SHIFT) & 0x3) as u8,
            raw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_reports_mqes_and_timeout() {
        // MQES (23:0) = 0xFFF, TO (31:24) = 1.
        assert_eq!(CAP_VALUE & 0x00FF_FFFF, 0x0FFF);
        assert_eq!((CAP_VALUE >> 24) & 0xFF, 1);
    }

    #[test]
    fn cc_decodes_enable_and_page_size() {
        let cc = CtrlConfig::from_raw(0);
        assert!(!cc.enable);
        // EN=1, MPS=0, CSS=0
        let cc = CtrlConfig::from_raw(1);
        assert!(cc.enable);
        assert_eq!(cc.mps, 0);
        assert_eq!(cc.css, 0);
        // MPS=4 would mean 256 KiB pages — decodes, is refused later.
        let cc = CtrlConfig::from_raw((4 << CC_MPS_SHIFT) | 1);
        assert!(cc.enable);
        assert_eq!(cc.mps, 4);
    }

    #[test]
    fn aqa_packs_0based_sizes() {
        // 4-entry queues → fields hold 3.
        let aqa = aqa_from_sizes(3, 3);
        assert_eq!(aqa, (3 << 24) | (3 << 16));
    }
}
