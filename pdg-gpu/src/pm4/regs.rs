//! gfx10.3 (RDNA 2) GC register model.
//!
//! The PS5's GPU is an RDNA 2 (GFX10.3) part. PDG models the command
//! processor's register spaces exactly as the public AMD sources define
//! them:
//!
//! * **mm offsets + `BASE_IDX` segments** come from the Linux kernel's
//!   `gc_10_3_0_offset.h` (Sienna-Cichlid-class headers).
//! * **The segment bases** (`SEG0 = 0x1260`, `SEG1 = 0xA000` for GC) come
//!   from `navi10_ip_offset.h` — the navi family IP layout that the
//!   gfx10.3 parts (Dimgrey Cavefish et al.) share.
//! * **The CP packet spaces** then follow the kernel's own ring code:
//!   a `SET_*_REG` payload offset is `(segment_base + mm) − space_start`,
//!   verified against the kernel's `SET_UCONFIG_REG(mmSCRATCH_REG0)` and
//!   `SET_CONTEXT_REG(mmPA_SC_TILE_STEERING_OVERRIDE)` emissions.
//!
//! That yields the gfx10.3 native packet-offset formulas implemented in
//! [`RegSpace`]:
//!
//! | space   | BASE_IDX | packet offset      |
//! |---------|----------|--------------------|
//! | config  | 0        | `mm − 0xDA0`       |
//! | context | 1        | `mm`               |
//! | sh      | 0        | `mm − 0x19A0`      |
//! | uconfig | 1        | `mm − 0x2000`      |
//!
//! (`0xDA0 = 0x2000 − 0x1260` and `0x19A0 = 0x2C00 − 0x1260`; the
//! context space needs no adjustment because `SEG1` equals its start.)
//!
//! AMD hardware also accepts the older "gfx6-compatibility" register
//! window that Mesa programs through gfx10; PDG does not model it yet —
//! it becomes relevant the moment a real console GNM stream can be
//! compared against public sources, and the `GpuContext` design keeps
//! room for it (registers are keyed by space+offset, not by enum).

use std::collections::HashMap;

/// GC segment bases for gfx10 (navi family), from `navi10_ip_offset.h`.
pub const GC_SEG0: u32 = 0x1260;
pub const GC_SEG1: u32 = 0xA000;

/// `SET_CONFIG_REG` window start (kernel `soc15d.h`).
pub const SET_CONFIG_REG_START: u32 = 0x2000;
/// `SET_CONTEXT_REG` window start (kernel `soc15d.h`).
pub const SET_CONTEXT_REG_START: u32 = 0xA000;
/// `SET_SH_REG` window start (kernel `soc15d.h`).
pub const SET_SH_REG_START: u32 = 0x2C00;
/// `SET_UCONFIG_REG` window start (kernel `soc15d.h`).
pub const SET_UCONFIG_REG_START: u32 = 0xC000;

/// A PM4 register space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegSpace {
    /// `SET_CONFIG_REG` (0x68) — config registers.
    Config,
    /// `SET_CONTEXT_REG` (0x69) — context-state registers.
    Context,
    /// `SET_SH_REG` (0x76) — per-shader (SQ) registers.
    Sh,
    /// `SET_UCONFIG_REG` (0x79) — user configuration registers.
    UConfig,
}

impl RegSpace {
    /// Space opcode for `SET_*_REG` packets.
    #[must_use]
    pub const fn set_opcode(self) -> u8 {
        match self {
            Self::Config => 0x68,
            Self::Context => 0x69,
            Self::Sh => 0x76,
            Self::UConfig => 0x79,
        }
    }

    /// Absolute GRBM start of this space.
    #[must_use]
    pub const fn start(self) -> u32 {
        match self {
            Self::Config => SET_CONFIG_REG_START,
            Self::Context => SET_CONTEXT_REG_START,
            Self::Sh => SET_SH_REG_START,
            Self::UConfig => SET_UCONFIG_REG_START,
        }
    }
}

/// A register address inside one of the four CP spaces, carrying the raw
/// `mm` offset and its `BASE_IDX` segment (gfx10.3 native scheme).
///
/// `mm` values are the ones from the kernel `gc_10_3_0_offset.h`; the
/// packet-visible offset is derived per [`RegSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegAddr {
    /// Register space.
    pub space: RegSpace,
    /// `mm` offset from the kernel header.
    pub mm: u32,
    /// `BASE_IDX` from the kernel header.
    pub base_idx: u32,
}

impl RegAddr {
    /// Build a register address.
    #[must_use]
    pub const fn new(space: RegSpace, mm: u32, base_idx: u32) -> Self {
        Self {
            space,
            mm,
            base_idx,
        }
    }

    /// Convert a raw `SET_*_REG` payload offset in `space` to a
    /// [`RegAddr`], if it lands in a segment PDG can map.
    #[must_use]
    pub fn from_packet_offset(space: RegSpace, offset: u32) -> Option<Self> {
        // gfx10.3 native scheme: packet offset = (seg_base + mm) - space_start.
        let (seg_base, base_idx) = match space {
            // Config and SH registers live in GC SEG0; context and uconfig in SEG1.
            RegSpace::Config => (GC_SEG0, 0),
            RegSpace::Context => (GC_SEG1, 1),
            RegSpace::Sh => (GC_SEG0, 0),
            RegSpace::UConfig => (GC_SEG1, 1),
        };
        let abs = space.start().checked_add(offset)?;
        if abs < seg_base {
            return None;
        }
        Some(Self::new(space, abs - seg_base, base_idx))
    }

    /// The `SET_*_REG` payload offset for this register.
    ///
    /// Returns `None` when the register's segment does not project into
    /// this space (e.g. a `BASE_IDX 0` register asked for as context).
    #[must_use]
    pub fn packet_offset(self) -> Option<u32> {
        let seg_base = match (self.space, self.base_idx) {
            (RegSpace::Config, 0) | (RegSpace::Sh, 0) => GC_SEG0,
            (RegSpace::Context, 1) | (RegSpace::UConfig, 1) => GC_SEG1,
            _ => return None,
        };
        let abs = seg_base.checked_add(self.mm)?;
        let off = abs.checked_sub(self.space.start())?;
        (off < 0x1_0000).then_some(off)
    }
}

/// Context-space registers PDG gives typed meaning (kernel `gc_10_3_0`).
pub mod ctx {
    use super::{RegAddr, RegSpace};

    /// `mmPA_SC_SCREEN_SCISSOR_TL` (BASE_IDX 1).
    pub const PA_SC_SCREEN_SCISSOR_TL: RegAddr = RegAddr::new(RegSpace::Context, 0x000C, 1);
    /// `mmPA_SC_SCREEN_SCISSOR_BR` (BASE_IDX 1).
    pub const PA_SC_SCREEN_SCISSOR_BR: RegAddr = RegAddr::new(RegSpace::Context, 0x000D, 1);
    /// `mmVGT_DRAW_INITIATOR` (BASE_IDX 1).
    pub const VGT_DRAW_INITIATOR: RegAddr = RegAddr::new(RegSpace::Context, 0x01FC, 1);
    /// `mmDB_DEPTH_CONTROL` (BASE_IDX 1).
    pub const DB_DEPTH_CONTROL: RegAddr = RegAddr::new(RegSpace::Context, 0x0200, 1);
    /// `mmCB_COLOR_CONTROL` (BASE_IDX 1).
    pub const CB_COLOR_CONTROL: RegAddr = RegAddr::new(RegSpace::Context, 0x0202, 1);
    /// `mmVGT_EVENT_INITIATOR` (BASE_IDX 1).
    pub const VGT_EVENT_INITIATOR: RegAddr = RegAddr::new(RegSpace::Context, 0x02A4, 1);
    /// `mmPA_SC_AA_CONFIG` (BASE_IDX 1).
    pub const PA_SC_AA_CONFIG: RegAddr = RegAddr::new(RegSpace::Context, 0x02F8, 1);
    /// `mmCB_COLOR0_BASE` (BASE_IDX 1).
    pub const CB_COLOR0_BASE: RegAddr = RegAddr::new(RegSpace::Context, 0x0318, 1);
    /// `mmCB_COLOR0_PITCH` (BASE_IDX 1).
    pub const CB_COLOR0_PITCH: RegAddr = RegAddr::new(RegSpace::Context, 0x0319, 1);
    /// `mmCB_COLOR0_SLICE` (BASE_IDX 1).
    pub const CB_COLOR0_SLICE: RegAddr = RegAddr::new(RegSpace::Context, 0x031A, 1);
    /// `mmCB_COLOR0_VIEW` (BASE_IDX 1).
    pub const CB_COLOR0_VIEW: RegAddr = RegAddr::new(RegSpace::Context, 0x031B, 1);
    /// `mmCB_COLOR0_INFO` (BASE_IDX 1).
    pub const CB_COLOR0_INFO: RegAddr = RegAddr::new(RegSpace::Context, 0x031C, 1);
    /// `mmCB_COLOR0_ATTRIB` (BASE_IDX 1).
    pub const CB_COLOR0_ATTRIB: RegAddr = RegAddr::new(RegSpace::Context, 0x031D, 1);
    /// `mmCB_COLOR0_CLEAR_WORD0` (BASE_IDX 1).
    pub const CB_COLOR0_CLEAR_WORD0: RegAddr = RegAddr::new(RegSpace::Context, 0x0323, 1);
    /// `mmCB_COLOR0_CLEAR_WORD1` (BASEIdx 1).
    pub const CB_COLOR0_CLEAR_WORD1: RegAddr = RegAddr::new(RegSpace::Context, 0x0324, 1);
    /// `mmCB_TARGET_MASK` (BASE_IDX 1).
    pub const CB_TARGET_MASK: RegAddr = RegAddr::new(RegSpace::Context, 0x008E, 1);
    /// `mmVGT_PRIMITIVE_TYPE` (BASE_IDX 1).
    pub const VGT_PRIMITIVE_TYPE: RegAddr = RegAddr::new(RegSpace::Context, 0x2242, 1);
    /// `mmVGT_INDEX_TYPE` (BASE_IDX 1).
    pub const VGT_INDEX_TYPE: RegAddr = RegAddr::new(RegSpace::Context, 0x2243, 1);
    /// `mmVGT_NUM_INSTANCES` (BASE_IDX 1).
    pub const VGT_NUM_INSTANCES: RegAddr = RegAddr::new(RegSpace::Context, 0x224D, 1);
}

/// SH-space registers PDG gives typed meaning (kernel `gc_10_3_0`).
pub mod sh {
    use super::{RegAddr, RegSpace};

    /// `mmSPI_SHADER_PGM_LO_PS` (BASE_IDX 0).
    pub const SPI_SHADER_PGM_LO_PS: RegAddr = RegAddr::new(RegSpace::Sh, 0x19A8, 0);
    /// `mmSPI_SHADER_PGM_HI_PS` (BASE_IDX 0).
    pub const SPI_SHADER_PGM_HI_PS: RegAddr = RegAddr::new(RegSpace::Sh, 0x19A9, 0);
    /// `mmSPI_SHADER_USER_DATA_PS_0` (BASE_IDX 0); 32 consecutive.
    pub const SPI_SHADER_USER_DATA_PS_0: RegAddr = RegAddr::new(RegSpace::Sh, 0x19AC, 0);
    /// `mmSPI_SHADER_PGM_LO_VS` (BASE_IDX 0).
    pub const SPI_SHADER_PGM_LO_VS: RegAddr = RegAddr::new(RegSpace::Sh, 0x19E8, 0);
    /// `mmSPI_SHADER_PGM_HI_VS` (BASE_IDX 0).
    pub const SPI_SHADER_PGM_HI_VS: RegAddr = RegAddr::new(RegSpace::Sh, 0x19E9, 0);
    /// `mmSPI_SHADER_USER_DATA_VS_0` (BASE_IDX 0); 16 consecutive.
    pub const SPI_SHADER_USER_DATA_VS_0: RegAddr = RegAddr::new(RegSpace::Sh, 0x19EC, 0);
    /// `mmCOMPUTE_DIM_X` (BASE_IDX 0).
    pub const COMPUTE_DIM_X: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA1, 0);
    /// `mmCOMPUTE_DIM_Y` (BASE_IDX 0).
    pub const COMPUTE_DIM_Y: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA2, 0);
    /// `mmCOMPUTE_DIM_Z` (BASE_IDX 0).
    pub const COMPUTE_DIM_Z: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA3, 0);
    /// `mmCOMPUTE_NUM_THREAD_X` (BASE_IDX 0).
    pub const COMPUTE_NUM_THREAD_X: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA7, 0);
    /// `mmCOMPUTE_NUM_THREAD_Y` (BASE_IDX 0).
    pub const COMPUTE_NUM_THREAD_Y: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA8, 0);
    /// `mmCOMPUTE_NUM_THREAD_Z` (BASE_IDX 0).
    pub const COMPUTE_NUM_THREAD_Z: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BA9, 0);
    /// `mmCOMPUTE_PGM_LO` (BASE_IDX 0).
    pub const COMPUTE_PGM_LO: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BAC, 0);
    /// `mmCOMPUTE_PGM_HI` (BASE_IDX 0).
    pub const COMPUTE_PGM_HI: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BAD, 0);
    /// `mmCOMPUTE_RESOURCE_LIMITS` (BASE_IDX 0).
    pub const COMPUTE_RESOURCE_LIMITS: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BB5, 0);
    /// `mmCOMPUTE_USER_DATA_0` (BASE_IDX 0); 16 consecutive.
    pub const COMPUTE_USER_DATA_0: RegAddr = RegAddr::new(RegSpace::Sh, 0x1BE0, 0);
}

/// CB/texture data-format numbers (shared T#/CB encoding, public GCN
/// documentation as also mirrored by shadPS4's `DataFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DataFormat {
    /// 1 channel, 8 bits.
    Fmt8 = 1,
    /// 1 channel, 16 bits.
    Fmt16 = 2,
    /// 2 channels, 8:8.
    Fmt8_8 = 3,
    /// 1 channel, 32 bits.
    Fmt32 = 4,
    /// 2 channels, 16:16.
    Fmt16_16 = 5,
    /// 10:11:11 packed.
    Fmt10_11_11 = 6,
    /// 11:11:10 packed.
    Fmt11_11_10 = 7,
    /// 10:10:10:2 packed.
    Fmt10_10_10_2 = 8,
    /// 2:10:10:10 packed.
    Fmt2_10_10_10 = 9,
    /// 4 channels, 8:8:8:8.
    Fmt8_8_8_8 = 10,
    /// 2 channels, 32:32.
    Fmt32_32 = 11,
    /// 4 channels, 16:16:16:16.
    Fmt16_16_16_16 = 12,
    /// 3 channels, 32:32:32.
    Fmt32_32_32 = 13,
    /// 4 channels, 32:32:32:32.
    Fmt32_32_32_32 = 14,
    /// Unknown format number.
    Unknown = 0,
}

impl DataFormat {
    /// Decode a format number.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            1 => Self::Fmt8,
            2 => Self::Fmt16,
            3 => Self::Fmt8_8,
            4 => Self::Fmt32,
            5 => Self::Fmt16_16,
            6 => Self::Fmt10_11_11,
            7 => Self::Fmt11_11_10,
            8 => Self::Fmt10_10_10_2,
            9 => Self::Fmt2_10_10_10,
            10 => Self::Fmt8_8_8_8,
            11 => Self::Fmt32_32,
            12 => Self::Fmt16_16_16_16,
            13 => Self::Fmt32_32_32,
            14 => Self::Fmt32_32_32_32,
            _ => Self::Unknown,
        }
    }

    /// Bits per element (sum over channels).
    #[must_use]
    pub const fn bits_per_element(self) -> u32 {
        match self {
            Self::Fmt8 | Self::Unknown => 8,
            Self::Fmt16 | Self::Fmt8_8 => 16,
            Self::Fmt32
            | Self::Fmt16_16
            | Self::Fmt10_11_11
            | Self::Fmt11_11_10
            | Self::Fmt10_10_10_2
            | Self::Fmt2_10_10_10
            | Self::Fmt8_8_8_8 => 32,
            Self::Fmt32_32 | Self::Fmt16_16_16_16 => 64,
            Self::Fmt32_32_32 => 96,
            Self::Fmt32_32_32_32 => 128,
        }
    }

    /// Channel count.
    #[must_use]
    pub const fn channels(self) -> u32 {
        match self {
            Self::Fmt8 | Self::Fmt16 | Self::Fmt32 | Self::Unknown => 1,
            Self::Fmt8_8 | Self::Fmt16_16 | Self::Fmt32_32 => 2,
            Self::Fmt10_11_11 | Self::Fmt11_11_10 | Self::Fmt32_32_32 => 3,
            _ => 4,
        }
    }
}

/// Number-format numbers (shared T#/CB encoding, public GCN
/// documentation as also mirrored by shadPS4's `NumberFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NumberFormat {
    /// Unsigned normalized.
    Unorm = 0,
    /// Signed normalized.
    Snorm = 1,
    /// Unsigned scaled to float.
    Uscaled = 2,
    /// Signed scaled to float.
    Sscaled = 3,
    /// Unsigned integer.
    Uint = 4,
    /// Signed integer.
    Sint = 5,
    /// Signed normalized, zero preserved.
    SnormNz = 6,
    /// IEEE float.
    Float = 7,
    /// Unknown number format.
    Unknown = 0xFF,
}

impl NumberFormat {
    /// Decode a number-format number.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Unorm,
            1 => Self::Snorm,
            2 => Self::Uscaled,
            3 => Self::Sscaled,
            4 => Self::Uint,
            5 => Self::Sint,
            6 => Self::SnormNz,
            7 => Self::Float,
            _ => Self::Unknown,
        }
    }
}

/// One of the four CP register files.
#[derive(Debug, Default, Clone)]
pub struct RegisterFile {
    values: HashMap<(RegSpace, u32, u32), u32>,
}

impl RegisterFile {
    /// Read a register (0 when never written).
    #[must_use]
    pub fn get(&self, addr: RegAddr) -> u32 {
        self.values
            .get(&(addr.space, addr.mm, addr.base_idx))
            .copied()
            .unwrap_or(0)
    }

    /// Write a register; returns the previous value.
    pub fn set(&mut self, addr: RegAddr, value: u32) -> u32 {
        self.values
            .insert((addr.space, addr.mm, addr.base_idx), value)
            .unwrap_or(0)
    }

    /// Reset to power-on state (all defaults are zero in PDG's model;
    /// the real defaults differ per register and are tracked where they
    /// matter as typed state instead).
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Remove every register in `space` (used by CLEAR_STATE).
    pub fn remove_space(&mut self, space: RegSpace) {
        self.values.retain(|(s, _, _), _| *s != space);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's own ring emissions must round-trip through the
    /// packet-offset formulas (SCRATCH_REG0 via SET_UCONFIG_REG and
    /// PA_SC_TILE_STEERING_OVERRIDE via SET_CONTEXT_REG).
    #[test]
    fn kernel_verified_offsets_roundtrip() {
        // mmSCRATCH_REG0 = 0x2040, BASE_IDX 1, written by the kernel as
        // SET_UCONFIG_REG offset 0x40 (0xA000 + 0x2040 - 0xC000).
        let scratch = RegAddr::new(RegSpace::UConfig, 0x2040, 1);
        assert_eq!(scratch.packet_offset(), Some(0x40));
        assert_eq!(
            RegAddr::from_packet_offset(RegSpace::UConfig, 0x40),
            Some(scratch)
        );

        // mmPA_SC_TILE_STEERING_OVERRIDE = 0x00D7, BASE_IDX 1, written by
        // the kernel as SET_CONTEXT_REG offset 0xD7.
        let tile = RegAddr::new(RegSpace::Context, 0x00D7, 1);
        assert_eq!(tile.packet_offset(), Some(0xD7));
        assert_eq!(
            RegAddr::from_packet_offset(RegSpace::Context, 0xD7),
            Some(tile)
        );
    }

    /// The SH-space registers the e2e path binds shaders through.
    #[test]
    fn sh_space_shader_registers() {
        // SPI_SHADER_PGM_LO_VS = 0x19E8 BASE_IDX 0 → SEG0 + mm - 0x2C00.
        assert_eq!(sh::SPI_SHADER_PGM_LO_VS.packet_offset(), Some(0x48));
        // SPI_SHADER_PGM_LO_PS = 0x19A8 → 0x08.
        assert_eq!(sh::SPI_SHADER_PGM_LO_PS.packet_offset(), Some(0x08));
        // COMPUTE_PGM_LO = 0x1BAC → 0x20C (matches the Vega kernel's own
        // SET_SH_REG computation pattern).
        assert_eq!(sh::COMPUTE_PGM_LO.packet_offset(), Some(0x20C));
        // COMPUTE_DIM_X = 0x1BA1 → 0x201.
        assert_eq!(sh::COMPUTE_DIM_X.packet_offset(), Some(0x201));
    }

    /// Context registers project at their raw `mm` value.
    #[test]
    fn context_registers_project_at_mm() {
        for (addr, mm) in [
            (ctx::CB_COLOR0_BASE, 0x0318),
            (ctx::CB_COLOR0_INFO, 0x031C),
            (ctx::VGT_DRAW_INITIATOR, 0x01FC),
            (ctx::VGT_PRIMITIVE_TYPE, 0x2242),
            (ctx::VGT_INDEX_TYPE, 0x2243),
            (ctx::VGT_NUM_INSTANCES, 0x224D),
            (ctx::CB_TARGET_MASK, 0x008E),
            (ctx::PA_SC_SCREEN_SCISSOR_TL, 0x000C),
        ] {
            assert_eq!(addr.packet_offset(), Some(mm), "context projection");
            assert_eq!(
                RegAddr::from_packet_offset(RegSpace::Context, mm),
                Some(addr)
            );
        }
    }

    /// A `BASE_IDX`/space combination the hardware cannot express.
    #[test]
    fn mismatched_segments_rejected() {
        // A BASE_IDX 0 register cannot be written through the context space.
        let vs_pgm = RegAddr::new(RegSpace::Sh, 0x19E8, 0);
        assert_eq!(
            RegAddr {
                space: RegSpace::Context,
                ..vs_pgm
            }
            .packet_offset(),
            None
        );
    }

    #[test]
    fn register_file_read_write() {
        let mut rf = RegisterFile::default();
        assert_eq!(rf.get(ctx::CB_COLOR0_INFO), 0);
        rf.set(ctx::CB_COLOR0_INFO, 0xABCD_1234);
        assert_eq!(rf.get(ctx::CB_COLOR0_INFO), 0xABCD_1234);
        assert_eq!(rf.set(ctx::CB_COLOR0_INFO, 1), 0xABCD_1234);
        rf.clear();
        assert_eq!(rf.get(ctx::CB_COLOR0_INFO), 0);
    }
}
