//! Semantic IR and typed GPU context.
//!
//! The command processor decodes PM4 into [`GpuCommand`]s executed against
//! a [`GpuContext`] — the typed view over the raw gfx10.3 register files
//! (see [`crate::pm4::regs`]). A [`crate::GpuDevice`] renders a batch of
//! commands against the context snapshot.
//!
//! Design notes:
//!
//! * The typed view is **derived on read** from the register file rather
//!   than materialized on write: there is one source of truth, and
//!   `CLEAR_STATE` is a register-file operation instead of a dual update.
//! * Registers outside PDG's typed subset stay readable as raw words —
//!   the CP never pretends state it does not model.

use crate::pm4::regs::{DataFormat, NumberFormat, RegAddr, RegSpace, RegisterFile, ctx, sh};

/// Index-buffer element size (`VGT_INDEX_TYPE.INDEX_TYPE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexType {
    /// 16-bit indices.
    U16,
    /// 32-bit indices.
    U32,
    /// 8-bit indices (gfx9+).
    U8,
    /// Not programmed yet.
    #[default]
    Unset,
}

impl IndexType {
    /// Decode `VGT_INDEX_TYPE.INDEX_TYPE`.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        match raw & 3 {
            0 => Self::U16,
            1 => Self::U32,
            2 => Self::U8,
            _ => Self::Unset,
        }
    }
}

/// Primitive topology (`VGT_PRIMITIVE_TYPE.DI_PT_*`, Mesa `sid.h` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveType {
    /// Point list.
    PointList,
    /// Line list.
    LineList,
    /// Line strip.
    LineStrip,
    /// Triangle list.
    TriList,
    /// Triangle strip.
    TriStrip,
    /// Triangle fan.
    TriFan,
    /// Programmed but not supported by the backend yet.
    #[default]
    Unsupported,
}

impl PrimitiveType {
    /// Decode the `DI_PT_*` number.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            1 => Self::PointList,
            2 => Self::LineList,
            3 => Self::LineStrip,
            4 => Self::TriList,
            5 => Self::TriStrip,
            6 => Self::TriFan,
            _ => Self::Unsupported,
        }
    }
}

/// Color-buffer component swap (`CB_COLORn_INFO.COMP_SWAP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompSwap {
    /// Standard order.
    #[default]
    Standard,
    /// Alternate order (ALT1).
    Alt1,
    /// Alternate order (ALT2).
    Alt2,
    /// Alternate order (ALT3).
    Alt3,
}

impl CompSwap {
    /// Decode the 2-bit field.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        match raw & 3 {
            0 => Self::Standard,
            1 => Self::Alt1,
            2 => Self::Alt2,
            _ => Self::Alt3,
        }
    }

    /// Permutation applied to a color written to the target.
    #[must_use]
    pub fn apply(self, rgba: [f32; 4]) -> [f32; 4] {
        match self {
            Self::Standard => rgba,
            Self::Alt1 => [rgba[2], rgba[1], rgba[0], rgba[3]],
            Self::Alt2 => [rgba[1], rgba[0], rgba[2], rgba[3]],
            Self::Alt3 => [rgba[0], rgba[1], rgba[2], rgba[3]],
        }
    }
}

/// Tiling of a color target (`CB_COLORn_ATTRIB.TILE_MODE_INDEX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileMode {
    /// Linear (micro) tiling — the only mode PDG computes dimensions for.
    Linear,
    /// A tiled mode PDG cannot yet size; the backend must reject the target.
    Tiled,
}

/// One bound color target (typed view of the `CB_COLORn_*` family).
#[derive(Debug, Clone, PartialEq)]
pub struct TargetState {
    /// Guest address of the surface (`CB_COLORn_BASE` + `_BASE_EXT`).
    pub base: u64,
    /// Data format (`CB_COLORn_INFO.FORMAT`).
    pub format: DataFormat,
    /// Number format (`CB_COLORn_INFO.NUMBER_TYPE`).
    pub number: NumberFormat,
    /// Component swap (`CB_COLORn_INFO.COMP_SWAP`).
    pub comp_swap: CompSwap,
    /// Width in pixels, `None` for unsupported tiling.
    pub width: Option<u32>,
    /// Height in pixels, `None` for unsupported tiling.
    pub height: Option<u32>,
    /// Tile mode.
    pub tile_mode: TileMode,
    /// `CB_COLORn_CLEAR_WORD0`.
    pub clear_word0: u32,
    /// `CB_COLORn_CLEAR_WORD1`.
    pub clear_word1: u32,
}

impl TargetState {
    /// Whether the target can be rendered to by the current backends.
    #[must_use]
    pub fn is_renderable(&self) -> bool {
        self.tile_mode == TileMode::Linear
            && self.width.is_some()
            && self.height.is_some()
            && self.format != DataFormat::Unknown
    }
}

/// Shader program binding (`SPI_SHADER_PGM_LO/HI_*` + user data).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderBinding {
    /// Guest address of the shader program.
    pub pgm: u64,
    /// User-SGPR dwords (`SPI_SHADER_USER_DATA_*`).
    pub user_data: Vec<u32>,
    /// First user-data register (for range checks).
    pub user_data_base: RegAddr,
}

/// Compute dispatch state (typed view of the `COMPUTE_*` SH registers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputeState {
    /// Guest address of the compute program.
    pub pgm: u64,
    /// Grid dimensions (COMPUTE_DIM_X/Y/Z).
    pub groups: [u32; 3],
    /// Block dimensions (COMPUTE_NUM_THREAD_X/Y/Z).
    pub threads: [u32; 3],
}

/// Shader stages PDG binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Vertex shader.
    Vertex,
    /// Pixel shader.
    Pixel,
    /// Compute shader.
    Compute,
}

/// Draw parameters snapshot (VGT state at `DRAW_*` time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawParams {
    /// Vertex/index count.
    pub count: u32,
    /// Index type (`Unset` for auto draws).
    pub index_type: IndexType,
    /// Instance count.
    pub instances: u32,
    /// Topology.
    pub prim_type: PrimitiveType,
    /// Base vertex (`DRAW_INDEX_OFFSET_2`).
    pub base_vertex: i32,
    /// Index-buffer guest address (`INDEX_BASE`), zero for auto draws.
    pub index_base: u64,
}

/// A semantic GPU command produced by the command processor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuCommand {
    /// No-op.
    Nop,
    /// Execute a draw with the context's current state.
    Draw {
        /// VGT snapshot.
        params: DrawParams,
    },
    /// Execute a compute dispatch.
    Dispatch {
        /// Grid size in groups.
        groups: [u32; 3],
    },
    /// Write `value` to `addr` when everything before it completed.
    Fence {
        /// Guest physical address to write.
        addr: u64,
        /// Value to write (32 or 64 bits; high word ignored for 32).
        value: u64,
        /// True for a 64-bit write.
        wide: bool,
    },
    /// Wait until `*addr` compares true against `value`.
    SemaphoreWait {
        /// Guest physical address to poll.
        addr: u64,
        /// Value compared for equality.
        value: u64,
    },
    /// Copy `len` bytes from `src` to `dst` through the memory port.
    Copy {
        /// Destination address.
        dst: u64,
        /// Source address.
        src: u64,
        /// Length in bytes.
        len: u32,
    },
    /// Present the framebuffer (frontend hook).
    Present {
        /// Swapchain/target index.
        index: u32,
    },
}

/// The typed GPU context: raw register files + derived views.
#[derive(Debug, Default, Clone)]
pub struct GpuContext {
    regs: RegisterFile,
}

impl GpuContext {
    /// A fresh, power-on context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Raw register files.
    #[must_use]
    pub fn regs(&self) -> &RegisterFile {
        &self.regs
    }

    /// Raw register files (mutating access for the CP).
    pub(crate) fn regs_mut(&mut self) -> &mut RegisterFile {
        &mut self.regs
    }

    /// Deep snapshot of the register state (used when handing the
    /// context to a backend while the CP stays mutable).
    #[must_use]
    pub fn clone_state(&self) -> Self {
        self.clone()
    }

    /// Reset all context-space registers (CLEAR_STATE).
    pub fn clear_context(&mut self) {
        self.regs.remove_space(RegSpace::Context);
    }

    /// Shader binding for a stage (address + user-SGPR window).
    #[must_use]
    pub fn shader_binding(&self, stage: Stage) -> ShaderBinding {
        let (lo, hi, ud0, ud_n): (RegAddr, RegAddr, RegAddr, usize) = match stage {
            Stage::Vertex => (
                sh::SPI_SHADER_PGM_LO_VS,
                sh::SPI_SHADER_PGM_HI_VS,
                sh::SPI_SHADER_USER_DATA_VS_0,
                32,
            ),
            Stage::Pixel => (
                sh::SPI_SHADER_PGM_LO_PS,
                sh::SPI_SHADER_PGM_HI_PS,
                sh::SPI_SHADER_USER_DATA_PS_0,
                32,
            ),
            Stage::Compute => (
                sh::COMPUTE_PGM_LO,
                sh::COMPUTE_PGM_HI,
                sh::COMPUTE_USER_DATA_0,
                16,
            ),
        };
        // PGM_LO/HI pack the address in units of 256 bytes (shader-base
        // granularity, public AMD documentation).
        let pgm =
            (u64::from(self.regs.get(lo)) << 8) | ((u64::from(self.regs.get(hi)) & 0xFFFF) << 40);
        let user_data = (0..ud_n as u32)
            .map(|i| {
                self.regs
                    .get(RegAddr::new(ud0.space, ud0.mm + i, ud0.base_idx))
            })
            .collect();
        ShaderBinding {
            pgm,
            user_data,
            user_data_base: ud0,
        }
    }

    /// Compute state.
    #[must_use]
    pub fn compute(&self) -> ComputeState {
        ComputeState {
            pgm: (u64::from(self.regs.get(sh::COMPUTE_PGM_LO)) << 8)
                | ((u64::from(self.regs.get(sh::COMPUTE_PGM_HI)) & 0xFFFF) << 40),
            groups: [
                self.regs.get(sh::COMPUTE_DIM_X),
                self.regs.get(sh::COMPUTE_DIM_Y),
                self.regs.get(sh::COMPUTE_DIM_Z),
            ],
            threads: [
                self.regs.get(sh::COMPUTE_NUM_THREAD_X),
                self.regs.get(sh::COMPUTE_NUM_THREAD_Y),
                self.regs.get(sh::COMPUTE_NUM_THREAD_Z),
            ],
        }
    }

    /// Typed view of color target `n` (0..8).
    ///
    /// Dimension decoding supports the linear tile mode only:
    /// a linear `256`-byte tile row is `(PITCH.TILE_MAX + 1)` tiles wide
    /// and `(SLICE.TILE_MAX + 1)` micro-tile rows (8 lines) tall.
    #[must_use]
    pub fn target(&self, n: u8) -> TargetState {
        debug_assert!(n < 8, "eight color targets exist");
        // gfx10.3 CB_COLORn register stride (kernel gc_10_3_0 header:
        // CB_COLOR0_BASE 0x0318, CB_COLOR1_BASE 0x0327, ... = 0x0F).
        let step = 0x0F;
        let info = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_INFO.mm + u32::from(n) * step,
            1,
        );
        let base = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_BASE.mm + u32::from(n) * step,
            1,
        );
        let base_ext = RegAddr::new(RegSpace::Context, 0x0390 + u32::from(n), 1);
        let pitch = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_PITCH.mm + u32::from(n) * step,
            1,
        );
        let slice = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_SLICE.mm + u32::from(n) * step,
            1,
        );
        let attrib = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_ATTRIB.mm + u32::from(n) * step,
            1,
        );
        let clear0 = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_CLEAR_WORD0.mm + u32::from(n) * step,
            1,
        );
        let clear1 = RegAddr::new(
            RegSpace::Context,
            ctx::CB_COLOR0_CLEAR_WORD1.mm + u32::from(n) * step,
            1,
        );

        let info_v = self.regs.get(info);
        let attrib_v = self.regs.get(attrib);

        let format = DataFormat::from_raw((info_v & 0x0000_007C) >> 2);
        let number = NumberFormat::from_raw((info_v & 0x0000_0700) >> 8);
        let comp_swap = CompSwap::from_raw((info_v & 0x0000_1800) >> 11);
        let tile_mode_index = attrib_v & 0x1F;
        let tile_mode = if tile_mode_index == 0 {
            TileMode::Linear
        } else {
            TileMode::Tiled
        };

        let (width, height) = if tile_mode == TileMode::Linear {
            let bytes_per_px = format.bits_per_element() / 8;
            let tile_cols = u64::from(self.regs.get(pitch) & 0x7FF) + 1; // PITCH.TILE_MAX
            let tile_rows = u64::from(self.regs.get(slice) & 0x7FF) + 1; // SLICE.TILE_MAX
            let w = (tile_cols * 256 / u64::from(bytes_per_px.max(1))) as u32;
            let h = (tile_rows * 8) as u32;
            (Some(w), Some(h))
        } else {
            (None, None)
        };

        let base_addr =
            (u64::from(self.regs.get(base_ext) & 0xFF) << 32) | u64::from(self.regs.get(base));

        TargetState {
            base: base_addr,
            format,
            number,
            comp_swap,
            width,
            height,
            tile_mode,
            clear_word0: self.regs.get(clear0),
            clear_word1: self.regs.get(clear1),
        }
    }

    /// Draw state (`VGT_*`).
    #[must_use]
    pub fn draw_state(&self) -> (IndexType, PrimitiveType, u32) {
        (
            IndexType::from_raw(self.regs.get(ctx::VGT_INDEX_TYPE) & 3),
            PrimitiveType::from_raw(self.regs.get(ctx::VGT_PRIMITIVE_TYPE) & 0x3F),
            self.regs.get(ctx::VGT_NUM_INSTANCES).max(1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_type_decodes() {
        assert_eq!(IndexType::from_raw(0), IndexType::U16);
        assert_eq!(IndexType::from_raw(1), IndexType::U32);
        assert_eq!(IndexType::from_raw(2), IndexType::U8);
        assert_eq!(IndexType::from_raw(3), IndexType::Unset);
    }

    #[test]
    fn primitive_type_decodes() {
        assert_eq!(PrimitiveType::from_raw(4), PrimitiveType::TriList);
        assert_eq!(PrimitiveType::from_raw(5), PrimitiveType::TriStrip);
        assert_eq!(PrimitiveType::from_raw(6), PrimitiveType::TriFan);
        assert_eq!(PrimitiveType::from_raw(0x20), PrimitiveType::Unsupported);
    }

    #[test]
    fn comp_swap_permutes() {
        let rgba = [0.1, 0.2, 0.3, 1.0];
        assert_eq!(CompSwap::Standard.apply(rgba), rgba);
        assert_eq!(CompSwap::Alt1.apply(rgba), [0.3, 0.2, 0.1, 1.0]);
    }

    #[test]
    fn shader_binding_assembles_address() {
        let mut ctx = GpuContext::new();
        ctx.regs_mut().set(sh::SPI_SHADER_PGM_LO_VS, 0x0001_2345);
        ctx.regs_mut().set(sh::SPI_SHADER_PGM_HI_VS, 0x0000_0002);
        let b = ctx.shader_binding(Stage::Vertex);
        assert_eq!(b.pgm, (0x12345 << 8) | (0x2 << 40));
        // User data window reads back consecutive SH registers.
        for i in 0..4u32 {
            ctx.regs_mut().set(
                RegAddr::new(RegSpace::Sh, sh::SPI_SHADER_USER_DATA_VS_0.mm + i, 0),
                i + 7,
            );
        }
        let b = ctx.shader_binding(Stage::Vertex);
        assert_eq!(&b.user_data[..4], &[7, 8, 9, 10]);
    }

    #[test]
    fn compute_state_from_registers() {
        let mut ctx = GpuContext::new();
        ctx.regs_mut().set(sh::COMPUTE_PGM_LO, 0x1000);
        ctx.regs_mut().set(sh::COMPUTE_DIM_X, 8);
        ctx.regs_mut().set(sh::COMPUTE_DIM_Y, 4);
        ctx.regs_mut().set(sh::COMPUTE_DIM_Z, 2);
        ctx.regs_mut().set(sh::COMPUTE_NUM_THREAD_X, 64);
        ctx.regs_mut().set(sh::COMPUTE_NUM_THREAD_Y, 1);
        ctx.regs_mut().set(sh::COMPUTE_NUM_THREAD_Z, 1);
        let c = ctx.compute();
        assert_eq!(c.pgm, 0x1000 << 8);
        assert_eq!(c.groups, [8, 4, 2]);
        assert_eq!(c.threads, [64, 1, 1]);
    }

    #[test]
    fn target_linear_dims_roundtrip() {
        let mut ctx = GpuContext::new();
        // 64x64 32bpp linear target: PITCH.TILE_MAX = 64*4/256-1 = 0,
        // SLICE.TILE_MAX = 64/8-1 = 7.
        ctx.regs_mut().set(ctx_reg(0x031C), 10 << 2); // FORMAT = 8_8_8_8
        ctx.regs_mut().set(ctx_reg(0x031D), 0); // TILE_MODE_INDEX = LINEAR
        ctx.regs_mut().set(ctx_reg(0x0319), 0); // PITCH.TILE_MAX
        ctx.regs_mut().set(ctx_reg(0x031A), 7); // SLICE.TILE_MAX
        ctx.regs_mut().set(ctx_reg(0x0318), 0x2000_0000);
        let t = ctx.target(0);
        assert_eq!(t.format, DataFormat::Fmt8_8_8_8);
        assert_eq!(t.width, Some(64));
        assert_eq!(t.height, Some(64));
        assert_eq!(t.tile_mode, TileMode::Linear);
        assert!(t.is_renderable());
    }

    #[test]
    fn target_tiled_has_no_dims() {
        let mut ctx = GpuContext::new();
        ctx.regs_mut().set(ctx_reg(0x031D), 4); // some tiled mode
        let t = ctx.target(0);
        assert_eq!(t.tile_mode, TileMode::Tiled);
        assert!(t.width.is_none() && t.height.is_none());
        assert!(!t.is_renderable());
    }

    #[test]
    fn clear_state_only_touches_context_space() {
        let mut ctx = GpuContext::new();
        ctx.regs_mut().set(ctx_reg(0x031C), 40 << 2);
        ctx.regs_mut().set(sh::COMPUTE_DIM_X, 8);
        ctx.clear_context();
        assert_eq!(ctx.regs().get(ctx_reg(0x031C)), 0);
        assert_eq!(ctx.regs().get(sh::COMPUTE_DIM_X), 8, "SH space survives");
    }

    fn ctx_reg(mm: u32) -> RegAddr {
        RegAddr::new(RegSpace::Context, mm, 1)
    }
}
