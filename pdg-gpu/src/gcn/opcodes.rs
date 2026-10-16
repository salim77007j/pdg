//! gfx10 (RDNA) opcode numbers per instruction space.
//!
//! Every number was extracted from the real encodings in LLVM's public
//! AMDGPU backend (`VOP1/2/3Instructions.td`, `SOPInstructions.td`,
//! `SMInstructions.td`, `DSInstructions.td`, `BUFInstructions.td`) —
//! specifically the `*_Real_gfx10` / `*_Real_gfx6_gfx7_gfx10` classes,
//! whose numbers gfx10 shares. Instructions outside this table decode
//! to a numeric opcode and the lifter refuses them by name; the decoder
//! still frames them correctly.
//!
//! Renames LLVM applies for gfx10 are noted (e.g. `s_add_u32` was
//! `s_add_co_u32`-shaped on gfx9; `v_add_u32` is `v_add_nc_u32`).

/// SOP1 opcodes.
pub mod sop1 {
    /// `s_mov_b32` (gfx6-10).
    pub const S_MOV_B32: u8 = 0x03;
    /// `s_mov_b64`.
    pub const S_MOV_B64: u8 = 0x04;
    /// `s_not_b32`.
    pub const S_NOT_B32: u8 = 0x07;
    /// `s_swappc_b64`.
    pub const S_SWAPPC_B64: u8 = 0x21;
    /// `s_and_saveexec_b64`.
    pub const S_AND_SAVEEXEC_B64: u8 = 0x24;
}

/// SOP2 opcodes.
pub mod sop2 {
    /// `s_add_u32` (gfx10 name of gfx9 `s_add_co_u32`).
    pub const S_ADD_U32: u8 = 0x00;
    /// `s_sub_u32`.
    pub const S_SUB_U32: u8 = 0x01;
    /// `s_add_i32`.
    pub const S_ADD_I32: u8 = 0x02;
    /// `s_sub_i32`.
    pub const S_SUB_I32: u8 = 0x03;
    /// `s_min_u32`.
    pub const S_MIN_U32: u8 = 0x07;
    /// `s_max_u32`.
    pub const S_MAX_U32: u8 = 0x09;
    /// `s_cselect_b32`.
    pub const S_CSELECT_B32: u8 = 0x0A;
    /// `s_and_b32`.
    pub const S_AND_B32: u8 = 0x0E;
    /// `s_or_b32`.
    pub const S_OR_B32: u8 = 0x10;
    /// `s_xor_b32`.
    pub const S_XOR_B32: u8 = 0x12;
    /// `s_lshl_b32`.
    pub const S_LSHL_B32: u8 = 0x1E;
    /// `s_lshr_b32`.
    pub const S_LSHR_B32: u8 = 0x20;
    /// `s_ashr_i32`.
    pub const S_ASHR_I32: u8 = 0x22;
}

/// SOPC opcodes.
pub mod sopc {
    /// `s_cmp_eq_u32`.
    pub const S_CMP_EQ_U32: u8 = 0x06;
    /// `s_cmp_lg_u32`.
    pub const S_CMP_LG_U32: u8 = 0x07;
    /// `s_cmp_gt_u32`.
    pub const S_CMP_GT_U32: u8 = 0x08;
    /// `s_cmp_ge_u32`.
    pub const S_CMP_GE_U32: u8 = 0x09;
    /// `s_cmp_lt_u32`.
    pub const S_CMP_LT_U32: u8 = 0x0A;
    /// `s_cmp_le_u32`.
    pub const S_CMP_LE_U32: u8 = 0x0B;
}

/// SOPK opcodes.
pub mod sopk {
    /// `s_movk_i32`.
    pub const S_MOVK_I32: u8 = 0x00;
    /// `s_cmpk_eq_u32`.
    pub const S_CMPK_EQ_U32: u8 = 0x09;
    /// `s_cmpk_lg_u32`.
    pub const S_CMPK_LG_U32: u8 = 0x0A;
    /// `s_addk_i32`.
    pub const S_ADDK_I32: u8 = 0x0F;
}

/// SOPP opcodes.
pub mod sopp {
    /// `s_nop`.
    pub const S_NOP: u8 = 0x00;
    /// `s_endpgm`.
    pub const S_ENDPGM: u8 = 0x01;
    /// `s_branch`.
    pub const S_BRANCH: u8 = 0x02;
    /// `s_cbranch_scc0`.
    pub const S_CBRANCH_SCC0: u8 = 0x04;
    /// `s_cbranch_scc1`.
    pub const S_CBRANCH_SCC1: u8 = 0x05;
    /// `s_cbranch_execz`.
    pub const S_CBRANCH_EXECZ: u8 = 0x08;
    /// `s_cbranch_execnz`.
    pub const S_CBRANCH_EXECNZ: u8 = 0x09;
    /// `s_barrier`.
    pub const S_BARRIER: u8 = 0x0A;
    /// `s_waitcnt`.
    pub const S_WAITCNT: u8 = 0x0C;
}

/// SMEM opcodes (gfx10).
pub mod smem {
    /// `s_load_dword`.
    pub const S_LOAD_DWORD: u8 = 0x00;
    /// `s_load_dwordx2`.
    pub const S_LOAD_DWORDX2: u8 = 0x01;
    /// `s_load_dwordx4`.
    pub const S_LOAD_DWORDX4: u8 = 0x02;
    /// `s_load_dwordx8`.
    pub const S_LOAD_DWORDX8: u8 = 0x03;
    /// `s_load_dwordx16`.
    pub const S_LOAD_DWORDX16: u8 = 0x04;
}

/// VOP1 opcodes.
pub mod vop1 {
    /// `v_mov_b32`.
    pub const V_MOV_B32: u8 = 0x01;
    /// `v_cvt_f32_i32`.
    pub const V_CVT_F32_I32: u8 = 0x05;
    /// `v_cvt_i32_f32`.
    pub const V_CVT_I32_F32: u8 = 0x08;
    /// `v_not_b32`.
    pub const V_NOT_B32: u8 = 0x37;
}

/// VOP2 opcodes.
pub mod vop2 {
    /// `v_add_f32`.
    pub const V_ADD_F32: u8 = 0x03;
    /// `v_mul_f32`.
    pub const V_MUL_F32: u8 = 0x08;
    /// `v_mul_u32_u24`.
    pub const V_MUL_U32_U24: u8 = 0x0B;
    /// `v_min_u32`.
    pub const V_MIN_U32: u8 = 0x13;
    /// `v_max_u32`.
    pub const V_MAX_U32: u8 = 0x14;
    /// `v_lshrrev_b32`.
    pub const V_LSHRREV_B32: u8 = 0x16;
    /// `v_ashrrev_i32`.
    pub const V_ASHRREV_I32: u8 = 0x18;
    /// `v_lshlrev_b32`.
    pub const V_LSHLREV_B32: u8 = 0x1A;
    /// `v_and_b32`.
    pub const V_AND_B32: u8 = 0x1B;
    /// `v_or_b32`.
    pub const V_OR_B32: u8 = 0x1C;
    /// `v_xor_b32`.
    pub const V_XOR_B32: u8 = 0x1D;
    /// `v_mac_f32`.
    pub const V_MAC_F32: u8 = 0x1F;
    /// `v_add_u32` (`v_add_nc_u32`).
    pub const V_ADD_U32: u8 = 0x25;
    /// `v_sub_u32` (`v_sub_nc_u32`).
    pub const V_SUB_U32: u8 = 0x26;
}

/// VOP3 opcodes (gfx10, 10-bit).
pub mod vop3 {
    /// `v_mad_f32`.
    pub const V_MAD_F32: u16 = 0x141;
    /// `v_mad_u32_u24`.
    pub const V_MAD_U32_U24: u16 = 0x143;
    /// `v_bfe_u32`.
    pub const V_BFE_U32: u16 = 0x148;
    /// `v_fma_f32`.
    pub const V_FMA_F32: u16 = 0x14B;
    /// `v_interp_p0_f32` (RDNA moved VINTRP here).
    pub const V_INTERP_P0_F32: u16 = 0x200;
    /// `v_interp_p1_f32`.
    pub const V_INTERP_P1_F32: u16 = 0x201;
    /// `v_interp_mov_f32`.
    pub const V_INTERP_MOV_F32: u16 = 0x202;

    /// Base added to a VOP2 opcode to reach its VOP3 (e64) form
    /// (LLVM `VOP2_Real_e64_gfx10` packs `{0,1,0,0, op}`).
    pub const VOP2_TO_VOP3: u16 = 0x100;
}

/// DS opcodes.
pub mod ds {
    /// `ds_write_b32`.
    pub const DS_WRITE_B32: u8 = 0x0D;
    /// `ds_write2_b32`.
    pub const DS_WRITE2_B32: u8 = 0x0E;
    /// `ds_read_b32`.
    pub const DS_READ_B32: u8 = 0x36;
    /// `ds_read2_b32`.
    pub const DS_READ2_B32: u8 = 0x37;
    /// `ds_write_b64`.
    pub const DS_WRITE_B64: u8 = 0x4D;
    /// `ds_read_b64`.
    pub const DS_READ_B64: u8 = 0x76;
}

/// MUBUF opcodes.
pub mod mubuf {
    /// `buffer_load_dword`.
    pub const BUFFER_LOAD_DWORD: u8 = 0x0C;
    /// `buffer_load_dwordx2`.
    pub const BUFFER_LOAD_DWORDX2: u8 = 0x0D;
    /// `buffer_load_dwordx4`.
    pub const BUFFER_LOAD_DWORDX4: u8 = 0x0E;
    /// `buffer_store_dword`.
    pub const BUFFER_STORE_DWORD: u8 = 0x1C;
    /// `buffer_store_dwordx2`.
    pub const BUFFER_STORE_DWORDX2: u8 = 0x1D;
}

/// EXP targets (`EXP.tgt` field, 0..63).
pub mod exp_tgt {
    /// First param export (position).
    pub const PARAM0: u8 = 0x0;
    /// First pixel export (color target 0).
    pub const MRTZ0: u8 = 0x8;
    /// Null export.
    pub const NULL: u8 = 0x09;
}

/// Name an SOP1 opcode, if known.
#[must_use]
pub fn sop1_name(op: u8) -> Option<&'static str> {
    Some(match op {
        sop1::S_MOV_B32 => "s_mov_b32",
        sop1::S_MOV_B64 => "s_mov_b64",
        sop1::S_NOT_B32 => "s_not_b32",
        sop1::S_SWAPPC_B64 => "s_swappc_b64",
        sop1::S_AND_SAVEEXEC_B64 => "s_and_saveexec_b64",
        _ => return None,
    })
}

/// Name an SOP2 opcode, if known.
#[must_use]
pub fn sop2_name(op: u8) -> Option<&'static str> {
    Some(match op {
        sop2::S_ADD_U32 => "s_add_u32",
        sop2::S_SUB_U32 => "s_sub_u32",
        sop2::S_ADD_I32 => "s_add_i32",
        sop2::S_SUB_I32 => "s_sub_i32",
        sop2::S_MIN_U32 => "s_min_u32",
        sop2::S_MAX_U32 => "s_max_u32",
        sop2::S_CSELECT_B32 => "s_cselect_b32",
        sop2::S_AND_B32 => "s_and_b32",
        sop2::S_OR_B32 => "s_or_b32",
        sop2::S_XOR_B32 => "s_xor_b32",
        sop2::S_LSHL_B32 => "s_lshl_b32",
        sop2::S_LSHR_B32 => "s_lshr_b32",
        sop2::S_ASHR_I32 => "s_ashr_i32",
        _ => return None,
    })
}

/// Name a SOPC opcode, if known.
#[must_use]
pub fn sopc_name(op: u8) -> Option<&'static str> {
    Some(match op {
        sopc::S_CMP_EQ_U32 => "s_cmp_eq_u32",
        sopc::S_CMP_LG_U32 => "s_cmp_lg_u32",
        sopc::S_CMP_GT_U32 => "s_cmp_gt_u32",
        sopc::S_CMP_GE_U32 => "s_cmp_ge_u32",
        sopc::S_CMP_LT_U32 => "s_cmp_lt_u32",
        sopc::S_CMP_LE_U32 => "s_cmp_le_u32",
        _ => return None,
    })
}

/// Name a SOPK opcode, if known.
#[must_use]
pub fn sopk_name(op: u8) -> Option<&'static str> {
    Some(match op {
        sopk::S_MOVK_I32 => "s_movk_i32",
        sopk::S_CMPK_EQ_U32 => "s_cmpk_eq_u32",
        sopk::S_CMPK_LG_U32 => "s_cmpk_lg_u32",
        sopk::S_ADDK_I32 => "s_addk_i32",
        _ => return None,
    })
}

/// Name a SOPP opcode, if known.
#[must_use]
pub fn sopp_name(op: u8) -> Option<&'static str> {
    Some(match op {
        sopp::S_NOP => "s_nop",
        sopp::S_ENDPGM => "s_endpgm",
        sopp::S_BRANCH => "s_branch",
        sopp::S_CBRANCH_SCC0 => "s_cbranch_scc0",
        sopp::S_CBRANCH_SCC1 => "s_cbranch_scc1",
        sopp::S_CBRANCH_EXECZ => "s_cbranch_execz",
        sopp::S_CBRANCH_EXECNZ => "s_cbranch_execnz",
        sopp::S_BARRIER => "s_barrier",
        sopp::S_WAITCNT => "s_waitcnt",
        _ => return None,
    })
}

/// Name an SMEM opcode, if known.
#[must_use]
pub fn smem_name(op: u8) -> Option<&'static str> {
    Some(match op {
        smem::S_LOAD_DWORD => "s_load_dword",
        smem::S_LOAD_DWORDX2 => "s_load_dwordx2",
        smem::S_LOAD_DWORDX4 => "s_load_dwordx4",
        smem::S_LOAD_DWORDX8 => "s_load_dwordx8",
        smem::S_LOAD_DWORDX16 => "s_load_dwordx16",
        _ => return None,
    })
}

/// Name a VOP1 opcode, if known.
#[must_use]
pub fn vop1_name(op: u8) -> Option<&'static str> {
    Some(match op {
        vop1::V_MOV_B32 => "v_mov_b32",
        vop1::V_CVT_F32_I32 => "v_cvt_f32_i32",
        vop1::V_CVT_I32_F32 => "v_cvt_i32_f32",
        vop1::V_NOT_B32 => "v_not_b32",
        _ => return None,
    })
}

/// Name a VOP2 opcode, if known.
#[must_use]
pub fn vop2_name(op: u8) -> Option<&'static str> {
    Some(match op {
        vop2::V_ADD_F32 => "v_add_f32",
        vop2::V_MUL_F32 => "v_mul_f32",
        vop2::V_MUL_U32_U24 => "v_mul_u32_u24",
        vop2::V_MIN_U32 => "v_min_u32",
        vop2::V_MAX_U32 => "v_max_u32",
        vop2::V_LSHRREV_B32 => "v_lshrrev_b32",
        vop2::V_ASHRREV_I32 => "v_ashrrev_i32",
        vop2::V_LSHLREV_B32 => "v_lshlrev_b32",
        vop2::V_AND_B32 => "v_and_b32",
        vop2::V_OR_B32 => "v_or_b32",
        vop2::V_XOR_B32 => "v_xor_b32",
        vop2::V_MAC_F32 => "v_mac_f32",
        vop2::V_ADD_U32 => "v_add_u32",
        vop2::V_SUB_U32 => "v_sub_u32",
        _ => return None,
    })
}

/// Name a VOP3 opcode, if known.
#[must_use]
pub fn vop3_name(op: u16) -> Option<&'static str> {
    Some(match op {
        vop3::V_MAD_F32 => "v_mad_f32",
        vop3::V_MAD_U32_U24 => "v_mad_u32_u24",
        vop3::V_BFE_U32 => "v_bfe_u32",
        vop3::V_FMA_F32 => "v_fma_f32",
        vop3::V_INTERP_P0_F32 => "v_interp_p0_f32",
        vop3::V_INTERP_P1_F32 => "v_interp_p1_f32",
        vop3::V_INTERP_MOV_F32 => "v_interp_mov_f32",
        0x103 => "v_add_f32(e64)",
        0x108 => "v_mul_f32(e64)",
        0x11B => "v_and_b32(e64)",
        0x11C => "v_or_b32(e64)",
        0x11D => "v_xor_b32(e64)",
        0x113 => "v_min_u32(e64)",
        0x114 => "v_max_u32(e64)",
        0x11A => "v_lshlrev_b32(e64)",
        0x116 => "v_lshrrev_b32(e64)",
        0x118 => "v_ashrrev_i32(e64)",
        _ => return None,
    })
}

/// Name a DS opcode, if known.
#[must_use]
pub fn ds_name(op: u8) -> Option<&'static str> {
    Some(match op {
        ds::DS_WRITE_B32 => "ds_write_b32",
        ds::DS_WRITE2_B32 => "ds_write2_b32",
        ds::DS_READ_B32 => "ds_read_b32",
        ds::DS_READ2_B32 => "ds_read2_b32",
        ds::DS_WRITE_B64 => "ds_write_b64",
        ds::DS_READ_B64 => "ds_read_b64",
        _ => return None,
    })
}

/// Name an MUBUF opcode, if known.
#[must_use]
pub fn mubuf_name(op: u8) -> Option<&'static str> {
    Some(match op {
        mubuf::BUFFER_LOAD_DWORD => "buffer_load_dword",
        mubuf::BUFFER_LOAD_DWORDX2 => "buffer_load_dwordx2",
        mubuf::BUFFER_LOAD_DWORDX4 => "buffer_load_dwordx4",
        mubuf::BUFFER_STORE_DWORD => "buffer_store_dword",
        mubuf::BUFFER_STORE_DWORDX2 => "buffer_store_dwordx2",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors() {
        // Anchors cross-checked against the LLVM extraction.
        assert_eq!(sop1_name(0x03), Some("s_mov_b32"));
        assert_eq!(sop2_name(0x00), Some("s_add_u32"));
        assert_eq!(sop2_name(0x22), Some("s_ashr_i32"));
        assert_eq!(sopp_name(0x01), Some("s_endpgm"));
        assert_eq!(sopp_name(0x0C), Some("s_waitcnt"));
        assert_eq!(smem_name(0x02), Some("s_load_dwordx4"));
        assert_eq!(vop1_name(0x01), Some("v_mov_b32"));
        assert_eq!(vop2_name(0x1B), Some("v_and_b32"));
        assert_eq!(vop2_name(0x25), Some("v_add_u32"));
        assert_eq!(vop3_name(0x200), Some("v_interp_p0_f32"));
        assert_eq!(vop3_name(0x103), Some("v_add_f32(e64)"));
        assert_eq!(ds_name(0x36), Some("ds_read_b32"));
        assert_eq!(mubuf_name(0x0E), Some("buffer_load_dwordx4"));
        // Unknown stays None.
        assert_eq!(vop1_name(0x99), None);
        assert_eq!(sop2_name(0x63), None);
    }
}
