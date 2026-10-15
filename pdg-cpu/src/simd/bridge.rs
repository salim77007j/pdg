//! Bridges between the vector file and the GPRs/mem: MOVD/MOVQ, MOVMSK*,
//! PEXTR/PINSR, scalar conversions, COMISs/PTEST flags, variable blends,
//! broadcasts, permutes, variable shifts, widen/narrow and gathers.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use crate::fault::{AccessError, Fault, MemoryAccess};
use crate::interpreter::{Action, effective_address};
use crate::state::CpuState;

use super::{V256, finish_vec, immediate_of, read_vec, write_vec};

/// Try to execute `i` if it belongs to this family.
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn exec(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    wide: bool,
) -> Result<Option<Action>, Fault> {
    use Mnemonic::*;
    let m = i.mnemonic();
    let imm = immediate_of(i);
    let a = read_vec(i, first_src(i), next_ip, state, mem)?;
    let b = read_vec(i, last_src(i), next_ip, state, mem)?;

    let result: V256 = match m {
        // ---- variable blends (implicit XMM0 mask) -------------------------
        Blendvps | Vblendvps => blendv(a, b, mask_xmm0(state, wide), 4),
        Blendvpd | Vblendvpd => blendv(a, b, mask_xmm0(state, wide), 8),
        Pblendvb | Vpblendvb => blendv(a, b, mask_xmm0(state, wide), 1),

        // ---- broadcasts ---------------------------------------------------
        Vbroadcastss => broadcast32(a),
        Vbroadcastsd => broadcast64(a),
        Vbroadcastf128 | Vbroadcasti128 => [a[0], b[0]],
        Vpbroadcastb => broadcast8(a),
        Vpbroadcastw => broadcast16(a),
        Vpbroadcastd => broadcast32(a),
        Vpbroadcastq => broadcast64(a),

        // ---- 128-bit lane permutes ----------------------------------------
        Vpermq | Vpermpd => {
            // imm selects qwords per 128-bit half (2 per half).
            let mut r = [0u128; 2];
            for h in 0..2usize {
                for j in 0..2u32 {
                    let sel = ((imm >> (h as u32 * 2 + j)) & 1) as usize + h * 2;
                    let v = (b[sel / 2] >> ((sel % 2) * 64)) & 0xFFFF_FFFF_FFFF_FFFF;
                    r[h] |= v << (j * 64);
                }
            }
            r
        }
        Vpermd | Vpermps => {
            // a = indices, b = data: per 32-bit lane, a selects from b.
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..4 {
                    let idx = ((a[h] >> (j * 32)) & 7) as u32;
                    let v = (b[(idx / 4) as usize] >> ((idx % 4) * 32)) & 0xFFFF_FFFF;
                    r[h] |= v << (j * 32);
                }
            }
            r
        }
        Vperm2f128 | Vperm2i128 => {
            let sel = |n: u32| -> u128 {
                let c = ((imm >> (n * 4)) & 3) as usize;
                b[c]
            };
            [sel(0), sel(1)]
        }
        Vinsertf128 | Vinserti128 => {
            // dst = a with 128-bit lane `imm & 1` replaced by src2's low lane.
            let mut r = a;
            r[imm as usize & 1] = b[0];
            r
        }
        Vextracti128 | Vextractf128 => {
            // Memory or register destination; the imm picks the half.
            let v = a[imm as usize & 1];
            if i.op_kind(0) == OpKind::Memory {
                let addr = effective_address(i, next_ip, state);
                mem.write_buf(addr, &v.to_le_bytes())
                    .map_err(AccessError::into_fault)?;
                return Ok(Some(Action::Next));
            }
            [v, 0]
        }

        // ---- variable shifts (AVX2) ---------------------------------------
        Vpsllvd | Vpsllvq => {
            if m == Vpsllvd {
                shift_var(a, b, 4, |x, c| x << (c & 31))
            } else {
                shift_var(a, b, 8, |x, c| x << (c & 63))
            }
        }
        Vpsrlvd | Vpsrlvq => {
            if m == Vpsrlvd {
                shift_var(a, b, 4, |x, c| x >> (c & 31))
            } else {
                shift_var(a, b, 8, |x, c| x >> (c & 63))
            }
        }
        Vpsravd => shift_var(a, b, 4, |x, c| ((x as i32) >> (c & 31)) as u32 as u64),

        // ---- widen / narrow (PMOVZX/PMOVSX) --------------------------------
        Pmovzxbw | Vpmovzxbw => widen(a, 1, false),
        Pmovsxbw | Vpmovsxbw => widen(a, 1, true),
        Pmovzxwd | Vpmovzxwd => widen(a, 2, false),
        Pmovsxwd | Vpmovsxwd => widen(a, 2, true),
        Pmovzxdq | Vpmovzxdq => widen(a, 4, false),
        Pmovsxdq | Vpmovsxdq => widen(a, 4, true),
        Pmovzxbq => widen2(a, 1, false),
        Pmovsxbq => widen2(a, 1, true),
        Pmovzxwq => widen2(a, 2, false),
        Pmovsxwq => widen2(a, 2, true),

        // ---- horizontal / misc SSSE3 --------------------------------------
        Pabsb | Pabsw | Pabsd => {
            let w = match m {
                Pabsb => 1,
                Pabsw => 2,
                _ => 4,
            };
            let mut r = [0u128; 2];
            let n = 16 / w;
            for h in 0..2 {
                for j in 0..n {
                    let sh = j * w * 8;
                    let v = ((a[h] >> sh) & mask_lane(w)) as u64;
                    let abs = match w {
                        1 => (v as i8).unsigned_abs() as u64,
                        2 => (v as i16).unsigned_abs() as u64,
                        _ => (v as i32).unsigned_abs() as u64,
                    };
                    r[h] |= (abs as u128) << sh;
                }
            }
            r
        }
        Phaddw | Vphaddw => hadd(a, b, 2, false),
        Phaddd | Vphaddd => hadd(a, b, 4, false),
        Phaddsw => hadd(a, b, 2, true),
        Phsubw | Vphsubw => hsub(a, b, 2),
        Phsubd | Vphsubd => hsub(a, b, 4),
        Phsubsw => hsub_saturating(a, b),
        Psadbw | Vpsadbw => {
            // Sum of absolute byte differences per 64-bit group.
            let mut r = [0u128; 2];
            for h in 0..2 {
                for g in 0..2 {
                    let mut acc: u64 = 0;
                    for j in 0..8 {
                        let x = ((a[h] >> (g * 64 + j * 8)) & 0xFF) as u64;
                        let y = ((b[h] >> (g * 64 + j * 8)) & 0xFF) as u64;
                        acc += x.abs_diff(y);
                    }
                    r[h] |= u128::from(acc) << (g * 64);
                }
            }
            r
        }

        _ => {
            return match exec_gathers(i, next_ip, state, mem, a, b, wide)? {
                Some(result) => finish_vec(i, state, mem, next_ip, result, wide).map(Some),
                None => Ok(None),
            };
        }
    };
    let action = finish_vec(i, state, mem, next_ip, result, wide)?;
    Ok(Some(action))
}

fn mask_lane(w: u32) -> u128 {
    (1u128 << (w * 8)) - 1
}

fn first_src(i: &Instruction) -> u32 {
    if i.op_count() >= 3 && !matches!(i.op_kind(2), OpKind::Immediate8) {
        1
    } else {
        0
    }
}

fn last_src(i: &Instruction) -> u32 {
    let n = i.op_count();
    if matches!(
        i.op_kind(n - 1),
        OpKind::Immediate8 | OpKind::Immediate8to64 | OpKind::Immediate32to64
    ) {
        n - 2
    } else {
        n - 1
    }
}

fn mask_xmm0(state: &CpuState, wide: bool) -> V256 {
    let v = state.ymm_idx(0);
    if wide { v } else { [v[0], 0] }
}

/// Variable blend against an XMM0 mask at `lane` bytes.
fn blendv(a: V256, b: V256, mask: V256, lane: u32) -> V256 {
    let n = 16 / lane;
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..n {
            let sh = j * lane * 8;
            let bit = (mask[h] >> sh) & 1;
            let src = if bit != 0 { b } else { a };
            r[h] |= src[h] & (mask_lane(lane) << sh);
        }
    }
    r
}

fn broadcast8(a: V256) -> V256 {
    let v = a[0] & 0xFF;
    [
        v * 0x0101_0101_0101_0101_0101_0101_0101_0101,
        v * 0x0101_0101_0101_0101_0101_0101_0101_0101,
    ]
}

fn broadcast16(a: V256) -> V256 {
    let v = a[0] & 0xFFFF;
    [
        v * 0x0001_0001_0001_0001_0001_0001_0001_0001,
        v * 0x0001_0001_0001_0001_0001_0001_0001_0001,
    ]
}

fn broadcast32(a: V256) -> V256 {
    let v = a[0] & 0xFFFF_FFFF;
    [
        v * 0x0000_0001_0000_0001_0000_0001_0000_0001,
        v * 0x0000_0001_0000_0001_0000_0001_0000_0001,
    ]
}

fn broadcast64(a: V256) -> V256 {
    [a[0] & 0xFFFF_FFFF_FFFF_FFFF, a[0] & 0xFFFF_FFFF_FFFF_FFFF]
}

fn shift_var(a: V256, b: V256, w: u32, f: impl Fn(u64, u64) -> u64) -> V256 {
    let n = 16 / w;
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..n {
            let sh = j * w * 8;
            let x = ((a[h] >> sh) & mask_lane(w)) as u64;
            let c = ((b[h] >> sh) & mask_lane(w)) as u64;
            r[h] |= ((f(x, c) & mask_u64(w)) as u128) << sh;
        }
    }
    r
}

fn mask_u64(w: u32) -> u64 {
    (1u64 << (w * 8)) - 1
}

/// PMOVZX/PMOVSX with one widening step (byte→word, word→dword, dword→qword).
fn widen(a: V256, from: u32, signed: bool) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..16 / (from * 2) {
            let sh = j * from * 8;
            let v = ((a[h] >> sh) & mask_lane(from)) as u64;
            let ext = if signed {
                match from {
                    1 => v as i8 as i64 as u64,
                    2 => v as i16 as i64 as u64,
                    _ => v as i32 as i64 as u64,
                }
            } else {
                v
            };
            r[h] |= (ext as u128) << (j * from * 2 * 8);
        }
    }
    r
}

/// PMOVZX/PMOVSX with two widening steps (byte→qword, word→qword).
fn widen2(a: V256, from: u32, signed: bool) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..2 {
            let sh = j * from * 8;
            let v = ((a[h] >> sh) & mask_lane(from)) as u64;
            let ext = if signed {
                match from {
                    1 => v as i8 as i64 as u64,
                    _ => v as i16 as i64 as u64,
                }
            } else {
                v
            };
            r[h] |= (ext as u128) << (j * 64);
        }
    }
    r
}

fn hadd(a: V256, b: V256, w: u32, sat: bool) -> V256 {
    let n = 16 / w;
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..n / 2 {
            let x0 = ((a[h] >> (j * 2 * w * 8)) & mask_lane(w)) as u64;
            let x1 = ((a[h] >> ((j * 2 + 1) * w * 8)) & mask_lane(w)) as u64;
            let sum = match (w, sat) {
                (2, false) => x0.wrapping_add(x1),
                (2, true) => ((x0 as i16).saturating_add(x1 as i16)) as u16 as u64,
                _ => x0.wrapping_add(x1),
            };
            r[h] |= (sum as u128) << (j * w * 8);
        }
        for j in 0..n / 2 {
            let y0 = ((b[h] >> (j * 2 * w * 8)) & mask_lane(w)) as u64;
            let y1 = ((b[h] >> ((j * 2 + 1) * w * 8)) & mask_lane(w)) as u64;
            let sum = match (w, sat) {
                (2, false) => y0.wrapping_add(y1),
                (2, true) => ((y0 as i16).saturating_add(y1 as i16)) as u16 as u64,
                _ => y0.wrapping_add(y1),
            };
            r[h] |= (sum as u128) << ((n / 2 + j) * w * 8);
        }
    }
    r
}

fn hsub(a: V256, b: V256, w: u32) -> V256 {
    let n = 16 / w;
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..n / 2 {
            let x0 = ((a[h] >> (j * 2 * w * 8)) & mask_lane(w)) as u64;
            let x1 = ((a[h] >> ((j * 2 + 1) * w * 8)) & mask_lane(w)) as u64;
            r[h] |= (x0.wrapping_sub(x1) as u128) << (j * w * 8);
        }
        for j in 0..n / 2 {
            let y0 = ((b[h] >> (j * 2 * w * 8)) & mask_lane(w)) as u64;
            let y1 = ((b[h] >> ((j * 2 + 1) * w * 8)) & mask_lane(w)) as u64;
            r[h] |= (y0.wrapping_sub(y1) as u128) << ((n / 2 + j) * w * 8);
        }
    }
    r
}

fn hsub_saturating(a: V256, b: V256) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..4 {
            let x0 = ((a[h] >> (j * 32)) & 0xFFFF) as u16 as i16;
            let x1 = ((a[h] >> (j * 32 + 16)) & 0xFFFF) as u16 as i16;
            r[h] |= ((x0.saturating_sub(x1)) as u16 as u128) << (j * 16);
        }
        for j in 0..4 {
            let y0 = ((b[h] >> (j * 32)) & 0xFFFF) as u16 as i16;
            let y1 = ((b[h] >> (j * 32 + 16)) & 0xFFFF) as u16 as i16;
            r[h] |= ((y0.saturating_sub(y1)) as u16 as u128) << (64 + j * 16);
        }
    }
    r
}

// ---------------------------------------------------------------------------
// GPR bridge instructions (called directly from the dispatcher prelude)
// ---------------------------------------------------------------------------

/// COMISs/UCOMISs: compare lane 0, set ZF/PF/CF, clear OF/SF/AF.
pub(crate) fn exec_comis(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    use crate::state::rflags;
    let a = read_vec(i, 0, next_ip, state, mem)?;
    let b = read_vec(i, 1, next_ip, state, mem)?;
    let (zf, pf, cf) = match i.mnemonic() {
        Mnemonic::Comiss | Mnemonic::Ucomiss => {
            let x = f32::from_bits((a[0] & 0xFFFF_FFFF) as u32);
            let y = f32::from_bits((b[0] & 0xFFFF_FFFF) as u32);
            comis_flags(x.is_nan() || y.is_nan(), x > y, x < y)
        }
        _ => {
            let x = f64::from_bits((a[0] & 0xFFFF_FFFF_FFFF_FFFF) as u64);
            let y = f64::from_bits((b[0] & 0xFFFF_FFFF_FFFF_FFFF) as u64);
            comis_flags(x.is_nan() || y.is_nan(), x > y, x < y)
        }
    };
    state.rflags = (state.rflags
        & !(rflags::ZF | rflags::PF | rflags::CF | rflags::OF | rflags::SF | rflags::AF))
        | (u64::from(zf) * rflags::ZF)
        | (u64::from(pf) * rflags::PF)
        | (u64::from(cf) * rflags::CF);
    Ok(Action::Next)
}

/// PTEST: ZF = (dst & src) == 0; CF = (dst & !src) == 0.
pub(crate) fn exec_ptest(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    use crate::state::rflags;
    let a = read_vec(i, 0, next_ip, state, mem)?;
    let b = read_vec(i, 1, next_ip, state, mem)?;
    let zf = (a[0] & b[0] == 0) && (a[1] & b[1] == 0);
    let nand = (!a[0] & b[0] == 0) && (!a[1] & b[1] == 0);
    state.rflags = (state.rflags & !(rflags::ZF | rflags::CF))
        | (u64::from(zf) * rflags::ZF)
        | (u64::from(nand) * rflags::CF);
    Ok(Action::Next)
}

/// MOVD: 32-bit GPR/mem ↔ low lane of XMM (zero-extended into XMM).
pub(crate) fn exec_movd(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    match i.op_kind(0) {
        OpKind::Register => {
            // xmm, r/m32: zero the upper 96 bits.
            let v = gpr_or_mem32(i, 1, next_ip, state, mem)?;
            let r = i.op_register(0);
            state.set_xmm(r, u128::from(v)).ok_or_else(|| bad(i))?;
        }
        _ => {
            // r/m32, xmm: store the low 32 bits.
            let a = read_vec(i, 1, next_ip, state, mem)?;
            let v = (a[0] & 0xFFFF_FFFF) as u32;
            store_gpr_or_mem32(i, 0, next_ip, state, mem, v)?;
        }
    }
    Ok(Action::Next)
}

/// MOVQ: 64-bit GPR/mem ↔ low lane of XMM.
pub(crate) fn exec_movq(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    match i.op_kind(0) {
        OpKind::Register => {
            let v = gpr_or_mem64(i, 1, next_ip, state, mem)?;
            let r = i.op_register(0);
            state.set_xmm(r, u128::from(v)).ok_or_else(|| bad(i))?;
        }
        _ => {
            let a = read_vec(i, 1, next_ip, state, mem)?;
            let v = (a[0] & 0xFFFF_FFFF_FFFF_FFFF) as u64;
            store_gpr_or_mem64(i, 0, next_ip, state, mem, v)?;
        }
    }
    Ok(Action::Next)
}

/// MOVMSKPS/PD/PMOVMSKB: pack sign bits into a GPR.
#[allow(clippy::needless_range_loop)]
pub(crate) fn exec_movmsk(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    let _ = next_ip;
    let a = read_vec(i, 1, next_ip, state, mem)?;
    let mut bits = 0u64;
    match i.mnemonic() {
        Mnemonic::Movmskps | Mnemonic::Vmovmskps => {
            for h in 0..2usize {
                for j in 0..4u32 {
                    bits |= (((a[h] >> (j * 32 + 31)) & 1) as u64) << (h as u32 * 4 + j);
                }
            }
        }
        Mnemonic::Movmskpd | Mnemonic::Vmovmskpd => {
            for h in 0..2usize {
                for j in 0..2u32 {
                    bits |= (((a[h] >> (j * 64 + 63)) & 1) as u64) << (h as u32 * 2 + j);
                }
            }
        }
        _ => {
            // Pmovmskb: byte sign bits.
            for h in 0..2usize {
                for j in 0..8u32 {
                    bits |= (((a[h] >> (j * 8 + 7)) & 1) as u64) << (h as u32 * 8 + j);
                }
            }
        }
    }
    let r = i.op_register(0);
    state.set_reg(r, bits).ok_or_else(|| bad(i))?;
    Ok(Action::Next)
}

/// PEXTRB/W/D/Q and EXTRACTPS: lane → GPR/mem.
pub(crate) fn exec_pextr(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    let imm = immediate_of(i);
    let a = read_vec(i, 1, next_ip, state, mem)?;
    let w = match i.mnemonic() {
        Mnemonic::Pextrb | Mnemonic::Vpextrb => 1,
        Mnemonic::Pextrw | Mnemonic::Vpextrw => 2,
        _ => 4,
    };
    let idx = imm as usize & if w == 1 { 31 } else { 7 };
    let v = extract_lane(a, w, idx);
    match i.op_kind(0) {
        OpKind::Register => {
            let r = i.op_register(0);
            state.set_reg(r, v).ok_or_else(|| bad(i))?;
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            let bytes = (v as u32).to_le_bytes()[..w as usize].to_vec();
            mem.write_buf(addr, &bytes)
                .map_err(AccessError::into_fault)?;
        }
    }
    Ok(Action::Next)
}

/// PINSRB/W/D/Q and INSERTPS: GPR/mem lane → vector.
pub(crate) fn exec_pinsr(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    let imm = immediate_of(i);
    let a = read_vec(i, 0, next_ip, state, mem)?;
    let w: u32 = match i.mnemonic() {
        Mnemonic::Pinsrb | Mnemonic::Vpinsrb => 1,
        Mnemonic::Pinsrw | Mnemonic::Vpinsrw => 2,
        _ => 4,
    };
    let idx = imm as usize & if w == 1 { 15 } else { 7 };
    let v = match i.op_kind(1) {
        OpKind::Register => {
            let r = i.op_register(1);
            state.reg(r).ok_or_else(|| bad(i))?
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            let mut buf = [0u8; 8];
            mem.read_buf(addr, &mut buf[..w as usize])
                .map_err(AccessError::into_fault)?;
            let mut v = [0u8; 8];
            v[..w as usize].copy_from_slice(&buf[..w as usize]);
            u64::from_le_bytes(v) & mask_u64(w)
        }
    };
    let mut r = a;
    let sh = idx * w as usize * 8;
    r[0] = (r[0] & !(mask_lane(w) << sh)) | ((u128::from(v) & mask_lane(w)) << sh);
    write_vec(i, 0, state, r, false)?;
    Ok(Action::Next)
}

/// CVTSS2SI family: lane 0 → GPR (rounded or truncated).
pub(crate) fn exec_cvt_to_gpr(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    let a = read_vec(i, 1, next_ip, state, mem)?;
    let single = matches!(
        i.mnemonic(),
        Mnemonic::Cvtss2si | Mnemonic::Cvttss2si | Mnemonic::Vcvtss2si | Mnemonic::Vcvttss2si
    );
    let (_, dst_w, _) = crate::state::decode_reg(i.op_register(0)).ok_or_else(|| bad(i))?;
    let sixtyfour = dst_w == 8;
    let trunc = matches!(
        i.mnemonic(),
        Mnemonic::Cvttss2si | Mnemonic::Cvttsd2si | Mnemonic::Vcvttss2si | Mnemonic::Vcvttsd2si
    );
    let v: i64 = if single {
        let f = f32::from_bits((a[0] & 0xFFFF_FFFF) as u32);
        if trunc {
            f as i64
        } else {
            match (state.mxcsr >> 13) & 3 {
                0 => f.round_ties_even() as i64,
                1 => f.floor() as i64,
                2 => f.ceil() as i64,
                _ => f.trunc() as i64,
            }
        }
    } else {
        let f = f64::from_bits((a[0] & 0xFFFF_FFFF_FFFF_FFFF) as u64);
        if trunc {
            f as i64
        } else {
            match (state.mxcsr >> 13) & 3 {
                0 => f.round_ties_even() as i64,
                1 => f.floor() as i64,
                2 => f.ceil() as i64,
                _ => f.trunc() as i64,
            }
        }
    };
    let r = i.op_register(0);
    if sixtyfour {
        state.set_reg(r, v as u64).ok_or_else(|| bad(i))?;
    } else {
        state.set_reg(r, (v as u32) as u64).ok_or_else(|| bad(i))?;
    }
    Ok(Action::Next)
}

/// CVTSI2SS/SD: GPR/mem → lane 0 of XMM (rest of dst preserved).
pub(crate) fn exec_cvt_from_gpr(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<Action, Fault> {
    let a = read_vec(i, 0, next_ip, state, mem)?;
    let src = match i.op_kind(1) {
        OpKind::Register => {
            let r = i.op_register(1);
            state.reg(r).ok_or_else(|| bad(i))?
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            let w = i.memory_size().size().min(8);
            let mut buf = [0u8; 8];
            mem.read_buf(addr, &mut buf[..w])
                .map_err(AccessError::into_fault)?;
            let mut v = [0u8; 8];
            v[..w].copy_from_slice(&buf[..w]);
            u64::from_le_bytes(v)
        }
    };
    let single = matches!(i.mnemonic(), Mnemonic::Cvtsi2ss | Mnemonic::Vcvtsi2ss);
    let sixtyfour = i.memory_size().size() == 8;
    let int_val: i64 = if sixtyfour {
        src as i64
    } else {
        (src & 0xFFFF_FFFF) as i32 as i64
    };
    let mut r = a;
    if single {
        let f = int_val as f32;
        r[0] = (a[0] & !0xFFFF_FFFF) | (f.to_bits() as u128);
    } else {
        let f = int_val as f64;
        r[0] = (a[0] & !0xFFFF_FFFF_FFFF_FFFF) | (f.to_bits() as u128);
    }
    write_vec(i, 0, state, r, false)?;
    Ok(Action::Next)
}

// ---------------------------------------------------------------------------
// gathers
// ---------------------------------------------------------------------------

/// VGATHER families: per-lane indexed loads with a mask register that is
/// zeroed on completion. The vindex vector is the ModRM index register of
/// the VSIB memory operand; the base/displacement come from the same operand.
fn exec_gathers(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    _a: V256,
    _b: V256,
    wide: bool,
) -> Result<Option<V256>, Fault> {
    use Mnemonic::*;
    let m = i.mnemonic();
    let (idx_lane, data_lane) = match m {
        Vpgatherdd | Vgatherdps => (4, 4),
        Vpgatherdq | Vgatherdpd => (4, 8),
        Vpgatherqd | Vgatherqps => (8, 4),
        Vpgatherqq | Vgatherqpd => (8, 8),
        _ => return Ok(None),
    };
    let dst = read_vec(i, 0, next_ip, state, mem)?;
    let vindex = state.ymm_halves(i.memory_index()).ok_or_else(|| bad(i))?;
    let mask = read_vec(i, 2, next_ip, state, mem)?;
    let scale = i.memory_index_scale() as u64;
    let disp = i.memory_displacement64();
    let base = if i.memory_base() != Register::None {
        state.reg(i.memory_base()).expect("GPR base")
    } else {
        0
    };
    let mut r = dst;
    let lanes = if wide { 32 / data_lane } else { 16 / data_lane };
    for l in 0..lanes {
        let dh = (l * data_lane / 16) as usize;
        let dj = ((l * data_lane) % 16) / data_lane;
        let mask_bit = (mask[dh] >> (dj * data_lane * 8 + data_lane * 8 - 1)) & 1;
        if mask_bit == 0 {
            continue; // dst lane keeps its original value
        }
        let ih = (l * idx_lane / 16) as usize;
        let ij = ((l * idx_lane) % 16) / idx_lane;
        let idx = ((vindex[ih] >> (ij * idx_lane * 8)) & mask_lane(idx_lane)) as u64;
        let addr = base
            .wrapping_add(idx.wrapping_mul(scale))
            .wrapping_add(disp);
        let mut buf = [0u8; 8];
        mem.read_buf(addr, &mut buf[..data_lane as usize])
            .map_err(AccessError::into_fault)?;
        let mut v = [0u8; 8];
        v[..data_lane as usize].copy_from_slice(&buf[..data_lane as usize]);
        let val = u64::from_le_bytes(v);
        let sh = dj * data_lane * 8;
        r[dh] = (r[dh] & !(mask_lane(data_lane) << sh)) | ((val as u128) << sh);
    }
    // The mask operand is zeroed architecturally.
    let mask_reg = i.op_register(2);
    state
        .set_ymm_halves(mask_reg, [0; 2])
        .ok_or_else(|| bad(i))?;
    Ok(Some(r))
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn bad(i: &Instruction) -> Fault {
    Fault::InvalidOpcode {
        mnemonic: format!("{:?}", i.mnemonic()),
        rip: i.ip(),
    }
}

/// COMISs flag pattern from (unordered, greater, less).
fn comis_flags(unordered: bool, greater: bool, less: bool) -> (bool, bool, bool) {
    if unordered {
        (true, true, true)
    } else if greater {
        (false, false, false)
    } else if less {
        (false, false, true)
    } else {
        (true, false, false)
    }
}

fn extract_lane(a: V256, w: u32, idx: usize) -> u64 {
    let h = idx * w as usize / 16;
    let j = (idx * w as usize % 16) / w as usize;
    ((a[h] >> (j * w as usize * 8)) & mask_lane(w)) as u64
}

fn gpr_or_mem32(
    i: &Instruction,
    n: u32,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<u32, Fault> {
    match i.op_kind(n) {
        OpKind::Register => {
            let r = i.op_register(n);
            Ok(state.reg(r).ok_or_else(|| bad(i))? as u32)
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            let mut buf = [0u8; 4];
            mem.read_buf(addr, &mut buf)
                .map_err(AccessError::into_fault)?;
            Ok(u32::from_le_bytes(buf))
        }
    }
}

fn gpr_or_mem64(
    i: &Instruction,
    n: u32,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<u64, Fault> {
    match i.op_kind(n) {
        OpKind::Register => {
            let r = i.op_register(n);
            Ok(state.reg(r).ok_or_else(|| bad(i))?)
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            let mut buf = [0u8; 8];
            mem.read_buf(addr, &mut buf)
                .map_err(AccessError::into_fault)?;
            Ok(u64::from_le_bytes(buf))
        }
    }
}

fn store_gpr_or_mem32(
    i: &Instruction,
    n: u32,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    v: u32,
) -> Result<(), Fault> {
    match i.op_kind(n) {
        OpKind::Register => {
            let r = i.op_register(n);
            state.set_reg(r, u64::from(v)).ok_or_else(|| bad(i))?;
            Ok(())
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            mem.write_buf(addr, &v.to_le_bytes())
                .map_err(AccessError::into_fault)
        }
    }
}

fn store_gpr_or_mem64(
    i: &Instruction,
    n: u32,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    v: u64,
) -> Result<(), Fault> {
    match i.op_kind(n) {
        OpKind::Register => {
            let r = i.op_register(n);
            state.set_reg(r, v).ok_or_else(|| bad(i))?;
            Ok(())
        }
        _ => {
            let addr = effective_address(i, next_ip, state);
            mem.write_buf(addr, &v.to_le_bytes())
                .map_err(AccessError::into_fault)
        }
    }
}
