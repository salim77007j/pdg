//! Shuffle, blend, pack, shift and integer-reorganisation families.

use iced_x86::{Instruction, Mnemonic, OpKind};

use crate::fault::{Fault, MemoryAccess};
use crate::interpreter::Action;
use crate::state::CpuState;

use super::{V256, finish_vec, immediate_of, read_vec};

/// Try to execute `i` if it belongs to this family.
pub(crate) fn exec(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    _a: V256,
    _b: V256,
    wide: bool,
) -> Result<Option<Action>, Fault> {
    use Mnemonic::*;
    let m = i.mnemonic();
    let imm = immediate_of(i);
    // Source operands: (a, b) with a = first source (dst for legacy).
    let a = read_vec(i, first_src(i), next_ip, state, mem)?;
    let b = read_vec(i, last_src(i), next_ip, state, mem)?;

    let result: V256 = match m {
        Shufps | Vshufps => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..2 {
                    let sel = ((imm >> (j * 2)) & 3) as u32;
                    r[h] |= ((a[h] >> (sel * 32)) & 0xFFFF_FFFF) << (j * 32);
                }
                for j in 0..2 {
                    let sel = ((imm >> (4 + j * 2)) & 3) as u32;
                    r[h] |= ((b[h] >> (sel * 32)) & 0xFFFF_FFFF) << (64 + j * 32);
                }
            }
            r
        }
        Shufpd | Vshufpd => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                let sa = ((imm >> h) & 1) as u32;
                let sb = ((imm >> (h + 2)) & 1) as u32;
                r[h] = (a[h] & 0xFFFF_FFFF_FFFF_FFFF)
                    | (((b[h] >> (sb * 64)) & 0xFFFF_FFFF_FFFF_FFFF) << 64);
                if sa == 1 {
                    r[h] = (r[h] & !0xFFFF_FFFF_FFFF_FFFF) | ((a[h] >> 64) & 0xFFFF_FFFF_FFFF_FFFF);
                }
            }
            r
        }
        Pshufd | Vpshufd => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..4 {
                    let sel = ((imm >> (j * 2)) & 3) as u32;
                    r[h] |= ((a[h] >> (sel * 32)) & 0xFFFF_FFFF) << (j * 32);
                }
            }
            r
        }
        Pshuflw | Vpshuflw => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..4 {
                    let sel = ((imm >> (j * 2)) & 3) as u32;
                    r[h] |= ((a[h] >> (sel * 16)) & 0xFFFF) << (j * 16);
                }
                r[h] |= a[h] & 0xFFFF_FFFF_0000_0000;
            }
            r
        }
        Pshufhw | Vpshufhw => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                r[h] = a[h] & 0xFFFF_FFFF;
                for j in 0..4 {
                    let sel = ((imm >> (j * 2)) & 3) as u32;
                    r[h] |= ((a[h] >> (64 + sel * 16)) & 0xFFFF) << (64 + j * 16);
                }
            }
            r
        }
        Pshufb | Vpshufb => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..16 {
                    let sel = ((b[h] >> (j * 8)) & 0x7F) as u32;
                    let byte = if (b[h] >> (j * 8)) & 0x80 != 0 {
                        0
                    } else {
                        (a[h] >> (sel * 8)) & 0xFF
                    };
                    r[h] |= byte << (j * 8);
                }
            }
            r
        }
        Palignr | Vpalignr => {
            let mut r = [0u128; 2];
            let c = imm as u32;
            for h in 0..2 {
                if c < 16 {
                    r[h] = (a[h] << (c * 8)) | if c == 0 { 0 } else { b[h] >> ((16 - c) * 8) };
                } else if c < 32 {
                    r[h] = b[h] << ((c - 16) * 8);
                }
                // >= 32: zero.
            }
            r
        }
        Punpcklbw | Vpunpcklbw => unpack(a, b, 1, 8),
        Punpcklwd | Vpunpcklwd => unpack(a, b, 2, 4),
        Punpckldq | Vpunpckldq => unpack(a, b, 4, 2),
        Punpcklqdq | Vpunpcklqdq => unpack(a, b, 8, 1),
        Punpckhbw | Vpunpckhbw => unpack_hi(a, b, 1, 8),
        Punpckhwd | Vpunpckhwd => unpack_hi(a, b, 2, 4),
        Punpckhdq | Vpunpckhdq => unpack_hi(a, b, 4, 2),
        Punpckhqdq | Vpunpckhqdq => unpack_hi(a, b, 8, 1),

        Packsswb | Vpacksswb => pack(a, b, |x, y| {
            let xo = (x as i16).clamp(i8::MIN as i16, i8::MAX as i16) as u8 as u64;
            let yo = (y as i16).clamp(i8::MIN as i16, i8::MAX as i16) as u8 as u64;
            xo | (yo << 8)
        }),
        Packuswb | Vpackuswb => pack(a, b, |x, y| x.clamp(0, 255) | y.clamp(0, 255) << 8),
        Packssdw | Vpackssdw => pack32(a, b, |x, y| {
            let xo = (x as i32).clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u32;
            let yo = (y as i32).clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u32;
            (xo | (yo << 16)) as u64
        }),
        Packusdw | Vpackusdw => pack32(a, b, |x, y| x.clamp(0, 65535) | (y.clamp(0, 65535) << 16)),

        Pblendw | Vpblendw => blend_lanes(a, b, 2, imm),
        Vpblendd => blend_lanes(a, b, 4, imm),
        Blendps | Vblendps => blend_lanes(a, b, 4, imm),
        Blendpd | Vblendpd => blend_lanes(a, b, 8, imm),

        _ => return Ok(None),
    };
    let action = finish_vec(i, state, mem, next_ip, result, wide)?;
    Ok(Some(action))
}

/// First source operand index for the shuffle family (legacy: 0; VEX: 1).
fn first_src(i: &Instruction) -> u32 {
    if i.op_count() >= 3 { 1 } else { 0 }
}

/// Last source operand index, skipping a trailing immediate.
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

/// PUNPCKL shape: interleave the low elements of each 128-bit half.
fn unpack(a: V256, b: V256, lane: u32, n: u32) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        let mut out = 0u128;
        for j in 0..n {
            let sh = j * lane * 8;
            let x = (a[h] >> sh) & mask128(lane);
            let y = (b[h] >> sh) & mask128(lane);
            out |= x << (2 * j * lane * 8);
            out |= y << (2 * j * lane * 8 + lane * 8);
        }
        r[h] = out;
    }
    r
}

/// PUNPCKH shape: interleave the high elements of each 128-bit half.
fn unpack_hi(a: V256, b: V256, lane: u32, n: u32) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        let mut out = 0u128;
        for j in 0..n {
            let sh = 128 - (j + 1) * lane * 8;
            let x = (a[h] >> sh) & mask128(lane);
            let y = (b[h] >> sh) & mask128(lane);
            out |= x << (2 * (n - 1 - j) * lane * 8);
            out |= y << (2 * (n - 1 - j) * lane * 8 + lane * 8);
        }
        r[h] = out;
    }
    r
}

const fn mask128(lane: u32) -> u128 {
    (1u128 << (lane * 8)) - 1
}

/// PACK shape: combine saturated 16→8 lanes of a and b into one result.
fn pack(a: V256, b: V256, f: impl Fn(u64, u64) -> u64) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..8 {
            let x = ((a[h] >> (j * 16)) & 0xFFFF) as u64;
            let y = ((b[h] >> (j * 16)) & 0xFFFF) as u64;
            r[h] |= ((f(x, y) & 0xFFFF) as u128) << (j * 16);
        }
    }
    r
}

/// PACK shape for 32→16 lanes.
fn pack32(a: V256, b: V256, f: impl Fn(u64, u64) -> u64) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..4 {
            let x = ((a[h] >> (j * 32)) & 0xFFFF_FFFF) as u64;
            let y = ((b[h] >> (j * 32)) & 0xFFFF_FFFF) as u64;
            r[h] |= ((f(x, y) & 0xFFFF_FFFF) as u128) << (j * 32);
        }
    }
    r
}

/// Imm-masked per-lane blend at `lane` bytes.
fn blend_lanes(a: V256, b: V256, lane: u32, imm: u8) -> V256 {
    let n = 16 / lane;
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..n {
            let bit = if h == 0 {
                imm >> j & 1
            } else {
                imm >> (n + j) & 1
            };
            let src = if bit == 1 { b } else { a };
            r[h] |= src[h] & (mask128(lane) << (j * lane * 8));
        }
    }
    r
}
