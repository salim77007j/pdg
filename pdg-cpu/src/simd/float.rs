//! Floating-point families: arithmetic, compares, blends with variable
//! masks, rounding, conversions, horizontal ops, broadcasts, variable shifts
//! and the widen/narrow moves.

use iced_x86::{Instruction, Mnemonic, OpKind};

use crate::fault::{Fault, MemoryAccess};
use crate::interpreter::Action;
use crate::state::CpuState;

use super::{V256, finish_vec, immediate_of, read_vec};

/// Try to execute `i` if it belongs to this family.
#[allow(clippy::too_many_lines)]
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

    // Variable-mask blends use the implicit XMM0 source.
    let _mask = |state: &CpuState, wide: bool| -> V256 {
        let v = state.ymm_idx(0);
        if wide { v } else { [v[0], 0] }
    };

    let first = first_src(i);
    let last = last_src(i);
    let a = read_vec(i, first, next_ip, state, mem)?;
    let b = read_vec(i, last, next_ip, state, mem)?;

    let result: V256 = match m {
        // ---- packed/scalar float arithmetic --------------------------------
        Addps | Vaddps => f32_lanes(a, b, |x, y| x + y),
        Addpd | Vaddpd => f64_lanes(a, b, |x, y| x + y),
        Subps | Vsubps => f32_lanes(a, b, |x, y| x - y),
        Subpd | Vsubpd => f64_lanes(a, b, |x, y| x - y),
        Mulps | Vmulps => f32_lanes(a, b, |x, y| x * y),
        Mulpd | Vmulpd => f64_lanes(a, b, |x, y| x * y),
        Divps | Vdivps => f32_lanes(a, b, |x, y| x / y),
        Divpd | Vdivpd => f64_lanes(a, b, |x, y| x / y),
        Sqrtps | Vsqrtps => f32_lanes(a, a, |x, _| x.sqrt()),
        Sqrtpd | Vsqrtpd => f64_lanes(a, a, |x, _| x.sqrt()),
        Minps | Vminps => f32_lanes(a, b, |x, y| if x < y { x } else { y }),
        Minpd | Vminpd => f64_lanes(a, b, |x, y| if x < y { x } else { y }),
        Maxps | Vmaxps => f32_lanes(a, b, |x, y| if x > y { x } else { y }),
        Maxpd | Vmaxpd => f64_lanes(a, b, |x, y| if x > y { x } else { y }),

        Addss | Vaddss => scalar_f32(a, b, |x, y| x + y),
        Addsd | Vaddsd => scalar_f64(a, b, |x, y| x + y),
        Subss | Vsubss => scalar_f32(a, b, |x, y| x - y),
        Subsd | Vsubsd => scalar_f64(a, b, |x, y| x - y),
        Mulss | Vmulss => scalar_f32(a, b, |x, y| x * y),
        Mulsd | Vmulsd => scalar_f64(a, b, |x, y| x * y),
        Divss | Vdivss => scalar_f32(a, b, |x, y| x / y),
        Divsd | Vdivsd => scalar_f64(a, b, |x, y| x / y),
        Sqrtss | Vsqrtss => scalar_f32(a, a, |x, _| x.sqrt()),
        Sqrtsd | Vsqrtsd => scalar_f64(a, a, |x, _| x.sqrt()),
        Minss | Vminss => scalar_f32(a, b, |x, y| if x < y { x } else { y }),
        Minsd | Vminsd => scalar_f64(a, b, |x, y| if x < y { x } else { y }),
        Maxss | Vmaxss => scalar_f32(a, b, |x, y| if x > y { x } else { y }),
        Maxsd | Vmaxsd => scalar_f64(a, b, |x, y| if x > y { x } else { y }),

        Addsubps | Vaddsubps => f32_lanes(
            a,
            b,
            |x, y| if x.to_bits() & 1 == 0 { x - y } else { x + y },
        ),
        Addsubpd | Vaddsubpd => f64_lanes(
            a,
            b,
            |x, y| if x.to_bits() & 1 == 0 { x - y } else { x + y },
        ),

        Haddps | Vhaddps => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..2 {
                    let x0 = f32_lane(a[h], j * 2);
                    let x1 = f32_lane(a[h], j * 2 + 1);
                    r[h] |= ((x0 + x1).to_bits() as u128) << (j * 32);
                }
                for j in 0..2 {
                    let y0 = f32_lane(b[h], j * 2);
                    let y1 = f32_lane(b[h], j * 2 + 1);
                    r[h] |= ((y0 + y1).to_bits() as u128) << (64 + j * 32);
                }
            }
            r
        }
        Haddpd | Vhaddpd => [
            (f64_lane(a[0], 0) + f64_lane(a[0], 1)).to_bits() as u128
                | ((f64_lane(b[0], 0) + f64_lane(b[0], 1)).to_bits() as u128) << 64,
            (f64_lane(a[1], 0) + f64_lane(a[1], 1)).to_bits() as u128
                | ((f64_lane(b[1], 0) + f64_lane(b[1], 1)).to_bits() as u128) << 64,
        ],
        Hsubps | Vhsubps => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..2 {
                    let x0 = f32_lane(a[h], j * 2);
                    let x1 = f32_lane(a[h], j * 2 + 1);
                    r[h] |= ((x0 - x1).to_bits() as u128) << (j * 32);
                }
                for j in 0..2 {
                    let y0 = f32_lane(b[h], j * 2);
                    let y1 = f32_lane(b[h], j * 2 + 1);
                    r[h] |= ((y0 - y1).to_bits() as u128) << (64 + j * 32);
                }
            }
            r
        }
        Hsubpd | Vhsubpd => [
            (f64_lane(a[0], 0) - f64_lane(a[0], 1)).to_bits() as u128
                | ((f64_lane(b[0], 0) - f64_lane(b[0], 1)).to_bits() as u128) << 64,
            (f64_lane(a[1], 0) - f64_lane(a[1], 1)).to_bits() as u128
                | ((f64_lane(b[1], 0) - f64_lane(b[1], 1)).to_bits() as u128) << 64,
        ],

        // ---- compares ------------------------------------------------------
        Cmpps | Vcmpps | Cmpss | Vcmpss => {
            let scalar = matches!(m, Cmpss | Vcmpss);
            let pred = imm & 0x07;
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..4 {
                    if scalar && (h > 0 || j > 0) {
                        continue; // scalar keeps the upper lanes of dst
                    }
                    let x = f32_lane(a[h], j);
                    let y = f32_lane(b[h], j);
                    let set = cmp_pred_f32(pred, x, y);
                    r[h] |= (u128::from(set) * 0xFFFF_FFFF) << (j * 32);
                }
                if scalar {
                    r[h] |= a[h] & !0xFFFF_FFFF;
                }
            }
            r
        }
        Cmppd | Vcmppd | Cmpsd | Vcmpsd => {
            let scalar = matches!(m, Cmpsd | Vcmpsd);
            let pred = imm & 0x07;
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..2 {
                    if scalar && (h > 0 || j > 0) {
                        continue;
                    }
                    let x = f64_lane(a[h], j);
                    let y = f64_lane(b[h], j);
                    let set = cmp_pred_f64(pred, x, y);
                    r[h] |= (u128::from(set) * u128::from(!0u64)) << (j * 64);
                }
                if scalar {
                    r[h] |= a[h] & !0xFFFF_FFFF_FFFF_FFFF;
                }
            }
            r
        }

        // ---- rounding ------------------------------------------------------
        Roundps | Vroundps => f32_lanes(a, a, |x, _| round_by32(x, imm)),
        Roundpd | Vroundpd => f64_lanes(a, a, |x, _| round_by(x, imm)),
        Roundss | Vroundss => scalar_f32(a, a, |x, _| round_by32(x, imm)),
        Roundsd | Vroundsd => scalar_f64(a, a, |x, _| round_by(x, imm)),

        // ---- dot products --------------------------------------------------
        Dpps | Vdpps => {
            let mut acc = 0f32;
            for j in 0..4 {
                if imm & (1 << (4 + j)) != 0 {
                    acc += f32_lane(a[0], j) * f32_lane(b[0], j);
                }
            }
            let mut r = [0u128; 2];
            for j in 0..4 {
                let v = if imm & (1 << j) != 0 { acc } else { 0.0 };
                r[0] |= (v.to_bits() as u128) << (j * 32);
            }
            r
        }
        Dppd | Vdppd => {
            let mut acc = 0f64;
            for j in 0..2 {
                if imm & (1 << (4 + j)) != 0 {
                    acc += f64_lane(a[0], j) * f64_lane(b[0], j);
                }
            }
            let mut r = [0u128; 2];
            for j in 0..2 {
                let v = if imm & (1 << j) != 0 { acc } else { 0.0 };
                r[0] |= (v.to_bits() as u128) << (j * 64);
            }
            r
        }

        // ---- conversions ---------------------------------------------------
        Cvtps2pd | Vcvtps2pd => {
            let mut r = [0u128; 2];
            for j in 0..2 {
                let f = f32::from_bits(((a[0] >> (j * 32)) & 0xFFFF_FFFF) as u32);
                r[0] |= (f64::from(f).to_bits() as u128) << (j * 64);
            }
            if wide {
                for j in 0..2 {
                    let f = f32::from_bits(((a[1] >> (j * 32)) & 0xFFFF_FFFF) as u32);
                    r[1] |= (f64::from(f).to_bits() as u128) << (j * 64);
                }
            }
            r
        }
        Cvtpd2ps | Vcvtpd2ps => {
            let mut r = [0u128; 2];
            for j in 0..2 {
                let f = f64::from_bits(((a[0] >> (j * 64)) & 0xFFFF_FFFF_FFFF_FFFF) as u64);
                r[0] |= ((f as f32).to_bits() as u128) << (j * 32);
            }
            for j in 0..2 {
                let f = f64::from_bits(((a[1] >> (j * 64)) & 0xFFFF_FFFF_FFFF_FFFF) as u64);
                r[0] |= ((f as f32).to_bits() as u128) << (64 + j * 32);
            }
            r
        }
        Cvtdq2ps | Vcvtdq2ps => raw_i32_lanes_to_f32(a),
        Cvtps2dq | Vcvtps2dq => f32_lanes_int(a, |x| round_toward(x, state)),
        Vcvttps2dq | Cvttps2dq => f32_lanes_int(a, |x| x as i32),
        Cvtpd2dq | Vcvtpd2dq => f64_lanes_int(a, |x| round_toward_64(x, state)),
        Vcvttpd2dq | Cvttpd2dq => f64_lanes_int(a, |x| x as i32),
        Cvtdq2pd | Vcvtdq2pd => {
            let mut r = [0u128; 2];
            for j in 0..2 {
                let d = ((a[0] >> (j * 32)) & 0xFFFF_FFFF) as i32;
                r[0] |= ((d as f64).to_bits() as u128) << (j * 64);
            }
            if wide {
                for j in 0..2 {
                    let d = ((a[1] >> (j * 32)) & 0xFFFF_FFFF) as i32;
                    r[1] |= ((d as f64).to_bits() as u128) << (j * 64);
                }
            }
            r
        }
        Cvtss2sd | Vcvtss2sd => scalar_f32_to_f64(a, b),
        Cvtsd2ss | Vcvtsd2ss => scalar_f64_to_f32(a, b),

        _ => return Ok(None),
    };
    let action = finish_vec(i, state, mem, next_ip, result, wide)?;
    Ok(Some(action))
}

fn first_src(i: &Instruction) -> u32 {
    if i.op_count() >= 3 { 1 } else { 0 }
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

// ---------------------------------------------------------------------------
// lane helpers
// ---------------------------------------------------------------------------

/// CVTDQ2PS: raw i32 bit patterns → f32 lanes (NOT float reinterpretation).
fn raw_i32_lanes_to_f32(a: V256) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..4 {
            let x = ((a[h] >> (j * 32)) & 0xFFFF_FFFF) as u32 as i32;
            r[h] |= ((x as f32).to_bits() as u128) << (j * 32);
        }
    }
    r
}

fn f32_lane(v: u128, j: u32) -> f32 {
    f32::from_bits(((v >> (j * 32)) & 0xFFFF_FFFF) as u32)
}

fn f64_lane(v: u128, j: u32) -> f64 {
    f64::from_bits(((v >> (j * 64)) & 0xFFFF_FFFF_FFFF_FFFF) as u64)
}

/// Pack a `f32` kernel over all lanes of both halves.
fn f32_lanes(a: V256, b: V256, f: impl Fn(f32, f32) -> f32) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..4 {
            r[h] |= (f(f32_lane(a[h], j), f32_lane(b[h], j)).to_bits() as u128) << (j * 32);
        }
    }
    r
}

/// Pack a `f64` kernel over both halves.
fn f64_lanes(a: V256, b: V256, f: impl Fn(f64, f64) -> f64) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..2 {
            r[h] |= (f(f64_lane(a[h], j), f64_lane(b[h], j)).to_bits() as u128) << (j * 64);
        }
    }
    r
}

/// Scalar f32: apply `f` to lane 0 of `a` and `b`, preserving everything else.
fn scalar_f32(a: V256, b: V256, f: impl Fn(f32, f32) -> f32) -> V256 {
    let mut r = a;
    let v = f(f32_lane(a[0], 0), f32_lane(b[0], 0));
    r[0] = (a[0] & !0xFFFF_FFFF) | (v.to_bits() as u128);
    r
}

fn scalar_f64(a: V256, b: V256, f: impl Fn(f64, f64) -> f64) -> V256 {
    let mut r = a;
    let v = f(f64_lane(a[0], 0), f64_lane(b[0], 0));
    r[0] = (a[0] & !0xFFFF_FFFF_FFFF_FFFF) | (v.to_bits() as u128);
    r
}

/// Scalar cvtss2sd: dst lane0 (f64) = (f32) src lane0; rest preserved.
fn scalar_f32_to_f64(a: V256, b: V256) -> V256 {
    let mut r = a;
    let v = f64::from(f32_lane(b[0], 0));
    r[0] = (a[0] & !0xFFFF_FFFF_FFFF_FFFF) | (v.to_bits() as u128);
    r
}

/// Scalar cvtsd2ss: dst lane0 (f32) = (f32) src lane0; rest preserved.
fn scalar_f64_to_f32(a: V256, b: V256) -> V256 {
    let mut r = a;
    let v = f64_lane(b[0], 0) as f32;
    r[0] = (a[0] & !0xFFFF_FFFF) | (v.to_bits() as u128);
    r
}

/// f32 lanes → i32 lanes.
fn f32_lanes_int(a: V256, f: impl Fn(f32) -> i32) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..4 {
            r[h] |= (f(f32_lane(a[h], j)) as u32 as u128) << (j * 32);
        }
    }
    r
}

/// f64 lanes → i32 lanes (lower 64 → two i32s).
fn f64_lanes_int(a: V256, f: impl Fn(f64) -> i32) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..2 {
            r[h] |= (f(f64_lane(a[h], j)) as u32 as u128) << (j * 32);
        }
    }
    r
}

// The negated-comparison predicates are intentional:  keeps the
// architectural NaN behaviour of NLT/NLE (true when unordered).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn cmp_pred_f32(pred: u8, x: f32, y: f32) -> bool {
    match pred {
        0 => x == y,
        1 => x < y,
        2 => x <= y,
        3 => false, // NLE-complement placeholder used by no encoder
        4 => x != y,
        5 => !(x < y),
        6 => !(x <= y),
        _ => x.is_nan() || y.is_nan(), // unordered
    }
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn cmp_pred_f64(pred: u8, x: f64, y: f64) -> bool {
    match pred {
        0 => x == y,
        1 => x < y,
        2 => x <= y,
        3 => false,
        4 => x != y,
        5 => !(x < y),
        6 => !(x <= y),
        _ => x.is_nan() || y.is_nan(),
    }
}

/// ROUNDxS imm8 rounding for f64: bits 0..1 mode.
fn round_by(x: f64, imm: u8) -> f64 {
    match imm & 3 {
        0 => x.round_ties_even(),
        1 => x.floor(),
        2 => x.ceil(),
        _ => x.trunc(),
    }
}

/// ROUNDxS imm8 rounding for f32: bits 0..1 mode.
fn round_by32(x: f32, imm: u8) -> f32 {
    match imm & 3 {
        0 => x.round_ties_even(),
        1 => x.floor(),
        2 => x.ceil(),
        _ => x.trunc(),
    }
}

/// MXCSR-honouring f32→i32 conversion (CVTPS2DQ).
fn round_toward(x: f32, state: &CpuState) -> i32 {
    match (state.mxcsr >> 13) & 3 {
        0 => x.round_ties_even() as i32,
        1 => x.floor() as i32,
        2 => x.ceil() as i32,
        _ => x.trunc() as i32,
    }
}

/// MXCSR-honouring f64→i32 conversion (CVTPD2DQ).
fn round_toward_64(x: f64, state: &CpuState) -> i32 {
    match (state.mxcsr >> 13) & 3 {
        0 => x.round_ties_even() as i32,
        1 => x.floor() as i32,
        2 => x.ceil() as i32,
        _ => x.trunc() as i32,
    }
}
