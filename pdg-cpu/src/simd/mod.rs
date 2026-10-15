//! SIMD execution: the SSE → AVX2 vector families.
//!
//! This module owns everything vector: the XMM/YMM register views on
//! [`CpuState`], operand plumbing for register/memory/immediate vector
//! operands, and the lane-wise kernels for the implemented instruction
//! families (see `pdg-cpu/src/cpuid.rs` for the exact advertised set).
//!
//! Semantics notes:
//! * Legacy-SSE writes to XMM zero the upper 128 bits (AVX-capable CPU
//!   rule); VEX.128 writes zero it too; VEX.256 writes the full register.
//! * Scalar ops (`addss` etc.) operate on lane 0 only and preserve the rest,
//!   except loads that are architecturally zeroing (`movss` from memory).
//! * Floating point follows IEEE-754 via Rust's `f32`/`f64`; MXCSR rounding
//!   mode is honoured by the float→int conversions; untrapped exception
//!   *flags* are not raised (documented simplification).

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use crate::{
    fault::{AccessError, Fault, MemoryAccess},
    state::CpuState,
};

mod bridge;
mod crypto;
mod float;
mod shuffle;
mod super_dispatch;

/// A 256-bit vector value: `[low 128 bits, high 128 bits]`.
pub type V256 = [u128; 2];

/// True when the instruction is one of the vector families this module owns.
pub fn is_vector_mnemonic(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        // moves
        Movaps | Movups | Movapd | Movupd | Movdqa | Movdqu
            | Movd | Movq
            | Movhpd | Movlpd | Movhps | Movlps | Movlhps | Movhlps
            | Movss | Movsd
            | Movmskps | Movmskpd
            | Vmovaps | Vmovups | Vmovapd | Vmovupd | Vmovdqu | Vmovdqa
            | Vmovd | Vmovq
            | Vmovss | Vmovsd | Vmovhpd | Vmovlpd | Vmovhps | Vmovlps
            | Vmovmskps | Vmovmskpd | Vpmovmskb
            | Vzeroupper | Vzeroall
            | Vbroadcastss | Vbroadcastsd | Vbroadcastf128
            | Vpbroadcastb | Vpbroadcastw | Vpbroadcastd | Vpbroadcastq
            | Vbroadcasti128
            | Vinsertf128 | Vinserti128 | Vextractf128 | Vextracti128
            | Vperm2f128 | Vperm2i128 | Vpermq | Vpermd | Vpermps | Vpermpd
            | Vpshufb | Vpshufd | Vpshuflw | Vpshufhw
            | Vshufps | Vshufpd
        // integer arithmetic
            | Paddb | Paddw | Paddd | Paddq
            | Psubb | Psubw | Psubd | Psubq
            | Paddsb | Paddsw | Paddusb | Paddusw
            | Psubsb | Psubsw | Psubusb | Psubusw
            | Vpaddb | Vpaddw | Vpaddd | Vpaddq
            | Vpsubb | Vpsubw | Vpsubd | Vpsubq
            | Vpaddsb | Vpaddsw | Vpaddusb | Vpaddusw
            | Vpsubsb | Vpsubsw | Vpsubusb | Vpsubusw
            | Pmulhw | Pmulhuw | Pmullw | Pmulld | Pmuludq | Pmuldq | Pmaddwd | Pmaddubsw
            | Pmulhrsw
            | Vpmulhw | Vpmulhuw | Vpmullw | Vpmulld | Vpmuludq | Vpmuldq
            | Vpmaddwd | Vpmaddubsw | Vpmulhrsw
        // logic
            | Pand | Por | Pxor | Pandn
            | Vandps | Vandpd | Vorps | Vorpd | Vxorps | Vxorpd | Vandnps | Vandnpd
            | Vpand | Vpor | Vpxor | Vpandn
        // comparison
            | Pcmpeqb | Pcmpeqw | Pcmpeqd | Pcmpeqq
            | Pcmpgtb | Pcmpgtw | Pcmpgtd | Pcmpgtq
            | Ptest
            | Vpcmpeqb | Vpcmpeqw | Vpcmpeqd | Vpcmpeqq
            | Vpcmpgtb | Vpcmpgtw | Vpcmpgtd | Vpcmpgtq
            | Vptest
            | Comiss | Comisd | Ucomiss | Ucomisd
        // min/max/avg
            | Pmaxsb | Pmaxsw | Pmaxsd | Pmaxub | Pmaxuw | Pmaxud
            | Pminsb | Pminsw | Pminsd | Pminub | Pminuw | Pminud
            | Vpmaxsb | Vpmaxsw | Vpmaxsd | Vpmaxub | Vpmaxuw | Vpmaxud
            | Vpminsb | Vpminsw | Vpminsd | Vpminub | Vpminuw | Vpminud
            | Pavgb | Pavgw | Vpavgb | Vpavgw
        // shuffles / packs
            | Punpcklbw | Punpcklwd | Punpckldq | Punpcklqdq
            | Punpckhbw | Punpckhwd | Punpckhdq | Punpckhqdq
            | Vpunpcklbw | Vpunpcklwd | Vpunpckldq | Vpunpcklqdq
            | Vpunpckhbw | Vpunpckhwd | Vpunpckhdq | Vpunpckhqdq
            | Packsswb | Packssdw | Packuswb | Packusdw
            | Vpacksswb | Vpackssdw | Vpackuswb | Vpackusdw
            | Pshufb | Pshufd | Pshuflw | Pshufhw
            | Palignr | Vpalignr
            | Vpblendw | Vblendps | Vblendpd | Vpblendd
            | Pblendw | Blendps | Blendpd | Blendvps | Blendvpd | Pblendvb
        // shifts
            | Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad
            | Vpsllw | Vpslld | Vpsllq | Vpsrlw | Vpsrld | Vpsrlq | Vpsraw | Vpsrad
            | Vpsllvd | Vpsllvq | Vpsrlvd | Vpsrlvq | Vpsravd
        // extract/insert/widen-narrow
            | Pextrb | Pextrw | Pextrd | Pextrq | Extractps
            | Vpextrb | Vpextrw | Vpextrd | Vpextrq | Vextractps
            | Pinsrb | Pinsrw | Pinsrd | Pinsrq | Insertps
            | Vpinsrb | Vpinsrw | Vpinsrd | Vpinsrq | Vinsertps
            | Pmovzxbw | Pmovzxwd | Pmovzxdq | Pmovzxbq | Pmovzxwq
            | Pmovsxbw | Pmovsxwd | Pmovsxdq | Pmovsxbq | Pmovsxwq
            | Vpmovzxbw | Vpmovzxwd | Vpmovzxdq | Vpmovsxbw | Vpmovsxwd | Vpmovsxdq
        // SSSE3 extras
            | Pabsb | Pabsw | Pabsd
            | Psignb | Psignw | Psignd
            | Phaddw | Phaddd | Phsubw | Phsubd | Phaddsw | Phsubsw
            | Vphaddw | Vphaddd | Vphsubw | Vphsubd
            | Psadbw | Vpsadbw
        // float arithmetic
            | Addps | Addpd | Addss | Addsd
            | Subps | Subpd | Subss | Subsd
            | Mulps | Mulpd | Mulss | Mulsd
            | Divps | Divpd | Divss | Divsd
            | Sqrtps | Sqrtpd | Sqrtss | Sqrtsd
            | Minps | Minpd | Minss | Minsd
            | Maxps | Maxpd | Maxss | Maxsd
            | Addsubps | Addsubpd | Haddps | Haddpd | Hsubps | Hsubpd
            | Vaddps | Vaddpd | Vaddss | Vaddsd
            | Vsubps | Vsubpd | Vsubss | Vsubsd
            | Vmulps | Vmulpd | Vmulss | Vmulsd
            | Vdivps | Vdivpd | Vdivss | Vdivsd
            | Vsqrtps | Vsqrtpd | Vsqrtss | Vsqrtsd
            | Vminps | Vminpd | Vminss | Vminsd
            | Vmaxps | Vmaxpd | Vmaxss | Vmaxsd
            | Vaddsubps | Vaddsubpd | Vhaddps | Vhaddpd | Vhsubps | Vhsubpd
            | Vroundps | Vroundss | Vroundpd | Vroundsd
            | Roundps | Roundss | Roundpd | Roundsd
            | Vdpps | Vdppd | Dpps | Dppd
        // float compares
            | Cmpps | Cmppd | Cmpss | Cmpsd
            | Vcmpps | Vcmppd | Vcmpss | Vcmpsd
        // float conversions
            | Cvtps2pd | Cvtpd2ps | Cvtdq2ps | Cvtps2dq | Cvttps2dq | Cvtpd2dq | Cvttpd2dq
            | Cvtdq2pd | Cvtsi2ss | Cvtsi2sd | Cvtss2si | Cvtsd2si | Cvttss2si | Cvttsd2si
            | Cvtss2sd | Cvtsd2ss
            | Vcvtps2pd | Vcvtpd2ps | Vcvtdq2ps | Vcvtps2dq | Vcvttps2dq | Vcvtpd2dq
            | Vcvttpd2dq | Vcvtdq2pd | Vcvtsi2ss | Vcvtsi2sd | Vcvtss2si | Vcvtsd2si
            | Vcvttss2si | Vcvttsd2si | Vcvtss2sd | Vcvtsd2ss
        // AES / crypto
            | Aesenc | Aesenclast | Aesdec | Aesdeclast | Aesimc | Aeskeygenassist
            | Vaesenc | Vaesenclast | Vaesdec | Vaesdeclast | Vaesimc | Vaeskeygenassist
            | Pclmulqdq | Vpclmulqdq
        // gathers
            | Vpgatherdd | Vpgatherdq | Vpgatherqd | Vpgatherqq
            | Vgatherdps | Vgatherdpd | Vgatherqps | Vgatherqpd
    )
}

// ---------------------------------------------------------------------------
// operand plumbing
// ---------------------------------------------------------------------------

/// Vector operand width of instruction `i` in bytes (16 or 32).
fn vec_width(i: &Instruction) -> u32 {
    fn is_ymm(r: Register) -> bool {
        matches!(
            r,
            Register::YMM0
                | Register::YMM1
                | Register::YMM2
                | Register::YMM3
                | Register::YMM4
                | Register::YMM5
                | Register::YMM6
                | Register::YMM7
                | Register::YMM8
                | Register::YMM9
                | Register::YMM10
                | Register::YMM11
                | Register::YMM12
                | Register::YMM13
                | Register::YMM14
                | Register::YMM15
        )
    }
    for n in 0..i.op_count() {
        if i.op_kind(n) == OpKind::Register && is_ymm(i.op_register(n)) {
            return 32;
        }
        if i.op_kind(n) == OpKind::Memory && i.memory_size().size() >= 32 {
            return 32;
        }
    }
    16
}

/// Read a vector operand (XMM/YMM register or memory) as a 256-bit value,
/// zero-extending 128-bit sources.
fn read_vec(
    i: &Instruction,
    n: u32,
    next_ip: u64,
    state: &CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<V256, Fault> {
    match i.op_kind(n) {
        OpKind::Register => {
            let r = i.op_register(n);
            if let Some(idx) = CpuState::xmm_index(r) {
                let v = state.ymm_idx(idx);
                // A YMM-aliased read gets the full register; an XMM read
                // gets only the low half (upper zeroed by the caller's
                // width logic where it matters).
                if matches!(
                    r,
                    Register::YMM0
                        | Register::YMM1
                        | Register::YMM2
                        | Register::YMM3
                        | Register::YMM4
                        | Register::YMM5
                        | Register::YMM6
                        | Register::YMM7
                        | Register::YMM8
                        | Register::YMM9
                        | Register::YMM10
                        | Register::YMM11
                        | Register::YMM12
                        | Register::YMM13
                        | Register::YMM14
                        | Register::YMM15
                ) {
                    Ok(v)
                } else {
                    Ok([v[0], 0])
                }
            } else {
                Err(Fault::InvalidOpcode {
                    mnemonic: format!("{:?} op{n} not a vector register", i.mnemonic()),
                    rip: i.ip(),
                })
            }
        }
        OpKind::Memory => {
            let w = i.memory_size().size().min(32);
            let mut buf = [0u8; 32];
            let addr = crate::interpreter::effective_address(i, next_ip, state);
            mem.read_buf(addr, &mut buf[..w])
                .map_err(AccessError::into_fault)?;
            let lo = u128::from_le_bytes(buf[..16].try_into().expect("16 bytes"));
            let hi = if w == 32 {
                u128::from_le_bytes(buf[16..32].try_into().expect("16 bytes"))
            } else {
                0
            };
            Ok([lo, hi])
        }
        _ => Err(Fault::InvalidOpcode {
            mnemonic: format!("{:?} op{n} not a vector operand", i.mnemonic()),
            rip: i.ip(),
        }),
    }
}

/// Write a full-width vector result to operand `n`, honouring VEX/legacy
/// upper-half rules (`wide` = VEX.256).
fn write_vec(
    i: &Instruction,
    n: u32,
    state: &mut CpuState,
    v: V256,
    wide: bool,
) -> Result<(), Fault> {
    if i.op_kind(n) != OpKind::Register {
        return Err(Fault::InvalidOpcode {
            mnemonic: format!("{:?} vector dst must be a register", i.mnemonic()),
            rip: i.ip(),
        });
    }
    let r = i.op_register(n);
    if wide {
        state.set_ymm_halves(r, v).ok_or_else(|| bad_op(i))?;
    } else {
        // Legacy & VEX.128: write low half, zero the upper.
        state.set_xmm(r, v[0]).ok_or_else(|| bad_op(i))?;
    }
    Ok(())
}

fn bad_op(i: &Instruction) -> Fault {
    Fault::InvalidOpcode {
        mnemonic: format!("{:?}", i.mnemonic()),
        rip: i.ip(),
    }
}

// ---------------------------------------------------------------------------
// lane helpers
// ---------------------------------------------------------------------------

/// Number of elements per 128-bit half for a lane byte-width.
const fn per_half(lane_bytes: u32) -> usize {
    (128 / 8 / lane_bytes) as usize
}

/// Apply a binary lane kernel at byte-width `w` (1, 2, 4 or 8).
fn zip_lanes(a: V256, b: V256, w: u32, f: impl Fn(u64, u64) -> u64) -> V256 {
    let n = per_half(w);
    let mut r = [0u128; 2];
    for h in 0..2usize {
        for j in 0..n {
            let sh = j * w as usize * 8;
            let (x, y) = (
                ((a[h] >> sh) & u128::from(mask64(w))) as u64,
                ((b[h] >> sh) & u128::from(mask64(w))) as u64,
            );
            r[h] |= u128::from(f(x, y) & mask64(w)) << sh;
        }
    }
    r
}

/// Apply a binary lane kernel at 128-bit granularity (2 lanes per half).
fn zip_q(a: V256, b: V256, f: impl Fn(u128, u128) -> u128) -> V256 {
    [f(a[0], b[0]), f(a[1], b[1])]
}

const fn mask64(w: u32) -> u64 {
    match w {
        1 => 0xFF,
        2 => 0xFFFF,
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

/// Execute one vector instruction. Returns `Action::Next` semantics via
/// `Ok(())`-style: the caller advances RIP (no vector instruction branches).
pub(crate) fn execute_vector(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
) -> Result<crate::interpreter::Action, Fault> {
    use Mnemonic::*;
    let wide = vec_width(i) == 32;
    let m = i.mnemonic();

    // ---- GPR-flag-setting compares --------------------------------------
    match m {
        Comiss | Comisd | Ucomiss | Ucomisd => {
            return bridge::exec_comis(i, next_ip, state, mem);
        }
        Ptest | Vptest => {
            return bridge::exec_ptest(i, next_ip, state, mem);
        }
        _ => {}
    }

    // ---- special moves with GPR/mem/shaped operands ------------------------
    match m {
        Vzeroupper => {
            for k in 0..16 {
                let v = state.ymm_idx(k);
                state.set_ymm_idx(k, [v[0], 0]);
            }
            return Ok(crate::interpreter::Action::Next);
        }
        Vzeroall => {
            for k in 0..16 {
                state.set_ymm_idx(k, [0; 2]);
            }
            return Ok(crate::interpreter::Action::Next);
        }
        Movd | Vmovd => return bridge::exec_movd(i, next_ip, state, mem),
        Movq | Vmovq => return bridge::exec_movq(i, next_ip, state, mem),
        Movmskps | Movmskpd | Vmovmskps | Vmovmskpd | Pmovmskb | Vpmovmskb => {
            return bridge::exec_movmsk(i, next_ip, state, mem);
        }
        Pextrb | Pextrw | Pextrd | Pextrq | Extractps | Vpextrb | Vpextrw | Vpextrd | Vpextrq
        | Vextractps => {
            return bridge::exec_pextr(i, next_ip, state, mem);
        }
        Pinsrb | Pinsrw | Pinsrd | Pinsrq | Insertps | Vpinsrb | Vpinsrw | Vpinsrd | Vpinsrq => {
            return bridge::exec_pinsr(i, next_ip, state, mem);
        }
        Cvtss2si | Cvtsd2si | Cvttss2si | Cvttsd2si | Vcvtss2si | Vcvtsd2si | Vcvttss2si
        | Vcvttsd2si => {
            return bridge::exec_cvt_to_gpr(i, next_ip, state, mem);
        }
        Cvtsi2ss | Cvtsi2sd | Vcvtsi2ss | Vcvtsi2sd => {
            return bridge::exec_cvt_from_gpr(i, next_ip, state, mem);
        }
        Comiss | Comisd | Ucomiss | Ucomisd | Ptest | Vptest => unreachable!(),
        _ => {}
    }

    // ---- unary / source-only forms ---------------------------------------
    match m {
        Movaps | Movups | Movapd | Movupd | Movdqa | Movdqu | Vmovaps | Vmovups | Vmovapd
        | Vmovupd | Vmovdqu | Vmovdqa => {
            let v = read_vec(i, 1, next_ip, state, mem)?;
            // Store to memory or register; width from operand size.
            return write_vec_or_mem(i, state, mem, next_ip, v, wide);
        }
        Aesimc | Vaesimc => {
            let v = read_vec(i, 1, next_ip, state, mem)?;
            let r = crypto::aes_inv_mix_columns(v[0]);
            write_vec(
                i,
                0,
                state,
                [
                    r,
                    if wide {
                        crypto::aes_inv_mix_columns(v[1])
                    } else {
                        0
                    },
                ],
                wide,
            )?;
            return Ok(crate::interpreter::Action::Next);
        }
        Aeskeygenassist | Vaeskeygenassist => {
            let v = read_vec(i, 1, next_ip, state, mem)?;
            let imm = i.immediate8();
            let f = |x: u128| crypto::keygen_assist(x, imm);
            let lo = f(v[0]);
            let hi = if wide { f(v[1]) } else { 0 };
            write_vec(i, 0, state, [lo, hi], wide)?;
            return Ok(crate::interpreter::Action::Next);
        }
        _ => {}
    }

    // ---- scalar / half / lane moves (store forms must not pre-read op0) --
    match m {
        Movss | Vmovss | Movsd | Vmovsd | Movhpd | Vmovhpd | Movlpd | Vmovlpd | Movhps
        | Vmovhps | Movlps | Vmovlps | Movlhps | Vmovlhps | Movhlps | Vmovhlps => {
            return exec_vector_stage2(i, next_ip, state, mem, [0; 2], [0; 2], wide);
        }
        _ => {}
    }

    // ---- immediate-controlled forms take their own operand paths ----------
    if has_imm_control(i) {
        if let Some(action) = super_dispatch::try_all(i, next_ip, state, mem, [0; 2], [0; 2], wide)?
        {
            return Ok(action);
        }
        return Err(bad_op(i));
    }

    // ---- binary (and binary+imm) forms ------------------------------------
    let src2 = read_vec(
        i,
        if i.op_count() >= 3 { 2 } else { 1 },
        next_ip,
        state,
        mem,
    )?;
    let src1 = read_vec(i, 0, next_ip, state, mem)?; // legacy: op0 is also dst

    // Three-operand VEX: op0 = dst, op1 = src1, op2 = src2.
    let (a, b) = if i.op_count() >= 3 && i.op_kind(1) != OpKind::Immediate8 {
        (read_vec(i, 1, next_ip, state, mem)?, src2)
    } else {
        (src1, src2)
    };

    let result: V256 = match m {
        // ---- packed integer arithmetic --------------------------------
        Paddb | Vpaddb => zip_lanes(a, b, 1, |x, y| x.wrapping_add(y)),
        Paddw | Vpaddw => zip_lanes(a, b, 2, |x, y| x.wrapping_add(y)),
        Paddd | Vpaddd => zip_lanes(a, b, 4, |x, y| x.wrapping_add(y)),
        Paddq | Vpaddq => zip_q(a, b, |x, y| x.wrapping_add(y)),
        Psubb | Vpsubb => zip_lanes(a, b, 1, |x, y| x.wrapping_sub(y)),
        Psubw | Vpsubw => zip_lanes(a, b, 2, |x, y| x.wrapping_sub(y)),
        Psubd | Vpsubd => zip_lanes(a, b, 4, |x, y| x.wrapping_sub(y)),
        Psubq | Vpsubq => zip_q(a, b, |x, y| x.wrapping_sub(y)),
        Paddsb | Vpaddsb => zip_lanes(a, b, 1, |x, y| {
            (x as i8).saturating_add(y as i8) as u8 as u64
        }),
        Paddsw | Vpaddsw => zip_lanes(a, b, 2, |x, y| {
            (x as i16).saturating_add(y as i16) as u16 as u64
        }),
        Paddusb | Vpaddusb => zip_lanes(a, b, 1, |x, y| x.saturating_add(y) & 0xFF),
        Paddusw | Vpaddusw => zip_lanes(a, b, 2, |x, y| x.saturating_add(y) & 0xFFFF),
        Psubsb | Vpsubsb => zip_lanes(a, b, 1, |x, y| {
            (x as i8).saturating_sub(y as i8) as u8 as u64
        }),
        Psubsw | Vpsubsw => zip_lanes(a, b, 2, |x, y| {
            (x as i16).saturating_sub(y as i16) as u16 as u64
        }),
        Psubusb | Vpsubusb => zip_lanes(a, b, 1, |x, y| x.saturating_sub(y) & 0xFF),
        Psubusw | Vpsubusw => zip_lanes(a, b, 2, |x, y| x.saturating_sub(y) & 0xFFFF),

        Pmullw | Vpmullw => zip_lanes(a, b, 2, |x, y| x.wrapping_mul(y) & 0xFFFF),
        Pmulhw | Vpmulhw => zip_lanes(a, b, 2, |x, y| {
            (((x as i16 as i32) * (y as i16 as i32)) >> 16) as u16 as u64
        }),
        Pmulhuw | Vpmulhuw => zip_lanes(a, b, 2, |x, y| (x * y) >> 16),
        Pmulhrsw | Vpmulhrsw => zip_lanes(a, b, 2, |x, y| {
            let p = (x as i16 as i32) * (y as i16 as i32);
            ((p + 0x4000) >> 15) as u16 as u64
        }),
        Pmulld | Vpmulld => zip_lanes(a, b, 4, |x, y| x.wrapping_mul(y)),
        Pmuludq | Vpmuludq => zip_lanes_64_pairs(a, b, |x, y| x.wrapping_mul(y)),
        Pmuldq | Vpmuldq => {
            zip_lanes_64_pairs(a, b, |x, y| (x as i64).wrapping_mul(y as i64) as u64)
        }

        Pmaddwd | Vpmaddwd => {
            // per 32-bit result: a16[0]*b16[0] + a16[1]*b16[1]
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..4 {
                    let w0 = ((a[h] >> (j * 32)) & 0xFFFF) as u16 as i32 as i64;
                    let w1 = ((a[h] >> (j * 32 + 16)) & 0xFFFF) as u16 as i32 as i64;
                    let v0 = ((b[h] >> (j * 32)) & 0xFFFF) as u16 as i32 as i64;
                    let v1 = ((b[h] >> (j * 32 + 16)) & 0xFFFF) as u16 as i32 as i64;
                    r[h] |= ((w0.wrapping_mul(v0).wrapping_add(w1.wrapping_mul(v1))) as u32
                        as u128)
                        << (j * 32);
                }
            }
            r
        }
        Pmaddubsw | Vpmaddubsw => {
            let mut r = [0u128; 2];
            for h in 0..2 {
                for j in 0..8 {
                    let u = ((a[h] >> (j * 16)) & 0xFF) as u8 as u32 as i32;
                    let s0 = ((b[h] >> (j * 16)) & 0xFF) as i8 as i32;
                    let u1 = ((a[h] >> (j * 16 + 8)) & 0xFF) as u8 as i32;
                    let s1 = ((b[h] >> (j * 16 + 8)) & 0xFF) as i8 as i32;
                    let p = u.wrapping_mul(s0).wrapping_add(u1.wrapping_mul(s1));
                    r[h] |= (p.clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u128) << (j * 16);
                }
            }
            r
        }

        // ---- logic ------------------------------------------------------
        Pand | Vpand | Andps | Vandps | Andpd | Vandpd => zip_q(a, b, |x, y| x & y),
        Por | Vpor | Orps | Vorps | Orpd | Vorpd => zip_q(a, b, |x, y| x | y),
        Pxor | Vpxor | Xorps | Vxorps | Xorpd | Vxorpd => zip_q(a, b, |x, y| x ^ y),
        Pandn | Vpandn | Andnps | Vandnps | Andnpd | Vandnpd => zip_q(a, b, |x, y| !x & y),

        // ---- comparisons --------------------------------------------------
        Pcmpeqb | Vpcmpeqb => zip_lanes(a, b, 1, |x, y| if x == y { 0xFF } else { 0 }),
        Pcmpeqw | Vpcmpeqw => zip_lanes(a, b, 2, |x, y| if x == y { 0xFFFF } else { 0 }),
        Pcmpeqd | Vpcmpeqd => zip_lanes(a, b, 4, |x, y| if x == y { 0xFFFF_FFFF } else { 0 }),
        Pcmpeqq | Vpcmpeqq => zip_q(a, b, |x, y| if x == y { !0 } else { 0 }),
        Pcmpgtb | Vpcmpgtb => {
            zip_lanes(a, b, 1, |x, y| if (x as i8) > (y as i8) { 0xFF } else { 0 })
        }
        Pcmpgtw | Vpcmpgtw => zip_lanes(
            a,
            b,
            2,
            |x, y| if (x as i16) > (y as i16) { 0xFFFF } else { 0 },
        ),
        Pcmpgtd | Vpcmpgtd => zip_lanes(a, b, 4, |x, y| {
            if (x as i32) > (y as i32) {
                0xFFFF_FFFF
            } else {
                0
            }
        }),
        Pcmpgtq | Vpcmpgtq => zip_q(a, b, |x, y| if (x as i128) > (y as i128) { !0 } else { 0 }),

        // ---- min/max/avg ---------------------------------------------------
        Pmaxsb | Vpmaxsb => zip_lanes(a, b, 1, |x, y| x.max((y as i8 as u8) as u64)),
        Pmaxsw | Vpmaxsw => zip_lanes(a, b, 2, |x, y| x.max((y as i16 as u16) as u64)),
        Pmaxsd | Vpmaxsd => zip_lanes(a, b, 4, |x, y| x.max((y as i32 as u32) as u64)),
        Pmaxub | Vpmaxub => zip_lanes(a, b, 1, |x, y| x.max(y)),
        Pmaxuw | Vpmaxuw => zip_lanes(a, b, 2, |x, y| x.max(y)),
        Pmaxud | Vpmaxud => zip_lanes(a, b, 4, |x, y| x.max(y)),
        Pminsb | Vpminsb => zip_lanes(a, b, 1, |x, y| x.min((y as i8 as u8) as u64)),
        Pminsw | Vpminsw => zip_lanes(a, b, 2, |x, y| x.min((y as i16 as u16) as u64)),
        Pminsd | Vpminsd => zip_lanes(a, b, 4, |x, y| x.min((y as i32 as u32) as u64)),
        Pminub | Vpminub => zip_lanes(a, b, 1, |x, y| x.min(y)),
        Pminuw | Vpminuw => zip_lanes(a, b, 2, |x, y| x.min(y)),
        Pminud | Vpminud => zip_lanes(a, b, 4, |x, y| x.min(y)),
        Pavgb | Vpavgb => zip_lanes(a, b, 1, |x, y| (x + y).div_ceil(2)),
        Pavgw | Vpavgw => zip_lanes(a, b, 2, |x, y| (x + y).div_ceil(2)),

        _ => return exec_vector_stage2(i, next_ip, state, mem, a, b, wide),
    };
    write_vec(i, 0, state, result, wide)?;
    Ok(crate::interpreter::Action::Next)
}

/// 64-bit-lane pair helper: operates on even 32-bit lanes' products forming
/// 64-bit results from 32-bit inputs (PMULUDQ/PMULDQ shape).
fn zip_lanes_64_pairs(a: V256, b: V256, f: impl Fn(u64, u64) -> u64) -> V256 {
    let mut r = [0u128; 2];
    for h in 0..2 {
        for j in 0..2 {
            let x = ((a[h] >> (j * 64)) & 0xFFFF_FFFF) as u64;
            let y = ((b[h] >> (j * 64)) & 0xFFFF_FFFF) as u64;
            r[h] |= (f(x, y) as u128) << (j * 64);
        }
    }
    r
}

/// Writes either a vector register or, for store-form moves, guest memory.
fn write_vec_or_mem(
    i: &Instruction,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    next_ip: u64,
    v: V256,
    wide: bool,
) -> Result<crate::interpreter::Action, Fault> {
    if i.op_kind(0) == OpKind::Memory {
        // Store form: op0 is memory, op1 is the source register.
        let addr = crate::interpreter::effective_address(i, next_ip, state);
        let w = i.memory_size().size().min(32);
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&v[0].to_le_bytes());
        buf[16..32].copy_from_slice(&v[1].to_le_bytes());
        mem.write_buf(addr, &buf[..w])
            .map_err(AccessError::into_fault)?;
        return Ok(crate::interpreter::Action::Next);
    }
    write_vec(i, 0, state, v, wide)?;
    Ok(crate::interpreter::Action::Next)
}

/// Stage 2: instruction shapes that need custom operand handling (immediate
/// controls, scalar lanes, memory bridges, crypto, gathers).
#[allow(clippy::too_many_lines)]
fn exec_vector_stage2(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    a: V256,
    b: V256,
    wide: bool,
) -> Result<crate::interpreter::Action, Fault> {
    use Mnemonic::*;
    let m = i.mnemonic();
    let _imm = immediate_of(i);

    let result: V256 = match m {
        // ---- scalar / half moves ----------------------------------------
        Movss | Vmovss => {
            // reg,mem zeroes upper lanes; reg,reg preserves them; store form
            // writes 4 bytes.
            if i.op_kind(0) == OpKind::Memory {
                let src = read_vec(i, 1, next_ip, state, mem)?;
                let addr = crate::interpreter::effective_address(i, next_ip, state);
                let bytes = (src[0] as u32).to_le_bytes();
                mem.write_buf(addr, &bytes)
                    .map_err(AccessError::into_fault)?;
                return Ok(crate::interpreter::Action::Next);
            }
            let src = read_vec(i, 1, next_ip, state, mem)?;
            if i.op_kind(1) == OpKind::Memory {
                [src[0], 0]
            } else {
                [a[0] & !0xFFFF_FFFF | (src[0] & 0xFFFF_FFFF), a[1]]
            }
        }
        Movsd | Vmovsd => {
            if i.op_kind(0) == OpKind::Memory {
                let src = read_vec(i, 1, next_ip, state, mem)?;
                let addr = crate::interpreter::effective_address(i, next_ip, state);
                mem.write_buf(addr, &src[0].to_le_bytes()[..8])
                    .map_err(AccessError::into_fault)?;
                return Ok(crate::interpreter::Action::Next);
            }
            let src = read_vec(i, 1, next_ip, state, mem)?;
            [
                a[0] & !0xFFFF_FFFF_FFFF_FFFF | (src[0] & 0xFFFF_FFFF_FFFF_FFFF),
                a[1],
            ]
        }
        Movhpd | Vmovhpd | Movhps | Vmovhps => {
            if i.op_kind(0) == OpKind::Memory {
                // store bits 64:127 of the source register
                let src = read_vec(i, 1, next_ip, state, mem)?;
                let addr = crate::interpreter::effective_address(i, next_ip, state);
                mem.write_buf(addr, &(src[0] >> 64).to_le_bytes()[..8])
                    .map_err(AccessError::into_fault)?;
                return Ok(crate::interpreter::Action::Next);
            }
            // load: dst[64:127] = m64, dst[63:0] preserved.
            let src = read_vec(i, 1, next_ip, state, mem)?;
            [
                (a[0] & 0xFFFF_FFFF_FFFF_FFFF) | ((src[0] & 0xFFFF_FFFF_FFFF_FFFF) << 64),
                a[1],
            ]
        }
        Movlpd | Vmovlpd | Movlps | Vmovlps => {
            if i.op_kind(0) == OpKind::Memory {
                let src = read_vec(i, 1, next_ip, state, mem)?;
                let addr = crate::interpreter::effective_address(i, next_ip, state);
                mem.write_buf(addr, &src[0].to_le_bytes()[..8])
                    .map_err(AccessError::into_fault)?;
                return Ok(crate::interpreter::Action::Next);
            }
            // load: dst[63:0] = m64, dst[127:64] preserved.
            let src = read_vec(i, 1, next_ip, state, mem)?;
            [
                (a[0] & !0xFFFF_FFFF_FFFF_FFFF) | (src[0] & 0xFFFF_FFFF_FFFF_FFFF),
                a[1],
            ]
        }
        Movlhps | Vmovlhps => {
            // dst[63:0] preserved, dst[127:64] = src[63:0] (reg-reg only).
            let src = read_vec(i, 1, next_ip, state, mem)?;
            [
                (a[0] & 0xFFFF_FFFF_FFFF_FFFF) | ((src[0] & 0xFFFF_FFFF_FFFF_FFFF) << 64),
                a[1],
            ]
        }
        Movhlps | Vmovhlps => {
            // dst[127:64] preserved, dst[63:0] = src[127:64] (reg-reg only).
            let src = read_vec(i, 1, next_ip, state, mem)?;
            [(a[0] & !0xFFFF_FFFF_FFFF_FFFF) | (src[0] >> 64), a[1]]
        }

        // Anything else reaching stage2 belongs to the sibling modules.
        _ => {
            if let Some(action) = super_dispatch::try_all(i, next_ip, state, mem, a, b, wide)? {
                return Ok(action);
            }
            return Err(bad_op(i));
        }
    };
    write_vec(i, 0, state, result, wide)?;
    Ok(crate::interpreter::Action::Next)
}

/// True when the instruction's last operand is an immediate control byte
/// (shuffle selectors, blend masks, rounding modes, ...).
pub(crate) fn has_imm_control(i: &Instruction) -> bool {
    let n = i.op_count();
    n >= 1
        && matches!(
            i.op_kind(n - 1),
            OpKind::Immediate8
                | OpKind::Immediate8to16
                | OpKind::Immediate8to32
                | OpKind::Immediate8to64
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate32to64
        )
        && !matches!(
            i.mnemonic(),
            Mnemonic::Aeskeygenassist | Mnemonic::Vaeskeygenassist
        )
}

/// Immediate operand of instruction `i` (the last operand when immediate).
pub(crate) fn immediate_of(i: &Instruction) -> u8 {
    for n in 0..i.op_count() {
        if matches!(
            i.op_kind(n),
            OpKind::Immediate8 | OpKind::Immediate8to64 | OpKind::Immediate32to64
        ) {
            return i.immediate8();
        }
    }
    0
}

/// Last source operand index (skipping a trailing immediate).
pub(crate) fn last_src(i: &Instruction) -> u32 {
    let n = i.op_count();
    if matches!(
        i.op_kind(n - 1),
        OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate32to64
    ) {
        n - 2
    } else {
        n - 1
    }
}

/// Write the result to op0 as a vector (or memory, for store forms).
pub(crate) fn finish_vec(
    i: &Instruction,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    next_ip: u64,
    result: V256,
    wide: bool,
) -> Result<crate::interpreter::Action, Fault> {
    write_vec_or_mem(i, state, mem, next_ip, result, wide)
}
