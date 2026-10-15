//! AES-NI and carry-less multiply (PCLMULQDQ) families, pure Rust kernels.

use iced_x86::{Instruction, Mnemonic};

use crate::fault::{Fault, MemoryAccess};
use crate::interpreter::Action;
use crate::state::CpuState;

use super::{V256, finish_vec, immediate_of, read_vec};

/// AES S-box.
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

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
    let a = read_vec(
        i,
        if i.op_count() >= 3 { 1 } else { 0 },
        next_ip,
        state,
        mem,
    )?;
    let b = read_vec(i, super::last_src(i), next_ip, state, mem)?;

    let round = |x: u128, key: u128, last: bool| {
        if last {
            aes_enc_last(x, key)
        } else {
            aes_enc_round(x, key)
        }
    };
    let result: V256 = match m {
        Aesenc | Vaesenc => [
            round(a[0], b[0], false),
            if wide { round(a[1], b[1], false) } else { 0 },
        ],
        Aesenclast | Vaesenclast => [
            round(a[0], b[0], true),
            if wide { round(a[1], b[1], true) } else { 0 },
        ],
        Aesdec | Vaesdec => [
            aes_dec_round(a[0], b[0]),
            if wide { aes_dec_round(a[1], b[1]) } else { 0 },
        ],
        Aesdeclast | Vaesdeclast => [
            aes_dec_last(a[0], b[0]),
            if wide { aes_dec_last(a[1], b[1]) } else { 0 },
        ],
        Pclmulqdq | Vpclmulqdq => {
            let imm = immediate_of(i);
            let sel = |v: u128, half: u8| -> u64 {
                if half == 1 {
                    (v >> 64) as u64
                } else {
                    v as u64
                }
            };
            let x = sel(a[0], imm & 1);
            let y = sel(b[0], (imm >> 4) & 1);
            let p = clmul64(x, y);
            [
                p,
                if wide {
                    let x2 = sel(a[1], imm & 1);
                    let y2 = sel(b[1], (imm >> 4) & 1);
                    clmul64(x2, y2)
                } else {
                    0
                },
            ]
        }
        _ => return Ok(None),
    };
    let action = finish_vec(i, state, mem, next_ip, result, wide)?;
    Ok(Some(action))
}

/// One AESENC round: ShiftRows → SubBytes → MixColumns → AddRoundKey.
pub(crate) fn aes_enc_round(state: u128, key: u128) -> u128 {
    let shifted = shift_rows(state);
    let subbed = sub_bytes(shifted);
    let mixed = mix_columns(subbed);
    mixed ^ key
}

/// Final AESENCLAST round: ShiftRows → SubBytes → AddRoundKey.
pub(crate) fn aes_enc_last(state: u128, key: u128) -> u128 {
    sub_bytes(shift_rows(state)) ^ key
}

/// AESDEC (equivalent inverse cipher): the round key applies *first*, then
/// InvMixColumns, InvSubBytes, InvShiftRows — the exact inverse of
/// [`aes_enc_round`]'s ShiftRows → SubBytes → MixColumns → AddRoundKey.
pub(crate) fn aes_dec_round(state: u128, key: u128) -> u128 {
    inv_shift_rows(inv_sub_bytes(inv_mix_columns_u128(state ^ key)))
}

/// AESDECLAST: AddRoundKey, InvSubBytes, InvShiftRows (no InvMixColumns).
pub(crate) fn aes_dec_last(state: u128, key: u128) -> u128 {
    inv_shift_rows(inv_sub_bytes(state ^ key))
}

/// AESIMC: inverse MixColumns applied to each of the four columns.
pub(crate) fn aes_inv_mix_columns(state: u128) -> u128 {
    inv_mix_columns_u128(state)
}

/// AESKEYGENASSIST: X1 = rot(subword(X0)), X3 = rot(subword(X2)).
pub(crate) fn keygen_assist(x: u128, rcon: u8) -> u128 {
    let subword = |w: u32| -> u32 {
        let b = w.to_le_bytes();
        u32::from_le_bytes([
            SBOX[b[0] as usize],
            SBOX[b[1] as usize],
            SBOX[b[2] as usize],
            SBOX[b[3] as usize],
        ])
    };
    let rot = |w: u32| w.rotate_left(8);
    let x0 = x as u32;
    let x2 = (x >> 64) as u32;
    let r = u32::from(rcon);
    let t1 = rot(subword(x0));
    let t3 = rot(subword(x2));
    // Result: [0]=t3rcon, [1]=t3, [2]=t1rcon, [3]=t1 — per Intel layout.
    (t3 as u128) | ((t3 ^ r) as u128) << 32 | ((t1 as u128) << 64) | ((t1 ^ r) as u128) << 96
}

fn shift_rows(s: u128) -> u128 {
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for c in 0..4 {
        for row in 0..4 {
            // state[col][row]: byte index = col*4 + row in AES layout.
            let src_col = (c + row) % 4;
            r[c * 4 + row] = b[src_col * 4 + row];
        }
    }
    u128::from_le_bytes(r)
}

fn inv_shift_rows(s: u128) -> u128 {
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for c in 0..4 {
        for row in 0..4 {
            let src_col = (c + 4 - row) % 4;
            r[c * 4 + row] = b[src_col * 4 + row];
        }
    }
    u128::from_le_bytes(r)
}

fn sub_bytes(s: u128) -> u128 {
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for (i, x) in b.iter().enumerate() {
        r[i] = SBOX[*x as usize];
    }
    u128::from_le_bytes(r)
}

fn inv_sub_bytes(s: u128) -> u128 {
    let inv = inv_sbox();
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for (i, x) in b.iter().enumerate() {
        r[i] = inv[*x as usize];
    }
    u128::from_le_bytes(r)
}

/// Inverse S-box, derived from [`SBOX`] once per process.
fn inv_sbox() -> &'static [u8; 256] {
    static INV: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    INV.get_or_init(|| {
        let mut t = [0u8; 256];
        for (v, s) in SBOX.iter().enumerate() {
            t[*s as usize] = v as u8;
        }
        t
    })
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    p
}

fn mix_column(col: [u8; 4]) -> [u8; 4] {
    [
        gf_mul(col[0], 2) ^ gf_mul(col[1], 3) ^ col[2] ^ col[3],
        col[0] ^ gf_mul(col[1], 2) ^ gf_mul(col[2], 3) ^ col[3],
        col[0] ^ col[1] ^ gf_mul(col[2], 2) ^ gf_mul(col[3], 3),
        gf_mul(col[0], 3) ^ col[1] ^ col[2] ^ gf_mul(col[3], 2),
    ]
}

fn inv_mix_column(col: [u8; 4]) -> [u8; 4] {
    [
        gf_mul(col[0], 14) ^ gf_mul(col[1], 11) ^ gf_mul(col[2], 13) ^ gf_mul(col[3], 9),
        gf_mul(col[0], 9) ^ gf_mul(col[1], 14) ^ gf_mul(col[2], 11) ^ gf_mul(col[3], 13),
        gf_mul(col[0], 13) ^ gf_mul(col[1], 9) ^ gf_mul(col[2], 14) ^ gf_mul(col[3], 11),
        gf_mul(col[0], 11) ^ gf_mul(col[1], 13) ^ gf_mul(col[2], 9) ^ gf_mul(col[3], 14),
    ]
}

fn mix_columns(s: u128) -> u128 {
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for c in 0..4 {
        let col = mix_column([b[c * 4], b[c * 4 + 1], b[c * 4 + 2], b[c * 4 + 3]]);
        r[c * 4..c * 4 + 4].copy_from_slice(&col);
    }
    u128::from_le_bytes(r)
}

fn inv_mix_columns_u128(s: u128) -> u128 {
    let b = s.to_le_bytes();
    let mut r = [0u8; 16];
    for c in 0..4 {
        let col = inv_mix_column([b[c * 4], b[c * 4 + 1], b[c * 4 + 2], b[c * 4 + 3]]);
        r[c * 4..c * 4 + 4].copy_from_slice(&col);
    }
    u128::from_le_bytes(r)
}

/// 64×64 carry-less multiplication → 128-bit result.
fn clmul64(a: u64, b: u64) -> u128 {
    // For every set bit at position p in `a`, xor in `b` shifted by p.
    let mut r = 0u128;
    let mut a = a as u128;
    let bb = b as u128;
    while a != 0 {
        let p = a.trailing_zeros();
        r ^= bb << p;
        a &= a - 1; // clear the lowest set bit
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS-197 C.1: full AES-128 encryption built from our round
    /// primitives must match the reference implementation exactly.
    #[test]
    fn aes128_full_encryption_matches_reference() {
        use aes::cipher::{BlockEncrypt, KeyInit};
        let plain = u128::from_le_bytes([
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ]);
        let key_bytes = [
            0x2bu8, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        // Reference: RustCrypto AES-128.
        let cipher = aes::Aes128::new_from_slice(&key_bytes).unwrap();
        let mut block = plain.to_le_bytes();
        cipher.encrypt_block(aes::Block::from_mut_slice(&mut block));
        let expect = u128::from_le_bytes(block);

        let rks = aes128_key_schedule(&key_bytes);
        let mut st = plain ^ rks[0];
        for rk in rks.iter().take(10).skip(1) {
            st = aes_enc_round(st, *rk);
        }
        st = aes_enc_last(st, rks[10]);
        assert_eq!(st, expect, "AES-128 rounds must match the reference");
    }

    /// Textbook AES-128 key schedule (FIPS-197 §5.2).
    fn aes128_key_schedule(key: &[u8; 16]) -> Vec<u128> {
        let subw = |w: u32| -> u32 {
            let b = w.to_be_bytes();
            u32::from_be_bytes([
                SBOX[b[0] as usize],
                SBOX[b[1] as usize],
                SBOX[b[2] as usize],
                SBOX[b[3] as usize],
            ])
        };
        let mut w = [0u32; 44];
        for i in 0..4 {
            w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        }
        for i in 4..44 {
            let mut t = w[i - 1];
            if i % 4 == 0 {
                t = subw(t.rotate_left(8)) ^ (u32::from(RCON[i / 4 - 1]) << 24);
            }
            w[i] = w[i - 4] ^ t;
        }
        (0..11)
            .map(|r| {
                let mut k = [0u8; 16];
                for j in 0..4 {
                    k[4 * j..4 * j + 4].copy_from_slice(&w[4 * r + j].to_be_bytes());
                }
                u128::from_le_bytes(k)
            })
            .collect()
    }

    /// Round constants for the AES-128 key schedule.
    const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36];

    #[test]
    fn clmul_basics() {
        // (1 * 1) = 1; (2 * 4) = 8 (carryless XOR of shifted).
        assert_eq!(clmul64(1, 1), 1);
        assert_eq!(clmul64(2, 4), 8);
        // (a * b) with overlap: 0b11 * 0b11 = x^2 + 1 = 0b101.
        assert_eq!(clmul64(3, 3), 0b101);
    }

    #[test]
    fn sbox_is_a_permutation() {
        let inv = inv_sbox();
        for v in 0..256u16 {
            let s = SBOX[v as usize];
            assert_eq!(inv[s as usize], v as u8);
        }
    }

    #[test]
    fn dec_round_inverts_enc_round() {
        let st = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        let key = 0xdead_beef_cafe_babe_0123_4567_89ab_cdefu128;
        let e = aes_enc_round(st, key);
        assert_eq!(aes_dec_round(e, key), st);
        let el = aes_enc_last(st, key);
        assert_eq!(aes_dec_last(el, key), st);
    }
}
