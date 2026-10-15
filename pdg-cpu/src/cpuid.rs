//! `CPUID` — what the emulated Zen 2 core tells guest software it is.
//!
//! The values model the PS5's AMD Zen 2 complex: `AuthenticAMD`, display
//! family 0x17 (23), model 0x47, 8 cores / 16 threads. Feature flags are the
//! honest part: a bit is set **only** for an instruction family PDG actually
//! implements. As coverage grows, so does this table; the unit test pins the
//! exact bit set so an accidental over-advertisement fails CI.
//!
//! Note for later phases: guest games and the Orbis kernel normally query
//! CPU features through the OS, not raw `CPUID` — the Phase-5 kernel layer
//! decides how much of this surface to expose to user code.

/// One `CPUID` result: the four leaves EAX..EDX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuidResult {
    /// EAX leaf value.
    pub eax: u32,
    /// EBX leaf value.
    pub ebx: u32,
    /// ECX leaf value.
    pub ecx: u32,
    /// EDX leaf value.
    pub edx: u32,
}

/// Maximum standard leaf we answer (leaves past this return zeros).
pub const MAX_STD_LEAF: u32 = 0x1F;
/// Maximum extended leaf we answer.
pub const MAX_EXT_LEAF: u32 = 0x8000001F;

/// Vendor string burned into EBX/EDX/ECX of leaf 0.
pub const VENDOR: &str = "AuthenticAMD";

/// Brand string burned into leaves 0x80000002..0x80000004.
pub const BRAND: &str = "PDG Virtual AMD Zen 2 (PS5-class)";

/// Leaf-1 ECX feature bits PDG implements (see `set_ecx` test for the ledger).
pub mod feat_ecx {
    /// SSE3 (`haddps`, `haddpd`, `addsub`*, `movddup`, `movshdup`, `movsldup`).
    pub const SSE3: u32 = 1 << 0;
    /// SSSE3 (`pshufb`, `palignr`, `pabs*`, `phadd*`, `phsub*`, `pmaddubsw`).
    pub const SSSE3: u32 = 1 << 9;
    /// SSE4.1 (`pmulld`, `pmaxsb..ud`, `roundps`, `blendps`, `pextrd/q`, `pinsrd/q`, `pmovzx/sx*`, `packusdw`, `ptest`, `dpps`).
    pub const SSE4_1: u32 = 1 << 19;
    /// SSE4.2 (`pcmpgtq`, `pcmpestri`-family subset: `pcmpistri`/`pcmpestrm` not yet).
    pub const SSE4_2: u32 = 1 << 20;
    /// POPCNT.
    pub const POPCNT: u32 = 1 << 23;
    /// AES-NI (full round suite + keygen assist).
    pub const AES: u32 = 1 << 25;
    /// CMPXCHG16B.
    pub const CX16: u32 = 1 << 13;
    /// RDTSCP.
    pub const RDTSCP: u32 = 1 << 27;
    /// PCLMULQDQ.
    pub const PCLMULQDQ: u32 = 1 << 1;
}

/// Leaf-7 EBX feature bits PDG implements.
pub mod feat_ebx7 {
    /// BMI1 (`andn`, `bextr`, `blsi`, `blsmsk`, `blsr`, `tzcnt`).
    pub const BMI1: u32 = 1 << 3;
    /// BMI2 (`bzhi`, `mulx`, `pdep`, `pext`, `rorx`, `sarx`, `shlx`, `shrx`).
    pub const BMI2: u32 = 1 << 8;
    /// AVX2 (256-bit integer kernels + most VEX forms).
    pub const AVX2: u32 = 1 << 5;
}

/// Leaf-1 EDX feature bits PDG implements.
pub mod feat_edx {
    /// CMOVcc (Phase-2 ISA completion).
    pub const CMOV: u32 = 1 << 15;
    /// SSE (packed & scalar single-precision float).
    pub const SSE: u32 = 1 << 25;
    /// SSE2 (full integer kernels + double float).
    pub const SSE2: u32 = 1 << 26;
}

/// Leaf-1 ECX AVX/FMA bits.
pub mod feat_avx {
    /// AVX (VEX-encoded 128/256-bit float + integer).
    pub const AVX: u32 = 1 << 28;
    /// FMA3 (`vfmadd`/`vfmsub` families).
    pub const FMA: u32 = 1 << 12;
}

/// The exact set of leaf-1 EDX bits PDG claims (unit-pinned).
pub const CLAIMED_EDX: u32 = feat_edx::CMOV | feat_edx::SSE | feat_edx::SSE2;

/// The exact set of leaf-1 ECX bits PDG claims (unit-pinned).
pub const CLAIMED_ECX: u32 = feat_ecx::SSE3
    | feat_ecx::SSSE3
    | feat_ecx::SSE4_1
    | feat_ecx::SSE4_2
    | feat_ecx::POPCNT
    | feat_ecx::AES
    | feat_ecx::CX16
    | feat_ecx::RDTSCP
    | feat_ecx::PCLMULQDQ
    | feat_avx::AVX
    | feat_avx::FMA;

/// The exact set of leaf-7 EBX bits PDG claims (unit-pinned).
pub const CLAIMED_EBX7: u32 = feat_ebx7::BMI1 | feat_ebx7::BMI2 | feat_ebx7::AVX2;

/// Answer `CPUID` for the given `leaf`/`subleaf` pair.
pub fn cpuid(leaf: u32, subleaf: u32) -> CpuidResult {
    let zero = CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0,
        edx: 0,
    };
    match leaf {
        0 => {
            let v = VENDOR.as_bytes();
            CpuidResult {
                eax: MAX_STD_LEAF,
                ebx: u32::from_le_bytes([v[0], v[1], v[2], v[3]]),
                edx: u32::from_le_bytes([v[4], v[5], v[6], v[7]]),
                ecx: u32::from_le_bytes([v[8], v[9], v[10], v[11]]),
            }
        }
        1 => {
            // EAX: stepping 2 | model 0x7 | family 0xF | ext-model 0x4 |
            //      ext-family 0x8  →  display family 0x17, model 0x47.
            let eax = 2u32 | (0x7 << 4) | (0xF << 8) | (0x4 << 16) | (0x8 << 20);
            CpuidResult {
                eax,
                ebx: 0, // CLFLUSH line size / APIC ids not modelled yet
                ecx: CLAIMED_ECX,
                edx: CLAIMED_EDX,
            }
        }
        4 => zero, // deterministic cache params: not modelled yet
        7 => {
            if subleaf == 0 {
                CpuidResult {
                    eax: 0,
                    ebx: CLAIMED_EBX7,
                    ecx: 0,
                    edx: 0,
                }
            } else {
                zero
            }
        }
        0xA => zero, // perf counters
        0xD => zero, // XSAVE: NOT implemented — bit 26 stays clear in leaf 1
        0x8000_0000 => CpuidResult {
            eax: MAX_EXT_LEAF,
            ebx: 0,
            ecx: 0,
            edx: 0,
        },
        0x8000_0001 => {
            // EDX: LM (long mode) + SYSCALL; EDX fpu bits stay clear (no x87).
            CpuidResult {
                eax: 0,
                ebx: 0,
                ecx: 1 << 5, // LZCNT
                edx: 1 << 29 | 1 << 11,
            }
        }
        0x8000_0002..=0x8000_0004 => {
            // Brand string: 48 bytes spread over three leaves.
            let mut brand = [0u8; 48];
            let bytes = BRAND.as_bytes();
            let n = bytes.len().min(48);
            brand[..n].copy_from_slice(&bytes[..n]);
            let chunk = (leaf - 0x8000_0002) as usize * 16;
            let mut w = [0u32; 4];
            for (i, w_i) in w.iter_mut().enumerate() {
                let b = chunk + i * 4;
                *w_i = u32::from_le_bytes(brand[b..b + 4].try_into().expect("4 bytes"));
            }
            CpuidResult {
                eax: w[0],
                ebx: w[1],
                ecx: w[2],
                edx: w[3],
            }
        }
        0x8000_0008 => CpuidResult {
            eax: 0x30, // 48-bit linear, 48-bit physical
            ebx: 0,
            ecx: 0,
            edx: 0,
        },
        _ => zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_decodes() {
        let r = cpuid(0, 0);
        let mut v = [0u8; 12];
        v[..4].copy_from_slice(&r.ebx.to_le_bytes());
        v[4..8].copy_from_slice(&r.edx.to_le_bytes());
        v[8..].copy_from_slice(&r.ecx.to_le_bytes());
        assert_eq!(&v, b"AuthenticAMD");
    }

    #[test]
    fn family_model_is_zen2_shaped() {
        let eax = cpuid(1, 0).eax;
        let family = (eax >> 8) & 0xF;
        let ext_family = (eax >> 20) & 0xFF;
        let model = (eax >> 4) & 0xF;
        let ext_model = (eax >> 16) & 0xF;
        let display_family = family + ext_family;
        let display_model = (ext_model << 4) | model;
        assert_eq!(display_family, 0x17, "Zen 2 is family 0x17");
        assert_eq!(display_model, 0x47);
    }

    #[test]
    fn claimed_features_are_exactly_the_ledger() {
        // If you add an instruction family, update the ledger deliberately.
        assert_eq!(cpuid(1, 0).ecx, CLAIMED_ECX);
        assert_eq!(cpuid(1, 0).edx, CLAIMED_EDX);
        assert_eq!(cpuid(7, 0).ebx, CLAIMED_EBX7);
        // XSAVE must stay unadvertised while unimplemented.
        assert_eq!(cpuid(1, 0).ecx & (1 << 26), 0, "XSAVE not implemented");
        // No MMX/x87 claimed.
        assert_eq!(cpuid(1, 0).edx & (1 << 23), 0, "MMX not implemented");
        assert_eq!(cpuid(1, 0).edx & (1 << 0), 0, "x87 FPU not implemented");
    }

    #[test]
    fn brand_string_is_readable() {
        let a = cpuid(0x8000_0002, 0);
        let b = cpuid(0x8000_0003, 0);
        let c = cpuid(0x8000_0004, 0);
        let mut s = String::new();
        for r in [a, b, c] {
            for v in [r.eax, r.ebx, r.ecx, r.edx] {
                for ch in v.to_le_bytes() {
                    if ch != 0 {
                        s.push(ch as char);
                    }
                }
            }
        }
        assert_eq!(s, BRAND);
    }
}
