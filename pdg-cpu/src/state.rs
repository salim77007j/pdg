//! Architectural CPU state: the Zen 2 register file as seen by guest code.

use iced_x86::Register;

/// GPR array indices (x86-64 encoding order).
pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;
// R8..R15 use their own index values 8..=15.

/// RFLAGS bit positions (as defined by the architecture).
pub mod rflags {
    /// Bit 0 — carry.
    pub const CF: u64 = 1 << 0;
    /// Bit 1 — reserved, always 1.
    pub const RESERVED_1: u64 = 1 << 1;
    /// Bit 2 — parity.
    pub const PF: u64 = 1 << 2;
    /// Bit 3 — reserved.
    pub const RESERVED_3: u64 = 1 << 3;
    /// Bit 4 — auxiliary carry.
    pub const AF: u64 = 1 << 4;
    /// Bit 5 — reserved.
    pub const RESERVED_5: u64 = 1 << 5;
    /// Bit 6 — zero.
    pub const ZF: u64 = 1 << 6;
    /// Bit 7 — sign.
    pub const SF: u64 = 1 << 7;
    /// Bit 8 — trap.
    pub const TF: u64 = 1 << 8;
    /// Bit 9 — interrupt enable.
    pub const IF: u64 = 1 << 9;
    /// Bit 10 — direction.
    pub const DF: u64 = 1 << 10;
    /// Bit 11 — overflow.
    pub const OF: u64 = 1 << 11;
}

/// Segment descriptor-table register (GDTR or IDTR): 16-bit limit + 64-bit
/// base, as loaded/stored by `LGDT`/`LIDT`/`SGDT`/`SIDT` (10-byte memory
/// operand in 64-bit mode).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DescriptorTableRegister {
    /// Linear base of the table.
    pub base: u64,
    /// One less than the table length in bytes.
    pub limit: u16,
}

/// MXCSR (SSE control/status) bit positions.
pub mod mxcsr {
    /// Invalid-operation flag.
    pub const IE: u32 = 1 << 0;
    /// Denormal flag.
    pub const DE: u32 = 1 << 1;
    /// Divide-by-zero flag.
    pub const ZE: u32 = 1 << 2;
    /// Overflow flag.
    pub const OE: u32 = 1 << 3;
    /// Underflow flag.
    pub const UE: u32 = 1 << 4;
    /// Precision flag.
    pub const PE: u32 = 1 << 5;
    /// Denormals-are-zero.
    pub const DAZ: u32 = 1 << 6;
    /// Invalid-operation mask.
    pub const IM: u32 = 1 << 7;
    /// Denormal mask.
    pub const DM: u32 = 1 << 8;
    /// Divide-by-zero mask.
    pub const ZM: u32 = 1 << 9;
    /// Overflow mask.
    pub const OM: u32 = 1 << 10;
    /// Underflow mask.
    pub const UM: u32 = 1 << 11;
    /// Precision mask.
    pub const PM: u32 = 1 << 12;
    /// Flush-to-zero.
    pub const FZ: u32 = 1 << 15;
    /// Bits defined by the architecture (write mask).
    pub const ALL: u32 = 0xFFFF;
    /// Power-on value: all exceptions masked.
    pub const RESET: u32 = 0x1F80;
}

/// Model-specific register ids PDG knows by name (values from AMD manuals).
pub mod msr {
    /// Extended feature enables (NX, SCE, ...).
    pub const EFER: u32 = 0xC000_0080;
    /// Legacy syscall target CS.
    pub const STAR: u32 = 0xC000_0081;
    /// `syscall` target RIP.
    pub const LSTAR: u32 = 0xC000_0082;
    /// Compatibility-mode syscall target (unused by 64-bit OSes).
    pub const CSTAR: u32 = 0xC000_0083;
    /// `syscall` flags mask.
    pub const SFMASK: u32 = 0xC000_0084;
    /// FS base.
    pub const FS_BASE: u32 = 0xC000_0100;
    /// GS base.
    pub const GS_BASE: u32 = 0xC000_0101;
    /// Kernel-mode GS base (swapgs pair).
    pub const KERNEL_GS_BASE: u32 = 0xC000_0102;
}

/// Architectural register state of one logical processor (one hardware
/// thread of a Zen 2 core).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuState {
    /// General-purpose registers in x86-64 encoding order (RAX, RCX, RDX,
    /// RBX, RSP, RBP, RSI, RDI, R8..R15). Upper halves follow x86-64 rules:
    /// 32-bit writes zero-extend, 8/16-bit writes merge.
    pub gpr: [u64; 16],
    /// Instruction pointer.
    pub rip: u64,
    /// Flags register (see [`rflags`] for bit positions).
    pub rflags: u64,
    /// Vector registers, full 256-bit view: `ymm[n][0]` is the low 128 bits
    /// (the XMM register), `ymm[n][1]` the upper half (AVX). Non-VEX
    /// (legacy SSE) writes to XMMn zero the upper half, as on AVX-capable
    /// silicon like Zen 2.
    pub ymm: [[u128; 2]; 16],
    /// SSE control/status register.
    pub mxcsr: u32,
    /// Global descriptor-table register.
    pub gdtr: DescriptorTableRegister,
    /// Interrupt descriptor-table register.
    pub idtr: DescriptorTableRegister,
    /// FS segment base (TLS pointer convention on x86-64).
    pub fs_base: u64,
    /// GS segment base (kernel per-CPU convention; see `swapgs`).
    pub gs_base: u64,
    /// Sparse model-specific register file (id → value); FS/GS bases are
    /// additionally mirrored into [`CpuState::fs_base`]/[`gs_base`] on
    /// access for the address-generation path.
    pub msrs: std::collections::BTreeMap<u32, u64>,
    /// Control register CR0 (PE/PG modelled as permanently enabled: PDG
    /// runs the guest in long mode from reset).
    pub cr0: u64,
    /// Control register CR2 — page-fault linear address.
    pub cr2: u64,
    /// Control register CR3 — page-table root (kept in sync with the
    /// platform layer's MMU through `MemoryAccess::reload_cr3`).
    pub cr3: u64,
    /// Control register CR4 (PAE/PGE/OSXSR bits mirrored for guest reads).
    pub cr4: u64,
    /// Virtual timestamp counter, advanced as guest instructions retire.
    /// `rdtsc` reads it; the JIT bumps it per block using its instruction
    /// count. Monotonic within a run, not host-time-anchored (documented in
    /// the Phase-2 report).
    pub tsc: u64,
}

impl Default for CpuState {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// Power-on MSR file: EFER reflects the enabled long-mode configuration
/// (SCE | LME | LMA | NXE).
fn seeded_msrs() -> std::collections::BTreeMap<u32, u64> {
    [(msr::EFER, 0xD01u64)].into_iter().collect()
}

impl CpuState {
    /// Power-on state: RIP at `entry`, RSP at `stack_top`, RFLAGS = bit 1 set.
    pub fn new(entry: u64, stack_top: u64) -> Self {
        let mut gpr = [0u64; 16];
        gpr[RSP] = stack_top;
        Self {
            gpr,
            rip: entry,
            rflags: rflags::RESERVED_1,
            ymm: [[0; 2]; 16],
            mxcsr: mxcsr::RESET,
            gdtr: DescriptorTableRegister::default(),
            idtr: DescriptorTableRegister::default(),
            fs_base: 0,
            gs_base: 0,
            msrs: seeded_msrs(),
            // Long mode from reset: PE | PG (+MP); PAE, PGE, OSFXSR, OSXMMEXCPT.
            cr0: 0x8000_0013,
            cr2: 0,
            cr3: 0,
            cr4: 0x6A0,
            tsc: 0,
        }
    }

    // ----- vector register views --------------------------------------------

    /// Read XMMn (low 128 bits of the vector register) by iced register id.
    /// Returns `None` for non-XMM ids.
    pub fn xmm(&self, r: Register) -> Option<u128> {
        Self::xmm_index(r).map(|i| self.ymm[i][0])
    }

    /// Legacy-SSE write of XMMn: sets the low 128 bits and zeroes the upper
    /// half (AVX register-zeroing rule).
    pub fn set_xmm(&mut self, r: Register, value: u128) -> Option<()> {
        let i = Self::xmm_index(r)?;
        self.ymm[i] = [value, 0];
        Some(())
    }

    /// Full 256-bit view of YMMn as `[low, high]`.
    pub fn ymm_halves(&self, r: Register) -> Option<[u128; 2]> {
        Self::xmm_index(r).map(|i| self.ymm[i])
    }

    /// Full 256-bit write of YMMn.
    pub fn set_ymm_halves(&mut self, r: Register, value: [u128; 2]) -> Option<()> {
        let i = Self::xmm_index(r)?;
        self.ymm[i] = value;
        Some(())
    }

    /// Raw YMM write by index (used by the SIMD kernels).
    pub fn set_ymm_idx(&mut self, i: usize, value: [u128; 2]) {
        self.ymm[i] = value;
    }

    /// Raw YMM read by index.
    pub fn ymm_idx(&self, i: usize) -> [u128; 2] {
        self.ymm[i]
    }

    /// Map an iced XMM/YMM register id to its index 0..16.
    pub fn xmm_index(r: Register) -> Option<usize> {
        use Register::*;
        Some(match r {
            XMM0 => 0,
            XMM1 => 1,
            XMM2 => 2,
            XMM3 => 3,
            XMM4 => 4,
            XMM5 => 5,
            XMM6 => 6,
            XMM7 => 7,
            XMM8 => 8,
            XMM9 => 9,
            XMM10 => 10,
            XMM11 => 11,
            XMM12 => 12,
            XMM13 => 13,
            XMM14 => 14,
            XMM15 => 15,
            YMM0 => 0,
            YMM1 => 1,
            YMM2 => 2,
            YMM3 => 3,
            YMM4 => 4,
            YMM5 => 5,
            YMM6 => 6,
            YMM7 => 7,
            YMM8 => 8,
            YMM9 => 9,
            YMM10 => 10,
            YMM11 => 11,
            YMM12 => 12,
            YMM13 => 13,
            YMM14 => 14,
            YMM15 => 15,
            _ => return Option::None,
        })
    }

    // ----- flag accessors -------------------------------------------------

    /// Carry flag.
    pub fn cf(&self) -> bool {
        self.rflags & rflags::CF != 0
    }
    fn set_cf(&mut self, v: bool) {
        self.set_flag(rflags::CF, v);
    }
    /// Parity flag.
    pub fn pf(&self) -> bool {
        self.rflags & rflags::PF != 0
    }
    fn set_pf(&mut self, v: bool) {
        self.set_flag(rflags::PF, v);
    }
    /// Auxiliary carry flag.
    pub fn af(&self) -> bool {
        self.rflags & rflags::AF != 0
    }
    fn set_af(&mut self, v: bool) {
        self.set_flag(rflags::AF, v);
    }
    /// Zero flag.
    pub fn zf(&self) -> bool {
        self.rflags & rflags::ZF != 0
    }
    fn set_zf(&mut self, v: bool) {
        self.set_flag(rflags::ZF, v);
    }
    /// Sign flag.
    pub fn sf(&self) -> bool {
        self.rflags & rflags::SF != 0
    }
    fn set_sf(&mut self, v: bool) {
        self.set_flag(rflags::SF, v);
    }
    /// Direction flag.
    pub fn df(&self) -> bool {
        self.rflags & rflags::DF != 0
    }
    /// Overflow flag.
    pub fn of(&self) -> bool {
        self.rflags & rflags::OF != 0
    }
    fn set_of(&mut self, v: bool) {
        self.set_flag(rflags::OF, v);
    }

    fn set_flag(&mut self, bit: u64, v: bool) {
        if v {
            self.rflags |= bit;
        } else {
            self.rflags &= !bit;
        }
    }

    // ----- flag computation -----------------------------------------------

    /// Flags after `a + b + carry_in` at operand width `w` bytes
    /// (1, 2, 4 or 8). `result` is the wrapped sum. Operands must be masked
    /// to `w` bytes by the caller.
    pub fn flags_add(&mut self, a: u64, b: u64, carry_in: u64, result: u64, w: u32) {
        let bits = w * 8;
        let sign = 1u64 << (bits - 1);
        let full = a as u128 + b as u128 + carry_in as u128;
        let limit = (1u128 << bits) - 1;
        let res = result & mask(w);
        self.set_cf(full > limit);
        self.set_zf(res == 0);
        self.set_sf(res & sign != 0);
        self.set_of(!(a ^ b) & (a ^ res) & sign != 0);
        self.set_af((a ^ b ^ res) & 0x10 != 0);
        self.set_pf(parity(res as u8));
    }

    /// Flags after `a - b - borrow_in` at operand width `w` bytes.
    pub fn flags_sub(&mut self, a: u64, b: u64, borrow_in: u64, result: u64, w: u32) {
        let bits = w * 8;
        let sign = 1u64 << (bits - 1);
        let res = result & mask(w);
        self.set_cf(b > a || (b == a && borrow_in == 1));
        self.set_zf(res == 0);
        self.set_sf(res & sign != 0);
        self.set_of((a ^ b) & (a ^ res) & sign != 0);
        self.set_af((a ^ b ^ res) & 0x10 != 0);
        self.set_pf(parity(res as u8));
    }

    /// Flags after AND/OR/XOR/TEST at width `w`: CF = OF = 0, AF undefined
    /// (cleared here), ZF/SF/PF from the result.
    pub fn flags_logic(&mut self, result: u64, w: u32) {
        let bits = w * 8;
        let sign = 1u64 << (bits - 1);
        let res = result & mask(w);
        self.set_cf(false);
        self.set_of(false);
        self.set_af(false);
        self.set_zf(res == 0);
        self.set_sf(res & sign != 0);
        self.set_pf(parity(res as u8));
    }

    // ----- register file --------------------------------------------------

    /// Read a GPR view by iced-x86 register id (`AL`, `AH`, `AX`, `EAX`,
    /// `RAX`, ...). Returns `None` for non-GPR registers.
    pub fn reg(&self, r: Register) -> Option<u64> {
        let (index, size, high) = decode_reg(r)?;
        let full = self.gpr[index];
        Some(if high {
            (full >> 8) & 0xFF
        } else {
            match size {
                1 => full & 0xFF,
                2 => full & 0xFFFF,
                4 => full & 0xFFFF_FFFF,
                _ => full,
            }
        })
    }

    /// Write a GPR view. Follows x86-64 register-write semantics: 64-bit
    /// writes replace, 32-bit writes zero-extend, 8/16-bit writes merge.
    pub fn set_reg(&mut self, r: Register, value: u64) -> Option<()> {
        let (index, size, high) = decode_reg(r)?;
        if high {
            self.gpr[index] = (self.gpr[index] & !0xFF00) | ((value & 0xFF) << 8);
        } else {
            self.gpr[index] = match size {
                8 => value,
                4 => value & 0xFFFF_FFFF,
                2 => (self.gpr[index] & !0xFFFF) | (value & 0xFFFF),
                1 => (self.gpr[index] & !0xFF) | (value & 0xFF),
                _ => return None,
            };
        }
        Some(())
    }

    /// Read RSP (commonly needed by the stack helpers).
    pub fn rsp(&self) -> u64 {
        self.gpr[RSP]
    }
}

/// Even-parity of a byte (PF is set when the low byte has even popcount).
#[inline]
fn parity(v: u8) -> bool {
    v.count_ones() % 2 == 0
}

/// Bit mask for an operand `w` bytes wide.
#[inline]
pub(crate) fn mask(w: u32) -> u64 {
    match w {
        1 => 0xFF,
        2 => 0xFFFF,
        4 => 0xFFFF_FFFF,
        8 => u64::MAX,
        _ => 0,
    }
}

/// Decompose an iced-x86 GPR register id into (gpr index, size in bytes,
/// is-high-byte). Returns `None` for non-GPR ids.
pub fn decode_reg(r: Register) -> Option<(usize, u32, bool)> {
    use Register::*;
    Some(match r {
        AL => (0, 1, false),
        AH => (0, 1, true),
        AX => (0, 2, false),
        EAX => (0, 4, false),
        RAX => (0, 8, false),
        CL => (1, 1, false),
        CH => (1, 1, true),
        CX => (1, 2, false),
        ECX => (1, 4, false),
        RCX => (1, 8, false),
        DL => (2, 1, false),
        DH => (2, 1, true),
        DX => (2, 2, false),
        EDX => (2, 4, false),
        RDX => (2, 8, false),
        BL => (3, 1, false),
        BH => (3, 1, true),
        BX => (3, 2, false),
        EBX => (3, 4, false),
        RBX => (3, 8, false),
        SPL => (4, 1, false),
        SP => (4, 2, false),
        ESP => (4, 4, false),
        RSP => (4, 8, false),
        BPL => (5, 1, false),
        BP => (5, 2, false),
        EBP => (5, 4, false),
        RBP => (5, 8, false),
        SIL => (6, 1, false),
        SI => (6, 2, false),
        ESI => (6, 4, false),
        RSI => (6, 8, false),
        DIL => (7, 1, false),
        DI => (7, 2, false),
        EDI => (7, 4, false),
        RDI => (7, 8, false),
        R8L => (8, 1, false),
        R8W => (8, 2, false),
        R8D => (8, 4, false),
        R8 => (8, 8, false),
        R9L => (9, 1, false),
        R9W => (9, 2, false),
        R9D => (9, 4, false),
        R9 => (9, 8, false),
        R10L => (10, 1, false),
        R10W => (10, 2, false),
        R10D => (10, 4, false),
        R10 => (10, 8, false),
        R11L => (11, 1, false),
        R11W => (11, 2, false),
        R11D => (11, 4, false),
        R11 => (11, 8, false),
        R12L => (12, 1, false),
        R12W => (12, 2, false),
        R12D => (12, 4, false),
        R12 => (12, 8, false),
        R13L => (13, 1, false),
        R13W => (13, 2, false),
        R13D => (13, 4, false),
        R13 => (13, 8, false),
        R14L => (14, 1, false),
        R14W => (14, 2, false),
        R14D => (14, 4, false),
        R14 => (14, 8, false),
        R15L => (15, 1, false),
        R15W => (15, 2, false),
        R15D => (15, 4, false),
        R15 => (15, 8, false),
        _ => return Option::None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_views() {
        let mut s = CpuState::new(0, 0);
        s.gpr[RAX] = 0x1122_3344_5566_7788;
        assert_eq!(s.reg(Register::AL), Some(0x88));
        assert_eq!(s.reg(Register::AH), Some(0x77));
        assert_eq!(s.reg(Register::AX), Some(0x7788));
        assert_eq!(s.reg(Register::EAX), Some(0x5566_7788));
        assert_eq!(s.reg(Register::RAX), Some(0x1122_3344_5566_7788));
        assert_eq!(s.reg(Register::RIP), None);
    }

    #[test]
    fn reg_write_semantics() {
        let mut s = CpuState::new(0, 0);
        s.set_reg(Register::RAX, u64::MAX).unwrap();
        // 32-bit write zero-extends.
        s.set_reg(Register::EAX, 0x1).unwrap();
        assert_eq!(s.gpr[RAX], 1);
        // 16-bit write merges.
        s.set_reg(Register::RAX, 0xAAAA_BBBB_CCCC_DDDD).unwrap();
        s.set_reg(Register::AX, 0x1111).unwrap();
        assert_eq!(s.gpr[RAX], 0xAAAA_BBBB_CCCC_1111);
        // 8-bit write merges; high byte merges into bits 8..15.
        s.set_reg(Register::AL, 0xEE).unwrap();
        assert_eq!(s.gpr[RAX], 0xAAAA_BBBB_CCCC_11EE);
        s.set_reg(Register::AH, 0xDD).unwrap();
        assert_eq!(s.gpr[RAX], 0xAAAA_BBBB_CCCC_DDEE);
    }

    #[test]
    fn add_flags_carry_and_zero() {
        let mut s = CpuState::new(0, 0);
        s.flags_add(1, u64::MAX - 1, 0, 0, 8); // 1 + (2^64-2) = 2^64-1 (no CF)
        assert!(!s.cf());
        s.flags_add(1, u64::MAX, 0, 0, 8); // wraps → CF, ZF
        assert!(s.cf());
        assert!(s.zf());
        assert!(!s.sf());
        assert!(!s.of());
    }

    #[test]
    fn add_flags_overflow_and_sign() {
        let mut s = CpuState::new(0, 0);
        s.flags_add(i64::MAX as u64, 1, 0, (i64::MAX as u64).wrapping_add(1), 8);
        assert!(s.of());
        assert!(s.sf());
        assert!(!s.zf());
    }

    #[test]
    fn add_flags_width_aware() {
        let mut s = CpuState::new(0, 0);
        // 0xFF + 1 in 8-bit arithmetic: carry, zero, no overflow.
        s.flags_add(0xFF, 1, 0, 0, 1);
        assert!(s.cf());
        assert!(s.zf());
        assert!(!s.of());
        // Same operands at 32-bit width: 0x100, no carry, SF=0, PF=0.
        s.flags_add(0xFF, 1, 0, 0x100, 4);
        assert!(!s.cf());
        assert!(!s.zf());
        assert!(!s.sf());
    }

    #[test]
    fn sub_flags_borrow() {
        let mut s = CpuState::new(0, 0);
        s.flags_sub(5, 7, 0, 5u64.wrapping_sub(7), 8);
        assert!(s.cf());
        assert!(s.sf());
        assert!(!s.zf());
        s.flags_sub(7, 7, 0, 0, 8);
        assert!(!s.cf());
        assert!(s.zf());
    }

    #[test]
    fn logic_flags_clear_cf_of() {
        let mut s = CpuState::new(0, 0);
        s.set_flag(rflags::CF, true);
        s.set_flag(rflags::OF, true);
        s.flags_logic(0x80, 1);
        assert!(!s.cf());
        assert!(!s.of());
        assert!(s.sf());
        assert!(!s.zf());
        assert!(!s.pf()); // 0x80 has one bit set → odd parity → PF clear
    }

    #[test]
    fn power_on_state() {
        let s = CpuState::new(0x1000, 0x8000);
        assert_eq!(s.rip, 0x1000);
        assert_eq!(s.rsp(), 0x8000);
        assert_eq!(s.rflags, rflags::RESERVED_1);
    }
}
