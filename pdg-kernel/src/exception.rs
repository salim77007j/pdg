//! x86-64 architectural exception vector numbers.

/// Exception/interrupt vector numbers (Intel SDMA Volume 3, Table 6-1).
pub mod vector {
    /// Divide error (#DE).
    pub const DIVIDE_ERROR: u8 = 0;
    /// Debug (#DB).
    pub const DEBUG: u8 = 1;
    /// NMI.
    pub const NMI: u8 = 2;
    /// Breakpoint (#BP).
    pub const BREAKPOINT: u8 = 3;
    /// Overflow (#OF).
    pub const OVERFLOW: u8 = 4;
    /// Bound range exceeded (#BR).
    pub const BOUND_RANGE: u8 = 5;
    /// Invalid opcode (#UD).
    pub const INVALID_OPCODE: u8 = 6;
    /// Device not available (#NM).
    pub const DEVICE_NOT_AVAILABLE: u8 = 7;
    /// Double fault (#DF).
    pub const DOUBLE_FAULT: u8 = 8;
    /// Invalid TSS (#TS).
    pub const INVALID_TSS: u8 = 10;
    /// Segment not present (#NP).
    pub const SEGMENT_NOT_PRESENT: u8 = 11;
    /// Stack-segment fault (#SS).
    pub const STACK_SEGMENT: u8 = 12;
    /// General protection (#GP).
    pub const GENERAL_PROTECTION: u8 = 13;
    /// Page fault (#PF).
    pub const PAGE_FAULT: u8 = 14;
    /// x87 FPU error (#MF).
    pub const X87_FPU: u8 = 16;
    /// Alignment check (#AC).
    pub const ALIGNMENT_CHECK: u8 = 17;
    /// Machine check (#MC).
    pub const MACHINE_CHECK: u8 = 18;
    /// SIMD FP exception (#XM).
    pub const SIMD_FP: u8 = 19;
    /// Virtualization exception (#VE).
    pub const VIRTUALIZATION: u8 = 20;
}

/// Frame pushed by the CPU when delivering an exception through the IDT
/// (used from Phase 2 onward).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterruptFrame {
    /// RIP to resume at.
    pub rip: u64,
    /// Code segment.
    pub cs: u64,
    /// RFLAGS at the fault point.
    pub rflags: u64,
    /// RSP at the fault point.
    pub rsp: u64,
}

#[cfg(test)]
mod tests {
    #[test]
    fn vectors_match_architecture() {
        assert_eq!(super::vector::PAGE_FAULT, 14);
        assert_eq!(super::vector::GENERAL_PROTECTION, 13);
        assert_eq!(super::vector::INVALID_OPCODE, 6);
        assert_eq!(super::vector::DIVIDE_ERROR, 0);
    }
}
