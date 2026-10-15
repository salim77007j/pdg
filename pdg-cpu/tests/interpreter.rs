//! Integration tests for the Phase-1 interpreter.
//!
//! Every test assembles a tiny x86-64 payload by hand (byte-level) and runs
//! it on a `FlatMemory`, then asserts architectural results: register
//! values, flag bits, control flow and fault behaviour.

use pdg_cpu::{CpuState, Fault, FlatMemory, Interpreter, MemoryAccess, RunOutcome, SyscallHook};

const STACK_TOP: u64 = 0x8000;
const IMAGE_LEN: usize = 0x10000;

/// Assemble a run: code at 0, stack top at STACK_TOP.
fn run_code(code: &[u8]) -> (CpuState, RunOutcome, FlatMemory) {
    run_code_with_state(code, CpuState::new(0, STACK_TOP))
}

fn run_code_with_state(code: &[u8], state: CpuState) -> (CpuState, RunOutcome, FlatMemory) {
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, code);
    let mut cpu = Interpreter::new(state);
    let outcome = cpu.run(&mut mem, 10_000);
    (cpu.state, outcome, mem)
}

fn halted(outcome: &RunOutcome) -> bool {
    matches!(outcome, RunOutcome::Halted { .. })
}

/// Doubles RAX (a stand-in HLE syscall).
struct Doubler;

impl SyscallHook for Doubler {
    fn on_syscall(&mut self, cpu: &mut CpuState, _mem: &mut dyn MemoryAccess) -> Result<(), Fault> {
        let v = cpu.reg(pdg_cpu::iced_x86::Register::RAX).expect("rax");
        cpu.set_reg(pdg_cpu::iced_x86::Register::RAX, v.wrapping_mul(2))
            .expect("rax");
        Ok(())
    }
}

fn run_code_with_hook(code: &[u8], hook: Doubler) -> (CpuState, RunOutcome) {
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, code);
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    cpu.set_syscall_hook(Box::new(hook));
    let outcome = cpu.run(&mut mem, 10_000);
    (cpu.state, outcome)
}

#[test]
fn mov_imm_reg_forms() {
    // mov rax, 0x12345678 (C7 with imm32)
    let (s, o, _) = run_code(&[0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12, 0xF4]);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0x1234_5678));

    // mov eax, 0x12345678 (B8: zero-extends into RAX)
    let (s, o, _) = run_code(&[0xB8, 0x78, 0x56, 0x34, 0x12, 0xF4]);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0x1234_5678));

    // mov ax, 0x1234 merges into the low 16 bits only
    let state = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[pdg_cpu::state::RAX] = u64::MAX;
        s
    };
    let (s, o, _) = run_code_with_state(&[0x66, 0xB8, 0x34, 0x12, 0xF4], state);
    assert!(halted(&o));
    assert_eq!(
        s.reg(pdg_cpu::iced_x86::Register::RAX),
        Some(0xFFFF_FFFF_FFFF_1234)
    );
}

#[test]
fn imm8_sign_extension_in_64bit_ops() {
    // mov rax, 0 ; add rax, -5 → rax = 0xFFFF...FFFB
    let code = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x83, 0xC0, 0xFB, // add rax, -5 (sign-extended imm8)
        0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(
        s.reg(pdg_cpu::iced_x86::Register::RAX),
        Some(0xFFFF_FFFF_FFFF_FFFB)
    );
    assert!(s.sf());
    assert!(!s.zf());
}

#[test]
fn add_flags_carry_zero() {
    // mov rax, 1 ; add rax, -1 → 0 with CF=ZF=1, OF=0
    let code = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x83, 0xC0, 0xFF, // add rax, -1
        0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0));
    assert!(s.cf());
    assert!(s.zf());
    assert!(!s.of());
    assert!(!s.sf());
}

#[test]
fn sub_flags_borrow_and_sign() {
    // mov rax, 5 ; sub rax, 7 → -2: CF=1, SF=1, ZF=0
    let code = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xE8, 0x07, 0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(
        s.reg(pdg_cpu::iced_x86::Register::RAX),
        Some(u64::from(5u32).wrapping_sub(7))
    );
    assert!(s.cf());
    assert!(s.sf());
    assert!(!s.zf());
}

#[test]
fn cmp_has_no_writeback() {
    // mov rax, 9 ; cmp rax, 9 → ZF=1 but RAX unchanged
    let code = [
        0x48, 0xC7, 0xC0, 0x09, 0x00, 0x00, 0x00, 0x48, 0x83, 0xF8, 0x09, 0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(9));
    assert!(s.zf());
}

#[test]
fn test_instruction_clears_cf_of() {
    // mov rax, 0xF0F0 ; test rax, 0x0FF0 → nonzero, flags cleared
    let code = [
        0x48, 0xC7, 0xC0, 0xF0, 0xF0, 0x00, 0x00, 0x48, 0xA9, 0xF0, 0x0F, 0x00,
        0x00, // test rax, 0x0FF0
        0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0xF0F0));
    assert!(!s.zf());
    assert!(!s.cf());
    assert!(!s.of());
}

#[test]
fn eight_bit_ops_set_flags_from_width() {
    // mov al, 0xFF ; add al, 1 → 0x00 with CF=1, ZF=1 (8-bit semantics)
    let code = [0xB0, 0xFF, 0x04, 0x01, 0xF4];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::AL), Some(0));
    assert!(s.cf());
    assert!(s.zf());
    assert!(!s.of());
}

#[test]
fn push_pop_roundtrip() {
    // mov rbx, 0x1122 ; push rbx ; pop rdi
    let code = [
        0x48, 0xC7, 0xC3, 0x22, 0x11, 0x00, 0x00, // mov rbx, 0x1122
        0x53, // push rbx
        0x5F, // pop rdi
        0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RDI), Some(0x1122));
    assert_eq!(s.rsp(), STACK_TOP);
}

#[test]
fn call_ret_returns_to_next_instruction() {
    // 0x0: call 0xA (E8 rel32=+5)  | 0x5: nop | 0x6: hlt | 0xA: ret
    let code = [
        0xE8, 0x05, 0x00, 0x00, 0x00, //
        0x90, //
        0xF4, //
        0x90, 0x90, 0x90, // padding 0x7..0x9
        0xC3, // ret at 0xA
    ];
    let (s, o, _) = run_code(&code);
    match o {
        RunOutcome::Halted { instructions, rip } => {
            assert_eq!(rip, 6);
            assert_eq!(instructions, 4); // call, ret, nop, hlt
        }
        other => panic!("expected Halted, got {other:?}"),
    }
    assert_eq!(s.rsp(), STACK_TOP);
}

#[test]
fn jne_taken_and_not_taken() {
    // taken: mov ecx, 0 ; dec rcx (→ -1, ZF=0) ; jne +2 → 0xC
    let taken = [
        0xB9, 0x00, 0x00, 0x00, 0x00, // 0x00: mov ecx, 0
        0x48, 0xFF, 0xC9, // 0x05: dec rcx
        0x75, 0x02, // 0x08: jne → 0x0C
        0xF4, // 0x0A: hlt (fall-through)
        0x90, // 0x0B: padding
        0xF4, // 0x0C: hlt (jump target)
    ];
    let (_, o, _) = run_code(&taken);
    match o {
        RunOutcome::Halted { rip, .. } => assert_eq!(rip, 0xC),
        other => panic!("expected Halted, got {other:?}"),
    }

    // not taken: mov ecx, 1 ; dec rcx (→ 0, ZF=1) ; jne +2 → falls through
    let not_taken = [
        0xB9, 0x01, 0x00, 0x00, 0x00, // 0x00: mov ecx, 1
        0x48, 0xFF, 0xC9, // 0x05: dec rcx
        0x75, 0x02, // 0x08: jne (not taken)
        0xF4, // 0x0A: hlt (fall-through)
        0xF4, // 0x0B
    ];
    let (_, o, _) = run_code(&not_taken);
    match o {
        RunOutcome::Halted { rip, .. } => assert_eq!(rip, 0xA),
        other => panic!("expected Halted, got {other:?}"),
    }
}

#[test]
fn counting_loop_retires_expected_instructions() {
    // mov ecx, 16 ; loop { dec rcx ; jne loop } ; hlt
    let code = [
        0xB9, 0x10, 0x00, 0x00, 0x00, // 0x0: mov ecx, 16
        0x48, 0xFF, 0xC9, // 0x5: dec rcx
        0x75, 0xFB, // 0x8: jne 0x5
        0xF4, // 0xA: hlt
    ];
    let (s, o, _) = run_code(&code);
    match o {
        RunOutcome::Halted { instructions, .. } => assert_eq!(instructions, 1 + 2 * 16 + 1),
        other => panic!("expected Halted, got {other:?}"),
    }
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::ECX), Some(0));
    assert!(s.zf());
}

#[test]
fn lea_computes_effective_address() {
    let state = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[pdg_cpu::state::RDX] = 0x1000;
        s
    };
    // lea rax, [rdx + 0x10]
    let (s, o, _) = run_code_with_state(&[0x48, 0x8D, 0x42, 0x10, 0xF4], state);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0x1010));
}

#[test]
fn shl_moves_bits_and_sets_cf() {
    let state = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[pdg_cpu::state::RAX] = 1;
        s
    };
    // shl rax, 63
    let (s, o, _) = run_code_with_state(&[0x48, 0xC1, 0xE0, 0x3F, 0xF4], state);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(1u64 << 63));
    assert!(s.sf());
    assert!(!s.zf());
    // CF = bit (64-63) of the original value = bit 1 of 1 = 0.
    assert!(!s.cf());
}

#[test]
fn shr_and_arith_sar() {
    let state = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[pdg_cpu::state::RAX] = 1u64 << 63;
        s
    };
    // shr rax, 63 → 1
    let (s, o, _) = run_code_with_state(&[0x48, 0xC1, 0xE8, 0x3F, 0xF4], state);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(1));

    let state = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[pdg_cpu::state::RAX] = 1u64 << 63;
        s
    };
    // sar rax, 63 → all ones (sign saturation)
    let (s, o, _) = run_code_with_state(&[0x48, 0xC1, 0xF8, 0x3F, 0xF4], state);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(u64::MAX));
    assert!(s.sf());
}

#[test]
fn xchg_swaps_registers() {
    // mov rax, 1 ; mov rcx, 2 ; xchg rax, rcx
    let code = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, //
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00, //
        0x48, 0x91, // xchg rax, rcx
        0xF4,
    ];
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(2));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RCX), Some(1));
}

#[test]
fn moffs_store_and_absolute_load() {
    // mov rax, 0xAABB ; mov [0x100], rax ; mov rcx, [0x100]
    let code = [
        0x48, 0xC7, 0xC0, 0xBB, 0xAA, 0x00, 0x00, // mov rax, 0xAABB
        0x48, 0xA3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov [0x100], rax
        0x48, 0x8B, 0x0C, 0x25, 0x00, 0x01, 0x00, 0x00, // mov rcx, [0x100]
        0xF4,
    ];
    let (s, o, mut mem) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RCX), Some(0xAABB));
    assert_eq!(mem.read(8, 0x100).unwrap(), 0xAABB);
}

#[test]
fn rip_relative_addressing() {
    // 0x0: mov rax, [rip + 0x10]  (target = 0x7 + 0x10 = 0x17)
    // 0x7: hlt
    // 0x17: qword 0x1234
    let mut code = vec![
        0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, // mov rax, [rip+0x10]
        0xF4, // hlt
    ];
    code.resize(0x17, 0x90);
    code.extend_from_slice(&0x1234u64.to_le_bytes());
    let (s, o, _) = run_code(&code);
    assert!(halted(&o));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(0x1234));
}

#[test]
fn syscall_hook_receives_and_mutates_state() {
    // mov eax, 21 ; syscall → hook doubles RAX → 42
    let code = [
        0xB8, 0x15, 0x00, 0x00, 0x00, // mov eax, 21
        0x0F, 0x05, // syscall
        0xF4,
    ];
    let (s, o) = run_code_with_hook(&code, Doubler);
    assert!(matches!(o, RunOutcome::Halted { .. }));
    assert_eq!(s.reg(pdg_cpu::iced_x86::Register::RAX), Some(42));
}

#[test]
fn syscall_without_hook_is_reported() {
    let (_, o, _) = run_code(&[0x0F, 0x05, 0xF4]);
    match o {
        RunOutcome::Fault(Fault::UnhandledSyscall { rip }) => assert_eq!(rip, 0),
        other => panic!("expected UnhandledSyscall, got {other:?}"),
    }
}

#[test]
fn unsupported_instruction_is_ud() {
    // ud2 (0F 0B) — valid encoding, outside the Phase-1 subset.
    let (_, o, _) = run_code(&[0x0F, 0x0B]);
    match o {
        RunOutcome::Fault(Fault::InvalidOpcode { rip, .. }) => assert_eq!(rip, 0),
        other => panic!("expected InvalidOpcode, got {other:?}"),
    }
}

#[test]
fn out_of_bounds_read_is_host_error() {
    // mov rax, [0x20000] against a 0x10000 image.
    let code = [0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00, 0xF4]; // mov rax, [0x20000]
    let (_, o, _) = run_code(&code);
    assert!(matches!(o, RunOutcome::Fault(Fault::HostError(_))));
}

#[test]
fn fetch_stops_near_end_of_image() {
    // Code placed at the last two bytes: nop + hlt both fit.
    let mut mem = FlatMemory::new(0x100);
    mem.load(0xFE, &[0x90, 0xF4]);
    let mut cpu = Interpreter::new(CpuState::new(0xFE, 0x80));
    let o = cpu.run(&mut mem, 100);
    assert!(matches!(o, RunOutcome::Halted { .. }));
}

mod prop {
    use super::*;
    use proptest::prelude::*;

    fn run_eax_op(a: u32, b: u32, opcode: u8) -> (u32, CpuState) {
        let mut code = Vec::new();
        code.push(0xB8); // mov eax, a
        code.extend_from_slice(&a.to_le_bytes());
        code.push(opcode); // 05 = add eax, imm32 ; 2D = sub eax, imm32
        code.extend_from_slice(&b.to_le_bytes());
        code.push(0xF4);
        let (s, o, _) = run_code(&code);
        assert!(matches!(o, RunOutcome::Halted { .. }));
        (s.reg(pdg_cpu::iced_x86::Register::EAX).unwrap() as u32, s)
    }

    proptest! {
        #[test]
        fn add32_matches_reference_model(a in any::<u32>(), b in any::<u32>()) {
            let (res, s) = run_eax_op(a, b, 0x05);
            prop_assert_eq!(res, a.wrapping_add(b));
            prop_assert_eq!(s.zf(), res == 0);
            prop_assert_eq!(s.sf(), res >> 31 == 1);
            prop_assert_eq!(s.cf(), (a as u64 + b as u64) > u32::MAX as u64);
            let of = (!(a ^ b) & (a ^ res) & 0x8000_0000) != 0;
            prop_assert_eq!(s.of(), of);
        }

        #[test]
        fn sub32_matches_reference_model(a in any::<u32>(), b in any::<u32>()) {
            let (res, s) = run_eax_op(a, b, 0x2D);
            prop_assert_eq!(res, a.wrapping_sub(b));
            prop_assert_eq!(s.zf(), res == 0);
            prop_assert_eq!(s.sf(), res >> 31 == 1);
            prop_assert_eq!(s.cf(), b > a);
            let of = ((a ^ b) & (a ^ res) & 0x8000_0000) != 0;
            prop_assert_eq!(s.of(), of);
        }
    }
}
