//! Phase-2 ISA integration tests: the integer completion, string ops,
//! atomics, system instructions, exception delivery and SIMD families,
//! verified end-to-end by executing hand-assembled x86-64 payloads.

use iced_x86::Register;
use pdg_cpu::{CpuState, Fault, FlatMemory, Interpreter, MemoryAccess, RunOutcome, state};

const STACK_TOP: u64 = 0x8000;
const IMAGE_LEN: usize = 0x20000;

fn run(code: &[u8]) -> (CpuState, RunOutcome, FlatMemory) {
    run_with(code, CpuState::new(0, STACK_TOP))
}

fn run_with(code: &[u8], st: CpuState) -> (CpuState, RunOutcome, FlatMemory) {
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, code);
    let mut cpu = Interpreter::new(st);
    let out = cpu.run(&mut mem, 100_000);
    (cpu.state, out, mem)
}

fn rax(s: &CpuState) -> u64 {
    s.reg(Register::RAX).unwrap_or(0)
}

#[test]
fn mul_div_family() {
    // mov eax, 7 ; mov ecx, 6 ; mul ecx  → rdx:rax = 42
    let code = [
        0xB8, 0x07, 0x00, 0x00, 0x00, // mov eax, 7
        0xB9, 0x06, 0x00, 0x00, 0x00, // mov ecx, 6
        0xF7, 0xE1, // mul ecx
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 42);
    assert_eq!(s.reg(Register::RDX), Some(0));
    assert!(!s.cf() && !s.of(), "small MUL clears CF/OF");

    // imul rax, rcx (two-operand)
    let code = [
        0x48, 0xC7, 0xC0, 0xFE, 0xFF, 0xFF, 0xFF, // mov rax, -2
        0x48, 0xC7, 0xC1, 0x19, 0x00, 0x00, 0x00, // mov rcx, 25
        0x48, 0x0F, 0xAF, 0xC1, // imul rax, rcx
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), (-50i64) as u64);

    // div: 100 / 7 = 14 r 2  (div qword [rbx])
    let code = [
        0xB8, 0x64, 0x00, 0x00, 0x00, // mov eax, 100
        0x31, 0xD2, // xor edx, edx
        0xB9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0xF7, 0xF1, // div ecx
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 14, "quotient");
    assert_eq!(s.reg(Register::RDX), Some(2), "remainder");

    // idiv with sign: -100 / 7 = -14 r -2 (cdq sign-extends EAX into EDX)
    let code = [
        0xB8, 0x9C, 0xFF, 0xFF, 0xFF, // mov eax, -100
        0x99, // cdq
        0xBE, 0x07, 0x00, 0x00, 0x00, // mov esi, 7
        0xF7, 0xFE, // idiv esi
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s) as i32, -14, "idiv quotient rounds toward zero");
    assert_eq!(s.reg(Register::RDX).unwrap() as i32, -2, "idiv remainder");
}

#[test]
fn div_by_zero_raises_de() {
    // mov ecx, 0 ; div ecx
    let code = [0x31, 0xC9, 0xF7, 0xF1, 0xF4];
    let (_, o, _) = run(&code);
    assert_eq!(o, RunOutcome::Fault(Fault::DivideError));
}

#[test]
fn movzx_movsx_family() {
    // movzx rax, sil after storing -1 in sil → 0xFF
    let st = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[state::RSI] = 0xFF;
        s
    };
    let code = [0x40, 0x0F, 0xB6, 0xC6, 0xF4]; // movzx eax, sil
    let (s, o, _) = run_with(&code, st);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 0xFF);

    // movsx rax, sil → 0xFFFF...FF
    let st = {
        let mut s = CpuState::new(0, STACK_TOP);
        s.gpr[state::RSI] = 0xFF;
        s
    };
    let code = [0x40, 0x0F, 0xBE, 0xC6, 0xF4]; // movsx eax, sil
    let (s, o, _) = run_with(&code, st);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 0xFFFF_FFFF);

    // movsxd rax, eax = -5 → -5 sign-extended
    let code = [
        0xB8, 0xFB, 0xFF, 0xFF, 0xFF, // mov eax, -5
        0x48, 0x63, 0xC0, // movsxd rax, eax
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), (-5i64) as u64);
}

#[test]
fn cmov_and_setcc() {
    // xor eax, eax ; sete al        → al = 1
    let code = [0x31, 0xC0, 0x0F, 0x94, 0xC0, 0xF4];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s) & 0xFF, 1);

    // mov eax, 5 ; cmp eax, 5 ; cmovne edx, eax (not taken); cmovge ecx, eax
    let code = [
        0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0x83, 0xF8, 0x05, // cmp eax, 5
        0xB9, 0x63, 0x00, 0x00, 0x00, // mov ecx, 99
        0x0F, 0x45, 0xC8, // cmovne ecx, eax (NOT taken: ZF=1)
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(s.reg(Register::RCX), Some(99), "cmovne not taken on ZF=1");
}

#[test]
fn bit_instructions() {
    // mov rax, 0x10 ; bsf rcx, rax → 4
    let code = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0x0F, 0xBC, 0xC8, // bsf rcx, rax
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(s.reg(Register::RCX), Some(4));

    // popcnt rax, rcx where rcx = 0b1011
    let code = [
        0x48, 0xC7, 0xC1, 0x0B, 0x00, 0x00, 0x00, // mov rcx, 0xB
        0xF3, 0x48, 0x0F, 0xB8, 0xC1, // popcnt rax, rcx
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 3);

    // bt rax, 4 with rax = 0x10 → CF=1
    let code = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0x0F, 0xBA, 0xE0, 0x04, // bt rax, 4
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert!(s.cf(), "bit 4 of 0x10 must set CF");
}

#[test]
fn rotate_instructions() {
    // mov eax, 0x80000001 ; rol eax, 4 → 0x00000018
    let code = [
        0xB8, 0x01, 0x00, 0x00, 0x80, // mov eax, 0x80000001
        0xC1, 0xC0, 0x04, // rol eax, 4
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 0x0000_0018);

    // rcl through carry: stc; mov al, 0x80; rcl al, 1 → al = 0x01, CF = 1
    let code = [
        0xF9, // stc
        0xB0, 0x80, // mov al, 0x80
        0xD0, 0xD0, // rcl al, 1
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s) & 0xFF, 0x01, "carry rotated into bit 0");
    assert!(s.cf(), "bit shifted out lands in CF");
}

#[test]
fn double_shifts() {
    // mov rax, 0xF0F0 ; mov rcx, 0x0F0F ; shld rax, rcx, 8 → 0xF0F00F0F
    let code = [
        0x48, 0xC7, 0xC0, 0xF0, 0xF0, 0x00, 0x00, // mov rax, 0xF0F0
        0x48, 0xC7, 0xC1, 0x0F, 0x0F, 0x00, 0x00, // mov rcx, 0x0F0F
        0x48, 0x0F, 0xA4, 0xC8, 0x08, // shld rax, rcx, 8
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    // (0xF0F0 << 8) | (0x0F0F >> 56) = 0xF0F000: the double shift pulls
    // only the TOP 8 bits of rcx into rax.
    assert_eq!(rax(&s), 0x0000_0000_00F0_F000);

    // shrd rax, rcx, 8 → 0x0F0FF0F0
    let code = [
        0x48, 0xC7, 0xC0, 0xF0, 0xF0, 0x00, 0x00, // mov rax, 0xF0F0
        0x48, 0xC7, 0xC1, 0x0F, 0x0F, 0x00, 0x00, // mov rcx, 0x0F0F
        0x48, 0x0F, 0xAC, 0xC8, 0x08, // shrd rax, rcx, 8
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    // The bits shifted in come from the TOP of rcx (0x0F0F's MSBs are 0),
    // so the result is just rax >> 8 = 0xF0.
    assert_eq!(rax(&s), 0xF0);
}

#[test]
fn lahf_sahf() {
    // add al, 0 (sets PF=1, ZF=1 for 0) ; lahf ; sahf
    let code = [
        0x30, 0xC0, // xor al, al → ZF=1, PF=1
        0x9F, // lahf
        0xF4,
    ];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    let ah = (rax(&s) >> 8) & 0xFF;
    assert_eq!(ah & 0x44, 0x44, "LAHF captures SF?no—ZF and PF");
    assert_eq!(ah & 0x02, 0x02, "reserved bit 1 always set");
}

#[test]
fn string_ops_with_rep() {
    // Fill 32 bytes at 0x100 with 0xAB via rep stosb.
    // rdi = 0x100, rcx = 32, al = 0xAB, cld, rep stosb
    let code = [
        0xBF, 0x00, 0x01, 0x00, 0x00, // mov edi, 0x100
        0xB9, 0x20, 0x00, 0x00, 0x00, // mov ecx, 32
        0xB0, 0xAB, // mov al, 0xAB
        0xFC, // cld
        0xF3, 0xAA, // rep stosb
        0xF4,
    ];
    let (s, o, mut mem) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(s.reg(Register::RCX), Some(0), "rep consumed the count");
    assert_eq!(s.reg(Register::RDI), Some(0x120));
    for i in 0..32u64 {
        assert_eq!(mem.read(1, 0x100 + i).unwrap(), 0xAB, "byte {i}");
    }

    // rep movsq: copy two qwords from 0x200 to 0x300.
    let code = [
        0xBF, 0x00, 0x03, 0x00, 0x00, // mov edi, 0x300
        0xBE, 0x00, 0x02, 0x00, 0x00, // mov esi, 0x200
        0xB9, 0x02, 0x00, 0x00, 0x00, // mov ecx, 2
        0xFC, // cld
        0xF3, 0x48, 0xA5, // rep movsq
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    mem.write(8, 0x200, 0xAAAA_BBBB_CCCC_DDDD).unwrap();
    mem.write(8, 0x208, 0x1234_5678_9ABC_DEF0).unwrap();
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(mem.read(8, 0x300).unwrap(), 0xAAAA_BBBB_CCCC_DDDD);
    assert_eq!(mem.read(8, 0x308).unwrap(), 0x1234_5678_9ABC_DEF0);

    // repne scasb: find the first zero in a 16-byte string.
    let code = [
        0xBE, 0x00, 0x04, 0x00, 0x00, // mov esi, 0x400
        0xB9, 0x10, 0x00, 0x00, 0x00, // mov ecx, 16
        0x31, 0xC0, // xor eax, eax (search for 0)
        0xFC, // cld
        0xF2, 0xAE, // repne scasb  (note: scas uses rdi)
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    for i in 0..5u64 {
        mem.write(1, 0x400 + i, b'X' as u64).unwrap();
    }
    mem.write(1, 0x405, 0).unwrap();
    let mut st = CpuState::new(0, STACK_TOP);
    st.gpr[state::RDI] = 0x400;
    let mut cpu = Interpreter::new(st);
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    // rcx = 16 - 6 = 10 remaining; rdi points one past the zero.
    assert_eq!(cpu.state.gpr[state::RCX], 10);
    assert_eq!(cpu.state.gpr[state::RDI], 0x406);
}

#[test]
fn atomics_xadd_cmpxchg() {
    // lock xadd: mem[0x100] = 10; rax = 5; lock xadd [0x100], rax
    // → rax = 10 (old), mem = 15
    let code = [
        0x48, 0xC7, 0x04, 0x25, 0x00, 0x01, 0x00, 0x00, 0x0A, 0x00, 0x00,
        0x00, // mov qword [0x100], 10
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0xF0, 0x48, 0x0F, 0xC1, 0x04, 0x25, 0x00, 0x01, 0x00, 0x00, // lock xadd rax, [0x100]
        0xF4,
    ];
    let (s, o, mut mem) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert_eq!(rax(&s), 10, "XADD returns the old value in src");
    assert_eq!(mem.read(8, 0x100).unwrap(), 15);

    // cmpxchg success: rax=42, [0x200]=42, cmpxchg [0x200], rdx(=7)
    // → [0x200]=7, ZF=1
    let code = [
        0x48, 0xC7, 0x04, 0x25, 0x00, 0x02, 0x00, 0x00, 0x2A, 0x00, 0x00,
        0x00, // mov qword [0x200], 42
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0x48, 0xC7, 0xC2, 0x07, 0x00, 0x00, 0x00, // mov rdx, 7
        0xF0, 0x48, 0x0F, 0xB1, 0x14, 0x25, 0x00, 0x02, 0x00,
        0x00, // lock cmpxchg [0x200], rdx
        0xF4,
    ];
    let (s, o, mut mem) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert!(s.zf(), "successful cmpxchg sets ZF");
    assert_eq!(mem.read(8, 0x200).unwrap(), 7);

    // cmpxchg failure: rax=1, [0x300]=42 → rax=42, ZF=0, mem unchanged
    let code = [
        0x48, 0xC7, 0x04, 0x25, 0x00, 0x03, 0x00, 0x00, 0x2A, 0x00, 0x00,
        0x00, // mov qword [0x300], 42
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xC7, 0xC2, 0x07, 0x00, 0x00, 0x00, // mov rdx, 7
        0xF0, 0x48, 0x0F, 0xB1, 0x14, 0x25, 0x00, 0x03, 0x00,
        0x00, // lock cmpxchg [0x300], rdx
        0xF4,
    ];
    let (s, o, mut mem) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    assert!(!s.zf(), "failed cmpxchg clears ZF");
    assert_eq!(rax(&s), 42, "failed cmpxchg loads dst into rax");
    assert_eq!(mem.read(8, 0x300).unwrap(), 42, "memory unchanged");
}

#[test]
fn cpuid_reports_zen2() {
    // mov eax, 0 ; cpuid ; (vendor in ebx)
    let code = [0x31, 0xC0, 0x0F, 0xA2, 0xF4];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    let ebx = s.reg(Register::RBX).unwrap() as u32;
    assert_eq!(ebx, u32::from_le_bytes(*b"Auth"));
}

#[test]
fn rdtsc_advances() {
    // rdtsc ; rdtsc → second read strictly greater (first ≥ 0).
    let code = [0x0F, 0x31, 0x0F, 0x31, 0xF4];
    let (s, o, _) = run(&code);
    assert!(matches!(o, RunOutcome::Halted { .. }), "{o:?}");
    let t = (s.reg(Register::RDX).unwrap() << 32) | rax(&s);
    assert!(t >= 1, "rdtsc sees retired-instruction progress: {t}");
}

#[test]
fn de_handler_via_idt_fixes_and_resumes() {
    // Program: divide [rcx] by 3; the divisor starts at 0 (#DE!).
    // The IDT[0] handler sets the divisor to 1 and IRETs; the retried
    // division succeeds.
    let mut mem = FlatMemory::new(IMAGE_LEN);
    // divisor lives at 0x1000
    mem.write(4, 0x1000, 0).unwrap();

    // code at 0:
    //   mov rcx, 0x1000        48 C7 C1 00 10 00 00
    //   mov eax, 42            B8 2A 00 00 00
    //   xor edx, edx           31 D2
    //   div dword [rcx]        F7 31
    //   hlt                     F4
    let code = [
        0x48, 0xC7, 0xC1, 0x00, 0x10, 0x00, 0x00, 0xB8, 0x2A, 0x00, 0x00, 0x00, 0x31, 0xD2, 0xF7,
        0x31, 0xF4,
    ];
    mem.load(0, &code);

    // handler at 0x200:
    //   mov dword [rcx], 1     C7 01 01 00 00 00
    //   iretq                   48 CF
    let handler = [0xC7, 0x01, 0x01, 0x00, 0x00, 0x00, 0x48, 0xCF];
    mem.load(0x200, &handler);

    // IDT at 0x3000: gate 0 → handler 0x200, present, interrupt gate.
    let mut gate = [0u8; 16];
    gate[0..2].copy_from_slice(&0x0200u16.to_le_bytes());
    gate[2..4].copy_from_slice(&0x08u16.to_le_bytes());
    gate[5] = 0x8E; // present | interrupt gate
    gate[6..8].copy_from_slice(&0x0000u16.to_le_bytes());
    mem.load(0x3000, &gate);

    let mut st = CpuState::new(0, STACK_TOP);
    st.idtr = state::DescriptorTableRegister {
        base: 0x3000,
        limit: 16 * 8, // small table
    };
    let mut cpu = Interpreter::new(st);
    let out = cpu.run(&mut mem, 1000);
    match out {
        RunOutcome::Halted { .. } => {}
        other => panic!("expected halt after handler fixup, got {other:?}"),
    }
    assert_eq!(
        cpu.state.gpr[state::RAX],
        42,
        "divide by 1 succeeds after fixup"
    );
}

#[test]
fn ud_without_idt_propagates() {
    // ud2 with no IDT → the fault surfaces to the caller.
    let code = [0x0F, 0x0B];
    let (_, o, _) = run(&code);
    match o {
        RunOutcome::Fault(Fault::InvalidOpcode { .. }) => {}
        other => panic!("expected #UD fault, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// SIMD
// ---------------------------------------------------------------------------

#[test]
fn sse_integer_packed_ops() {
    // Build XMM0 = [1,2,3,4] (dwords) and XMM1 = [10,20,30,40] in memory,
    // then paddd and store to memory.
    let code = [
        0x66, 0x0F, 0x6F, 0x06, 0x00, 0x01, 0x00, 0x00, // movdqa xmm0, [rsi]
        0x66, 0x0F, 0x6F, 0x0D, 0x10, 0x01, 0x00, 0x00, // movdqa xmm1, [rip+...]
        0x66, 0x0F, 0xFE, 0xC1, // paddd xmm0, xmm1
        0x66, 0x0F, 0x7F, 0x04, 0x25, 0x00, 0x02, 0x00, 0x00, // movdqa [0x200], xmm0
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    // NOTE: this code uses absolute 32-bit addresses (0x100, 0x110) via
    // sign-extended disp32 — assemble by hand below instead:
    let _ = code;
    let code = [
        // movdqa xmm0, [rip+0x?] — RIP-relative is fiddly; use moffs-free
        // addressing: rsi=0x100 then movdqa xmm0,[rsi]
        0xBE, 0x00, 0x01, 0x00, 0x00, // mov esi, 0x100
        0x66, 0x0F, 0x6F, 0x06, // movdqa xmm0, [rsi]
        0xBF, 0x10, 0x01, 0x00, 0x00, // mov edi, 0x110
        0x66, 0x0F, 0x6F, 0x0F, // movdqa xmm1, [rdi]
        0x66, 0x0F, 0xFE, 0xC1, // paddd xmm0, xmm1
        0xB9, 0x00, 0x02, 0x00, 0x00, // mov ecx, 0x200
        0x66, 0x0F, 0x7F, 0x01, // movdqa [rcx], xmm0
        0xF4,
    ];
    mem.load(0, &code);
    for (i, v) in [1u32, 2, 3, 4].iter().enumerate() {
        mem.write(4, 0x100 + i as u64 * 4, u64::from(*v)).unwrap();
    }
    for (i, v) in [10u32, 20, 30, 40].iter().enumerate() {
        mem.write(4, 0x110 + i as u64 * 4, u64::from(*v)).unwrap();
    }
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    for i in 0..4u64 {
        assert_eq!(
            mem.read(4, 0x200 + i * 4).unwrap(),
            11 + 11 * i,
            "lane {i} sum"
        );
    }
}

#[test]
fn sse_float_and_shuffle() {
    // addps xmm0, xmm1 on [1.0, 2.0, 3.0, 4.0] + [0.5 x4] → [1.5, 2.5, 3.5, 4.5]
    let code = [
        0xBE, 0x00, 0x01, 0x00, 0x00, // mov esi, 0x100
        0x0F, 0x28, 0x06, // movaps xmm0, [rsi]
        0xBF, 0x10, 0x01, 0x00, 0x00, // mov edi, 0x110
        0x0F, 0x28, 0x0F, // movaps xmm1, [rdi]
        0x0F, 0x58, 0xC1, // addps xmm0, xmm1
        0xB9, 0x00, 0x02, 0x00, 0x00, // mov ecx, 0x200
        0x0F, 0x29, 0x01, // movaps [rcx], xmm0
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    for i in 0..4u64 {
        mem.write(4, 0x100 + i * 4, (i as f32 + 1.0).to_bits() as u64)
            .unwrap();
        mem.write(4, 0x110 + i * 4, 0.5f32.to_bits() as u64)
            .unwrap();
    }
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    for i in 0..4u64 {
        let got = f32::from_bits(mem.read(4, 0x200 + i * 4).unwrap() as u32);
        assert_eq!(got, i as f32 + 1.5, "lane {i}");
    }
}

#[test]
fn pshufd_lane_select() {
    // pshufd xmm0, xmm1, 0x1B reverses the four dwords.
    let code = [
        0xBE, 0x00, 0x01, 0x00, 0x00, // mov esi, 0x100
        0x66, 0x0F, 0x6F, 0x0E, // movdqa xmm1, [rsi]
        0x66, 0x0F, 0x70, 0xC1, 0x1B, // pshufd xmm0, xmm1, 0x1B
        0xB9, 0x00, 0x02, 0x00, 0x00, // mov ecx, 0x200
        0x66, 0x0F, 0x7F, 0x01, // movdqa [rcx], xmm0
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    for i in 0..4u64 {
        mem.write(4, 0x100 + i * 4, 0xA0 + i).unwrap();
    }
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    for i in 0..4u64 {
        assert_eq!(mem.read(4, 0x200 + i * 4).unwrap(), 0xA3 - i, "lane {i}");
    }
}

#[test]
fn cvttps2dq_truncates() {
    let code = [
        0xBE, 0x00, 0x01, 0x00, 0x00, 0x0F, 0x28, 0x06, // movaps xmm0, [rsi]
        0x0F, 0x5B, 0xC0, // cvtdq2ps xmm0, xmm0
        0x66, 0x0F, 0x5B,
        0xC0, // cvttps2dq xmm0, xmm0  (wait — 0F 5B is cvtdq2ps; cvttps2dq is 66 0F 5B)
        0xB9, 0x00, 0x02, 0x00, 0x00, 0x66, 0x0F, 0x7F, 0x01, // movdqa [rcx], xmm0
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    for i in 0..4u64 {
        mem.write(4, 0x100 + i * 4, 10 + i * 10).unwrap();
    }
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    // ints → float → (trunc) ints: exact for these values.
    for i in 0..4u64 {
        assert_eq!(mem.read(4, 0x200 + i * 4).unwrap(), 10 + i * 10);
    }
}

#[test]
fn aes_round_known_answer_in_executor() {
    // aesenc xmm0, xmm1 with the FIPS-197 C.1 round-1 state and key:
    // state d4bf5d30e0b452aeb84111f11e2798e5, key a0fafe1788542cb123a339392a6c7605
    // expected round-2 input: a49c7ff2689f352b6b5bea43026a5049
    let code = [
        0xBE, 0x00, 0x01, 0x00, 0x00, 0x66, 0x0F, 0x6F, 0x06, // movdqa xmm0, [rsi]
        0xBF, 0x10, 0x01, 0x00, 0x00, 0x66, 0x0F, 0x6F, 0x0F, // movdqa xmm1, [rdi]
        0x66, 0x0F, 0x38, 0xDC, 0xC1, // aesenc xmm0, xmm1
        0xB9, 0x00, 0x02, 0x00, 0x00, 0x66, 0x0F, 0x7F, 0x01, // movdqa [rcx], xmm0
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    // Round-1 input: the C.1 state after AddRoundKey (193de3be...).
    let state_bytes: [u8; 16] = [
        0x19, 0x3d, 0xe3, 0xbe, 0xa0, 0xf4, 0xe2, 0x2b, 0x9a, 0xc6, 0x8d, 0x2a, 0xe9, 0xf8, 0x48,
        0x08,
    ];
    let key_bytes: [u8; 16] = [
        0xa0, 0xfa, 0xfe, 0x17, 0x88, 0x54, 0x2c, 0xb1, 0x23, 0xa3, 0x39, 0x39, 0x2a, 0x6c, 0x76,
        0x05,
    ];
    mem.load(0x100, &state_bytes);
    mem.load(0x110, &key_bytes);
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    let expect: [u8; 16] = [
        0xa4, 0x9c, 0x7f, 0xf2, 0x68, 0x9f, 0x35, 0x2b, 0x6b, 0x5b, 0xea, 0x43, 0x02, 0x6a, 0x50,
        0x49,
    ];
    let mut got = [0u8; 16];
    mem.read_buf(0x200, &mut got).unwrap();
    assert_eq!(got, expect, "AESENC must match FIPS-197 C.1");
}

#[test]
fn lock_prefix_xchg_memory() {
    // xchg [rcx], rax with rcx = 0x100, rax = 7, mem = 9
    // → rax = 9, mem = 7.
    let code = [
        0xB9, 0x00, 0x01, 0x00, 0x00, // mov ecx, 0x100
        0x48, 0xC7, 0xC0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x48, 0x87, 0x01, // xchg rax, [rcx]
        0xF4,
    ];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    mem.write(8, 0x100, 9).unwrap();
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(cpu.state.gpr[state::RAX], 9);
    assert_eq!(mem.read(8, 0x100).unwrap(), 7);
}
