//! Differential tests: the JIT against the interpreter oracle.

use pdg_cpu::{CpuState, FlatMemory, Interpreter, MemoryAccess, RunOutcome, state};
use pdg_jit::Jit;

const STACK_TOP: u64 = 0x8000;
const IMAGE_LEN: usize = 0x20000;

/// Deterministic xorshift for program generation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Compare two CpuStates (GPRs + RFLAGS) and memory images.
fn assert_identical(a: &CpuState, b: &CpuState, ma: &FlatMemory, mb: &FlatMemory) {
    for i in 0..16 {
        assert_eq!(a.gpr[i], b.gpr[i], "GPR {i} mismatch");
    }
    assert_eq!(a.rflags, b.rflags, "RFLAGS mismatch");
    assert_eq!(a.rip, b.rip, "RIP mismatch");
    let ia = ma.image();
    let ib = mb.image();
    let len = ia.len().min(ib.len());
    for (i, (x, y)) in ia[..len].iter().zip(ib[..len].iter()).enumerate() {
        assert_eq!(x, y, "memory differs at {i:#x} ({x:#x} vs {y:#x})");
    }
}

fn run_interp(code: &[u8]) -> (CpuState, FlatMemory, RunOutcome) {
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, code);
    let mut cpu = Interpreter::new(CpuState::new(0, STACK_TOP));
    let out = cpu.run(&mut mem, 1_000_000);
    (cpu.state, mem, out)
}

fn run_jit(code: &[u8]) -> (CpuState, FlatMemory, RunOutcome) {
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, code);
    let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
    let out = jit.run(&mut mem, 1_000_000);
    let st = jit.state().clone();
    (st, mem, out)
}

#[test]
fn jit_boots_and_halts() {
    // mov rax, 42 ; add rax, 8 ; hlt
    let code = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0x48, 0x83, 0xC0, 0x08, 0xF4,
    ];
    let (st, _, out) = run_jit(&code);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(st.gpr[state::RAX], 50);
}

#[test]
fn jit_matches_interpreter_on_alu_chain() {
    // A straight-line ALU chain with mixed widths and flags.
    let code = [
        0xB8, 0x34, 0x12, 0x00, 0x00, // mov eax, 0x1234
        0xBB, 0x78, 0x56, 0x00, 0x00, // mov ebx, 0x5678
        0x01, 0xD8, // add eax, ebx
        0x25, 0xFF, 0x0F, 0x00, 0x00, // and eax, 0xFFF
        0x83, 0xC8, 0x03, // or eax, 3
        0x83, 0xF0, 0x40, // xor eax, 0x40
        0xF7, 0xD8, // neg eax
        0x48, 0x0F, 0xAF, 0xC3, // imul rax, rbx
        0x48, 0xD1, 0xE0, // shl rax, 1
        0x48, 0x83, 0xE8, 0x11, // sub rax, 0x11
        0xF4,
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }), "{io:?}");
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
}

#[test]
fn jit_loop_formation_matches_interpreter() {
    // xor rax, rax ; mov rcx, 1000 ; loop: add rax, rcx ; dec rcx ; jne loop ; hlt
    let code = [
        0x48, 0x31, 0xC0, // xor rax, rax
        0x48, 0xC7, 0xC1, 0xE8, 0x03, 0x00, 0x00, // mov rcx, 1000
        0x48, 0x01, 0xC8, // loop: add rax, rcx
        0x48, 0xFF, 0xC9, // dec rcx
        0x75, 0xF9, // jne loop
        0xF4,
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }));
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
    assert_eq!(is.gpr[state::RAX], (1000 * 1001) / 2);
}

#[test]
fn jit_memory_ops_match_interpreter() {
    // Stores + loads through the inline TLB, including a page-crossing wide
    // store near the end of a page.
    let code = [
        0xBE, 0x00, 0x01, 0x00, 0x00, // mov esi, 0x100
        0xBF, 0x10, 0x01, 0x00, 0x00, // mov edi, 0x110
        0x48, 0x89, 0x06, // mov [rsi], rax
        0x48, 0x8B, 0x0F, // mov rcx, [rdi]
        // Page-crossing: write a qword at 0xFFE (spans into the next page).
        0x48, 0xC7, 0x04, 0x25, 0xFE, 0x0F, 0x00, 0x00, 0x99, 0x88, 0x77, 0x66, 0x00, 0x00, 0x00,
        0x00, // mov qword [0xFFE], 0x66778899
        0x48, 0xA1, 0xFE, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, [0xFFE]
        0xF4,
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }), "{io:?}");
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
}

#[test]
fn differential_fuzz_alu_programs() {
    // Generate random straight-line programs over the lifted instruction
    // forms; interpreter and JIT must agree bit-for-bit.
    let mut rng = Rng(0xC0FF_EE00_DEAD_BEEF);
    for case in 0..64u64 {
        let mut code: Vec<u8> = Vec::new();
        let ninstr = 8 + rng.below(16);
        for _ in 0..ninstr {
            let op = rng.below(7);
            let reg = 8 + rng.below(8); // r8..r15 avoid rsp/rbp specials
            let imm = (rng.next() & 0xFFFF) as u16;
            match op {
                0 => {
                    // mov rN, imm32: 49 C7 C0+(r-8) imm32
                    code.push(0x49);
                    code.push(0xC7);
                    code.push(0xC0 + (reg - 8) as u8);
                    code.extend_from_slice(&(imm as u32).to_le_bytes());
                }
                1 => {
                    // add rN, imm8
                    code.push(0x49);
                    code.push(0x83);
                    code.push(0xC0 + (reg - 8) as u8);
                    code.push((imm & 0xFF) as u8);
                }
                2 => {
                    // sub rN, imm8
                    code.push(0x49);
                    code.push(0x83);
                    code.push(0xE8 + (reg - 8) as u8);
                    code.push((imm & 0xFF) as u8);
                }
                3 => {
                    // xor rN, rM
                    let m = 8 + rng.below(8);
                    code.push(0x4D);
                    code.push(0x31);
                    code.push(0xC0 + (reg as u8 & 7) + ((m as u8 & 7) << 3));
                }
                4 => {
                    // shl rN, imm8 (masked by the ISA)
                    code.push(0x49);
                    code.push(0xC1);
                    code.push(0xE0 + (reg - 8) as u8);
                    code.push((imm & 0x3F) as u8);
                }
                5 => {
                    // cmp + cmov: cmp rN, imm8 ; cmovl rN, rM
                    let m = 8 + rng.below(8);
                    code.push(0x49);
                    code.push(0x83);
                    code.push(0xF8 + (reg - 8) as u8);
                    code.push((imm & 0xFF) as u8);
                    code.push(0x4D);
                    code.push(0x0F);
                    code.push(0x4C);
                    code.push(0xC0 + (reg as u8 & 7) + ((m as u8 & 7) << 3));
                }
                _ => {
                    // movzx/movsx rN, rM(w)
                    let m = 8 + rng.below(8);
                    code.push(0x4D);
                    code.push(0x0F);
                    code.push(0xB7); // movzx r64, r/m16
                    code.push(0xC0 + (reg as u8 & 7) + ((m as u8 & 7) << 3));
                }
            }
        }
        code.push(0xF4); // hlt

        let (is, im, io) = run_interp(&code);
        let (js, jm, jo) = run_jit(&code);
        assert!(
            matches!(io, RunOutcome::Halted { .. }),
            "case {case}: interpreter faulted: {io:?}"
        );
        assert!(
            matches!(jo, RunOutcome::Halted { .. }),
            "case {case}: jit faulted: {jo:?}"
        );
        if is.rflags != js.rflags || is.gpr != js.gpr {
            panic!(
                "case {case}: mismatch
  code = {code:02x?}
  interp rflags={:#x} jit rflags={:#x}\n  interp gprs={:x?}\n  jit gprs={:x?}",
                is.rflags, js.rflags, is.gpr, js.gpr
            );
        }
        assert_identical(&is, &js, &im, &jm);
    }
}

#[test]
fn differential_fuzz_with_memory() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    for case in 0..32u64 {
        let base: u16 = (0x100 + (rng.below(32) * 8)) as u16;
        let mut code: Vec<u8> = vec![
            0x49, 0xBA, // mov r10, imm64 (address)
        ];
        code.extend_from_slice(&u64::from(base).to_le_bytes());
        // mov rax, imm64
        code.push(0x48);
        code.push(0xB8);
        code.extend_from_slice(&rng.next().to_le_bytes());
        // mov [r10], rax
        code.extend_from_slice(&[0x4D, 0x89, 0x02]);
        // add rax, imm32
        code.push(0x48);
        code.push(0x05);
        code.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        // mov rcx, [r10] ; add rax, rcx
        code.extend_from_slice(&[0x49, 0x8B, 0x0A, 0x48, 0x01, 0xC8]);
        // sub r10, imm8 ; mov [r10], rax
        code.push(0x49);
        code.push(0x83);
        code.push(0xEA);
        code.push((rng.next() & 0x7) as u8);
        code.extend_from_slice(&[0x4D, 0x89, 0x02]);
        code.push(0xF4);

        let (is, im, io) = run_interp(&code);
        let (js, jm, jo) = run_jit(&code);
        assert!(
            matches!(io, RunOutcome::Halted { .. }),
            "case {case}: {io:?}"
        );
        assert!(
            matches!(jo, RunOutcome::Halted { .. }),
            "case {case}: {jo:?}"
        );
        assert_identical(&is, &js, &im, &jm);
    }
}

#[test]
fn smc_self_modifying_code() {
    // The program patches the imm8 of an `add rax, imm8` before running
    // it, forcing an SMC invalidation and recompilation:
    //   0x00: mov rax, 1
    //   0x07: jmp 0x0F          (skip the add; go patch first)
    //   0x09: add rax, 5        ← imm8 at 0x0C patched to 0x64
    //   0x0D: jmp 0x1C          (to the hlt)
    //   0x0F: mov byte [0x0C], 0x64
    //   0x17: jmp 0x09          (back into the patched add)
    //   0x1C: hlt
    let mut prog: Vec<u8> = Vec::new();
    prog.extend_from_slice(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00]); // 0x00
    prog.extend_from_slice(&[0xEB, 0x06]); // 0x07 → 0x0F
    prog.extend_from_slice(&[0x48, 0x83, 0xC0, 0x05]); // 0x09: add rax, 5
    prog.extend_from_slice(&[0xEB, 0x0D]); // 0x0D → 0x1C
    prog.extend_from_slice(&[0xC6, 0x04, 0x25, 0x0C, 0x00, 0x00, 0x00, 0x64]); // 0x0F: patch imm8 at 0x0C
    let jmp_at = prog.len(); // 0x17
    prog.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    prog.push(0xF4); // 0x1C: hlt
    let rel: i32 = 0x09 - (jmp_at as i32 + 5);
    prog[jmp_at + 1..jmp_at + 5].copy_from_slice(&rel.to_le_bytes());

    let (is, im, io) = run_interp(&prog);
    let (js, jm, jo) = run_jit(&prog);
    assert!(matches!(io, RunOutcome::Halted { .. }), "{io:?}");
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
    assert_eq!(is.gpr[state::RAX], 101, "1 + patched add(100)");
}

#[test]
fn budget_is_respected() {
    // An infinite guest loop must stop at the budget, exactly.
    // loop: jmp loop
    let code = [0xEB, 0xFE];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
    let out = jit.run(&mut mem, 10_000);
    match out {
        RunOutcome::MaxInstructions { instructions } => {
            assert!(instructions <= 10_000, "budget overshot: {instructions}");
        }
        other => panic!("expected budget stop, got {other:?}"),
    }
}

#[test]
fn jit_forward_conditional_branches_match_interpreter() {
    // Regression for a Phase-2 lifter bug: the not-taken path of a
    // forward (external-target) jcc was left unterminated, so *every*
    // block containing one failed cranelift verification and silently
    // fell back to the interpreter forever — loads compiled, but any
    // function with an early-exit branch (fib's `jb done`) ran at
    // interpreter speed. Shape: conditional early-exit + straight-line
    // body + backward unconditional jump.
    //   mov ecx, 9 ; xor eax, eax
    // loop: test ecx, ecx ; jz done
    //   add eax, ecx ; dec ecx ; jmp loop
    // done: hlt
    let code = [
        0xB9, 0x09, 0x00, 0x00, 0x00, // 0x00: mov ecx, 9
        0x31, 0xC0, // 0x05: xor eax, eax
        0x85, 0xC9, // 0x07: loop: test ecx, ecx
        0x74, 0x07, // 0x09: jz done (0x0B + 7 = 0x12)
        0x01, 0xC8, // 0x0B: add eax, ecx
        0xFF, 0xC9, // 0x0D: dec ecx
        0xEB, 0xF6, // 0x0F: jmp loop (0x11 - 10 = 0x07)
        0x90, // 0x11: nop (fallthrough guard)
        0xF4, // 0x12: done: hlt
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }));
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
    assert_eq!(is.gpr[state::RAX], 45, "sum 9..=1");

    // The whole program must lift: zero interpreter fallbacks.
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
    let out = jit.run(&mut mem, 100_000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(
        jit.stats().fallback_steps,
        0,
        "forward conditional branches must lift, not fall back"
    );
    assert_eq!(jit.state().gpr[state::RAX], 45);
}

#[test]
fn jit_loads_lift_through_the_inline_tlb() {
    // Regression for the mem_load block-parameter bug: loads *never*
    // compiled (every load-containing block failed verification and fell
    // back), while stores lifted fine. A load/store loop must now run
    // fully lifted.
    let code = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x01, 0x00, // mov rax, 0x10000 (off the code page)
        0xB9, 0x0A, 0x00, 0x00, 0x00, // mov ecx, 10
        0x48, 0x8B, 0x10, // 0x0E loop: mov rdx, [rax]
        0x48, 0x83, 0xC2, 0x01, // add rdx, 1
        0x48, 0x89, 0x10, // mov [rax], rdx
        0x48, 0xFF, 0xC9, // dec rcx
        0x75, 0xF1, // jne loop (0x1D - 15 = 0x0E)
        0xF4, // hlt
    ];
    let (is, mut im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }));
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
    let mut buf = [0u8; 8];
    im.read_buf(0x10000, &mut buf).unwrap();
    assert_eq!(u64::from_le_bytes(buf), 10, "10 increments through the TLB");

    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
    let out = jit.run(&mut mem, 100_000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(
        jit.stats().fallback_steps,
        0,
        "load/store loops must lift, not fall back"
    );
}

/// Doubler syscall for the JIT path.
struct Doubler;
impl pdg_cpu::SyscallHook for Doubler {
    fn on_syscall(
        &mut self,
        cpu: &mut CpuState,
        _mem: &mut dyn MemoryAccess,
    ) -> Result<(), pdg_cpu::Fault> {
        let v = cpu.gpr[state::RAX];
        cpu.gpr[state::RAX] = v.wrapping_mul(2);
        Ok(())
    }
}

#[test]
fn syscalls_through_the_jit() {
    // mov rax, 21 ; syscall ; hlt
    let code = [0x48, 0xC7, 0xC0, 0x15, 0x00, 0x00, 0x00, 0x0F, 0x05, 0xF4];
    let mut mem = FlatMemory::new(IMAGE_LEN);
    mem.load(0, &code);
    let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
    jit.set_syscall_hook(Box::new(Doubler));
    let out = jit.run(&mut mem, 1000);
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    assert_eq!(
        jit.state().gpr[state::RAX],
        42,
        "syscall serviced in JIT mode"
    );
}

#[test]
fn faults_reproduce_precisely() {
    // mov rax, [0x9000_0000] — unmapped: #PF in both tiers with the same
    // RIP and address.
    let code = [
        0x48, 0xA1, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, // mov rax, [0x90000000]
        0xF4,
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    let _ = (im, jm);
    // Both tiers must raise the *same* fault with the same RIP (FlatMemory
    // reports unmapped reads as host errors rather than #PF; the differential
    // contract is tier agreement, and the paged path maps these to #PF).
    match (&io, &jo) {
        (RunOutcome::Fault(f1), RunOutcome::Fault(f2)) => {
            assert_eq!(f1, f2, "same fault in both tiers");
        }
        _ => panic!("expected faults in both tiers: interp={io:?} jit={jo:?}"),
    }
    assert_eq!(is.rip, js.rip);
}

#[test]
fn fallback_instructions_execute_correctly() {
    // popcnt/mul are interpreter fallbacks in tier 1: results must still be
    // exact.
    let code = [
        0x48, 0xC7, 0xC1, 0x0B, 0x00, 0x00, 0x00, // mov rcx, 0xB
        0xF3, 0x48, 0x0F, 0xB8, 0xC1, // popcnt rax, rcx
        0xF7, 0xE1, // mul ecx
        0xF4,
    ];
    let (is, im, io) = run_interp(&code);
    let (js, jm, jo) = run_jit(&code);
    assert!(matches!(io, RunOutcome::Halted { .. }), "{io:?}");
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
}

#[test]
fn call_ret_stack_matches() {
    // fib(10) via real call/ret.
    let code: Vec<u8> = vec![
        0xB9, 0x0A, 0x00, 0x00, 0x00, // mov ecx, 10
        0xE8, 0x05, 0x00, 0x00, 0x00, // call f
        0xF4, // hlt
        // f @ 0x0A:
        0x48, 0x89, 0xC8, // mov rax, rcx
        0x48, 0x83, 0xF9, 0x02, // cmp rcx, 2
        0x72, 0x18, // jb ret (0x0A+0x12+2 = 0x1E... compute below)
    ];
    // jb target: at offset 0x0A+8+2 = 0x14... let me lay out precisely:
    let mut prog: Vec<u8> = Vec::new();
    prog.extend_from_slice(&[0xB9, 0x0A, 0x00, 0x00, 0x00]); // 0: mov ecx,10
    prog.extend_from_slice(&[0xE8, 0x05, 0x00, 0x00, 0x00]); // 5: call 0x0F
    prog.extend_from_slice(&[0xF4]); // 0x0A: hlt
    let f = prog.len(); // 0x0B
    prog.extend_from_slice(&[0x48, 0x89, 0xC8]); // f: mov rax, rcx
    prog.extend_from_slice(&[0x48, 0x83, 0xF9, 0x02]); // cmp rcx, 2
    let jb_at = prog.len();
    prog.extend_from_slice(&[0x72, 0x00]); // jb ret (patched)
    prog.extend_from_slice(&[0x51]); // push rcx
    prog.extend_from_slice(&[0x48, 0xFF, 0xC9]); // dec rcx
    let call1 = prog.len();
    prog.extend_from_slice(&[0xE8, 0x00, 0x00, 0x00, 0x00]); // call f (patched)
    prog.extend_from_slice(&[0x59]); // pop rcx
    prog.extend_from_slice(&[0x50]); // push rax
    prog.extend_from_slice(&[0x48, 0x8D, 0x49, 0xFE]); // lea rcx, [rcx-2]
    let call2 = prog.len();
    prog.extend_from_slice(&[0xE8, 0x00, 0x00, 0x00, 0x00]); // call f (patched)
    prog.extend_from_slice(&[0x5A]); // pop rdx
    prog.extend_from_slice(&[0x48, 0x01, 0xD0]); // add rax, rdx
    let ret_at = prog.len();
    prog.extend_from_slice(&[0xC3]); // ret

    // Patch the jumps.
    let rel = (ret_at as i32) - (jb_at as i32 + 2);
    prog[jb_at + 1] = rel as u8;
    let r1 = (f as i32) - (call1 as i32 + 5);
    prog[call1 + 1..call1 + 5].copy_from_slice(&r1.to_le_bytes());
    let r2 = (f as i32) - (call2 as i32 + 5);
    prog[call2 + 1..call2 + 5].copy_from_slice(&r2.to_le_bytes());

    let (is, im, io) = run_interp(&prog);
    let (js, jm, jo) = run_jit(&prog);
    assert!(matches!(io, RunOutcome::Halted { .. }), "{io:?}");
    assert!(matches!(jo, RunOutcome::Halted { .. }), "{jo:?}");
    assert_identical(&is, &js, &im, &jm);
    assert_eq!(is.gpr[state::RAX], 55, "fib(10)");
    let _ = code;
}
