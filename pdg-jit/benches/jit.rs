//! JIT vs interpreter benchmarks (the Phase-2 ≥50× target evidence).

use criterion::{Criterion, criterion_group, criterion_main};
use pdg_cpu::{CpuState, FlatMemory, MemoryAccess, RunOutcome};
use pdg_jit::Jit;

const STACK_TOP: u64 = 0x8000;
const IMAGE_LEN: usize = 0x20000;

/// `loop: dec rcx ; jne loop ; hlt` — 100M iterations.
fn alu_loop() -> Vec<u8> {
    let mut code = vec![
        0x48, 0xC7, 0xC1, 0x00, 0x00, 0x00, 0x00, // mov rcx, imm32 (patched)
    ];
    let n: u32 = 100_000_000;
    code[3..7].copy_from_slice(&n.to_le_bytes());
    code.extend_from_slice(&[
        0x48, 0xFF, 0xC9, // loop: dec rcx
        0x75, 0xFB, // jne loop
        0xF4, // hlt
    ]);
    code
}

/// fib(30) with the classic push/pop call sequence.
///
/// Layout (verified by pdg-disasm):
/// `0x00 mov ecx,30; 0x05 call f(+1); 0x0A hlt;
/// f@0x0B: mov rax,rcx; cmp rcx,2; jb done(+0x18); push rcx; dec rcx;
/// call f(-18); pop rcx; push rax; lea rcx,[rcx-2]; call f(-29);
/// pop rdx; add rax,rdx; done@0x2C: ret`
fn fib30() -> Vec<u8> {
    vec![
        0xB9, 0x1E, 0x00, 0x00, 0x00, // 0x00: mov ecx, 30
        0xE8, 0x01, 0x00, 0x00, 0x00, // 0x05: call f (0x0A + 1 = 0x0B)
        0xF4, // 0x0A: hlt
        // f @ 0x0B:
        0x48, 0x89, 0xC8, // mov rax, rcx
        0x48, 0x83, 0xF9, 0x02, // cmp rcx, 2
        0x72, 0x18, // jb done (0x14 + 0x18 = 0x2C)
        0x51, // push rcx
        0x48, 0xFF, 0xC9, // dec rcx
        0xE8, 0xEE, 0xFF, 0xFF, 0xFF, // call f (0x1D - 18 = 0x0B)
        0x59, // pop rcx
        0x50, // push rax
        0x48, 0x8D, 0x49, 0xFE, // lea rcx, [rcx-2]
        0xE8, 0xE3, 0xFF, 0xFF, 0xFF, // call f (0x28 - 29 = 0x0B)
        0x5A, // pop rdx
        0x48, 0x01, 0xD0, // add rax, rdx
        0xC3, // 0x2C: done: ret
    ]
}

/// load/add/store loop: memory throughput through the inline TLB.
/// The store target lives on a page away from the code so the SMC
/// (self-modifying-code) path stays out of the measurement.
fn mem_loop() -> Vec<u8> {
    let mut code = vec![
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x01, 0x00, // mov rax, 0x10000
        0x48, 0xC7, 0xC1, 0x00, 0x00, 0x00, 0x00, // mov rcx, imm32 (patched)
    ];
    let n: u32 = 20_000_000;
    code[10..14].copy_from_slice(&n.to_le_bytes());
    code.extend_from_slice(&[
        0x48, 0x8B, 0x10, // loop: mov rdx, [rax]
        0x48, 0x83, 0xC2, 0x01, // add rdx, 1
        0x48, 0x89, 0x10, // mov [rax], rdx
        0x48, 0xFF, 0xC9, // dec rcx
        0x75, 0xF1, // jne loop (0x1D - 15 = 0x0E)
        0xF4,
    ]);
    code
}

fn bench_alu(c: &mut Criterion) {
    let code = alu_loop();
    c.bench_function("jit/alulu", |b| {
        b.iter(|| {
            let mut mem = FlatMemory::new(IMAGE_LEN);
            mem.load(0, &code);
            let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
            let out = jit.run(&mut mem, u64::MAX / 4);
            assert!(matches!(out, RunOutcome::Halted { .. }));
            assert_eq!(jit.state().gpr[1], 0);
        })
    });
}

fn bench_mem(c: &mut Criterion) {
    let code = mem_loop();
    c.bench_function("jit/memloop", |b| {
        b.iter(|| {
            let mut mem = FlatMemory::new(IMAGE_LEN);
            mem.load(0, &code);
            let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
            let out = jit.run(&mut mem, u64::MAX / 4);
            assert!(matches!(out, RunOutcome::Halted { .. }));
            let v = mem.read(8, 0x10000).unwrap();
            assert_eq!(v, 20_000_000);
        })
    });
}

fn bench_fib(c: &mut Criterion) {
    let code = fib30();
    c.bench_function("jit/fib30", |b| {
        b.iter(|| {
            let mut mem = FlatMemory::new(IMAGE_LEN);
            mem.load(0, &code);
            let mut jit = Jit::with_flat(CpuState::new(0, STACK_TOP), &mut mem);
            let out = jit.run(&mut mem, u64::MAX / 4);
            assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
            assert_eq!(jit.state().gpr[0], 832_040, "fib(30)");
        })
    });
}

criterion_group!(benches, bench_alu, bench_mem, bench_fib);
criterion_main!(benches);
