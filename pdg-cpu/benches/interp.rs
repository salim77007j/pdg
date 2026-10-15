//! Interpreter throughput benchmark: a tight `dec/jne` counting loop,
//! the classic integer-core workload.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use pdg_cpu::{CpuState, FlatMemory, Interpreter, RunOutcome};

const COUNT: u32 = 0x1_0000; // 64 Ki iterations = 128 Ki guest instructions

fn code() -> Vec<u8> {
    let mut code = Vec::new();
    code.extend_from_slice(&[0xB9]); // mov ecx, COUNT (zero-extends into rcx)
    code.extend_from_slice(&COUNT.to_le_bytes());
    code.extend_from_slice(&[0x48, 0xFF, 0xC9]); // 0x5: dec rcx
    code.extend_from_slice(&[0x75, 0xFB]); // 0x8: jne 0x5
    code.push(0xF4); // 0xA: hlt
    code
}

fn bench_interp_loop(c: &mut Criterion) {
    let code = code();
    let mut group = c.benchmark_group("interpreter");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);
    group.throughput(criterion::Throughput::Elements(COUNT as u64 * 2 + 3));
    group.bench_function("dec_jne_loop_128Ki_instructions", |b| {
        b.iter(|| {
            let mut mem = FlatMemory::new(0x1000);
            mem.load(0, &code);
            let mut cpu = Interpreter::new(CpuState::new(0, 0x800));
            let outcome = cpu.run(&mut mem, u64::MAX - 1);
            assert!(matches!(outcome, RunOutcome::Halted { .. }));
        })
    });
    group.finish();
}

criterion_group!(benches, bench_interp_loop);
criterion_main!(benches);
