//! Quick timing harness (not part of the test suite).
use pdg_cpu::{CpuState, FlatMemory, RunOutcome};
use pdg_jit::Jit;

fn main() {
    let n: u32 = 1000;
    let mut code = vec![0x48, 0xC7, 0xC1, 0, 0, 0, 0];
    code[3..7].copy_from_slice(&n.to_le_bytes());
    code.extend_from_slice(&[0x48, 0xFF, 0xC9, 0x75, 0xFB, 0xF4]);
    let mut mem = FlatMemory::new(0x20000);
    mem.load(0, &code);
    let mut jit = Jit::with_flat(CpuState::new(0, 0x8000), &mut mem);
    let t0 = std::time::Instant::now();
    let out = jit.run(&mut mem, u64::MAX / 4);
    let dt = t0.elapsed();
    assert!(matches!(out, RunOutcome::Halted { .. }), "{out:?}");
    println!(
        "dec/jne iterations: {dt:?} ({:.3} ns/iter; {} blocks, {} calls, {} fallbacks)",
        dt.as_secs_f64() * 1e9 / f64::from(n),
        jit.stats().blocks_compiled,
        jit.stats().block_calls,
        jit.stats().fallback_steps
    );
    assert_eq!(jit.state().gpr[1], 0);
}
