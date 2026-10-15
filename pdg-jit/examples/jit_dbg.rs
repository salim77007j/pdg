//! Bisect which instruction diverges (debug tool).
use pdg_cpu::{CpuState, FlatMemory, Interpreter};
use pdg_jit::Jit;

fn main() {
    // add r13, -20 with r13 = 0 → OF must be 0.
    let code = [
        0x4D, 0x31, 0xED, // xor r13, r13
        0x49, 0x83, 0xC5, 0xEC, // add r13, -20
        0xF4,
    ];
    let mut mi = FlatMemory::new(0x20000);
    mi.load(0, &code);
    let mut ci = Interpreter::new(CpuState::new(0, 0x8000));
    ci.run(&mut mi, 1000);
    let mut mj = FlatMemory::new(0x20000);
    mj.load(0, &code);
    let mut cj = Jit::with_flat(CpuState::new(0, 0x8000), &mut mj);
    cj.run(&mut mj, 1000);
    println!(
        "interp rflags={:#x} OF={} jit rflags={:#x} OF={}",
        ci.state.rflags,
        ci.state.of(),
        cj.state().rflags,
        cj.state().of(),
    );
}
