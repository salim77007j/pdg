//! Phase-2 SMP integration: real host threads per logical core over one
//! shared bus, cross-core atomicity, IPI delivery, and mixed
//! JIT/interpreter execution tiers.

use pdg_core::config::PdgConfig;
use pdg_core::events::Event;
use pdg_core::smp::{CoreSpec, SmpSystem};
use pdg_cpu::RunOutcome;

/// Guest-physical address of the shared test counter (mapped RW by the
/// bootstrap tables, well away from images and stacks).
const COUNTER: u64 = 0x0030_0000;

fn config() -> PdgConfig {
    PdgConfig::default()
}

fn counter_of(sys: &SmpSystem) -> u64 {
    let mut buf = [0u8; 8];
    sys.bus().ram_read_buf(COUNTER, &mut buf).unwrap();
    u64::from_le_bytes(buf)
}

/// `mov rcx, iters ; movabs rdi, COUNTER ; loop { lock inc [rdi] } ; hlt`
///
/// Every iteration is a LOCK-prefixed RMW on the *shared* bus, so N cores
/// running this concurrently must produce exactly `N * iters` increments —
/// a lost update is a bug, not a flake.
fn atomic_loop(iters: u32) -> Vec<u8> {
    let mut code = Vec::new();
    code.extend_from_slice(&[0x48, 0xC7, 0xC1]); // mov rcx, imm32
    code.extend_from_slice(&iters.to_le_bytes());
    code.extend_from_slice(&[0x48, 0xBF]); // movabs rdi, imm64
    code.extend_from_slice(&COUNTER.to_le_bytes());
    code.extend_from_slice(&[0xF0, 0x48, 0xFF, 0x07]); // lock inc qword [rdi]
    code.extend_from_slice(&[0xE2, 0xFA]); // loop -6
    code.push(0xF4); // hlt
    code
}

#[test]
fn sixteen_threads_share_one_counter_exactly() {
    let sys = SmpSystem::new(config()).unwrap();
    let code = atomic_loop(1000);
    sys.boot_image(&code, 0x0010_0000).unwrap();

    let reports = sys.run_all(0x0010_0000, 200_000);

    assert_eq!(reports.len(), 16, "one host thread per logical core");
    for r in &reports {
        assert!(
            matches!(r.outcome, RunOutcome::Halted { .. }),
            "core {} did not halt: {:?}",
            r.core,
            r.outcome
        );
    }
    assert_eq!(
        counter_of(&sys),
        16 * 1000,
        "every LOCK increment must be counted exactly once"
    );
}

#[test]
fn ipi_between_cores_with_event_mirror() {
    let sys = SmpSystem::new(config()).unwrap();
    let ipi_base = sys.config().memory.ipi_mmio_base;

    // Core 0: send vector 0x42 to core 1, then halt.
    //   movabs rax, ipi_base      (48 B8 imm64)
    //   mov ecx, (0x42 << 8) | 1  (B9 imm32)
    //   mov [rax], rcx            (48 89 08)
    //   hlt                       (F4)
    let mut sender = Vec::new();
    sender.extend_from_slice(&[0x48, 0xB8]);
    sender.extend_from_slice(&ipi_base.to_le_bytes());
    sender.extend_from_slice(&[0xB9, 0x01, 0x42, 0x00, 0x00]);
    sender.extend_from_slice(&[0x48, 0x89, 0x08]);
    sender.push(0xF4);

    // Core 1: poll own mailbox (base + 0x100 + 1*8) until non-zero,
    // store the observed value at COUNTER, then halt.
    //   movabs rax, mailbox       (48 B8 imm64)
    //   movabs rdi, COUNTER       (48 BF imm64)
    // loop:
    //   mov rcx, [rax]            (48 8B 08)
    //   test rcx, rcx             (48 85 C9)
    //   jz loop                    (74 F8)
    //   mov [rdi], rcx            (48 89 0F)
    //   hlt                       (F4)
    let mailbox = ipi_base + 0x100 + 8;
    let mut receiver = Vec::new();
    receiver.extend_from_slice(&[0x48, 0xB8]);
    receiver.extend_from_slice(&mailbox.to_le_bytes());
    receiver.extend_from_slice(&[0x48, 0xBF]);
    receiver.extend_from_slice(&COUNTER.to_le_bytes());
    receiver.extend_from_slice(&[0x48, 0x8B, 0x08]);
    receiver.extend_from_slice(&[0x48, 0x85, 0xC9]);
    receiver.extend_from_slice(&[0x74, 0xF8]);
    receiver.extend_from_slice(&[0x48, 0x89, 0x0F]);
    receiver.push(0xF4);

    sys.boot_image(&sender, 0x0010_0000).unwrap();
    sys.boot_image(&receiver, 0x0011_0000).unwrap();

    let rx = sys.events().subscribe();
    // Receiver first: it spins until the IPI lands, whatever the host
    // scheduling order is.
    let reports = sys.run_cores(&[
        CoreSpec {
            core: 1,
            entry: 0x0011_0000,
            budget: 50_000_000,
        },
        CoreSpec {
            core: 0,
            entry: 0x0010_0000,
            budget: 100,
        },
    ]);

    for r in &reports {
        assert!(
            matches!(r.outcome, RunOutcome::Halted { .. }),
            "core {} did not halt: {:?}",
            r.core,
            r.outcome
        );
    }
    // The receiver observed pending-bit | vector (0x142 for vector 0x42).
    assert_eq!(counter_of(&sys), 0x142, "receiver saw the IPI mailbox");
    assert!(!sys.ipi().pending(1), "the poll consumed the mailbox");

    // The delivery was mirrored onto the host event bus.
    let mut saw_ipi = false;
    while let Ok(ev) = rx.try_recv() {
        if let Event::IpiSent { target, vector } = ev {
            assert_eq!((target, vector), (1, 0x42));
            saw_ipi = true;
        }
    }
    assert!(saw_ipi, "IpiSent must be mirrored on the event bus");
}

#[test]
fn mixed_tier_cores_share_the_bus_exactly() {
    // Core 0 executes on the JIT tier, core 1 on the interpreter tier.
    // Unliftable instructions (the LOCK inc) fall back to precise
    // interpreter steps through the *same* SharedGuest port, so both tiers
    // funnel into the same hardware atomics.
    use pdg_core::smp::SharedGuest;
    use pdg_cpu::{CpuState, Interpreter, MemoryAccess};
    use pdg_jit::Jit;
    use std::sync::Arc;

    let mut cfg = config();
    cfg.cpu.cores = 2;
    cfg.cpu.threads = 2;
    let sys = SmpSystem::new(cfg).unwrap();
    let code = atomic_loop(500);
    sys.boot_image(&code, 0x0010_0000).unwrap();

    let pml4 = sys.config().memory.pml4_base;
    let stack_top = sys.config().boot.stack_top;

    let bus_interp = Arc::clone(sys.bus());
    let bus_jit = Arc::clone(sys.bus());

    let interp_thread = std::thread::spawn(move || {
        let mut guest = SharedGuest::new(bus_interp, pml4);
        let mut cpu = Interpreter::new(CpuState::new(0x0010_0000, stack_top));
        cpu.run(&mut guest, 100_000)
    });

    let jit_thread = std::thread::spawn(move || {
        let mut guest = SharedGuest::new(bus_jit, pml4);
        let port = &mut guest as &mut dyn MemoryAccess as *mut dyn MemoryAccess;
        let mut jit = Jit::new(CpuState::new(0x0010_0000, stack_top - 0x1_0000), port);
        jit.run(&mut guest, 100_000)
    });

    let oi = interp_thread.join().unwrap();
    let oj = jit_thread.join().unwrap();
    assert!(matches!(oi, RunOutcome::Halted { .. }), "interp: {oi:?}");
    assert!(matches!(oj, RunOutcome::Halted { .. }), "jit: {oj:?}");
    assert_eq!(
        counter_of(&sys),
        1000,
        "500 increments per core, zero lost updates across tiers"
    );
}
