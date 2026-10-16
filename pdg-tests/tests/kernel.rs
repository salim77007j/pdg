//! Phase-5 integration tests: the emulated CPU runs a real ELF64 process
//! that calls Orbis syscalls, reads files through the read-only PFS,
//! manages anonymous memory, drains kernel event queues, and receives
//! NVMe completions through guest-programmed MSI-X interrupts — every
//! stage verified end to end, exactly the tier-1 pattern every phase
//! before this one established.

use pdg_frontend::kernel_session::{KernelOutcome, KernelSession};
use pdg_kernel::demo::layout;

#[test]
fn cpu_boots_a_process_and_drives_the_kernel_plane() {
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let session = KernelSession::new(&ssd).expect("session");
    let report = session.run();

    // The process must exit cleanly with code 0.
    match &report.outcome {
        KernelOutcome::Exited { code, instructions } => {
            assert_eq!(*code, 0);
            assert!(
                *instructions < 200_000,
                "suspiciously long: {instructions} instructions"
            );
        }
        other => panic!("demo process did not exit cleanly: {other:?}"),
    }

    // Every stage stored a pass.
    let names = [
        "console write",
        "getpid",
        "open over the PFS",
        "read + contents",
        "lseek END/SET",
        "close",
        "getdirentries stream",
        "mmap + roundtrip",
        "mprotect cycles",
        "munmap",
        "kqueue/kevent",
        "RELATIVE relocation",
        "TPOFF64 + fs TLS",
        "IDT + MSI-X + NVMe",
    ];
    for (i, (status, name)) in report.statuses.iter().zip(names.iter()).enumerate() {
        assert_eq!(*status, 1, "stage {i} ({name}) failed");
    }

    // The console transcript: the stage-0 probe plus the final banner.
    assert_eq!(report.console, format!("PHAS{}", layout::BANNER));

    // The kernel really served syscalls; the loader really relocated.
    assert!(
        report.syscalls >= 30,
        "suspiciously few syscalls: {}",
        report.syscalls
    );
    assert_eq!(report.relocations_applied, 2);

    // The interrupt really went through the guest IDT, and the event
    // queue really delivered.
    assert!(report.interrupt_delivered);
    assert!(report.equeue_event_drained);

    // The SSD really moved the marker: one write + one read of 512 bytes
    // (plus the controller's own enable path).
    assert_eq!(report.ssd_stats.writes, 1);
    assert_eq!(report.ssd_stats.reads, 1);
    assert_eq!(report.ssd_stats.write_bytes, 512);
    assert_eq!(report.ssd_stats.read_bytes, 512);
}

#[test]
fn running_twice_is_deterministic() {
    // A fresh session over the same SSD image produces the same outcome:
    // no hidden state leaks between runs (the marker sector is rewritten
    // by the demo each time).
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let first = KernelSession::new(&ssd).unwrap().run();
    let second = KernelSession::new(&ssd).unwrap().run();
    assert_eq!(first.outcome, second.outcome);
    assert_eq!(first.statuses, second.statuses);
    assert_eq!(first.console, second.console);
}
