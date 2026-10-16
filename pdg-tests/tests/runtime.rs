//! Phase-6 integration tests: the emulated CPU runs a multi-module,
//! multi-threaded process — the main executable with a `DT_NEEDED` closure
//! loaded through the kernel's dynlib, a worker thread created through
//! `thr_create`, umtx mutex/futex choreography with real parking, `stat`
//! over the PFS — and the session snapshot/restore proves the run is
//! deterministic.

use pdg_frontend::kernel_session::{RuntimeOutcome, RuntimeSession};
use pdg_kernel::demo::layout6;

fn stage_names() -> [&'static str; layout6::ST_COUNT as usize] {
    [
        "console write",
        "getpid",
        "dynlib_get_list",
        "dynlib_get_info",
        "dlsym + indirect call",
        "JUMP_SLOT call through the GOT",
        "GLOB_DAT + depth-2 dlsym",
        "fstat",
        "stat",
        "thr_self",
        "main TLS through fs:",
        "thr_create",
        "worker thr_self == 2",
        "worker TLS bump",
        "umtx choreography",
        "TLS intact after join",
    ]
}

#[test]
fn cpu_runs_a_modular_multithreaded_process() {
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let session = RuntimeSession::new(&ssd).expect("session");
    let (report, _) = session.run(None);

    match &report.outcome {
        RuntimeOutcome::Exited { code, instructions } => {
            assert_eq!(*code, 0);
            assert!(
                *instructions < 400_000,
                "suspiciously long: {instructions} instructions"
            );
        }
        other => panic!("runtime demo did not exit cleanly: {other:?}"),
    }

    for (i, (status, name)) in report.statuses.iter().zip(stage_names().iter()).enumerate() {
        assert_eq!(*status, 1, "stage {i} ({name}) failed");
    }

    // The console transcript: the stage-0 probe plus the final banner.
    assert_eq!(report.console, format!("PHAS{}", layout6::BANNER));

    // Three modules — main plus its two-deep needed closure.
    assert_eq!(report.modules, 3);
    // Every module's relocations: main's three (JUMP_SLOT, GLOB_DAT,
    // TPOFF64) plus the PRX's two (GLOB_DAT, TPOFF64).
    assert_eq!(report.relocations, 5);
    // Two threads lived: main and the worker.
    assert_eq!(report.threads_created, 2);
    // None remain.
    assert_eq!(report.threads, 0);
    // The exact count: 4 file/module calls + 3 dlsym + 2 stat-family +
    // 4 open/identity + 8 umtx + 3 thread + exit — the whole dance.
    assert_eq!(report.syscalls, 24);
}

#[test]
fn runtime_run_is_deterministic() {
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let first = RuntimeSession::new(&ssd).unwrap().run(None).0;
    let second = RuntimeSession::new(&ssd).unwrap().run(None).0;
    assert_eq!(first.outcome, second.outcome);
    assert_eq!(first.statuses, second.statuses);
    assert_eq!(first.console, second.console);
}

#[test]
fn snapshot_restore_continues_deterministically() {
    // Run once to completion; run again with a mid-program snapshot after
    // the dlsym stage, restore the snapshot into a *fresh* machine, and
    // continue — the final states must agree exactly.
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let full = RuntimeSession::new(&ssd).unwrap().run(None).0;

    let snapper = RuntimeSession::new(&ssd).unwrap();
    let (partial, snapshot) = snapper.run(Some(layout6::ST_DLSYM));
    let snapshot = snapshot.expect("the snapshot point is reached before the budget");
    // The partial run continued to completion after capturing.
    assert_eq!(partial.outcome, full.outcome);
    assert_eq!(partial.statuses, full.statuses);

    // Restore into a brand-new machine (fresh devices, fresh kernel) and
    // resume from the captured CPU contexts. The instruction counts
    // differ by construction (the resumed run's counter starts at zero
    // mid-program), so outcomes compare by exit code.
    let resumed_session = RuntimeSession::new(&ssd).unwrap();
    resumed_session.restore(&snapshot);
    let resumed = resumed_session.run_restored(&snapshot);
    match (&resumed.outcome, &full.outcome) {
        (RuntimeOutcome::Exited { code: a, .. }, RuntimeOutcome::Exited { code: b, .. }) => {
            assert_eq!(a, b)
        }
        (a, b) => panic!("outcomes diverged: {a:?} vs {b:?}"),
    }
    assert_eq!(resumed.statuses, full.statuses);
    assert_eq!(resumed.console, full.console);
    assert_eq!(resumed.threads_created, full.threads_created);

    // The snapshot is honest about its moment: captured before the
    // worker existed, with a substantial RAM footprint (page tables,
    // segments, the module images, both stacks).
    assert_eq!(snapshot.threads.len(), 1);
    // Page tables, both module images, the data segment, the TLS arena
    // and the stacks — the working set of a mid-program machine.
    assert!(
        snapshot.ram.len() >= 15,
        "suspiciously small RAM capture: {} pages",
        snapshot.ram.len()
    );
    // And the captured console holds the stage-0 probe prefix only
    // (the banner prints at the end).
    assert!(snapshot.console.starts_with("PHAS"));
    assert!(!snapshot.console.contains("RUNTIME OK"));
}

#[test]
fn snapshot_file_format_roundtrips_and_rejects_corruption() {
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let session = RuntimeSession::new(&ssd).unwrap();
    let (_, snapshot) = session.run(Some(pdg_kernel::demo::layout6::ST_DLSYM));
    let snapshot = snapshot.expect("snapshot captured");

    let bytes = pdg_frontend::kernel_session::SessionSnapshot::to_bytes(&snapshot);
    assert!(bytes.starts_with(b"PDGS"));
    assert!(
        bytes.len() > 4096,
        "a real machine state: {} bytes",
        bytes.len()
    );
    // Round-trip.
    let decoded =
        pdg_frontend::kernel_session::SessionSnapshot::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.threads.len(), snapshot.threads.len());
    assert_eq!(decoded.threads[0].tid, snapshot.threads[0].tid);
    assert_eq!(decoded.threads[0].gpr, snapshot.threads[0].gpr);
    assert_eq!(decoded.ram.len(), snapshot.ram.len());
    assert_eq!(decoded.console, snapshot.console);
    // The decoded kernel tables carry the module list and the thread
    // table through the binary codec.
    assert_eq!(
        decoded.kernel.dynlib.modules.len(),
        snapshot.kernel.dynlib.modules.len()
    );
    assert_eq!(
        decoded.kernel.dynlib.modules[0].name,
        snapshot.kernel.dynlib.modules[0].name
    );
    assert_eq!(
        decoded.kernel.threads.live_tids().len(),
        snapshot.kernel.threads.live_tids().len()
    );
    // Corruption is refused: flip one payload byte, the digest fails.
    let mut corrupt = bytes.clone();
    let mid = corrupt.len() / 2;
    corrupt[mid] ^= 0xFF;
    assert!(pdg_frontend::kernel_session::SessionSnapshot::from_bytes(&corrupt).is_none());
    // So is a truncated file.
    assert!(
        pdg_frontend::kernel_session::SessionSnapshot::from_bytes(&bytes[..bytes.len() - 40])
            .is_none()
    );
    // And a resumed run from the DECODED snapshot still matches the
    // full-run final state (the file format loses nothing the live
    // snapshot carried).
    let full = RuntimeSession::new(&ssd).unwrap().run(None).0;
    let resumed_session = RuntimeSession::new(&ssd).unwrap();
    resumed_session.restore(&decoded);
    let resumed = resumed_session.run_restored(&decoded);
    assert_eq!(resumed.statuses, full.statuses);
    assert_eq!(resumed.console, full.console);
}
