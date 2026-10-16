//! Phase-4 integration tests: the emulated CPU drives the whole I/O stack
//! end to end — NVMe queues through doorbell MMIO with PRP DMA, the
//! DualSense window, and the Tempest-shaped audio hydrate path — exactly
//! the tier-1 pattern the GPU phase established.

use pdg_frontend::io_session::{IoSession, SessionOutcome};
use pdg_io::demo::layout as demo;
use pdg_io::dualsense::{Dpad, DualSenseState};
use pdg_io::input;

/// The pad state the demo driver expects: CROSS held, left stick pushed
/// fully right.
fn demo_pad() -> DualSenseState {
    DualSenseState {
        buttons: input::button::CROSS,
        dpad: Dpad::Neutral,
        left_stick: [255, 128],
        right_stick: [128, 128],
        ..DualSenseState::neutral()
    }
}

#[test]
fn cpu_drives_storage_input_and_audio_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let ssd = tmp.path().join("ssd.img");
    let session = IoSession::new(&ssd).expect("session");

    let report = session.run(demo_pad());

    // The driver must halt cleanly, not fault or spin out its budget.
    match &report.outcome {
        SessionOutcome::Halted {
            instructions: n, ..
        } => {
            assert!(*n < 1_000_000, "suspiciously long: {n} instructions");
        }
        other => panic!("demo driver did not halt cleanly: {other:?}"),
    }

    // Every stage stored a pass.
    let names = [
        "controller enable",
        "identify serial",
        "io queue creation",
        "write/read roundtrip",
        "pad read",
        "audio hydration",
    ];
    for (i, (status, name)) in report.statuses.iter().zip(names.iter()).enumerate() {
        assert_eq!(*status, 1, "stage {i} ({name}) failed");
    }

    // The console transcript.
    assert_eq!(report.console, "PHASE4 IO OK");

    // The SSD really moved the marker: one write + one read of 512 bytes.
    assert_eq!(report.ssd_stats.writes, 1);
    assert_eq!(report.ssd_stats.reads, 1);
    assert_eq!(report.ssd_stats.write_bytes, 512);
    assert_eq!(report.ssd_stats.read_bytes, 512);

    // The audio the guest hydrated, rendered stereo with the centered
    // constant-power pan (cos(π/4) ≈ 0.70710678).
    let center = std::f32::consts::FRAC_1_SQRT_2;
    let expected: Vec<i16> = [0.25f32, 0.5, -0.75]
        .iter()
        .flat_map(|s| [*s; 2])
        .map(|s| (s * center * 32767.0) as i16)
        .collect();
    assert_eq!(report.audio, expected, "hydrated samples rendered stereo");

    // The read-back data really landed in the guest's second buffer.
    let marker = pdg_memory::bus::AccessWidth::W64;
    let v = session
        .bus
        .read(marker, demo::DATA_B)
        .expect("marker readback");
    assert_eq!(v, u64::from_le_bytes(*b"PDGSSDIO"));

    // The backing file holds the marker at LBA 4.
    let disk = std::fs::read(&ssd).unwrap();
    assert_eq!(&disk[4 * 512..4 * 512 + 8], b"PDGSSDIO");
}

#[test]
fn nvme_payload_uses_only_verified_instruction_forms() {
    // Disassemble the demo payload through iced-x86 and assert every
    // instruction is in the documented emitter set (the Phase-2
    // interpreter's verified subset for this driver).
    use pdg_cpu::iced_x86::{Decoder, DecoderOptions, Mnemonic};

    let mmio = pdg_io::demo::DemoMmio {
        nvme: 0xE000_0000,
        pad: 0xE100_0000,
        audio: 0xE200_0000,
        console: 0xF000_0000,
    };
    let code = pdg_io::demo::io_demo_payload(&mmio);
    // The blob ends with the string; disassemble the code part.
    let code_len = code.len() - b"PHASE4 IO OK\0".len();
    let mut decoder = Decoder::with_ip(64, &code[..code_len], demo::CODE, DecoderOptions::NONE);
    let allowed = std::collections::HashSet::from([
        Mnemonic::Mov,
        Mnemonic::Push,
        Mnemonic::Pop,
        Mnemonic::Cmp,
        Mnemonic::Test,
        Mnemonic::Je,
        Mnemonic::Jne,
        Mnemonic::Jmp,
        Mnemonic::Inc,
        Mnemonic::Hlt,
    ]);
    let mut count = 0usize;
    while decoder.can_decode() {
        let instr = decoder.decode();
        assert!(
            allowed.contains(&instr.mnemonic()),
            "unexpected instruction {:?} at RIP {:#x}",
            instr.mnemonic(),
            instr.ip()
        );
        count += 1;
    }
    assert!(
        count > 100,
        "the driver is substantial: {count} instructions"
    );
}
