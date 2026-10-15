//! Boot integration tests: full stack — interpreter → MMU → page tables →
//! guest-physical bus → MMIO console.

use pdg_core::config::PdgConfig;
use pdg_core::system::System;
use pdg_cpu::RunOutcome;

const IMAGE_BASE: u64 = 0x0010_0000;

/// Phase-1 boot payload, hand-assembled x86-64 (true 64-bit immediates —
/// 64-bit mode sign-extends imm32/disp32, so the MMIO address must come
/// from a `mov r64, imm64`):
///
/// ```text
/// 0x100000: mov rsi, 0x100040        ; string pointer
/// 0x10000A: mov rdi, 0xF0000000      ; MMIO console (imm64!)
/// 0x100014: mov rcx, 16              ; register set-up coverage
/// 0x10001B: mov al, [rsi]             ; ┐
/// 0x10001D: test al, al               ; │ copy loop until NUL
/// 0x10001F: jz 0x10002B               ; │ (zero → done)
/// 0x100021: mov [rdi], al             ; │ write char to console
/// 0x100023: inc rsi                   ; │
/// 0x100026: inc rdi                   ; │
/// 0x100029: jmp 0x10001B              ; ┘
/// 0x10002B: hlt
/// 0x100040: "Hello from PDG\0"
/// ```
const HELLO_PAYLOAD: &[u8] = &[
    0x48, 0xBE, 0x40, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rsi, 0x100040
    0x48, 0xBF, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, // mov rdi, 0xF0000000
    0x48, 0xC7, 0xC1, 0x10, 0x00, 0x00, 0x00, // mov rcx, 16
    0x8A, 0x06, // mov al, [rsi]        (0x10001B)
    0x84, 0xC0, // test al, al
    0x74, 0x0A, // jz 0x10002B
    0x88, 0x07, // mov [rdi], al
    0x48, 0xFF, 0xC6, // inc rsi
    0x48, 0xFF, 0xC7, // inc rdi
    0xEB, 0xF0, // jmp 0x10001B
    0xF4, // hlt                       (0x10002B)
];
const HELLO_STRING: &[u8] = b"Hello from PDG\0";

fn hello_image() -> Vec<u8> {
    let mut image = HELLO_PAYLOAD.to_vec();
    image.resize(0x40, 0x00); // pad to the string address
    image.extend_from_slice(HELLO_STRING);
    image
}

#[test]
fn boots_hello_payload_through_full_stack() {
    let mut system = System::new(PdgConfig::default()).expect("system");
    let entry = system.boot_image(&hello_image(), IMAGE_BASE).expect("boot");
    assert_eq!(entry, IMAGE_BASE);

    let report = system.run(1_000_000);
    match report.outcome {
        RunOutcome::Halted { instructions, rip } => {
            assert_eq!(rip, 0x0010_002B, "halted at the expected hlt");
            // 3 setup + 14 iterations × 7 + 3 (final test) + 1 hlt.
            assert_eq!(instructions, 3 + 14 * 7 + 3 + 1);
        }
        other => panic!("expected Halted, got {other:?}"),
    }
    assert_eq!(report.console, "Hello from PDG");
}

#[test]
fn dma_moves_boot_string_inside_guest_ram() {
    let mut system = System::new(PdgConfig::default()).expect("system");
    system.boot_image(&hello_image(), IMAGE_BASE).expect("boot");

    let dma = &mut system.dma;
    let bus = &mut system.mem.bus;
    dma.submit(bus, IMAGE_BASE + 0x40, IMAGE_BASE + 0x1000, 15)
        .expect("dma");
    let mut out = [0u8; 15];
    bus.ram_read_buf(IMAGE_BASE + 0x1000, &mut out).unwrap();
    assert_eq!(&out, b"Hello from PDG\0");
    assert_eq!(dma.history().len(), 1);
}

#[test]
fn garbage_bios_executes_and_reports_gracefully() {
    // A file of zeros decodes as `add [rax], al` at the reset vector and
    // executes until RIP walks off the end of the ROM window at exactly
    // 4 GiB — where the machine raises a clean architectural #PF instead
    // of hanging or crashing the host.
    let tmp = tempfile::tempdir().unwrap();
    let bios = tmp.path().join("garbage.bios");
    std::fs::write(&bios, vec![0u8; 0x1000]).unwrap();

    let mut system = System::new(PdgConfig::default()).expect("system");
    let entry = system.load_bios(&bios).expect("bios");
    assert_eq!(entry, 0xFFFF_FFF0);

    let report = system.run(100_000);
    match report.outcome {
        RunOutcome::Fault(pdg_cpu::Fault::PageFault { addr, code }) => {
            assert_eq!(addr, 0x1_0000_0000, "ran off the end of the ROM window");
            assert_eq!(code, 0, "not-present fetch above 4 GiB");
        }
        other => panic!("expected clean page fault at 4 GiB, got {other:?}"),
    }
}

#[test]
fn missing_bios_errors_with_guidance() {
    let mut system = System::new(PdgConfig::default()).expect("system");
    let err = system
        .load_bios(std::path::Path::new("/nonexistent/ps5.bios"))
        .expect_err("missing bios must error");
    let msg = err.to_string();
    assert!(msg.contains("legally-dumped"), "message: {msg}");
    assert!(msg.contains("does not distribute"), "message: {msg}");
}

#[test]
fn syscall_integration_through_system() {
    // mov eax, 21 ; syscall ; hlt — with a kernel dispatcher hooked up that
    // doubles RAX through the FreeBSD-style convention.
    let code = [
        0xB8, 0x15, 0x00, 0x00, 0x00, // mov eax, 21
        0x0F, 0x05, // syscall
        0xF4, // hlt
    ];
    let mut system = System::new(PdgConfig::default()).expect("system");
    system.boot_image(&code, IMAGE_BASE).expect("boot");

    let mut dispatcher = pdg_kernel::SyscallDispatcher::new();
    dispatcher.register(21, |cpu, _| {
        let v = cpu.reg(pdg_cpu::iced_x86::Register::RAX).unwrap_or(0);
        Ok(v as i64 * 2)
    });
    system.set_syscall_hook(Box::new(dispatcher));

    let report = system.run(100);
    assert!(
        matches!(report.outcome, RunOutcome::Halted { .. }),
        "got {:?}",
        report.outcome
    );
    assert_eq!(
        system.cpu_state().reg(pdg_cpu::iced_x86::Register::RAX),
        Some(42)
    );
}

#[test]
fn page_fault_reaches_the_run_report() {
    // The first 2 MiB is RW+X, the rest of RAM is RW but NX — executing
    // from an NX page must surface a #PF with the instruction-fetch bit.
    // We build that by branching far into an NX page we prepared with code.
    let mut system = System::new(PdgConfig::default()).expect("system");

    // mov rax, 0x400000 ; jmp rax  → 0x400000 is NX-mapped RAM.
    let code = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x40, 0x00, // mov rax, 0x400000
        0xFF, 0xE0, // jmp rax
    ];
    system.boot_image(&code, IMAGE_BASE).expect("boot");
    // Write an `hlt` into the NX page (physical write before execution).
    system.mem.bus.ram_write_buf(0x400_000, &[0xF4]).unwrap();

    let report = system.run(100);
    match report.outcome {
        RunOutcome::Fault(pdg_cpu::Fault::PageFault { addr, code }) => {
            assert_eq!(addr, 0x400_000);
            assert_eq!(code, 0x11, "present | instruction-fetch");
        }
        other => panic!("expected NX page fault, got {other:?}"),
    }
}
