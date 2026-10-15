# PDG — A PlayStation 5 Emulator in Rust

<div align="center">

[![CI](https://github.com/salim77007j/pdg/actions/workflows/ci.yml/badge.svg)](https://github.com/salim77007j/pdg/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange)

**PDG** is a next-generation, high-performance PlayStation 5 emulator
written entirely in Rust.

*Impossible is just unwritten code.*

</div>

---

## ⚖️ Legal notice — read this first

**PDG is an independent emulator. We do not distribute BIOS, firmware, or game files.**

**Users must legally own and dump their own PS5 system files and games.**

**This emulator contains no proprietary Sony code.**

Emulators are legal tools (established precedent: *Sony v. Connectix*, 2000; *Sony v. Bleem*, 2001). PDG is original, clean-room engineering: every line is written from scratch against publicly documented hardware behavior. The project is GPL-3.0 to keep it — and any fork — free forever.

If you do not own a PS5 and the games you dump, you may not use PDG to run them. piracy is neither endorsed nor assisted here.

## Status: Phase 1 of 6 (infrastructure)

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Foundation: workspace, core framework, memory subsystem, boot harness, CI | ✅ **complete** |
| 2 | CPU: full x86-64 ISA, JIT recompiler, 8 cores/16 threads | ⏳ next |
| 3 | GPU: PM4 command translation, RDNA 2 shader recompilation, wgpu backend | ⬜ |
| 4 | I/O: custom-SSD model, DualSense, Tempest audio, network | ⬜ |
| 5 | Kernel: Orbis syscall translation, ELF/PRX loader, PFS | ⬜ |
| 6 | Frontend: launcher GUI, optimization, release engineering | ⬜ |

**PDG cannot run commercial PS5 games yet.** Phase 1 delivers the machine's foundation; see [PHASE_1_REPORT.md](PHASE_1_REPORT.md) for an honest, detailed account of exactly what works and what does not.

### What genuinely works today

- **Memory subsystem** — real x86-64 four-level page-table walks (PML4→PDPT→PD→PT) over guest RAM, permission semantics (present/read-write/no-execute enforced at *every* level), a software TLB, an MMIO bus with devices, ROM windows, and a DMA engine that crosses region boundaries correctly.
- **CPU core** — fetch → decode (iced-x86) → execute interpreter for the x86-64 integer heart: data movement, ALU with architecturally exact flags, shifts, stack, control flow, `syscall` hooks, `hlt`. Unimplemented instructions raise `#UD`; nothing silently no-ops.
- **Boot harness** — `pdg-boot --bios <your-dump>` loads your legally-dumped firmware into the ROM window and starts executing at the reset vector `0xFFFF_FFF0` through the entire stack.
- **Cross-cutting** — layered TOML configuration, `tracing` logging, event bus, virtual clock + scheduler, 93 tests, property-based ALU fuzzing, criterion benchmarks, CI on Linux and Windows.

## Quick start

```bash
# Build everything (Rust 1.85+)
cargo build --release

# Disassemble any blob as x86-64
cargo run --release --bin pdg-disasm -- /path/to/blob

# Boot a raw image through the full stack (MMU, page tables, MMIO)
cargo run --release --bin pdg-boot -- --image /path/to/image.bin

# Boot your own legally-dumped PS5 BIOS
cargo run --release --bin pdg-boot -- --bios /path/to/your/dump.bios

# Tests / lints / benchmarks
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo bench --workspace
```

A minimal boot image that prints to the PDG MMIO console looks like this
(hand-assembled; see `pdg-tests/tests/boot.rs` for the source bytes):

```
0x100000: mov rsi, 0x100040        ; string pointer
0x10000A: mov rdi, 0xF0000000      ; MMIO console
0x10001B: mov al, [rsi]            ; ┐
0x10001D: test al, al               ; │ copy until NUL
0x10001F: jz done                  ; │
0x100021: mov [rdi], al             ; │
0x100023: inc rsi                   ; │
0x100026: inc rdi                   ; │
0x100029: jmp 0x10001B              ; ┘
0x10002B: hlt                       ; → console prints "Hello from PDG"
```

## Workspace layout

```
pdg/
├── pdg-core/      # Config, logging, events, clocks, System orchestrator
├── pdg-cpu/       # Zen 2 (x86-64) register model + interpreter
├── pdg-gpu/       # RDNA 2 framework: device traits + PM4 framing
├── pdg-memory/    # Guest-physical bus, MMU, page tables, DMA
├── pdg-io/        # Host block devices, async I/O, input model
├── pdg-audio/     # Audio backend abstraction
├── pdg-kernel/    # HLE syscall dispatch framework, exception vectors
├── pdg-frontend/  # Game library model (GUI lands in Phase 6)
├── pdg-tests/     # Cross-crate integration tests
└── pdg-tools/     # pdg-boot (boot harness), pdg-disasm
```

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Technology

Rust (edition 2024) · iced-x86 (decoding) · memmap2 (guest RAM) · tokio (async I/O) · serde + toml (config) · tracing (logging) · criterion + proptest (QA). Phase 2 adds cranelift/dynasm (JIT); Phase 3 adds wgpu + naga (GPU). Rationale for every choice lives in the phase reports.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The bar is simple: real code, real tests, honest reporting.

## License

[GPL-3.0-or-later](LICENSE)
