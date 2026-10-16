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

## Status: Phase 3 of 6 (GPU)

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Foundation: workspace, core framework, memory subsystem, boot harness, CI | ✅ **complete** |
| 2 | CPU: full integer ISA, SSE→AVX2 SIMD, cranelift JIT, SMP (8 cores/16 threads) | ✅ **complete** |
| 3 | GPU: PM4 command processor, GCN/RDNA shader recompilation to WGSL, wgpu backend, GNM ring | ✅ **complete** |
| 4 | I/O: custom-SSD model, DualSense, Tempest audio, network | ⬜ |
| 5 | Kernel: Orbis syscall translation, ELF/PRX loader, PFS | ⬜ |
| 6 | Frontend: launcher GUI, optimization, release engineering | ⬜ |

**PDG cannot run commercial PS5 games yet.** Phase 3 delivers the GPU translation stack — a PM4 command processor with the verified opcode table, GCN→SPIR-V→WGSL shader recompilation, a GNM ring the emulated CPU drives through MMIO, and an offscreen wgpu backend that renders into guest RAM — but the kernel and I/O stacks are still ahead; see [PHASE_3_REPORT.md](PHASE_3_REPORT.md) for an honest, detailed account of exactly what works and what does not.

### What genuinely works today

- **CPU interpreter (reference tier)** — the x86-64 integer ISA complete (MUL/DIV families, bit tests, string ops with REP/REPE/REPNE restart semantics, CMOVcc/SETcc, atomics with LOCK, CPUID/RDTSC/MSR, exceptions delivered through the guest IDT with architectural frames and IRET) plus SSE→SSE4.2 and the AVX2/FMA/AES-NI/PCLMULQDQ SIMD families with VEX upper-half zeroing rules — all verified instruction-by-instruction against the architecture manuals, including AES against FIPS-197 test vectors.
- **CPU JIT (execution tier)** — cranelift recompilation of guest basic blocks with loop formation, an inline software TLB over host memory windows, precise fallback to the interpreter for anything tier-1 does not lift, self-modifying-code invalidation, and differential fuzzing against the interpreter oracle. Benchmarks: **~1.0 ns per instruction** on integer loops (≈**48×** the interpreter on ALU code, **76×** on load/store code, **6×** on call-heavy code).
- **SMP** — `SmpSystem` runs all 16 logical cores as host threads over one shared bus, with per-core MMUs, real cross-core hardware atomics (`lock inc`/`xadd`/`cmpxchg` — zero lost updates, tested at 16 threads), and guest-visible IPIs through an MMIO device mirrored onto the host event bus.
- **Memory subsystem** — real x86-64 four-level page-table walks (PML4→PDPT→PD→PT) over guest RAM, permission semantics (present/read-write/no-execute enforced at *every* level), a software TLB, an MMIO bus with devices, ROM windows, and a DMA engine that crosses region boundaries correctly.
- **GPU command path** — the PM4 command processor executes real packet streams (four `SET_*_REG` spaces, type-0 writes, `DRAW_INDEX_AUTO`/`DRAW_INDEX_2`, `DISPATCH_DIRECT`, indirect-buffer chains, fences/synchronization, `WRITE_DATA`/`DMA_DATA`/`COPY_DATA`, `CLEAR_STATE`) against a gfx10.3 register model with offsets verified against the public kernel `gc_10_3_0` headers. The emulated CPU drives the GNM ring through MMIO — programs the ring, rings the doorbell, polls the fence — exactly the console driver contract. Unverified opcodes are refused with precise errors.
- **Shader recompilation** — a GCN/RDNA decoder (encoding layouts verified against LLVM's public AMDGPU backend) lifts guest shader programs to SPIR-V, validates through naga, and emits WGSL for the wgpu backend; a content-addressed on-disk cache keyed by SHA-256 of the guest ISA. The lifted subset covers the Phase-3 mini-GNM vertex-fetch contract; everything outside it is refused by name, never faked.
- **Rendering** — the wgpu backend renders draws offscreen through Vulkan (or any adapter wgpu supports) and reads the target back into guest RAM; the end-to-end test renders a fullscreen triangle through a guest shader and verifies **every pixel** of the 64×64 target. On adapter-less machines (CI without a GPU) the same suite degrades to the honest `NullGpu` backend and still verifies the command path.
- **Boot harness** — `pdg-boot --bios <your-dump>` loads your legally-dumped firmware into the ROM window and starts executing at the reset vector `0xFFFF_FFF0` through the entire stack.
- **Cross-cutting** — layered TOML configuration, `tracing` logging, event bus, virtual clock + scheduler, **224 tests** (unit, integration, differential fuzz, SMP, GPU end-to-end), criterion benchmarks, CI on Linux and Windows.

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
├── pdg-core/      # Config, logging, events, clocks, System + SMP System orchestrators
├── pdg-cpu/       # Zen 2 (x86-64) model: interpreter, full ISA + SIMD, CPUID
├── pdg-jit/        # cranelift JIT tier: block lifting, inline TLB, SMC, fallback
├── pdg-gpu/       # PM4 command processor, gfx10.3 register model, GCN shader
│                  # recompilation (ISA→SPIR-V→WGSL), shader cache, wgpu backend,
│                  # GNM ring MMIO device
├── pdg-memory/    # Guest-physical bus, MMU, page tables, DMA
├── pdg-io/        # Host block devices, async I/O, input model
├── pdg-audio/     # Audio backend abstraction
├── pdg-kernel/    # HLE syscall dispatch framework, exception vectors
├── pdg-frontend/  # Game library model (GUI lands in Phase 6)
├── pdg-tests/     # Cross-crate integration tests
└── pdg-tools/     # pdg-boot (boot harness), pdg-disasm, pdg-wgsldump
```

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Technology

Rust (edition 2024) · iced-x86 (decoding) · cranelift (JIT codegen, since Phase 2) · wgpu + naga + rspirv + sha2 (GPU, since Phase 3) · memmap2 (guest RAM) · tokio (async I/O) · serde + toml (config) · tracing (logging) · criterion + proptest (QA). Rationale for every choice lives in the phase reports.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The bar is simple: real code, real tests, honest reporting.

## License

[GPL-3.0-or-later](LICENSE)
