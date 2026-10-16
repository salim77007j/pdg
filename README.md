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

## Status: Phase 6 of 6 (complete)

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Foundation: workspace, core framework, memory subsystem, boot harness, CI | ✅ **complete** |
| 2 | CPU: full integer ISA, SSE→AVX2 SIMD, cranelift JIT, SMP (8 cores/16 threads) | ✅ **complete** |
| 3 | GPU: PM4 command processor, GCN/RDNA shader recompilation to WGSL, wgpu backend, GNM ring | ✅ **complete** |
| 4 | I/O: NVMe-shaped SSD model, DualSense input, Tempest-shaped audio, I/O session | ✅ **complete** |
| 5 | Kernel: Orbis syscall translation, ELF/SELF loader, read-only PFS, kernel objects, MSI-X-shaped interrupts | ✅ **complete** |
| 6 | Runtime & frontend: dynlib/PRX modules, threads + umtx, snapshots, stat family, egui launcher, release engineering | ✅ **complete** |

**PDG cannot run commercial PS5 games yet.** The six planned planes are complete: foundation, CPU, GPU, I/O, kernel, and the Phase-6 runtime — a real ELF64 process with a `DT_NEEDED` closure of PRX shared libraries loaded through the kernel's dynlib layer (cross-module `JUMP_SLOT`/`GLOB_DAT` linking, `dlsym`, module info), a worker thread created through `thr_create` with its own copy of the aggregated TLS block, umtx mutex choreography with real parking and FIFO handoff, the verified `stat`/`fstat` layouts, and a session snapshot/restore that resumes a mid-program machine byte-identically — see [PHASE_6_REPORT.md](PHASE_6_REPORT.md) for an honest, detailed account of exactly what works and what does not.

### What genuinely works today

- **CPU interpreter (reference tier)** — the x86-64 integer ISA complete (MUL/DIV families, bit tests, string ops with REP/REPE/REPNE restart semantics, CMOVcc/SETcc, atomics with LOCK, CPUID/RDTSC/MSR, exceptions delivered through the guest IDT with architectural frames and IRET) plus SSE→SSE4.2 and the AVX2/FMA/AES-NI/PCLMULQDQ SIMD families with VEX upper-half zeroing rules — all verified instruction-by-instruction against the architecture manuals, including AES against FIPS-197 test vectors.
- **CPU JIT (execution tier)** — cranelift recompilation of guest basic blocks with loop formation, an inline software TLB over host memory windows, precise fallback to the interpreter for anything tier-1 does not lift, self-modifying-code invalidation, and differential fuzzing against the interpreter oracle. Benchmarks: **~1.0 ns per instruction** on integer loops (≈**48×** the interpreter on ALU code, **76×** on load/store code, **6×** on call-heavy code).
- **SMP** — `SmpSystem` runs all 16 logical cores as host threads over one shared bus, with per-core MMUs, real cross-core hardware atomics (`lock inc`/`xadd`/`cmpxchg` — zero lost updates, tested at 16 threads), and guest-visible IPIs through an MMIO device mirrored onto the host event bus.
- **Memory subsystem** — real x86-64 four-level page-table walks (PML4→PDPT→PD→PT) over guest RAM, permission semantics (present/read-write/no-execute enforced at *every* level), a software TLB, an MMIO bus with devices, ROM windows, and a DMA engine that crosses region boundaries correctly.
- **GPU command path** — the PM4 command processor executes real packet streams (four `SET_*_REG` spaces, type-0 writes, `DRAW_INDEX_AUTO`/`DRAW_INDEX_2`, `DISPATCH_DIRECT`, indirect-buffer chains, fences/synchronization, `WRITE_DATA`/`DMA_DATA`/`COPY_DATA`, `CLEAR_STATE`) against a gfx10.3 register model with offsets verified against the public kernel `gc_10_3_0` headers. The emulated CPU drives the GNM ring through MMIO — programs the ring, rings the doorbell, polls the fence — exactly the console driver contract. Unverified opcodes are refused with precise errors.
- **Shader recompilation** — a GCN/RDNA decoder (encoding layouts verified against LLVM's public AMDGPU backend) lifts guest shader programs to SPIR-V, validates through naga, and emits WGSL for the wgpu backend; a content-addressed on-disk cache keyed by SHA-256 of the guest ISA. The lifted subset covers the Phase-3 mini-GNM vertex-fetch contract; everything outside it is refused by name, never faked.
- **Rendering** — the wgpu backend renders draws offscreen through Vulkan (or any adapter wgpu supports) and reads the target back into guest RAM; the end-to-end test renders a fullscreen triangle through a guest shader and verifies **every pixel** of the 64×64 target. On adapter-less machines (CI without a GPU) the same suite degrades to the honest `NullGpu` backend and still verifies the command path; a dedicated CI job runs the pixel-verified tier on lavapipe.
- **Storage (Phase 4)** — a doorbell-driven NVMe 2.0-shaped SSD controller over a host file: the register file (CAP/VS/CC/CSTS/AQA/ASQ/ACQ, doorbells at `0x1000`), submission/completion queues with phase-tagged completions that wrap correctly, PRP-resolved DMA into guest RAM (first page + two-page special case + PRP-list chains), IDENTIFY Controller/Namespace, I/O queue creation/deletion, READ/WRITE with out-of-range refusal, ABORT (aborts nothing — nothing is ever in flight across a doorbell), GET FEATURES (Number-of-Queues). Everything outside the modeled set is refused with a precise NVMe status code, never guessed.
- **Input (Phase 4)** — the DualSense model: full state (buttons, dpad hat, sticks, analog triggers, two touch points, gyro/accel, battery), adaptive-trigger effects modeled by *behavior* (`resistance_at(pull, t)` per mode: feedback, weapon, vibration), a seqlock-shaped guest MMIO window, host mapping with deadzones, and HID report 0x01 encoding for the verified subset (touch/motion extensions deliberately not serialized until a real transport verifies their offsets).
- **Audio (Phase 4)** — the Tempest-shaped mix path: voice hydration, distance attenuation with constant-power panning, saturating master mix, a linear resampler with cross-block phase carry, and the guest-visible audio window (PCM blocks hydrated through MMIO, rendered to any `AudioBackend`); a cpal output stream sits behind the `backend-cpal` feature. HRTF needs a verified HRIR dataset and is honestly not faked.
- **I/O sessions** — `pdg-boot --demo-io <ssd.img>` boots a hand-assembled guest driver that enables the SSD, identifies it, creates I/O queues, writes and reads a sector through PRP DMA, reads the pad window, hydrates an audio block, and prints `PHASE4 IO OK` — the emulated CPU in the driver's seat across the whole I/O plane.
- **Kernel sessions (Phase 5)** — `pdg-boot --demo-kernel <ssd.img>` builds a real ELF64 (typed emitters, disassembler-verified instruction set), loads it through the loader, and runs it: the process writes to the console through `write(2)`, opens and reads files from a mounted read-only PFS, walks directories with `getdirentries`, manages anonymous memory with `mmap`/`mprotect`/`munmap`, registers on and drains a `kqueue`, reads relocated pointers and TLS through `fs:` addressing, then installs an IDT, programs an MSI-X vector, doorbells an NVMe write — and the completion arrives as an interrupt through that IDT; the handler runs, `iretq`s, and the driver reads the sector back and byte-compares it. Then it prints `PHASE5 KERNEL OK` and calls `exit(0)`.
- **Process runtime (Phase 6)** — the dynlib layer: the main executable plus its whole `DT_NEEDED` closure of PRX-shaped shared objects loads through the kernel (`sys_dynlib_load_prx` with the ABI the public payload SDKs pin), cross-module `JUMP_SLOT`/`GLOB_DAT` relocations bind against the others' exports (rtld search order), `dlsym` resolves by name across a two-deep dependency chain, and the module table answers `get_list`/`get_info` with the size-prefixed `OrbisKernelModuleInfo` layout. Threads: `thr_create` parses a genuine FreeBSD 9.2 `ucontext_t`, each thread gets its own copy of the aggregated TLS block (`fs:` addressing is per-thread), and `_umtx_op` parks would-blockers until a wake — the mutex unlock hands the lock FIFO to one waiter. `stat`/`fstat` fill the verified 120-byte FreeBSD 9.2 `struct stat`. The session snapshot format (`PDGS`, SHA-256-checked) captures the whole machine — CPU contexts, sparse nonzero RAM pages, the kernel tables — and restoring it into a fresh machine continues the run to byte-identical final state.
- **Launcher GUI (Phase 6)** — the egui launcher (`pdg-launcher`, `--features gui`): game library, settings persistence, save-state listing with header metadata, and a run button that streams the guest console into the log panel. The headless model (settings, save-state scanning) is unit-tested without a display.
- **Crash reporting (Phase 6)** — every binary installs a panic hook that logs the panic with a forced backtrace and writes a `pdg-crash-*.log` file; CI gained a GUI compile-check job and a tag-triggered release job.
- **Boot harness** — `pdg-boot --bios <your-dump>` loads your legally-dumped firmware into the ROM window and starts executing at the reset vector `0xFFFF_FFF0` through the entire stack.
- **Kernel syscall layer (Phase 5)** — the Orbis table with numbers verified against FreeBSD 9.2's public `syscalls.master` (the PS4/PS5 kernel's documented base) and the public Sony 588+ block: `exit`, `read`, `write`, `open`, `close`, `getpid`, `munmap`, `mprotect`, `getdirentries`, `kqueue`, `kevent`, `mmap`, `lseek` — implemented over the console, the read-only PFS, the process address space and the kernel object table; the Sony block is logged by name and refused with `ENOSYS`, never guessed. The raw FreeBSD return convention (CF + errno) is honored.
- **Process loading (Phase 5)** — a genuine ELF64 loader: per-segment page permissions, zero-filled tails, `R_X86_64_RELATIVE` / `R_X86_64_TPOFF64` relocations through a supervisor write path (segment data lands in RX pages the way real kernels do), TLS with a documented FS-base model, and symbol relocations through a resolver. The SELF container is parsed per its publicly documented layout and unwrapped — plain (`flags == 0`) entries only; encryption/compression bits are refused with the exact value, not decoded from conflicting public documentation.
- **PFS (Phase 5)** — the read-only PlayStation filesystem over a block device, implemented from the public on-disk layout (cross-checked against two independent public readers): superblock, inode blocks, dirents, contiguous/direct/first-indirect block maps. Signed (36-byte) and encrypted images are refused precisely. A builder produces images in the same layout for tests and the demo.
- **Kernel objects + interrupts (Phase 5)** — event flags, recursive owner-tracked mutexes, counting semaphores and kqueue-shaped event queues behind a handle-checked table; and MSI-X-shaped interrupt plumbing: a guest-programmable vector table (public entry layout, `0xFEE00000` message-address convention) whose latched vector is delivered through the guest IDT at instruction boundaries — the same `deliver_exception` path `#PF` uses.
- **Cross-cutting** — layered TOML configuration, `tracing` logging, event bus, virtual clock + scheduler, **344 tests** (unit, integration, differential fuzz, SMP, GPU end-to-end, CPU-driven I/O, kernel and runtime sessions, snapshot codecs), criterion benchmarks, CI on Linux and Windows (plus a lavapipe job for the pixel-verified GPU tier).

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

# Phase-4 I/O demo: the emulated CPU drives the SSD, pad and audio
cargo run --release --bin pdg-boot -- --demo-io ./ssd.img

# Phase-5 kernel demo: a real ELF64 process boots, calls Orbis syscalls,
# reads the PFS, and takes NVMe completions through its own IDT
cargo run --release --bin pdg-boot -- --demo-kernel ./ssd.img

# Phase-6 runtime demo: a modular process — main + DT_NEEDED PRX closure —
# spawns a thread, contends a umtx mutex, bumps per-thread TLS, stats
# files, and dlsyms across modules
cargo run --release --bin pdg-boot -- --demo-runtime ./ssd.img

# The Phase-6 launcher GUI (egui; library, settings, save states)
cargo run --release --bin pdg-launcher --features gui

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
├── pdg-io/        # NVMe SSD model, DualSense model + pad MMIO, input sources,
│                  # host block devices, the Phase-4 demo payload
├── pdg-audio/     # Tempest-shaped mixer, spatializer, resampler, audio MMIO,
│                  # cpal backend (feature)
├── pdg-kernel/    # Orbis syscall layer, ELF64/SELF loader, read-only PFS,
│                  # kernel objects, MSI-X-shaped interrupts, VM manager,
│                  # the Phase-5 demo ELF
├── pdg-frontend/  # Game library model, the I/O + kernel sessions, the
│                  # Phase-6 runtime session (threads, modules, snapshots)
│                  # and the egui launcher (gui feature)
├── pdg-tests/     # Cross-crate integration tests (boot, SMP, GPU, I/O, kernel)
└── pdg-tools/     # pdg-boot (boot harness + I/O/kernel demos), pdg-disasm, pdg-wgsldump
```

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Technology

Rust (edition 2024) · iced-x86 (decoding) · cranelift (JIT codegen, since Phase 2) · wgpu + naga + rspirv + sha2 (GPU, since Phase 3; sha2 also digests the Phase-6 save states and NIDs) · memmap2 (guest RAM) · tokio (async I/O) · serde + toml (config and the Phase-6 launcher settings) · eframe/egui (the Phase-6 launcher GUI, feature-gated `gui`/`launcher` — the default build stays lean) · tracing (logging) · criterion + proptest (QA). Rationale for every choice lives in the phase reports. Phase 5 added no new dependencies; Phase 6's only new runtime dependency is the optional GUI stack, off by default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The bar is simple: real code, real tests, honest reporting.

## License

[GPL-3.0-or-later](LICENSE)
