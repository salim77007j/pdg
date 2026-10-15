# PHASE 1 REPORT — Foundation, Architecture, and Core Infrastructure

Date: 2026-10-16
Branch: `main`
Scope delivered: workspace + licensing + core framework + memory subsystem +
CPU execution foundations + boot harness + tools + CI + documentation.

---

## 1. What we accomplished

### 1.1 Repository and workspace

- Initialized `github.com/salim77007j/pdg` as a 10-crate Cargo workspace
  (edition 2024, MSRV 1.85): `pdg-core`, `pdg-cpu`, `pdg-gpu`,
  `pdg-memory`, `pdg-io`, `pdg-audio`, `pdg-kernel`, `pdg-frontend`,
  `pdg-tests`, `pdg-tools`.
- Licensing: GPL-3.0-or-later (canonical text in `LICENSE`), chosen to keep
  the project and its forks free, per emulator-community convention.
  README and CLI carry the required legal disclaimers verbatim; `.gitignore`
  actively excludes user BIOS/game files to prevent accidental copyright
  violations by contributors.
- CI (GitHub Actions): lint (rustfmt + clippy `-D warnings`), test matrix
  (ubuntu-latest + windows-latest: debug tests + release build), coverage
  (cargo-llvm-cov, lcov artifact upload).

### 1.2 Core framework (`pdg-core`)

- **Configuration** — layered, self-documenting TOML (`serde` + `toml`),
  partial-override semantics, round-trip tested, with a validating loader
  (`ram_mib`, layout invariants, stack placement) and `config/pdg.example.toml`.
- **Logging** — `tracing` + `EnvFilter` with `RUST_LOG` precedence and
  double-init protection.
- **Event bus** — typed pub/sub (`EventBus`) with dead-subscriber pruning.
- **Clock + scheduler** — host-anchored virtual clock at the PS5's 3.5 GHz
  and a priority-queue scheduler with stable tie-breaking, unit tested for
  ordering and deadline semantics.
- **`System` orchestrator** — builds the machine (RAM, MMIO console,
  bootstrap page tables), implements `boot_image` (raw image at an aligned
  base) and `load_bios` (user dump mapped to end at the 4 GiB boundary so
  the reset vector `0xFFFF_FFF0` always lands inside it; clear, actionable
  errors when the dump is missing), runs the machine, and produces a
  `BootReport` (outcome + guest console output).

### 1.3 Memory subsystem (`pdg-memory`) — fully functional

- **Physical RAM** — anonymous zero-filled mmap, page-granular, lazily
  committed by the host kernel (16 GiB configs reserve address space only).
- **Guest-physical bus** — region decode of RAM / ROM / MMIO, width-aware
  (8/16/32/64) reads and writes, byte-wise MMIO fallback, region-end queries
  for DMA chunking, read-only enforcement on ROM.
- **MMIO console device** — collects guest writes for boot diagnostics.
- **Page tables** — a builder that constructs *genuine* x86-64 four-level
  structures in guest RAM (4 KiB and 2 MiB leaves; the walker also supports
  1 GiB leaves real kernels use).
- **MMU** — architectural four-level walk over the guest's own tables.
  Permissions are enforced **cumulatively across all levels** (write
  requires RW at every level; execute requires NX clear at every level —
  matching hardware), canonical-address checks, architecturally exact
  page-fault error codes (P/W/US/EXEC bits), CR2-shaped fault reporting,
  and a software TLB with full flush on CR3 reload. TLB behavior is
  demonstrated by a test that observes a stale translation until CR3
  reload.
- **DMA engine** — guest-physical transfers that chunk correctly across
  region boundaries (RAM → MMIO crossing tested), with a history log.

### 1.4 CPU foundations (`pdg-cpu`) — interpreter tier, integer subset

- **Register file** — all 16 GPRs with every x86-64 view (8/16/32/64-bit,
  high-byte `*H` registers), correct write semantics (32-bit writes
  zero-extend; 8/16-bit writes merge), RFLAGS bit helpers.
- **Decode pipeline** — `iced-x86` decode with the guest RIP wired through,
  16-byte fetch windows that stop at page boundaries and cross pages
  correctly.
- **Interpreter** — fetch → decode → execute with an `Action` model
  (Next / Jump / Halt). Implemented semantics with architectural flag
  fidelity (CF/PF/AF/ZF/SF/OF verified at all widths):
  - Data: `mov` (reg/imm/mem, moffs, RIP-relative), `lea`, `xchg`,
    `push`/`pop` (+`pushf`/`popf`)
  - ALU: `add`, `adc`, `sub`, `sbb`, `cmp`, `test`, `and`, `or`, `xor`,
    `not`, `neg`, `inc`, `dec` (INC/DEC preserve CF — tested)
  - Shifts: `shl`/`sal`, `shr`, `sar` (count masking, CF/OF rules,
    shift-by-zero leaves flags)
  - Control: `jmp`, all 16 condition families, `call`/`ret` (+imm16)
  - Misc: `nop`/`endbr`, flag ops (`clc`/`stc`/`cmc`/`cld`/`std`), `hlt`
  - `syscall` dispatches to an HLE hook (`SyscallHook`)
- **Faults** — typed `Fault` enum (#PF with address+error code, #UD, #GP,
  #DE, unhandled-syscall, host error). Unimplemented instructions raise #UD
  — **nothing silently no-ops**.
- **Notable correctness battle scars** (found by our own tests, fixed, and
  now guarded): 64-bit-mode sign extension of `imm32`/`disp32`
  (payloads must use true 64-bit immediates), iced's pre-computed
  RIP-relative displacement, parity-flag polarity, and the
  `Register::None`/`Option::None` shadowing trap.

### 1.5 Kernel, GPU, I/O, audio, frontend scaffolds (honest scope)

- `pdg-kernel` — registry-based `SyscallDispatcher` implementing
  `SyscallHook` with the FreeBSD-style ABI (number RAX, args
  RDI/RSI/RDX/R10/R8/R9, result RAX); x86-64 exception vector constants.
  **No Orbis syscall numbers are guessed** — the table is Phase 5 work
  built from public homebrew research.
- `pdg-gpu` — `GpuDevice` trait + honest `NullGpu`, plus the PM4 type-3
  packet framing layer (round-trip tested). Opcode semantics land in
  Phase 3 after verification against public AMD documentation.
- `pdg-io` — `BlockDevice` trait, host-file block device (sync), tokio
  async positioned I/O (the DirectStorage-shaped foundation), the
  DualSense-shaped `InputState` model.
- `pdg-audio` — `AudioBackend` trait + counting null sink (cpal in
  Phase 4).
- `pdg-frontend` — game-library scan model (egui GUI in Phase 6).

### 1.6 Tools and the "boot to BIOS" deliverable

- **`pdg-boot`** — loads `--image <raw>` or `--bios <your dump>` and runs
  it through the full stack. Smoke-tested end-to-end: a hand-assembled
  payload prints `Hello from PDG` via the MMIO console, halting at the
  exact expected RIP after exactly 105 instructions. A garbage BIOS runs
  until RIP walks off the 4 GiB ROM boundary and reports a clean #PF —
  graceful, deterministic, never a host crash.
- **`pdg-disasm`** — x86-64/32/16 disassembler (NASM syntax) used to
  verify every hand-assembled test payload byte-for-byte.

### 1.7 Numbers (development container, `bench` profile)

| Metric | Result |
|---|---|
| Tests passing | **93 / 93** (incl. 256-case proptest ALU fuzz ×2) |
| Lines of Rust (workspace) | ~5,500 |
| Crates | 10 |
| Interpreter throughput | **21.5 MIPS** (`dec/jne` loop, 6.10 ms / 131,075 instrs) |
| MMU translate, TLB hit | **9.95 ns** |
| MMU translate, full 4-level walk | **17.70 ns** |
| MMU 64-bit read via TLB | **13.45 ns** |

These numbers are reference points, not promises: they come from a shared
CI-style container CPU. They establish the Phase-1 baseline that Phase 2's
JIT must beat by orders of magnitude.

## 2. Key technical decisions and justifications

| Decision | Rationale | Alternatives rejected |
|---|---|---|
| iced-x86 for decoding | Fastest pure-Rust x86 decoder, complete ISA coverage, actively maintained; no C toolchain burden on Windows CI | yaxpeax-x86 (slower decode loop), hand-written decoder (Phase-2-sized effort, better later when JIT needs its own frontend anyway) |
| Real 4-level page-table walk | The MMU must behave architecturally for real kernels; walking guest RAM keeps permissions honest (cumulative RW/NX) and makes Phase-5 kernel bring-up meaningful | Policy-based "shadow" mapping (faster but wrong long-term) |
| Software TLB keyed on 4 KiB slots | Simple, measurable; hierarchical/VVIPT refinement is a localized change | Full hierarchical TLB now (unneeded complexity before the JIT exists) |
| `MemoryAccess` port owned by `pdg-cpu` | Inverts the dependency: CPU core is testable against `FlatMemory` and the platform layer stays swappable for the JIT's direct-map optimization | CPU depending on `pdg-memory` (couples tiers, blocks JIT work) |
| `PageTableBuilder` writes real PTEs into guest RAM | Bootstrap and later kernel code operate on the same structures hardware would; zero divergence between "loader" and "architectural" views | Shadow metadata structures (would diverge from real semantics) |
| Interpreted subset first, JIT next phase | Correctness oracle first; the differential harness in Phase 2 needs a trusted reference. Also de-risks iced integration early | JIT-first (classic trap: debugging codegen and semantics simultaneously) |
| BIOS image mapped to end at 4 GiB | Reset vector `0xFFFF_FFF0` must land inside the image regardless of dump size — the PC-firmware convention | Fixed base mapping (breaks small dumps) |
| tokio only in `pdg-io` | Emulator core is synchronous by nature; async belongs at the I/O edge (storage/network) | tokio throughout (fights the machine model) |
| Registry-based syscall dispatcher | Lets Phase 5 fill the Orbis table incrementally with per-syscall tests | Monolithic switch (untestable incrementally) |
| GPL-3.0 | Emulator-community standard; prevents proprietary forks of years of reverse-engineering work | MIT/Apache (permits closed forks) |
| No goblin/ELF parsing yet | Phase-5 deliverable; an unused dependency today is waste | Adding it "for later" |

## 3. Libraries used and why

| Crate | Where | Why |
|---|---|---|
| iced-x86 | pdg-cpu, pdg-tools | x86-64 decode + formatting, pure Rust, zero deps on C |
| memmap2 | pdg-memory | Cross-platform anonymous/file mappings for guest RAM |
| bitflags | pdg-memory | Typed page-flag bit sets |
| serde + toml | pdg-core | Layered human-editable configuration |
| tracing + tracing-subscriber | everywhere | Structured, level-filtered, ecosystem standard |
| thiserror / anyhow | everywhere | Typed guest faults vs ergonomic host errors |
| tokio | pdg-io | Async positioned I/O foundation for the SSD pipeline |
| proptest | pdg-cpu | Property-based ALU fuzzing vs reference models |
| criterion | pdg-cpu, pdg-memory | Defensible perf numbers per phase |
| clap | pdg-tools | Ergonomic CLI tooling |

## 4. Current state of emulation

**What runs today (all verified by tests or CLI smoke runs):**

- The full Phase-1 machine boots: image load → bootstrap page tables →
  CPU reset → interpreted execution through the paged guest path.
- Integer x86-64 code with correct architectural flag semantics, stack,
  calls, branches, RIP-relative addressing, and MMIO device interaction.
- `syscall` into HLE handlers (integration-tested through `System` with a
  kernel dispatcher).
- The boot-to-BIOS flow: your legally-dumped BIOS loads into the ROM
  window, the CPU starts at the reset vector, executes, and the machine
  reports halt/budget/fault states honestly.

**What does not run (and is not claimed to):**

- **No commercial PS5 games.** No SELF/ELF loading, no Orbis kernel, no
  GPU translation, no audio, no storage stack yet — these are Phases 2–5.
- No JIT: 21.5 MIPS is ~0.6% of the guest's 3.5 GHz; the interpreter is a
  correctness oracle, not the shipping execution engine.
- No SIMD, string ops, MUL/DIV, or privileged-instruction emulation yet
  (Phase 2).
- Single vCPU: the thread model is designed but SMP execution is Phase 2.
- PM4 packet *semantics* (draws, register writes) are Phase 3; only the
  framing layer exists.

## 5. Known issues and risks

1. **Fetch truncation edge**: an instruction straddling a page boundary
   into unmapped space currently reports #UD (truncated window) rather
   than #PF; real hardware faults the fetch. Fix planned with Phase 2's
   precise-exception work.
2. **TLB granularity**: full flush on CR3 writes only; INVLPG and
   per-entry invalidation are Phase 2. A 2 MiB leaf fills only the
   accessed 4 KiB slot (conservative, correct-but-slow; a hierarchical
   TLB is the fix).
3. **Oodle/Kraken licensing** (Phase 4 risk): the console's compression
   codec is proprietary. We will evaluate open Kraken-lineage decoders and
   user-supplied components; PDG will never ship infringing code.
4. **wgpu ray-tracing headroom** (Phase 3 risk): if wgpu's RT abstraction
   proves insufficient, the `GpuDevice` seam allows a thin `ash`/Vulkan
   layer without re-architecting.
5. **Windows commit charge**: 16 GiB guest RAM configs may stress page
   files; the default stays 256 MiB until the loader learns huge-page
   committing.
6. **`lea` write width** uses the destination register's natural width
   (correct in practice; a dedicated width-override path is Phase 2
   polish).
7. **Interpreter's `FlatMemory`** grows on out-of-bounds loads — fine for
   tooling, never used in the guest path.

## 6. Next phase plan (Phase 2 — CPU emulation)

1. **Full ISA coverage**: MUL/DIV family, string ops (REP MOVSB/STOSB/...),
   MOVZX/MOVSX, CMOVcc, SETcc, bit tests, LAHF/SAHF, RDTSC (virtualized),
   privileged instructions routed to the kernel layer; decoder fuzzing
   (proptest round-trips vs iced).
2. **SIMD**: SSE→AVX2 register file (XMM/YMM), execution via host intrinsics
   with a portable fallback; per-instruction tests from the architecture
   manuals.
3. **JIT tier 1**: basic-block cache, lift to IR (cranelift frontend or
   dynasm hand-codegen — benchmarked spike decides), with
   differential-testing against the interpreter as the oracle.
4. **SMC handling**: write-protected code pages, fault-driven block
   invalidation.
5. **SMP**: 8 cores / 16 threads, host thread per core, atomics and
   lock-prefixed ops, IPI plumbing through the event bus.
6. **Exception delivery**: IDT-based fault injection using
   `pdg-kernel::exception` vectors, precise RIP/CR2/error-code frames.
7. **Performance targets**: ≥50× interpreter on integer workloads before
   the phase closes, with benchmark evidence in `PHASE_2_REPORT.md`.

---

*Phase 1 complete. PDG is a machine you can boot, probe, and trust to tell
you the truth. The foundation is poured; Phase 2 builds the engine.*
