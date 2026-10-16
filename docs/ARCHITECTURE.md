# PDG Architecture

Version: Phase 1 (2026) — this document describes the target design and marks
what exists today versus what is planned per phase.

---

## 1. Mission and constraints

PDG emulates a PlayStation 5 (AMD Zen 2 CPU, custom RDNA 2 GPU, 16 GiB
unified GDDR6, custom NVMe SSD, Tempest audio DSP, Orbis OS) on modern PCs,
translating guest workloads to native host execution with maximum efficiency.

Hard constraints:

1. **Legal**: no proprietary code, keys, firmware, or copyrighted data.
   Users supply their own legally-dumped BIOS and games.
2. **Honesty**: an unimplemented feature faults loudly; nothing silently
   no-ops or fakes results.
3. **Portability**: Linux and Windows first-class; GPUs from NVIDIA, AMD,
   and Intel via the wgpu abstraction.

## 2. System overview

```
                       ┌──────────────────────────────────────────────┐
                       │                  pdg-frontend                │
                       │        launcher GUI (Phase 6, egui)          │
                       └───────────────┬──────────────────────────────┘
                                       │ config / control
                       ┌───────────────▼──────────────────────────────┐
                       │                   pdg-core                   │
                       │  System orchestrator · config · events ·     │
                       │  clock/scheduler · logging                   │
                       └───┬───────────┬───────────┬──────────┬───────┘
                           │           │           │          │
              ┌────────────▼──┐ ┌──────▼─────┐ ┌───▼────┐ ┌───▼─────┐
              │    pdg-cpu    │ │ pdg-memory │ │ pdg-gpu│ │pdg-kernel│
              │ Zen 2 model   │ │ bus/MMU/   │ │ PM4 →  │ │ Orbis →  │
              │ interp→JIT    │ │ TLB/DMA    │ │ wgpu   │ │ host     │
              └───────────────┘ └────────────┘ └────────┘ └─────────┘
                       │              │            │          │
              ┌────────▼──────────────▼────────────▼──────────▼───────┐
              │        pdg-io / pdg-audio: storage, pads, sound        │
              └───────────────────────────────────────────────────────┘
```

Dependency direction: subsystem crates stay independent of each other;
`pdg-core` is where they meet. The CPU defines a `MemoryAccess` port that
the core implements on top of the MMU + bus (ports-and-adapters), so the CPU
core is testable against flat memory and the platform layer is swappable
when the JIT arrives.

## 3. CPU subsystem (`pdg-cpu`)

Target: AMD Zen 2 — 8 cores / 16 threads @ ~3.5 GHz, x86-64 with AVX2, FMA3,
BMI1/2, SSE4.2, AES-NI, SHA. Because the guest ISA is x86-64, PDG can (and
will) exploit same-ISA host execution; the layering keeps that an
optimization, not a correctness dependency.

### 3.1 Execution pipeline

Phase 1 implements tier 0; tiers 1–2 are Phase 2:

| Tier | Mechanism | Status |
|------|-----------|--------|
| 0 | Interpreter: `iced-x86` decode → semantic dispatch | ✅ Phase 1 |
| 1 | Basic-block JIT: lift to IR → optimize → native code (cranelift or dynasm-rs) | Phase 2 |
| 2 | Trace/hot-path superblocks with profiling | Phase 2–3 |

The interpreter is the correctness oracle: the JIT must produce identical
architectural results (a differential-fuzzing harness enforces this).

### 3.2 Thread model

One host thread per emulated core (8 cores / 16 SMT threads), with
cooperative scheduling gates for cross-core events (IPIs, global TLB
shootdowns). Synchronization primitives used by real games — atomics, lock
cmpxchg, futex-style waits — execute natively on the host where legal, and
via a scheduler-aware path when a guest thread must block. Phase 2
delivered the shape: `SmpSystem` runs every logical core on its own host
thread with a per-core MMU/TLB over one shared bus, hardware atomics for
LOCK-prefixed RMWs, and a guest-visible IPI MMIO device mirrored onto the
event bus. SMT (two threads per core) and scheduler-aware blocking remain
future work.

### 3.3 Memory ordering and exceptions

- Faults (page fault with CR2 + error code, #UD, #GP, #DE) deliver
  through the guest IDT with architectural frames and `IRET` (Phase 2);
  the JIT reproduces them precisely by re-executing the faulting
  instruction in the interpreter tier.
- Self-modifying code: the JIT guards stores with a code-page bloom and
  invalidates the affected blocks before re-executing the store through
  the interpreter (Phase 2, tested differentially). Guest page-table
  writes through the JIT's host window are the remaining hole —
  write-protected page tables land with the kernel phase.
- SIMD: AVX2 maps to host AVX2 when present (nearly universal on 2026
  desktops); a soft-fallback covers hosts without it.

## 4. Memory subsystem (`pdg-memory`)

### 4.1 Address spaces

```
Guest virtual ──(4-level x86-64 walk)──► Guest physical ──► Host backing
   PML4→PDPT→PD→PT, 4K/2M/1G leaves      RAM / ROM / MMIO     anon mmap
```

- **Guest RAM** is one contiguous anonymous mapping (zero-filled like real
  DRAM, committed on demand). 16 GiB configurations reserve address space
  but only touch pages the guest dirties; the default dev config is 256 MiB.
- **Guest-physical bus** decodes RAM, ROM windows (the BIOS at the top of
  the 4 GiB space, mapped to *end* at the boundary so the reset vector
  always lands in the image) and MMIO device ranges.
- **MMU** walks the guest's own page tables in guest RAM — permission bits
  (present / RW / NX) are enforced cumulatively across all four levels,
  exactly like hardware. A software TLB caches leaf translations and is
  flushed on CR3 writes; per-entry INVLPG invalidation is a planned
  refinement.
- **DMA engine** copies between any guest-physical regions, chunking
  correctly across region boundaries; the async queue for the SSD pipeline
  arrives in Phase 4.

### 4.2 Phase-1 guest memory map (simplified "flat boot")

```
0x0000_0000 .. 0x0020_0000    RAM, RW+X (boot image region)
0x0020_0000 .. RAM end        RAM, RW NX (bootstrap page tables live here)
0xF000_0000 .. +0x1000         MMIO: boot console (identity-mapped)
0xFF00_0000 .. 0xFFFF_FFFF     ROM window (BIOS), read-only
   └ 0xFFFE_0000 .. 0x1_0000_0000  ROM, executable (reset vector page)
```

The Orbis-accurate application/kernel split (low app half, canonical kernel
half, and the real console memory map) lands with the Phase-5 kernel work.

### 4.3 Performance path (planned)

The JIT will map guest pages into a reserved host VA range with a mirrored
alias (RX window for execution, RW window for stores), giving native loads
and stores direct guest memory access without per-access translation —
the same trick used by the fastest production emulators. The interpreter's
measured costs today: ~9.9 ns per TLB-hit translation, ~17.7 ns per full
walk, ~21.5 MIPS integer execution (bench profile, development container).

## 5. GPU subsystem (`pdg-gpu`)

Target: RDNA 2 — 36 CUs @ ≤2.23 GHz, hardware ray tracing, primitive/mesh
pipelines, VRS.

### 5.1 Command path

The PS5's graphics command processor consumes **PM4 packets** (AMD's
publicly documented packet language). Translation pipeline, live since
Phase 3:

```
Guest PM4 ring → packet decoder → semantic IR (GpuCommand) → GpuDevice → wgpu
```

The command processor (`pm4/cp`) executes the verified packet subset
— the four `SET_*_REG` spaces, type-0 writes, draws, dispatches,
indirect-buffer chains, synchronization (`RELEASE_MEM`, `EVENT_WRITE`,
`MEM_SEMAPHORE`, `WAIT_REG_MEM`, `ACQUIRE_MEM`), data movement
(`WRITE_DATA`, `DMA_DATA`, `COPY_DATA`) and state control — against a
gfx10.3 register model whose offsets are verified against the public
kernel `gc_10_3_0` headers. Everything else is refused with an error
naming the opcode. The GNM ring (`ring`) is an MMIO device: the
emulated CPU programs the ring, rings the doorbell, and polls fences
in guest RAM — the console driver contract, verified end to end by
CPU-driven tests.

### 5.2 Shader path

Games ship precompiled native ISA (GCN/RDNA-lineage) shaders. PDG
recompiles them at first use (Phase 3):

```
Guest native ISA ──(decode)──► Inst ──(lifter, rspirv)──► SPIR-V ──(naga)──► WGSL ──► wgpu pipeline
```

The decoder covers the gfx10 encoding layouts verified against LLVM's
public AMDGPU backend (SOP1/2/K/C/P, VOP1/2/3, SMEM, MUBUF, EXP, ...);
the lifter compiles the subset it can translate *correctly* — the
Phase-3 mini-GNM vertex-fetch contract (descriptor-table `s_load` +
`buffer_load` fetches, ALU, `exp` exports) — and refuses everything else
by instruction name. A content-addressed on-disk shader cache
(SHA-256 of guest ISA ‖ stage ‖ specialization) avoids re-translation
across runs. Hardware ray tracing maps to Vulkan ray pipelines
(falling back to compute-based traversal where wgpu's abstraction is
insufficient — if the abstraction gap proves too wide, a thin Vulkan
layer via `ash` will be added under the same `GpuDevice` trait seam).

### 5.3 Scheduling

Async compute queues map to wgpu's queue abstraction; the scheduler honors
guest-issued fences and semaphores.

## 6. Kernel subsystem (`pdg-kernel`)

Orbis OS is FreeBSD-derived. Strategy: **HLE** — translate syscalls, not
emulate kernel binaries. The Phase-1 `SyscallDispatcher` is a registry with
the FreeBSD-style ABI (number in RAX, args in RDI/RSI/RDX/R10/R8/R9,
result in RAX).

Phase 5 delivered the kernel plane:

- **`orbis`** — the Orbis syscall layer: numbers verified against the
  FreeBSD 9.2 `syscalls.master` (the console kernel's documented base)
  plus the public Sony 588+ name table. Thirteen syscalls implemented
  (console + read-only-PFS file I/O with `getdirentries`, anonymous
  `mmap`/`mprotect`/`munmap`, `kqueue`/`kevent`, `exit`, `getpid`,
  `lseek`); everything else — implemented or unknown — is refused with
  `ENOSYS` and logged by name. The raw FreeBSD return convention
  (CF + errno) is honored.
- **`vm`** — the process address space: per-process page tables behind a
  shared MMU, tracked regions, an identity-mapped DMA window (devices
  speak physical; processes speak virtual), and the `ProcessMemory`
  port the interpreter runs against.
- **`loader`** — genuine ELF64 loading (per-segment permissions,
  `RELATIVE`/`TPOFF64` relocations through a supervisor write path,
  TLS with a documented FS-base model) and the SELF container unwrap
  (public layout; plain `flags == 0` entries only — the bit meanings
  are not consistently documented, so nonzero values are refused with
  the exact number).
- **`pfs`** — the read-only PlayStation filesystem over a block device,
  from the public on-disk layout (superblock, inodes, dirents,
  contiguous/direct/first-indirect block maps); signed and encrypted
  images refused precisely; a deterministic builder produces images in
  the same layout.
- **`objects`** — event flags, recursive mutexes, semaphores and
  kqueue-shaped equeues behind a handle-checked table (the substrate
  the Phase-6 NID layer binds `sceKernel*` onto).
- **`msix`** — the MSI-X-shaped vector table (public entry layout) and
  delivery through the guest IDT at instruction boundaries via
  `Interpreter::inject_interrupt` — the same architectural path as
  `#PF` delivery.

The dynlib/PRX module layer, NID resolution, threads and the `stat`
family remain Phase-6 work; see PHASE_5_REPORT.md for the honest gap
list.

## 7. I/O subsystem (`pdg-io`)

- **Storage**: the PS5's custom SSD (5.5 GB/s raw; effective 8–22 GB/s with
  compression) is emulated as an NVMe-shaped queue in front of host files.
  The Phase-1 `AsyncFileBlockDevice` (tokio positioned I/O) is its
  foundation. **Oodle/Kraken licensing note**: the console's Oodle codec is
  proprietary; PDG will evaluate open Kraken-lineage decoders and
  user-supplied components in Phase 4 — never ship infringing code.
- **Controllers**: DualSense state is modeled today (`InputState`); the
  Phase-4 driver maps host gamepads (gyro, touchpad, adaptive triggers,
  haptics) onto it.
- **Network**: stubbed until Phase 4.

## 8. Audio subsystem (`pdg-audio`)

Tempest is HLE'd at the Orbis audio API level (the same approach the
leading PS3/PS4 emulators use): guest audio calls produce PCM streams that
`cpal` outputs; HRTF post-processing is applied host-side.

## 9. Frontend (`pdg-frontend`)

Phase 6: egui launcher — game library grid, per-game settings, BIOS picker,
save states, compatibility list. Phase 1 ships the library scan model.

## 10. Quality strategy

- **Unit + integration tests** (93 today) — including instruction-level
  tests with architectural flag verification.
- **Property-based fuzzing** (proptest) of the ALU against a reference
  model; decoder fuzzing broadens in Phase 2.
- **Differential testing**: interpreter vs JIT once the JIT exists.
- **Benchmarks** (criterion): interpreter MIPS, MMU walk/TLB costs —
  tracked per phase in the reports.
- **CI**: rustfmt + clippy `-D warnings`, tests on Linux and Windows,
  release builds, llvm-cov coverage artifacts.

## 11. Phase roadmap

| Phase | Deliverables |
|-------|--------------|
| 2 | Full x86-64 ISA (integer, SIMD, string, mul/div), JIT tier 1, SMP, IDT delivery |
| 3 | PM4 semantics, wgpu device, shader recompilation + cache |
| 4 | SSD model + decompression, DualSense, cpal audio, network stubs |
| 5 | Orbis syscalls, ELF/PRX loader, PFS, boot chain |
| 6 | egui frontend, optimization, packaging, releases |
