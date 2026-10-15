# PHASE 2 REPORT — CPU Emulation: Full ISA, SIMD, JIT, SMP

Date: 2026-10-16
Branch: `main`
Scope delivered: the emulated Zen 2 CPU in full — complete integer ISA,
SSE→AVX2 SIMD families, a cranelift JIT tier with differential testing,
and a 16-thread SMP machine — plus the correctness and performance work
that made the benchmarks tell the truth.

---

## 1. What we accomplished

### 1.1 Interpreter: the complete x86-64 integer ISA (reference tier)

The Phase-1 subset grew into the whole integer architecture, each piece
test-driven against the architecture manuals:

- **Arithmetic** — `MUL`/`IMUL` (all forms) and `DIV`/`IDIV` (both `#DE`
  paths: divide-by-zero and quotient overflow), `NEG`, sign/zero extends
  (`MOVZX`/`MOVSX`/`MOVSXD`), `CBW`/`CWDE`/`CDQE`/`CWD`/`CDQ`/`CQO`.
- **Bit manipulation** — `BT`/`BTS`/`BTR`/`BTC` including the memory-form
  *bit-offset addressing* (register index scales beyond the operand), `BSF`,
  `BSR`, `POPCNT`, `TZCNT`, `LZCNT`.
- **Rotates and double shifts** — `ROL`/`ROR` (width-aware windows — the
  sub-64-bit rotate wrap was a real bug our tests caught), `RCL`/`RCR`
  (count modulo bits+1), `SHLD`/`SHRD`.
- **Conditionals and moves** — unified `cond_cc` for all 16 `Jcc`/`CMOVcc`/
  `SETcc` families, `LAHF`/`SAHF` (the ModRM assumption was another
  caught bug).
- **String operations** — `MOVS`/`STOS`/`LODS`/`SCAS`/`CMPS` at all widths
  with `REP`/`REPE`/`REPNE` restart semantics (interruptible mid-string,
  per-step caps) — including the iced `F3`-bit `rep`/`repe` aliasing trap.
- **Atomics** — `XCHG`, `XADD`, `CMPXCHG`, `CMPXCHG8B`/`CMPXCHG16B`, and
  `LOCK`-prefixed `INC`/`DEC`. SMP correctness fix this session: locked
  memory `INC`/`DEC` now execute as a *true* atomic RMW (`xadd ±1`)
  instead of a plain read-modify-write that loses updates across cores.
- **System** — `CPUID` (Zen 2-shaped vendor/family/model with a pinned
  *claimed-features ledger* that only advertises what is actually
  implemented), `RDTSC`/`RDTSCP`, `RDMSR`/`WRMSR` (FS/GS bases, swapgs
  support), `LGDT`/`LIDT`/`SGDT`/`SIDT`, `MOV CRn`, `INVLPG`/`INVLPGA`,
  `CLI`/`STI`, fences/hints as no-ops, `INT n`/`INT3`/`INTO`, `IRET`.
- **Exception delivery** — through the guest's own IDT: error-code vs
  no-error-code vectors, CR2 on `#PF`, interrupt-vs-trap-gate IF
  semantics, flat-model frames, and `IRET` return. Verified end-to-end
  by a `#DE` handler that fixes the divisor and resumes.
- Phase-1 issue #1 closed: instruction fetch straddling into unmapped
  space now raises `#PF` (page-boundary probing), not `#UD`.

### 1.2 SIMD: SSE → AVX2, FMA, AES-NI, PCLMULQDQ

The `simd` module (families: shuffle, float, crypto, bridge,
super-dispatch) implements SSE/SSE2 integer + float, SSSE3 (`pshufb`,
`palignr`, `phadd`, `pabs`, `psign`, `psadbw`, …), SSE4.1 (`pmovzx/sx`,
`pextr`/`pinsr`, blends, `roundps`, `dpps`, `ptest`), the AVX2 integer
core (`vpadd`…`vpshufb`, `vpermq/d`, `vpbroadcast`, `vinsert/extract128`,
variable shifts, gathers with destructive masks), all FMA families
(132/213/231 × add/sub/nmadd/nmsub × ps/pd/ss/sd), the complete AES-NI
round suite (pure-Rust S-box + GF(2^8) multiply kernels, verified against
FIPS-197 C.1 vectors and cross-checked vs the RustCrypto `aes` crate), and
`PCLMULQDQ`. Register-file rules are architectural: VEX.128 and legacy
forms zero/merge upper halves exactly as the manuals say, and MXCSR
rounding is honored by `cvtps2dq`.

Bugs found and fixed by our own per-family tests: `imul` 2-op computed
`src*src` instead of `dst*src`; sub-64 rotates used `u64::rotate`
(no wrap); `SHLD`/`SHRD` read the count from the wrong operand;
`cvtdq2ps` reinterpreted integer lanes as floats; `AESDEC` applied the
key at the wrong end (equivalent-inverse-cipher order); `clmul` looped on
bit 0 instead of the lowest set bit.

### 1.3 JIT tier (cranelift): block recompilation with precise fallback

`pdg-jit` recompiles guest basic blocks through cranelift 0.136:

- **Lifting** — the integer + control-flow subset maps to CLIF: ALU with
  exact flag formulas, shifts, `MOV`/`MOVZX`/`MOVSX`/`LEA`, `PUSH`/`POP`,
  `CALL`/`RET`/`JMP`/`Jcc`/`CMOVcc`, `XCHG` reg-reg, string-free memory
  operands. Everything else exits with a *precise* fallback: the block
  names the guest RIP, the dispatcher single-steps (or windows) the
  interpreter, and the JIT resumes — tier-1 correctness is defined as
  "the interpreter would have done exactly this".
- **Loop formation** — a jcc whose target lies inside the block becomes a
  real loop in CLIF with the instruction budget re-checked every
  iteration, so `run(n)` terminates on guest infinite loops.
- **Inline software TLB** — 8192 direct-mapped slots over host memory
  windows (`page_ptr` on the memory port); misses call a helper that
  fills from the MMU; MMIO/ROM and page-crossing accesses fall to helpers.
- **SMC** — compiled code stores check a 4096-bit bloom of guest code
  pages before writing; a hit exits with the store address, the
  dispatcher drops that page's blocks, single-steps the store through
  the interpreter, and re-enters. Self-modifying code is *tested*
  (differentially vs the interpreter).
- **Fault precision** — slow-path helpers carry the guest RIP of the
  instruction they serve; faults reproduce exactly by re-executing the
  single instruction in the interpreter tier (which also drives IDT
  delivery).
- **Differential testing** — fixed programs plus two seeded fuzzers
  (ALU chains; ALU + memory traffic) run both tiers and compare GPRs,
  RFLAGS, RIP, and full memory images. SMC, budget semantics, and
  syscall exits have dedicated tests.

**Three silent-lifter bugs found and fixed this session** (all masked
before because the fallback preserved correctness while destroying
performance):

1. `mem_load`'s fast-path block carried a stray I64 block parameter —
   **loads never compiled**; every load-bearing block failed cranelift's
   verifier and fell back forever. (Stores were fine, which is why the
   differential tests passed — fallback is *correct*, just slow.)
2. The not-taken path of a **forward (external-target) jcc was left
   unterminated** — every block containing a conditional early-exit
   (any function shaped like `if (x < 2) return`) failed verification.
   The fuzzers only generated back-edge loops, so the class was invisible
   until the fib benchmark.
3. Fallback bookkeeping **zeroed the 128 KiB env TLB on every interpreter
   step** (and re-attempted failed compiles per re-entry), making
   fallback-heavy code ~700× slower than the interpreter.

Fixes: the two verifier bugs above; a *translation epoch* on the memory
port (`GuestMemory`/`SharedGuest` bump it on CR3 reload / INVLPG /
flush) so the JIT validates its TLB and rip-keyed caches cheaply instead
of flushing; a negative cache for failed compiles; interpreter fallback
*windows* (32 instructions) that amortize tier switches; and cached
env-var probes (8M `getenv` calls were 40% of fib's runtime). Two
regression tests now assert these paths *lift* (`fallback_steps == 0`),
not merely produce correct results.

### 1.4 SMP: 16 host threads over one shared bus

- **Memory made shareable** (Phase-2 groundwork) — RAM reads/writes go
  through relaxed atomics with per-width fast paths; `Bus::write` is
  `&self`; atomic RMWs (`xchg`/`xadd`/`cmpxchg`) run on the host's
  hardware atomics, with a global-lock fallback for unaligned operands.
- **`SmpSystem`** (`pdg-core::smp`) — `cpu.threads` (default 16) host
  threads, each with its own `Mmu`/TLB over the shared bus, private
  64 KiB stack slots, optional per-core syscall hooks from a factory.
  `run_all`/`run_cores` spawn, execute, and join with per-core reports.
- **Cross-core communication is real** — 16 cores running `lock inc
  [shared]` in loops produce *exactly* `16 × N` increments (tested);
  a mixed-tier test runs one core on the JIT and one on the interpreter
  against the same counter with zero lost updates.
- **IPIs** — an `IpiDevice` MMIO window (per-core mailboxes) lets guest
  cores signal each other; deliveries are mirrored onto the host event
  bus (`Event::IpiSent`). Tested end-to-end: core 0 sends vector 0x42 to
  core 1, which observes it through the MMIO poll and halts.
- **Fixes that made it correct** — locked INC/DEC became true RMWs; the
  bus now dispatches contained MMIO accesses at full width (the byte
  split turned every 64-bit IPI store into eight byte-stores — found by
  the IPI test seeing `0x100` instead of `0x142`); `GuestMemory` now
  overrides the CR3/INVLPG/flush hooks (a latent stale-TLB bug on the
  single-vCPU path).

### 1.5 Numbers (development container, `bench` profile, criterion)

| Metric | Result |
|---|---|
| Tests passing | **153 / 153** across 29 suites (unit, integration, differential fuzz, SMP) |
| Lint / format | clippy `-D warnings` clean · rustfmt clean |
| Lines of Rust (workspace) | ~15,500 (from ~5,500 in Phase 1) |
| Interpreter throughput | **20.3 MIPS** (dec/jne loop, 6.45 ms / 131,075 instrs) |
| JIT — ALU loop (`dec/jne` ×100M) | **203.2 ms ≈ 1.02 ns/instr ≈ 984 MIPS — ~48× interpreter** |
| JIT — load/add/store loop (×20M) | **64.9 ms ≈ 0.65 ns/instr ≈ 1541 MIPS — ~76× interpreter** |
| JIT — recursive `fib(30)` | **194.1 ms / 22.9M instrs ≈ 8.5 ns/instr — ~5.8× interpreter**, 0 fallbacks |
| SMP atomicity | 16 threads × 1000 `lock inc` = 16,000/16,000 exactly |
| MMU translate | 9.95 ns TLB-hit · 17.70 ns full walk (Phase-1 numbers, unchanged) |

The Phase-2 target — ≥50× the interpreter on integer workloads — is met
on loop-form integer code (76× on the memory-touching loop; 48× on the
pure-register dec/jne microloop, just under on this container's CPU).
Call-heavy code runs at ~6× because every `call`/`ret`/`jcc` ends a
block; cross-block compilation is the Phase-3 JIT agenda.

## 2. Key technical decisions and justifications

| Decision | Rationale | Alternatives rejected |
|---|---|---|
| cranelift 0.136 as the JIT backend | Probed first (Phase-2-0 spike): JITBuilder + symbol registration + frontend vars validated, 572 ps/iter on a 10M-iteration native loop — the ≥50× target was proven feasible before committing | dynasm-rs (hand-written x86 emission is a Phase-3-sized effort with worse ergonomics), LLVM (C++ dependency burden) |
| Interpreter as the correctness oracle, JIT behind differential tests | Every JIT bug this phase was *found* by comparing against the interpreter — the oracle pays for itself | Testing the JIT against hand-computed values only (the three silent lifter bugs would have survived) |
| Translation epoch on the memory port | One `u64` comparison replaces a 128 KiB TLB flush per fallback; also gave `GuestMemory`/`SharedGuest` a place to hang CR3/INVLPG hooks | Unconditional flush (measured ~700× slowdown on fib), generation-tagged TLB entries (needed for Phase 3's ASID work, overkill now) |
| Fallback *windows* (32 instrs) instead of single-steps | Unliftable stretches (string ops, SIMD in the JIT, rare system instructions) run at interpreter speed with amortized tier-sync; the JIT returns promptly on liftable code | Per-instruction fallback (correct but ~30× overhead-bound), full-block fallback (never returns to the JIT) |
| Negative compile cache | Failed lifts cost cranelift work before erroring; without the cache, fallback-heavy code re-paid that on every re-entry | Retrying forever (the measured pathology), disabling compilation wholesale |
| Host thread per logical core | The PS5 is 8C/16T; the memory layer was made thread-safe for exactly this shape; thread-per-core keeps per-core MMU/TLB ownership trivial | M:N green threads (tokio) for CPU execution — wrong tool: the cores are compute-bound, not I/O-bound |
| MMIO at full width for contained accesses | Devices deserve register-shaped accesses, not byte streams; the IPI test caught the byte-split immediately | Byte-split everywhere (Phase-1 behavior; fine for the console, wrong for register devices) |
| CPUID feature ledger only claims implemented features | "No fake implementations" made architectural: we do not advertise AVX-512 we cannot execute | Advertising the real Zen 2 feature set (would make real firmware take paths we cannot honor) |
| SMP uses the interpreter tier; JIT-per-core is a test, not the default | Phase 2 closes correctness; tier selection/perf under SMP is Phase-3 profiling work (both tiers already pass the same shared-bus atomics test) | Forcing JIT everywhere now (unmeasured interaction with per-core caches under contention) |

## 3. Libraries used and why

| Crate | Where | Why |
|---|---|---|
| cranelift-codegen/frontend/jit/module/native | pdg-jit | Battle-tested codegen with a Rust-native IR, host ISA detection, and a JIT module with symbol registration; the probe spike validated every API we rely on |
| iced-x86 | pdg-cpu, pdg-jit, pdg-tools | Same decode layer for both tiers — the JIT and the interpreter cannot disagree about what an instruction *is* |
| memmap2 / bitflags / serde / toml / tracing / thiserror / anyhow / tokio / proptest / criterion | as in Phase 1 | Unchanged; no new leaf dependencies beyond cranelift |

## 4. Current state of emulation

**What runs today (all verified by tests or measured benches):**

- The complete x86-64 integer ISA and the SSE→AVX2 SIMD families listed
  above, with architectural flag, exception, and register-merge
  semantics — the reference tier for everything downstream.
- Guest code under the JIT: integer loops at ~1 ns/instruction, real
  loops formed in CLIF, self-modifying code invalidated and recompiled,
  faults precise to the instruction, syscall exits serviced through HLE
  hooks, budgets honored even on `jmp $` infinite loops.
- Sixteen logical cores executing concurrently on host threads over one
  shared bus — with cross-core lock-protocol correctness (hardware
  atomics, no lost updates) and guest-visible IPIs mirrored to the host
  event bus.
- Everything from Phase 1 (boot to BIOS through the full stack, MMIO
  console, DMA, ROM windows, CI) still passes.

**What does not run (and is not claimed to):**

- **No commercial PS5 games.** No GPU (Phase 3), no Orbis kernel or
  SELF/ELF loading (Phase 5), no storage/audio/input stacks (Phases 4–5).
- x87 FPU, 3DNow!, AVX-512, and SMT scheduling (two threads per core)
  are not implemented; nothing advertises them in CPUID.
- The JIT does not lift SIMD, string ops, or the rarer system
  instructions yet — they fall back to the interpreter *correctly and
  measurably* (the fallback window machinery exists precisely so that
  stays true).

## 5. Known issues and risks

1. **Block-cache keying** — blocks are keyed by guest RIP only; a CR3
   switch to a different address space drops the whole cache (correct
   but coarse). Phase 3 keys by address-space id once the kernel phase
   exercises context switches.
2. **Page-table stores through the JIT's inline TLB** — a guest store
   that rewrites a *page table* through the host window does not
   invalidate other pages' cached translations. Flat-memory and
   single-address-space workloads (everything testable today) are
   unaffected; the Phase-5 kernel bring-up needs write-protected page
   tables or epoch-per-page tagging.
3. **TLB shootdown** — `INVLPG`/CR3 reloads invalidate only the calling
   core's caches; no cross-core invalidation IPI yet (the Phase-5 kernel
   owns that policy; the IPI plumbing to build it on now exists).
4. **`cmpxchg16b`** is emulated as a non-atomic 128-bit RMW (documented;
   correct under the current single-writer-per-address guest patterns,
   needs a `cmpxchg16b` host primitive for full SMP fidelity).
5. **JIT code lifetime** — tier 1 never frees compiled blocks; the leak
   is bounded by code-page rewrites (SMC drops cache entries but the
   native memory stays until the `Jit` dies). Reclamation lands with the
   Phase-3 code cache manager.
6. **Instruction budget is a block-entry approximation** on the JIT tier
   (documented in `pdg-jit`'s crate docs); budgets are never overshot,
   but the count can differ from the interpreter's by the tail of a
   window.

## 6. Next phase plan (Phase 3 — GPU)

1. **PM4/GNM command processing** — the framing layer exists; implement
   the packet semantics (context registers, draw dispatch, indirect
   buffers) against public AMD documentation.
2. **RDNA 2 shader recompilation** — GCN-derived ISA → SPIR-V via a
   custom decoder, then naga → WGSL for the wgpu backend; the
   interpreter↔JIT differential-testing discipline moves to the shader
   translator.
3. **wgpu backend** — Vulkan/DX12/Metal selection, memory budgeting
   against the 16 GiB unified map, present path.
4. **JIT tier 2** — cross-block compilation (direct block chaining to
   beat the per-call block-entry cost that holds fib at ~6×), SIMD
   lifting, per-address-space code caches, block reclamation.
5. **Ray tracing posture** — evaluate wgpu's RT abstraction early; the
   `GpuDevice` seam allows a thin ash/Vulkan fallback without
   re-architecting.

---

*Phase 2 complete. The machine now has a brain: a full CPU that runs at
native-ish speed when it can and tells the truth when it can't. Phase 3
gives it eyes.*
