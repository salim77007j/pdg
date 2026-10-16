# PHASE 3 REPORT — GPU: PM4 Command Processor, Shader Recompilation, wgpu Backend

**Scope delivered:** the GPU translation stack end to end — the PM4
command processor with a verified opcode table and gfx10.3 register
model, GCN/RDNA shader recompilation (guest ISA → SPIR-V → WGSL),
a GNM ring the emulated CPU drives through MMIO, and an offscreen
wgpu backend that renders into guest RAM.

**Bottom line:** a full guest-driven draw now works — CPU-executed
driver code programs a PM4 ring in guest RAM, rings the doorbell
through MMIO, the command processor decodes the stream, the guest's
GCN vertex shader is recompiled to WGSL and rendered through wgpu
(Vulkan/lavapipe in the test container), the color target is read
back into guest RAM, and the CPU observes the EOP fence. The
end-to-end test verifies **every pixel** of the 64×64 target.

---

## 1. What we accomplished

### 1.1 PM4 command processor (`pdg-gpu/src/pm4/`, ~2,000 lines)

* **Framing (`mod.rs`)** — type-0 (raw register writes), type-2
  (padding) and type-3 (opcode) packet headers with validated
  encode/decode, including the one-dword-lookahead literal rules the
  decoder needs.
* **Opcode table (`opcode.rs`)** — 70 type-3 opcodes, every number
  cross-checked against the public Linux kernel `soc15d.h` and, for
  the two `EVENT_WRITE_*` variants the kernel header does not name,
  the shadPS4 project's Liverpool documentation. Unknown opcodes
  decode to `ItOpcode::Unknown` and the CP refuses them by name.
* **Register model (`regs.rs`)** — the gfx10.3 register spaces with
  bases from `navi10_ip_offset.h` (`GC_SEG0 = 0x1260`,
  `GC_SEG1 = 0xA000`) and `SET_{CONFIG,CONTEXT,SH,UCONFIG}_REG`
  windows from `soc15d.h`; named registers
  (`VGT_PRIMITIVE_TYPE`, `CB_COLOR0_*`, `SPI_SHADER_PGM_*`,
  `SPI_SHADER_USER_DATA_*`, `VGT_INDEX_TYPE`, ...) with a
  derived-on-read typed view. The gfx6-compatibility window is not
  modeled yet (documented in the module header).
* **Command processor (`cp.rs`)** — executes packet streams against
  the register context and emits semantic `GpuCommand`s: the four
  `SET_*_REG` spaces, type-0 writes, `DRAW_INDEX_AUTO`/`DRAW_INDEX_2`,
  `DISPATCH_DIRECT`, indirect-buffer chains (bounded depth), the
  synchronization family (`RELEASE_MEM`, `EVENT_WRITE`(+`_EOP`),
  `MEM_SEMAPHORE`, `WAIT_REG_MEM`, `ACQUIRE_MEM`, `PFP_SYNC_ME`),
  data movement (`WRITE_DATA`, `DMA_DATA`, `COPY_DATA`) and state
  control (`CLEAR_STATE`, `CONTEXT_CONTROL`, `SET_BASE`,
  `INDEX_BASE`/`INDEX_TYPE`/`INDEX_BUFFER_SIZE`, `NUM_INSTANCES`,
  `PREAMBLE_CNTL`). Memory-backed effects go through the `CpMemPort`
  trait so the CP is unit-testable against a fake port.

### 1.2 GCN shader recompilation (`pdg-gpu/src/gcn/`, ~3,200 lines)

```
guest ISA ──(decode)──► Inst ──(lifter, rspirv)──► SPIR-V ──(naga)──► WGSL
```

* **Decoder + test encoder (`decode.rs`)** — gfx10 encoding layouts
  verified against LLVM's public AMDGPU backend (e.g.
  `v_add_f32_e32 v1, v2, v3` → `0x06020702`); covers SOP1/SOP2/SOPC/
  SOPK/SOPP, VOP1/VOP2/VOP3, SMEM, MUBUF and EXP with literal-source
  dword consumption exactly like hardware. The `Encoder` is the
  inverse used by every test and the integration suite.
* **Opcode tables (`opcodes.rs`)** — 82 verified opcodes across the
  scalar and vector spaces actually exercised by the Phase-3 shader
  contract (`s_mov`, `s_add`, `s_waitcnt`, `s_load_dwordx4`,
  `v_mov_b32`, `v_add_f32`, `buffer_load_dwordx4`, `exp`, `s_endpgm`,
  ...).
* **Lifter (`lifter.rs`)** — GCN subset → SPIR-V with a scalar/vector
  register file, inline-constant handling, descriptor-table `s_load`
  elision, `buffer_load` vertex fetches compiled against the
  fetch-parameter storage buffer (`[stride, count]` pairs), and `exp`
  exports to `@builtin(position)` / `@location(n)`. Uninitialized
  reads are refused by register name; control flow and LDS are
  refused honestly with precise errors.
* **naga front-end (`naga_fe.rs`)** — SPIR-V → WGSL through naga's
  `spv-in` + `wgsl-out` with full validation.
* **Shader cache (`cache.rs`)** — content-addressed on-disk cache:
  SHA-256 of (guest ISA ‖ stage ‖ specialization) → WGSL text +
  manifest, versioned format.

### 1.3 GNM ring + wgpu backend (`ring.rs`, `wgpu_be.rs`, `ir.rs`, ~1,440 lines)

* **`ir.rs`** — the typed `GpuContext` (derived-on-read from the
  register file) and `GpuCommand` (Draw, Dispatch, Fence, Copy, ...).
* **`ring.rs`** — the ring MMIO device: the CPU programs
  base/size/write-pointer, writes the doorbell, and the host `pump`
  executes the ring through the CP and the bound backend. Doorbell
  generations make concurrent submissions safe; the device is shared
  through `Arc` on the bus.
* **`wgpu_be.rs`** — the `GpuDevice` implementation over wgpu 27:
  adapter enumeration (Vulkan preferred), pipeline caching per guest
  program, vertex data bound as a read-only storage buffer through
  the descriptor-table contract (GNM has no fixed-function vertex
  fetch), offscreen render to an `Rgba8Unorm` target, pixel-exact
  readback into guest RAM, EOP fences after the draw completes.
  `NullGpu` remains the honest no-adapter fallback (fences and copies
  still work; draws are skipped with a trace line).

### 1.4 Bugs found and fixed during closure

1. **Test MMIO window inside RAM** — the tier-1 test placed the ring
   MMIO at `0x90000`, inside the 1 MiB guest RAM; the bus decodes RAM
   before MMIO, so the CPU's doorbell write silently landed in RAM.
   Moved outside RAM (`0x4000_0000`) with a comment explaining the
   decode order.
2. **Fetch-parameter buffer as `Uniform`** — WGSL requires 16-byte
   array strides in the `uniform` address space; the packed u32
   `[stride, count]` pairs failed naga validation. The buffer is now
   a `NonWritable` read-only **storage** buffer on both the shader
   and the wgpu bind-group-layout sides.
3. **Phantom vertex-stage input** — the lifter built `@location(0)`
   **inputs** for declared PS inputs on the vertex stage too; wgpu
   refused the pipeline because the input assembler provides no such
   attribute (GNM vertex shaders *produce* interpolants, they do not
   consume them). PS inputs now only declare the input side for the
   fragment stage. Found by dumping the lifted WGSL with the new
   `pdg-wgsldump` tool.
4. **Driver `ret` fault** — the tier-1 CPU driver returned into a
   zeroed stack and faulted after leaving the fence poll; it now
   parks with `hlt`, and the test asserts a clean
   `RunOutcome::Halted`.

### 1.5 Numbers

* 71 new tests since Phase 2 — **224 total, all passing** (unit,
  integration, differential fuzz, SMP, GPU end-to-end).
* `cargo clippy --workspace --all-targets -- -D warnings` clean;
  `cargo fmt --all -- --check` clean.
* ~7,200 lines in `pdg-gpu` (6 modules, 16 files).
* End-to-end render (lavapipe, development container): the full
  guest draw — ring decode, shader decode, lift, naga round-trip,
  pipeline build, render, 64×64 readback — completes in **~40 ms**,
  dominated by the first-use pipeline compile (cached afterwards).

---

## 2. Key technical decisions and justifications

1. **Verified tables over coverage** — every opcode and register
   number is copied from public sources (kernel `soc15d.h`,
   `gc_10_3_0` headers, `navi10_ip_offset.h`, LLVM AMDGPU, shadPS4's
   Liverpool docs), and everything outside the verified sets is
   refused with a precise error. The alternative — guessing numbers
   from memory — produces a GPU that "works" on hand-written tests
   and fails on real captures.
2. **Semantic IR between CP and backend** — the CP emits typed
   `GpuCommand`s instead of calling wgpu directly. This keeps the CP
   unit-testable without a host adapter, lets `NullGpu` verify the
   command path in CI, and leaves the `GpuDevice` seam open for a
   thin Vulkan/`ash` fallback if wgpu's ray-tracing or scheduling
   abstraction proves insufficient later.
3. **Storage buffer, not uniform, for fetch parameters** — WGSL's
   16-byte uniform-stride rule would force padding the packed u32
   `[stride, count]` pairs; a read-only storage buffer keeps the
   guest-visible layout and the same load code. Documented in the
   lifter where the SPIR-V is built.
4. **The mini-GNM contract, explicit** — the Phase-3 driver contract
   (vertex id in SGPR 6, descriptor table at SGPR pair 0, T# at
   table + 0x10 dwords, `param0` → color) is fixed and declared in
   `LiftInputs` rather than inferred. Real GNM streams bring their
   own bindings once a kernel exists to parse them (Phase 5); the
   lifter then specializes per draw.
5. **`pdg-wgsldump` as a first-class tool** — when a lift looks
   wrong, the fastest honest check is to look at the WGSL. The tool
   that found bug #3 ships with the workspace.
6. **CPU-driven tier in tests** — the ring is not merely pushed from
   the host: an emulated x86-64 driver (hand-assembled) programs the
   ring MMIO, rings the doorbell, and polls the fence. This is the
   console driver contract, and it exercises the CPU, bus, MMIO, and
   GPU layers together.

---

## 3. Libraries used and why

| Crate | Role | Why |
|-------|------|-----|
| `wgpu 27` | rendering backend | one code path over Vulkan/DX12/Metal/GLES; matches the project's portability rule |
| `naga 27` (`spv-in`, `wgsl-out`) | SPIR-V → WGSL | the same validator wgpu uses — a shader that round-trips here is a shader the backend will accept |
| `rspirv 0.12` | SPIR-V construction | typed builder with word-level control for the lifter |
| `sha2 0.10` | shader cache keys | content-addressed cache identity |
| `pollster 0.4` | adapter/device init blocking | wgpu's async surface is not needed at boot; no tokio dependency in the GPU crate |

(Existing workspace crates: `thiserror` for typed errors, `tracing`
for the skip/trace lines, `pdg-memory` for the bus/MMIO types.)

---

## 4. Current state of emulation

What genuinely works, verified by tests:

* A guest ring built in guest RAM, decoded, and executed — including
  `SET_SH_REG`/`SET_CONTEXT_REG` state, a draw, and an EOP fence
  written back into guest RAM.
* The emulated CPU driving that ring through MMIO registers, ringing
  the doorbell, and observing the fence (tier-1 test).
* A guest GCN vertex shader (hand-assembled via the `Encoder`)
  decoded, lifted to SPIR-V, validated through naga to WGSL, compiled
  by wgpu, and executed — fetching vertices through the
  descriptor-table/storage-buffer contract (tier-2/3 tests).
* A rendered fullscreen triangle, read back into guest RAM, with
  **all 4096 pixels** verified against the guest shader's export
  values, on a software Vulkan adapter (lavapipe).

What does **not** work yet (the honest list):

* **No commercial PS5 games.** No Orbis kernel or SELF/ELF loading
  (Phase 5), no storage/input stacks (Phase 4) — nothing exists yet
  to boot a game through this GPU.
* **Render targets**: linear `8_8_8_8` UNORM, one color target, no
  depth/stencil, no blending, no tiling (GPU addresses are treated as
  guest-physical, `PITCH.TILE_MAX`-sized targets only).
* **Primitives**: point/line/triangle lists and strips; triangle
  fans are refused (need index expansion), not faked.
* **Shaders**: the lifted subset is the mini-GNM contract above —
  scalar/vector MOV/ALU, descriptor-table loads, `buffer_load`
  fetches, `exp` exports. Control flow (branches), LDS, most VOP3,
  interpolation modifiers, and every instruction outside the subset
  are refused by name. The pixel shader used by the backend is a
  hand-written pass-through until guest pixel shaders lift (the
  fragment-stage lifter exists and is unit-tested).
* **Compute**: `DISPATCH_DIRECT` decodes into a `GpuCommand`, but the
  wgpu backend refuses to render it (Phase 4).
* **No ray tracing** (RDNA 2 RT hardware is not modeled at all).

---

## 5. Known issues and risks

1. **Fixed driver contract** — the vertex-fetch contract is
   hardcoded in `LiftInputs` per backend call. Real GNM programs
   carry their own binding metadata; the specialization arrives with
   the Phase-5 loader, and the lifter refuses mismatched programs
   rather than mis-lifting them.
2. **Shader ISA coverage** — the decoder understands the gfx10
   encoding layouts, but opcode coverage is deliberately narrow (82
   opcodes). Real shaders will hit refusals; the error messages name
   the instruction, and the tables extend mechanically.
3. **wgpu backend is single-draw synchronous** — one draw per
   submit; render passes, batching, and async queues are future work
   (the `GpuDevice` contract already batches commands).
4. **GPU addresses are guest-physical** — no GPUMMU paging is
   modeled; targets/fetches resolve directly against the bus. Real
   streams will need the GART (tracked with the kernel phase).
5. **lavapipe-only CI** — the e2e test skips (with a reason printed)
   on machines without an adapter; the command-path and shader-path
   tiers still run everywhere. Adding lavapipe to the CI image is a
   Phase-4 infrastructure task.
6. **`TriFan` refusal** — correct-but-limited; the expansion needs an
   index-buffer rewrite in the CP.

---

## 6. Next phase plan (Phase 4 — I/O)

1. **Storage** — the NVMe-shaped SSD queue model over
   `AsyncFileBlockDevice`, DMA into guest RAM, and the Kraken/Oodle
   posture decision (user-supplied components only; never infringing
   code in-tree).
2. **Input** — the DualSense model (buttons, analog sticks, touch,
   gyro, adaptive-trigger basics) behind an input trait with
   host-controller mapping.
3. **Audio** — the Tempest 3D AudioTech-shaped mixer: channel
   hydration, the `pdg-audio` backend over cpal, ring-buffer
   exchange with the emulated kernel.
4. **GPU depth** (carry-over) — depth/stencil targets, blending, and
   batching in the wgpu backend; lavapipe in CI; compute dispatch
   rendering.
5. **Infrastructure** — extend `pdg-boot` to mount a disc image and
   drive an I/O session end to end.

---

*Phase 3 complete: the machine has eyes. Every command and shader
that runs is verified or refused — nothing pretends. Stopping per the
phase gate until "Continue to Phase 4".*
