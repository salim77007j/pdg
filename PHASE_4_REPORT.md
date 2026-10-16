# PHASE 4 REPORT — I/O: NVMe SSD, DualSense, Tempest Audio

**Scope delivered:** the console's I/O plane — the NVMe-shaped SSD
with doorbell-driven queues and PRP DMA into guest RAM, the DualSense
pad model with adaptive-trigger physics and a seqlocked guest window,
and the Tempest-shaped audio mix path with a guest-visible hydrate
interface — plus the infrastructure that proves it: a hand-assembled
guest driver that boots on the emulated CPU and drives all three
subsystems end to end.

**Bottom line:** an emulated x86-64 CPU now boots a driver that enables
a virtual NVMe controller through the register file, identifies it,
creates I/O queues, writes and reads a sector through real
Physical-Region-Page DMA into guest RAM, reads the gamepad window, and
hydrates an audio block that renders to stereo PCM — then prints
`PHASE4 IO OK` and halts. Every stage is verified by the integration
suite; every refusal is a precise NVMe status code, never a guess.

---

## 1. What we accomplished

### 1.1 Storage: the NVMe model (`pdg-io/src/nvme/`, ~1,400 lines)

* **Register file (`regs.rs`)** — the BAR0-shaped window at the public
  NVMe 2.0 offsets: CAP (MQES 4095, TO 1), VS (2.0), CC with EN/MPS/CSS/
  SHN decode, CSTS with the RDY handshake, AQA/ASQ/ACQ latching, and
  the doorbell array at `0x1000` (two 4-byte doorbells per queue,
  `DSTRD = 0`).
* **Queue entries (`cmd.rs`)** — the 64-byte submission entry (CDW0
  opcode/fuse/PSDT, NSID, PRP1/PRP2, CDW10–15) and the 16-byte
  completion entry (result, SQHD/SQID, CID, status-with-phase-tag) with
  encode/decode and the modeled opcode sets: admin `CREATE/DELETE I/O
  SQ/CQ`, `IDENTIFY`, `ABORT`, `GET/SET FEATURES`; NVM `READ`, `WRITE`,
  `FLUSH`.
* **PRP translation (`prp.rs`)** — the full DMA model: the first
  pointer may address any offset in its page; the two-page special case
  takes PRP2 as the page pointer directly; longer transfers walk a PRP
  *list* in guest RAM. Scatter (`dma_write`) and gather (`dma_read`)
  cross the bus against resolved segments, refusing misalignment.
* **IDENTIFY structures (`identify.rs`)** — Controller (VID/SSVID
  `0x5044`, serial `PDG-SSD-000001`, model `PDG Virtual NVMe SSD`,
  firmware `0.4.0`, NN=1) and Namespace (NSZE/NCAP/NUSE from the
  backing file, LBAF0 with LBADS=9). Only fields PDG can fill honestly
  are nonzero; every offset is documented.
* **Controller (`controller.rs`)** — the `NvmeController<D:
  BlockDevice>` MMIO device: CC.EN 0→1 builds the admin pair and raises
  CSTS.RDY (with CFS on refused configurations: unconfigured queues,
  non-4 KiB MPS, non-NVM command sets); 1→0 tears everything down.
  Doorbell writes latch; **`pump(&bus)`** consumes pending submissions,
  executes them against the backing block device, and posts completions
  with phase tags that invert on completion-queue wrap (unit-tested
  across the wrap). The queue *type* decides the opcode namespace —
  admin commands live on queue 0, NVM commands on I/O queues — because
  the byte values overlap (admin `CREATE I/O SQ` 0x01 vs NVM `WRITE`
  0x01; this collision was a real bug found by the queue-creation
  test). Out-of-range LBAs are refused with `LBA Out Of Range`, wrong
  namespaces with `Invalid Namespace or Format`, unknown opcodes with
  `Invalid Command Opcode`. `ABORT` completes having aborted nothing —
  correct, because no command is ever in flight across a doorbell in
  this model. `GET FEATURES` answers only Number-of-Queues; `SET
  FEATURES` and `GET LOG PAGE` are refused.
* **Kraken/Oodle posture** — the deliberate legal decision, in code and
  docs: retail PS5 storage streams are compressed with proprietary
  codecs; PDG ships none of them. The `Decompressor` trait is the seam a
  user's own legally-obtained component would plug into.

### 1.2 Input: the DualSense model (`pdg-io/src/dualsense.rs`, ~600 lines)

* **State** — the full pad: PDG button bitfield, dpad hat (with
  impossible diagonals staying neutral), 0–255 sticks, analog L2/R2, two
  touch points, gyro/accelerometer samples, battery.
* **Adaptive triggers by behavior** — `TriggerEffect::resistance_at(
  pull, t)` models what the player *feels* per mode: `Feedback`
  (constant resistance after a threshold), `Weapon` (ramp-stiffen-
  break), `Vibration` (oscillating). The wire-format output blocks
  carry the publicly documented mode bytes for the host path; the
  parameter-byte layout is a constants table that Phase-5 hardware
  captures will verify, not a guess presented as fact.
* **Guest window (`PadDevice`)** — a seqlock-shaped MMIO snapshot:
  the host `set_state` bumps a sequence number around the write; the
  guest reads SEQ, reads the state registers, re-reads SEQ, and retries
  if it moved. Guest writes are ignored.
* **HID report 0x01** — encode for the unambiguous subset (report id,
  both button bytes with the dpad hat, sticks, analog triggers, counter
  nibble). Touch and motion extensions are deliberately *not*
  serialized until a real transport verifies their offsets — the same
  refuse-don't-guess posture as the GPU tables.
* **Host mapping** — `map_host_input` maps a normalized `InputState`
  onto the pad with a stick deadzone, and hard trigger presses set
  their digital bits.

### 1.3 Audio: the Tempest-shaped mix path (`pdg-audio/`, ~700 lines)

* **Mixer (`mixer.rs`)** — voice hydration/kill, per-voice gain,
  distance attenuation (clamped inverse) with constant-power pan law,
  saturating stereo accumulation, master gain after the sum. The
  constant-power law means a centered voice plays at −3 dB per
  channel — the tests assert exactly that.
* **Spatializer** — `SpatialParams` with distance/pan math
  (`channel_gains` = attenuation × pan coefficients). **HRTF is
  honestly absent**: real HRTF needs a verified HRIR dataset; the pan
  law is what ships and the gap is documented, not papered over.
* **Resampler (`resample.rs`)** — linear, with a phase carry between
  blocks so streamed audio stays continuous across block boundaries
  (asserted on a two-block ramp).
* **Guest window (`device.rs`)** — the audio MMIO device: the guest
  programs block base/length, pan and distance, then writes HYDRATE;
  the host `pump(&bus)` DMAs the mono-f32 block out of guest RAM into
  the mixer; `flush(backend)` renders frames and delivers interleaved
  i16 to any `AudioBackend`. Registers expose voices, hydrated samples
  and rendered frames so guests can poll consumption.
* **cpal backend (`cpal_be.rs`, feature `backend-cpal`)** — a real
  host output stream (WASAPI/ALSA/CoreAudio via cpal's default host),
  pulling from a queue; the stream starts on `init` and refuses writes
  before it. Off by default so headless builds stay dependency-light;
  CI machines without audio hardware see the refusal path, which is
  also tested.

### 1.4 Infrastructure

* **Config (`pdg-core`)** — new `[storage]` (ssd path/size/MMIO base)
  and `[input]` (pad MMIO base) sections, `audio.mmio_base`; validation
  demands 4 KiB alignment, above-RAM placement, and distinct pages for
  all I/O windows, with tests for each refusal.
* **Events** — `BlockIo { write, lba, blocks }` and
  `AudioHydrated { samples }` join the bus for observers.
* **The demo payload (`pdg-io/src/demo.rs`)** — ~140 hand-assembled
  x86-64 instructions built by typed emitters (`mov rax, imm64`,
  64-bit moffs load/store, `push`/`pop`, `cmp`, `test`, short jumps,
  `inc`, `hlt`). The integration suite disassembles the payload through
  iced-x86 and asserts every instruction is in the documented emitter
  set. The driver polls phase bits exactly like a real NVMe driver and
  stores per-stage pass/fail qwords the host verifies independently.
* **The session (`pdg-frontend/src/io_session.rs`)** — the machine
  builder: bus + SSD + pad + audio + console, the pump loop (service
  devices between 200-instruction slices — the same contract the GPU
  tier-1 harness established), and a `SessionReport` carrying outcome,
  per-stage statuses, console transcript, the rendered audio and SSD
  statistics.
* **`pdg-boot --demo-io <ssd.img>`** — the boot harness boots the demo
  driver and prints the honest per-stage report. The SSD image is the
  user's file; PDG creates an empty one when missing and never provides
  content.
* **CI** — a dedicated `gpu-e2e` job installs lavapipe
  (`mesa-vulkan-drivers`) and runs the pixel-verified GPU tier with
  `VK_ICD_FILENAMES` pointed at it — the Phase-3 promise that the
  software-Vulkan adapter runs the full render test in CI, kept.

### 1.5 Bugs found and fixed during closure

1. **Admin/NVM opcode collision** — the first controller design decoded
   "is this an admin command?" from the opcode byte; admin `CREATE I/O
   SQ` (0x01) collided with NVM `WRITE` (0x01) and queue creation
   completed as a write. The fix is the spec's own rule: the queue
   type decides the namespace. Found by the queue-creation test
   refusing with the *wrong* status code.
2. **CQ-only queues divided by zero** — after `CREATE I/O CQ` and
   before `CREATE I/O SQ`, the pump's pending-entry arithmetic hit
   `% sq_size` with `sq_size == 0`. The pump now skips submission-less
   queues.
3. **`test al, imm8(0)` is always zero** — the demo's print loop used
   `test al, 0` for the NUL check, which sets ZF unconditionally; the
   loop exited before writing a single byte. Replaced with `test al,
   al`. Found by single-stepping the interpreter against the payload.
4. **Print-loop back-jump arithmetic** — the rel8 was computed from the
   bytes before the `jmp` (−14) instead of from the end of the `jmp`
   (−16), skipping the character reload on every iteration after the
   first.
5. **PRP two-page special case** — an early test expected a PRP2
   *list* walk for a transfer that fits in two pages; the spec (and
   now the code) takes PRP2 as the page pointer directly in that
   case. The test now covers both behaviors explicitly.

---

## 2. Key technical decisions and justifications

1. **Doorbell + pump, everywhere** — the NVMe queues and the audio
   hydrate path use the same split as the Phase-3 GNM ring: guest MMIO
   writes latch; the host services at its scheduling point. This is
   not a convenience: real I/O completions are asynchronous, MMIO
   writes cannot reach the DMA engine, and the pattern keeps every
   device unit-testable without the CPU in the loop. The guest
   observes phase tags and counters — the hardware contract.
2. **Behavior over wire format for the pad** — the emulator-relevant
   truth of adaptive triggers is the resistance curve, so that is what
   is implemented and tested; the wire blocks are a constants table
   marked for Phase-5 verification. The alternative (guessing
   parameter-byte layouts) produces a pad that "works" in tests and
   mis-feels on hardware.
3. **The seqlock window** — pad state is read by the guest without
   host cooperation (input is device state, unlike storage DMA), but a
   torn read would be worse than none: the sequence-numbered snapshot
   gives the guest a consistency protocol real hardware drivers
   already practice.
4. **One backing trait, two engines** — the SSD model is generic over
   the existing sync `BlockDevice` (doorbell-pumped, deterministic,
   test-friendly); the Phase-1 `AsyncFileBlockDevice` remains the
   DirectStorage-shaped path a future async batching layer builds on.
   The queue model does not care which engine feeds it.
5. **Honest audio numerics** — constant-power panning means centered
   sources sum at −3 dB per channel; the tests assert the exact
   values rather than hand-waving "sounds right". Saturating
   accumulation clamps to ±1.0 instead of wrapping.
6. **The demo payload is a product, not a test hack** — the driver
   lives in `pdg-io::demo` with typed emitters, is disassembled and
   instruction-set-checked by the test suite, and drives the same
   machine `pdg-boot --demo-io` ships to users.

---

## 3. Libraries used and why

| Crate | Role | Why |
|-------|------|-----|
| `iced-x86` (existing) | payload disassembly check | the integration test proves the demo driver contains only verified instruction forms |
| `cpal 0.15` (feature `backend-cpal`) | host audio output | the de-facto Rust audio host abstraction (WASAPI/ALSA/CoreAudio); optional so headless/CI builds stay light |
| existing stack | everything else | `thiserror`, `tracing`, and the Phase-1–3 crates; Phase 4 added no other dependencies |

No new mandatory dependencies. The storage, input and audio models
need nothing beyond the bus and block-device abstractions that already
existed.

---

## 4. Current state of emulation

What genuinely works, verified by tests:

* A guest driver enables the NVMe controller through the register
  file with the CSTS.RDY handshake, submits IDENTIFY, and checks the
  serial number DMA'd into guest RAM.
* The guest creates an I/O queue pair, WRITEs a marker sector from
  guest RAM through PRP DMA, and READs it back into a different page —
  the marker is byte-compared by the guest and re-checked in the
  backing file by the host.
* The phase-tag machinery wraps correctly across completion-queue
  cycles (unit-tested through a full wrap).
* Refusals are precise NVMe status codes: out-of-range LBAs, wrong
  namespaces, unknown opcodes, refused features and log pages.
* The guest reads the pad window (buttons + stick) the host injected,
  through the seqlock protocol.
* The guest hydrates a mono-f32 audio block through MMIO; the host
  pump moves it into the mixer; the rendered stereo i16 (with the
  constant-power pan applied) is exactly what a backend receives.
* `pdg-boot --demo-io` runs the whole session and prints a per-stage
  PASS/FAIL report — 271 workspace tests, all passing, clippy `-D
  warnings` clean, `cargo fmt --check` clean.

What does **not** work yet (the honest list):

* **No interrupt model** — completions are polled (phase tags,
  counters). Real NVMe interrupt delivery (MSI-X) arrives with the
  kernel phase's interrupt plumbing.
* **No async batching** — one doorbell ring services one pump pass;
  DirectStorage-style concurrent submission batches over
  `AsyncFileBlockDevice` are future work.
* **No network stack** — the README's Phase-4 sketch mentioned
  network; nothing of it shipped, and it is re-planned with the
  kernel phase (sockets are syscall-surface, not device-surface).
* **Kraken/Oodle decompression** — deliberately absent (see the
  posture above); decrypted, decompressed user dumps only.
* **Adaptive-trigger wire layout** — the effect *behavior* is real and
  tested; the output-block byte layout awaits hardware capture.
* **HRTF** — pan-law spatialization only until a verified HRIR dataset
  is wired in.
* **GPU carry-over** — depth/stencil targets, blending, batching and
  compute-dispatch rendering in the wgpu backend remain open (tracked
  from the Phase-3 report); this phase shipped the CI lavapipe job
  from that list but not the rendering depth.

---

## 5. Known issues and risks

1. **Single-namespace, single-port** — one namespace, one I/O queue
   pair in the demo; the model supports 8 queues but multi-queue
   drivers are unexercised. The tables extend mechanically.
2. **Test-bus sizes** — several unit tests initially placed queues
   beyond their 1 MiB test RAM; the buses were grown and the addresses
   now exercise multi-megabyte PRP walks, but the lesson (fixtures
   must match the geometry they claim) is worth remembering for the
   kernel phase.
3. **Audio `flush` granularity** — `flush_all` renders in ≤512-frame
   chunks; a real output thread will want a ring buffer with
   backpressure rather than drain-to-empty. The `AudioBackend` seam
   already hides this from the guest.
4. **The pad's touch/gyro units** — raw sensor units pass through
   uncalibrated; calibration lands with host-hardware verification.

---

## 6. Next phase plan (Phase 5 — Kernel)

1. **Orbis syscall layer** — the seeded, documented syscall table over
   the existing `SyscallDispatcher`, with the honest-refusal posture
   for unverified numbers.
2. **SELF/ELF loading** — the loader that maps user modules, applies
   relocations, and builds process state (threads, TLS).
3. **Kernel objects** — event flags, mutexes, semaphores, equeues
   with the same object discipline this phase gave storage/input.
4. **PFS (the filesystem)** — read-only PFS over the SSD model, the
   first consumer of the storage stack by a real guest.
5. **Interrupt plumbing** — MSI-X-shaped delivery so NVMe completions
   and future devices can signal the cores.

---

*Phase 4 complete: the machine has hands, ears, and a fast drive — and
every command, byte and sample that moves is verified or refused.
Stopping per the phase gate until "Continue to Phase 5".*
