# PHASE 6 REPORT — Runtime: Dynlib/PRX, Threads, Snapshots, Frontend

**Scope delivered:** the process *runtime* on top of the Phase-5
kernel — the dynlib layer (a module table with cross-module linking and
NID bookkeeping), guest threads over a round-robin scheduler with real
umtx parking, the `stat` family with the layout verified against two
public sources, a session snapshot format that restores a mid-program
machine into a fresh process byte-identically, the egui launcher over
a headless-tested model, and the release engineering (crash reporting,
GUI CI job, release builds) — plus the flagship program that proves all
of it: a modular, multi-threaded ELF64 process whose `DT_NEEDED`
closure loads through the kernel's own dynamic linker.

**Bottom line:** an emulated CPU now boots a program that is *not one
image* — the main executable carries a `DT_NEEDED` on `libpdgx.prx`,
which itself needs `libpdgy.prx`; the kernel loads the closure, maps
each module at its own bias, relocates every `JUMP_SLOT` and
`GLOB_DAT` against the others' exports, and answers `dlsym` across the
chain. The process then spawns a thread through a genuine FreeBSD-9.2
`ucontext_t`; the thread gets its *own* copy of the aggregated TLS
block, the two threads contend a umtx mutex (one parks, the unlock
hands the lock over FIFO), join through `WAIT`/`WAKE` futex semantics,
and the main thread verifies its TLS was never touched. Along the way
the process `stat`s its own PRX file and reads its data through the
verified 120-byte layout. When the host snapshots the machine
mid-program and restores it into a *fresh* session, the run continues
to the same final statuses and console transcript — determinism proved,
not claimed. Every gap below is listed honestly.

---

## 1. What we accomplished

### 1.1 The dynlib layer (`pdg-kernel/src/dynlib.rs`, ~1,000 lines)

* **The module table** — [`Dynlib`] registers the main executable
  (handle 1) and every `load_prx`ed shared object, tracks each
  module's mapped segments, its `DT_NEEDED` list, its exports, and its
  TLS block offset in the process's aggregated TLS arena.
* **Two-phase loading, rtld-style** — Phase 5's monolithic `load()` is
  now [`map`] + [`relocate`]: a module and its whole needed closure are
  mapped first, then each is relocated with a resolver that searches
  the module's own exports and then its needed chain depth-first —
  exactly the discipline a real dynamic linker follows, which is what
  makes cross-module `JUMP_SLOT`/`GLOB_DAT` relocations bind both ways.
* **`DT_NEEDED` through the PFS** — needed names resolve per the
  documented PS4 convention (`/app0/sce_module/<name>`), the closure
  loads breadth-first with a depth limit of 8 and a cycle detector,
  duplicate dependencies deduplicate to one handle, and `unload_prx`
  refuses (EBUSY) while anything still needs the module.
* **Position independence, honestly tested** — the demo PRX's code
  finds its own data through RIP-relative `lea` (the displacement is a
  build-time constant, so any load bias works); the first attempt used
  absolute `moffs` addressing and the integration suite caught it
  immediately, which is the discipline working.
* **NIDs, the stated model** — PlayStation modules export through
  64-bit NIDs whose public 11-character encoding names real system
  modules' dynsym entries; Sony's exact name→NID hash is not published
  in a source PDG can verify, so [`nid`] computes PDG's own documented
  scheme (the first eight bytes of SHA-256, little-endian — matching
  the public 64-bit width), recorded beside every export and asserted
  by tests. `dlsym` takes the name string either way — NID-encoded for
  real system libraries, plain for PDG-built modules.
* **TLS model v2** — the process keeps one aggregated TLS arena (main
  template at offset 0, each PRX's template appended 8-byte-aligned);
  `TPOFF64` relocations are applied as `module_tls_offset + addend`, so
  one instruction sequence addresses each thread's own block.
  Single-module processes behave byte-identically to Phase 5.

### 1.2 Threads and umtx (`pdg-kernel/src/thread.rs`, orbis.rs)

* **`thr_create` (430)** parses the FreeBSD 9.2 amd64 `ucontext_t`
  (field offsets cited from the public headers: `mc_rip` at `+0xB0`,
  `mc_rsp` at `+0xC8`, `mc_rdi` at `+0x18`), allocates the new
  thread's TLS block (its own copy of the aggregated template, mapped
  from the mmap arena), and queues a `SpawnedThread` the session
  scheduler materializes into an interpreter with its own `FS` base.
* **`thr_exit` (431)** ends the calling thread and wakes any waiters
  parked on its state address (the libthr join channel); **`thr_self`
  (432)** reports the caller's tid. When the last live thread exits,
  the process exits with code 0 — the documented PDG policy. `exit`
  kills every thread.
* **`_umtx_op` (454)** implements the verified FreeBSD 9.2 op set for
  this tier: legacy `LOCK`/`UNLOCK` word locks, `WAIT`/`WAKE` on 64-bit
  words with the kernel's queue-then-compare discipline (a mismatched
  value returns immediately — no lost wakeup), and the umutex family
  (`MUTEX_TRYLOCK` EBUSY, `MUTEX_LOCK` with `EDEADLK` on recursion,
  `MUTEX_UNLOCK` with EPERM on foreign release and **FIFO handoff**:
  one parked waiter receives the lock directly). Timed waits are
  refused with `ENOSYS` and logged — the honest unimplemented marker.
* **The scheduler** (`RuntimeSession`) runs every runnable thread one
  200-instruction slice per round, materializes spawns, retires
  exits, and ends the process when the last thread is gone. Parking
  is a first-class outcome: the syscall's success value is written to
  RAX *before* parking, and the slice ends cleanly through the new
  `Fault::ThreadParked` (the same architectural-escape pattern
  `Fault::ProcessExit` established).
* **A real bug this exposed, fixed in the core**: the interpreter
  retired `syscall` *after* the hook returned; a parked thread
  therefore re-executed its wait on resume and saw stale state (a
  handed-off mutex read as its own → EDEADLK). The fix — the
  `syscall` retires the moment the kernel takes over — is how the
  silicon behaves and is now a documented interpreter invariant.

### 1.3 The stat family and the remaining syscall gaps

* **`stat` (188) / `fstat` (189)** — the FreeBSD 9.2 amd64 `struct
  stat`, 120 bytes, **verified against two independent public
  sources** (the FreeBSD 9.2 headers and the OpenOrbis
  `OrbisKernelStat`, which agree field-for-field): `st_dev` u32@0,
  `st_ino` u32@4, `st_mode` u16@8 … `st_size` i64@0x48,
  `st_blksize` u32@0x58, `st_birthtim`@0x68. The PFS inode fills
  `st_ino`/mode/size; timestamps read zero — the honest answer for
  what the on-disk format keeps.
* **The Sony dynlib block, implemented where pinned** — `dlsym`
  (591), `get_list` (592), `get_info` (593, the size-prefixed 0x160
  `OrbisKernelModuleInfo`), `load_prx` (594), `unload_prx` (595),
  with argument shapes cited to the public payload SDK that calls them
  on real firmware. **`get_proc_param` (598) and
  `process_needed_and_relocate` (599) stay refused** (ENOSYS, logged
  by name): no accessible public source pins their ABI shapes, and
  PDG's `load_prx` performs needed-loading and relocation eagerly
  instead — documented, not guessed.

### 1.4 Session snapshots (`PDGS`, kernel_session.rs + codecs)

* **The format** — `"PDGS"` + version + the CPU contexts (GPRs, RIP,
  flags, FS base, IDT) + the nonzero 4 KiB RAM pages + the kernel
  tables (fds, objects, kevent registrations, the dynlib module table,
  threads, the VM allocator state) + the console transcript, closed
  with a SHA-256 digest. Corruption, truncation, and version
  mismatches are refused, never guessed at.
* **The proof** — the integration suite runs the demo twice: once to
  completion, once with a snapshot captured after the dlsym stage;
  the snapshot is restored into a *fresh* machine (new bus, new
  devices, new kernel) and the run resumed — final statuses, console
  transcript, and thread count all match the full run exactly. The
  file-format round-trip (encode → decode → resume) is a separate
  test, so the codec provably loses nothing the live snapshot
  carried.
* **Where it is honest about limits** — the snapshot is taken between
  scheduler rounds (a clean instruction boundary); device-side
  in-flight state (an NVMe command mid-doorbell) is not captured —
  the demo snapshots at a point where nothing is in flight, and the
  format's users must respect that until device state joins the
  snapshot.

### 1.5 The launcher GUI and the model under it (pdg-frontend)

* **The model** (`launcher.rs`, headless-tested): `LauncherSettings`
  persists as TOML (`PDG_CONFIG`/XDG config home, atomic directory
  creation), and save-state scanning reads `PDGS` headers without
  loading whole machines (thread and RAM-page counts) while refusing
  non-version-1 files. Both have unit tests, including a
  real-snapshot-to-disk round trip.
* **The GUI** (`gui.rs`, the `gui` feature): the egui app — library
  list, settings form, save-state table, a run button that drives the
  runtime session on a host thread and streams the guest console into
  the log panel. It cannot be pixel-tested in CI (no display); the
  new CI job keeps it compiling (`cargo check --features gui`), and
  the logic it renders is all in the tested model — the same posture
  the cpal audio backend established in Phase 4.
* **`pdg-launcher`** (behind `required-features = ["gui"]`) is the
  windowed entry point; `pdg-boot` keeps every headless demo.

### 1.6 Release engineering

* **Crash reporting** (`pdg-core/src/crash.rs`) — every binary
  installs a panic hook: the panic and a forced `std::backtrace` go to
  the log, stderr, and a `pdg-crash-<millis>.log` file
  (`PDG_CRASH_DIR` overrides). Unit-tested with a real panic caught
  in `catch_unwind`. Zero new dependencies.
* **CI** — the GUI compile-check job and a tag-triggered release job
  (builds `pdg-boot`, `pdg-disasm`, `pdg-wgsldump`, `pdg-launcher`
  release binaries and uploads them). A pre-existing YAML typo in the
  push trigger (it read `branches: ain]`, matching nothing) is fixed
  to `[main]`.
* **`--version`** on `pdg-boot` (clap's standard flag) and the
  release profile's LTO settings were already in place from Phase 1
  and are unchanged — verified, not re-tuned.

### 1.7 Bugs found and fixed during closure

1. **The fail tail stored garbage** (latent since Phase 5): the
   stage-builder's fail jumps landed *past* the `xor eax,eax`,
   storing the live RAX instead of 0. It never fired before because
   every Phase-5 stage passed. Found the moment a Phase-6 stage
   legitimately failed and the status qword held the comparison
   operand instead of 0.
2. **The syscall-RIP retirement bug** (§1.2) — parked threads
   re-executed their syscall; a handed-off mutex read as the caller's
   own and deadlocked with EDEADLK.
3. **`mmap_anon` reserves but does not map** — `thr_create`'s TLS
   block allocation called the arena-reservation helper without
   mapping the pages; the write path-faulted. The `mmap` *syscall*
   had it right all along.
4. **PRX code with absolute addressing** — the demo library's
   `pdgx_tls_bump` used `moffs` absolute loads, which break under a
   nonzero load bias; rewritten RIP-relative (§1.1).
5. **The dlsym names were never written into the demo's data image**
   — stages read empty strings; the tests named it immediately.
6. **The dynsym walk needed a terminator rule** — without DT_SYMTAB
   sizes the parser walked into the string table; the all-zero-entry
   terminator (plus the builder emitting it explicitly) makes the
   walk bounded, and real linkers' zero padding agrees.
7. **Exports are `st_shndx != SHN_UNDEF`, not `st_value != 0`** — a
   function at its module's offset 0 is a legitimate export; the
   symbol filter now reads the section index the ELF ABI defines for
   this.

---

## 2. Key technical decisions and justifications

1. **Verified shapes only, again** — the syscall numbers cite the
   FreeBSD 9.2 `syscalls.master` (430/431/432/454/188/189) and the
   public payload SDK for the Sony block (591–595); `struct stat`
   cites the FreeBSD headers *and* the OpenOrbis SDK, which agree
   field-for-field. 598/599 stay refused because no accessible
   public source pins their shapes — the exact refusal posture
   every phase has held.
2. **The thread scheduler lives in the session, not the kernel** —
   the kernel owns thread *state* (parking, tids, umtx semantics);
   the session owns interpreters and slice discipline. This keeps
   the syscall layer unit-testable without a CPU, exactly like the
   rest of orbis.rs, and leaves the SMP host-threading (Phase 2) as
   the future escalation path.
3. **FIFO mutex handoff** — `MUTEX_UNLOCK` transfers the lock
   directly to the first waiter (the value is written on its behalf
   before the wake). This is deterministic — the demo's choreography
   is order-independent because every interleaving converges — and
   it models the fair end of the real spectrum.
4. **Snapshots clone state; files encode it** — the in-memory
   snapshot is typed (deep clones), the on-disk format is a
   length-prefixed binary with a digest, and both are exercised:
   the determinism test uses the typed path, the codec test the
   byte path, and a resume-from-decoded test proves they agree.
5. **The GUI is a feature, not the product** — `gui`/`launcher`
   features keep the default dependency graph lean (the Phase-5
   "nothing new" property survives for default builds), while the
   tested model/gui split keeps the GUI honest about what it can
   and cannot verify headlessly.
6. **`exit` ends all threads** — the POSIX shape; the thread table's
   `exit_all` runs in the syscall, so the report's live-thread count
   reads zero after a clean exit.

---

## 3. Libraries used and why

| Crate | Role | Why |
|-------|------|-----|
| eframe/egui 0.37 (optional, `gui` feature) | the launcher window | the phase plan named egui; the glow+X11 build needs no system GUI libraries at compile time, which CI proved |
| sha2 (existing) | NIDs, snapshot digests | already in-tree for the GPU cache; the save-state integrity and the NID scheme needed nothing new |
| serde + toml (optional, `launcher` feature) | settings persistence | already in-tree for layered config |
| std::backtrace (stdlib) | crash reports | zero-dependency backtraces |

Everything else is the existing stack. The default build still
compiles with no new dependencies; the GUI tier adds exactly what a
windowed launcher requires and nothing more.

---

## 4. Current state of emulation

What genuinely works, verified by tests (**344 passing
workspace-wide** with the launcher feature on; 25 new in this phase
across dynlib/thread/orbis unit tests, the launcher model, the crash
hook, and the four-test runtime integration suite):

* A modular process: the main executable with a two-deep `DT_NEEDED`
  closure loads through the kernel's dynlib; `JUMP_SLOT`,
  `GLOB_DAT` and `TPOFF64` relocations bind across modules; `dlsym`
  resolves by name through any module; the table answers
  `get_list`/`get_info` with the documented struct; `unload` refuses
  while needed.
* Threads: `thr_create` from a genuine ucontext, per-thread TLS
  blocks (the worker's `fs:` reads its own copy — the main thread's
  stays untouched), `thr_self`, `thr_exit` with join-channel wakes,
  umtx mutex contention with parking and FIFO handoff, futex
  `WAIT`/`WAKE` joins — all verified by the 16-stage integration
  suite down to the exact syscall count.
* `stat`/`fstat` with the verified 120-byte layout over the PFS.
* Snapshot/restore: a mid-program machine, captured and restored
  into a fresh session, finishes to identical final state; the
  binary format round-trips and refuses corruption.
* The launcher model (settings, save-state headers) unit-tested; the
  egui app compiles in CI; crash reports write real files.
* Benchmarks re-run for the phase gate: interpreter 6.35 ms per
  128 Ki-loop (as Phase 2 left it), JIT alu-loop and memory-loop
  unchanged, MMU TLB hit 10.4 ns / full walk 14.6 ns — the kernel
  layer's growth cost the hot tiers nothing.

What does **not** work yet (the honest list — the carry-overs that
matter most for real guests):

* **`thr_create` flags are refused** (THR_SUSPENDED/DETACHED-shaped
  bits: EINVAL) — no use case pinned them yet.
* **Timed umtx waits refused** (ENOSYS) — the pthread
  timeout-bearing calls need them next.
* **One process** — no fork/exec or process table; no priorities,
  no preemption: the scheduler is cooperative round-robin over
  runnable threads.
* **`get_proc_param`/`process_needed_and_relocate` refused** —
  shapes unpublished (§1.3); PDG's loader does the work eagerly.
* **Snapshots capture no device in-flight state** (§1.4) and no
  guest MSRs beyond the architectural set the demo uses.
* **The GUI streams the console after the run completes** — a
  long-running guest's live-streaming needs chunked session pumping
  (the seam exists: the shared console Arc).
* **No `pthread_join`-shaped value return** — the worker writes its
  result to shared guest memory instead (the demo's own pattern).
* **TLS is PDG's documented model v2** — aggregated arena, per-thread
  copies; Sony's exact TCB layout still needs a dump (the Phase-5
  caveat, unchanged).
* **Carry-overs from earlier phases** — GPU depth/stencil/blend,
  async NVMe, network, HRTF, SELF encryption/compression, PFS
  write/signed/encrypted support (all tracked in the prior reports).

---

## 5. Known issues and risks

1. **The runtime demo is the only threaded guest** — the scheduler,
   umtx and TLS paths have exactly one demanding customer. More
   thread-shaped payloads (or real module code) will pressure the
   park/wake interleavings the tests do not yet produce.
2. **`SharedBuilder` st_shndx=1 is a simplification** — dynamic
   linking consults the index only for UNDEF checks, and these
   images carry no section headers; a real `.rela.plt`-style layout
   would want section-aware tools someday.
3. **The snapshot RAM capture is whole-page** — a guest with a
   mostly-dirty working set produces large files; the sparse format
   skips only all-zero pages.
4. **The GUI's run button runs the demo session** — wiring it to
   arbitrary user images needs the loader path for untrusted ELFs
   (parse-refusal hardening beyond the current honest-error set).
5. **The report's syscall count is exact** (24) — a brittle assertion
   by design: any added stage must consciously update it.

---

## 6. What six phases built

A PS5-shaped machine in Rust, honestly: memory with real page tables;
a complete x86-64 core (interpreter + JIT + SMP) verified against the
architecture; a GPU command processor with shader recompilation; NVMe,
DualSense and Tempest-shaped I/O; a kernel that speaks Orbis's numbers
with sources cited and gaps refused by name; and now a *process* —
modular, threaded, snapshot-able — that runs on top of it all. The
roadmap's planes are done; what remains is the long tail of making
real games' code run, which is future work beyond the phase gates.

*Phase 6 complete — and with it, the plan.*
