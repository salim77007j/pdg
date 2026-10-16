# PHASE 5 REPORT — Kernel: Orbis Syscalls, ELF/SELF Loader, PFS, Interrupts

**Scope delivered:** the console's kernel plane — the Orbis syscall
translation layer over verified FreeBSD-9.2-derived numbers, a genuine
ELF64 loader with relocations and TLS, the read-only PlayStation
filesystem, kernel synchronization objects, the process address space
with `mmap`/`mprotect`/`munmap`, and MSI-X-shaped interrupt delivery
through the guest's own IDT — plus the program that proves all of it:
a real ELF64 process that boots through the loader, calls thirteen
syscalls, mounts and reads a PFS, manages its own memory, drains an
event queue, and receives NVMe completions as interrupts.

**Bottom line:** an emulated x86-64 CPU now boots an ELF64 file —
segments mapped with per-page permissions, `R_X86_64_RELATIVE` and
`R_X86_64_TPOFF64` relocations applied, TLS reachable through `fs:`
addressing — and that process opens files from a read-only PFS, reads
them byte-exactly, walks directories through `getdirentries`, maps and
unmaps its own memory, registers on and drains a `kqueue`, then
installs an IDT, programs an MSI-X vector, doorbells an NVMe write —
and the completion arrives as a real interrupt through that IDT; the
handler runs, `iretq`s, and the driver reads the sector back and
byte-compares it. Then it prints `PHASE5 KERNEL OK` and calls
`exit(0)`. Every stage is verified by the integration suite; every
refusal carries a precise errno, never a guess.

---

## 1. What we accomplished

### 1.1 The syscall layer (`pdg-kernel/src/orbis.rs`, ~1,250 lines)

* **Verified numbering** — the implemented numbers
  (`exit` 1, `read` 3, `write` 4, `open` 5, `close` 6, `getpid` 20,
  `munmap` 73, `mprotect` 74, `getdirentries` 196, `kqueue` 362,
  `kevent` 363, `mmap` 477, `lseek` 478) were checked against the
  FreeBSD 9.2 `sys/kern/syscalls.master` — the PS4/PS5 kernel's
  documented FreeBSD base — and the Sony-defined block (588–672) is
  carried as a name table from the public psdevwiki-derived research,
  logged on refusal. Nothing is implemented from memory; every number
  cites its source in the code.
* **The raw ABI** — FreeBSD convention: number in RAX, arguments in
  RDI/RSI/RDX/R10/R8/R9, and the return contract the Orbis `libkernel`
  wrappers sit on: `CF = 0` + value in RAX on success, `CF = 1` +
  positive errno in RAX on failure. The demo driver's `jc`-after-
  `syscall` checks exercise exactly that contract.
* **File I/O over the PFS** — `open`/`close`/`read`/`lseek` resolve
  paths in the mounted read-only PFS through a handle-checked fd table
  (0–2 reserved as the honest headless stdio: EOF for stdin, console
  for stdout/stderr, `EBADF` on `close`). `lseek(fd, 0, SEEK_END)` is
  the documented size query — `fstat` is deliberately absent (see
  §4). `getdirentries` packs genuine FreeBSD `struct dirent` records
  (`d_fileno`, `d_reclen`, `d_type`, `d_namlen`) with whole-record
  resume across calls, and the demo byte-compares the full packed
  stream against a host-side encoding of the same directory.
* **Memory syscalls** — `mmap` (anonymous, `MAP_PRIVATE|MAP_ANON`;
  `MAP_FIXED` honored; `PROT_NONE` reserves fault-on-access regions),
  `mprotect` (rewrites live PTEs and flushes the software TLB at the
  syscall boundary), `munmap` (clears PTEs, flushes, refuses loader
  segments). File-backed and `MAP_SHARED` mappings are refused with
  `ENODEV`/`EINVAL` — the unified cache they need does not exist yet.
* **Event queues** — `kqueue` creates a queue in the kernel object
  table; `kevent` applies `EV_ADD` registrations (anything beyond
  `EV_ADD|EV_CLEAR` is `EINVAL`) and drains queued events whose
  `(ident, filter)` match a registration, non-blocking, with
  unregistered events re-queued rather than dropped. The session's
  host-side trigger is the kernel-event-source stand-in, and the demo
  drains it in a poll loop exactly the way a real waiter would.
* **The refusal posture** — unknown numbers and the entire Sony block
  return `ENOSYS` with the number (and name, when known) logged.
  `Fault::ProcessExit` — a new architectural escape — turns
  `exit(code)` into a clean session outcome instead of a fault.

### 1.2 The process address space (`pdg-kernel/src/vm.rs`, ~720 lines)

* `VmManager` — the process page-table owner: a PML4 in guest RAM, a
  bump frame allocator, tracked regions, `map_anon` (anonymous pages),
  `map_page` (explicit frames), `map_identity_2m` (the DMA window and
  MMIO mappings, 2 MiB leaves), `reserve` (`PROT_NONE`), `unmap`,
  `protect`, and a shared software MMU behind a mutex so syscall
  handlers can mutate tables mid-syscall while the interpreter holds
  the memory port.
* `ProcessMemory` — the interpreter's `MemoryAccess` port over the
  shared bus+MMU: the same contract `pdg-core`'s `GuestMemory`
  established for the boot path, now per-process. Uncontended on the
  single-core session; the lock is the seam SMP will need anyway.
* The **DMA window** insight: devices address guest RAM physically,
  but the process speaks virtual — so the session identity-maps a
  16 MiB window (va == pa, 2 MiB leaves) where the NVMe queues and
  PRP buffers live. Guest drivers get a coherent view; the device
  DMA path stays untouched.

### 1.3 The ELF64 / SELF loader (`pdg-kernel/src/loader.rs`, ~1,330 lines)

* **ELF64** — parses genuine images: header validation (class, data
  encoding, machine, type), program headers, `PT_DYNAMIC`,
  `DT_SYMTAB`/`DT_STRTAB`/`DT_RELA` with virtual-address-to-file-offset
  mapping through the containing `PT_LOAD`, overlap detection between
  load segments, and bounds checks that name the offending structure.
  Loading maps every `PT_LOAD` page-by-page with per-page permissions
  (PF_W/PF_X honored), zero-fills `memsz > filesz` tails, and applies
  `R_X86_64_RELATIVE` and `R_X86_64_TPOFF64` relocations plus
  `R_X86_64_64`/`GLOB_DAT`/`JUMP_SLOT` through an optional symbol
  resolver. Anything else is refused by number.
* **The supervisor write path** — segment data must land in RX pages,
  which the user mapping forbids; the loader writes through
  `write_va_loader` (translate-for-presence, write physical) — the
  `CR0.WP = 0` discipline real kernels load with. The demo proves it:
  its text executes after being copied into a page no user store can
  touch.
* **TLS, documented honestly** — `PT_TLS` gives the block; PDG sets
  `FS` to the block's base and `TPOFF64` addends are block-relative.
  That is PDG's Phase-5 model, stated as such — Sony's actual TCB
  layout needs a real dump and lands with the dynlib work. The demo
  reads a TLS qword through `fs:[tpoff]` and byte-compares it.
* **SELF** — the container is parsed per its publicly documented
  layout (magic `4F 15 3D 1D`, identification bytes `00 01 01 12`,
  key type, header/meta/file sizes, entry count, 32-byte entries
  `{flags, offset, file_size, mem_size}`), cross-checked between two
  independent public sources. **Plain entries only**: the public
  documentation *disagrees* on what the entry `flags` bits mean
  (encryption, block sizes, compression live there), so any nonzero
  value is refused with the exact number in the error. The embedded
  ELF is located, fully validated as ELF64, and loaded through the
  standard path. `ElfBuilder` produces test/demo images in the same
  shape, ABI congruence included.

### 1.4 PFS, the read-only PlayStation filesystem (`pdg-kernel/src/pfs.rs`, ~910 lines)

* Implemented from the public on-disk layout, cross-checked against
  two independent public readers (the `orbis-pfs` crate and shadPS4,
  both citing psdevwiki): the 0x380-byte superblock (version 1,
  format 20130315, mode flags, block size, block/inode counts,
  superroot), inode blocks from block 1, 100-byte inode headers with
  12 direct + 5 indirect pointers, 16-byte dirent headers
  (`ino`, `type`, `namelen`, `entsize`).
* Block maps: the contiguous form (`direct[1] == 0xFFFF_FFFF`), the
  12 direct pointers, and the first indirect block; deeper
  indirection is refused by name. **Signed** PFS (36-byte entries
  carrying per-block signatures) and **encrypted** PFS are refused
  with precise errors — verification needs a real dump, and PDG does
  not guess. Compressed files (`PFSC`) are not implemented either;
  all three refusals are logged gaps, not silent ones.
* `PfsBuilder` produces deterministic images (sorted paths, stable
  inode numbers) in the same layout — the demo PFS and the unit tests
  round-trip through it, so the reader is tested against the writer
  and both against the documented field offsets.

### 1.5 Kernel objects and interrupts

* **Objects (`objects.rs`, ~630 lines)** — event flags (AND/OR wait,
  CLEAR_ALL/CLEAR_PAT consumption — the publicly documented
  `sceKernelEventFlag` semantics), recursive owner-tracking mutexes
  (EPERM on foreign unlock, honest would-block on cross-thread
  contention in the single-guest-thread model), counting semaphores
  with EOVERFLOW, and kqueue-shaped event queues — all behind a
  handle-checked table that refuses unknown handles and wrong-kind
  handles by name. The syscall layer exposes the queues through
  `kqueue`/`kevent`; the flags/mutexes/semaphores are the substrate
  the Phase-6 NID layer (`sceKernel*`) will bind onto.
* **MSI-X-shaped interrupts (`msix.rs`, ~220 lines)** — a
  guest-programmable vector table in the public entry layout
  (`{message address, message data, vector control}` with the
  `0xFEE0_0000` x86 message-address convention and the low data byte
  as the IDT vector). `fire(entry)` latches the vector; the session
  pump hands it to the new `Interpreter::inject_interrupt` — which
  pushes the architectural frame through the guest IDT exactly like
  `#PF` delivery, no error code, IF cleared for interrupt gates —
  **only when the guest is ready** (STI ran, LIDT installed);
  otherwise the vector requeues. The PBA is not guest-visible and
  masked entries latch without delivering: documented gaps.
* **The NVMe seam** — `NvmeController::set_completion_hook`, invoked
  outside the device lock after a pump posts completions, is the
  interrupt source. The session wires it to entry 0 — the entry the
  guest programs — so the *guest's own configuration* decides the
  vector.

### 1.6 The demo program (`demo.rs`, ~900 lines) and the session (`kernel_session.rs`, ~300 lines)

* The Phase-4 discipline, upgraded: typed emitters build a real ELF64
  (RX text, RW data with status slots and a genuine IDT, a TLS
  segment, `PT_DYNAMIC` carrying the two relocations), and the test
  suite disassembles the entire text through iced-x86 asserting every
  instruction is in the documented set (`mov`, `syscall`, `lidt`,
  `sti`, `iretq`, `jcc`, `cmp`, `test`, `push`/`pop`, `xor`, `nop`).
  Fourteen stages store pass/fail qwords the host reads
  independently. The interrupt handler is a real ISRs-shaped stub:
  save, store, restore, `iretq`.
* The kernel session builds the process machine: 512 MiB bus,
  console, NVMe over the user's SSD image, the MSI-X table, the
  identity-mapped DMA window and MMIO windows, the stack, the
  loader, the mounted PFS, the Orbis kernel — then the same
  200-instruction-slice pump every phase has used, now with device
  servicing, event triggering and interrupt delivery between slices.
  `pdg-boot --demo-kernel <ssd.img>` runs it and prints the honest
  per-stage report.

### 1.7 Bugs found and fixed during closure

1. **The loader could not write RX segments** — copying text through
   the user page tables faulted on the write-permission check. The
   fix is the supervisor write path (`write_va_loader`: translate for
   presence, write physical), which is what real loaders do. Found by
   the loader unit test the moment it ran.
2. **`Pfs::mount` accepted short reads** — the superblock read
   ignored the byte count, so a truncated image mounted happily and
   failed much later. The mount now demands the full 0x380 bytes.
   Found by the truncated-image test refusing to fail.
3. **rel8 overflow in the stage builder** — the NVMe stage is ~700
   bytes; its fail-jumps could not reach the FAIL stub in 8-bit
   displacement. The emitter moved to `jcc rel32` throughout.
4. **The fd table handed out 0** — the first `open` returned fd 0,
   colliding with stdin. The table now reserves 0–2 at construction.
5. **Console byte order** — a reversed iteration in the write handler
   printed `!GDP` for `PDG!`. Caught by the byte-exact console test.
6. **The builder's file offsets broke ABI congruence** —
   `p_offset ≡ p_vaddr (mod page)` now holds for every segment; the
   one-page gaps between segments make it structural.
7. **`identical instructions` vs iced naming** — iced-x86 spells
   `jc` as `Jb`; the disassembler-check's allowed set learned it.
8. **Dirent expectation length** — the demo compared 64 bytes of
   packed dirents while the encoder produced 60; the expectation is
   padded (the syscall's write leaves the tail zero), and the length
   check pins the exact 60.

---

## 2. Key technical decisions and justifications

1. **Verified numbers, refused gaps** — the syscall table cites
   FreeBSD 9.2's public `syscalls.master` per number; the Sony block
   is carried as *names for diagnosis only* because the ABI shapes of
   those calls are not publicly pinned (their parameters live in
   `libkernel` wrappers, not the raw table). Implementing them from
   guesswork would produce a kernel that "works" against our own
   demos and nothing else — the exact failure mode this project
   refuses.
2. **`fstat` deliberately absent** — the FreeBSD 9.2 amd64
   `struct stat` layout is the next thing to verify against a real
   dump; `lseek(SEEK_END)` serves size queries honestly in the
   meantime. A wrong `st_mode` offset would corrupt real guests
   silently.
3. **The shared-MMU kernel** — syscall handlers run inside the
   interpreter's memory-port borrow, so the kernel cannot reach the
   address space through that reference. Ownership + interior
   mutability (Arc'd bus, mutexed MMU) lets `mprotect` flush the TLB
   at the exact instruction boundary the syscall executes at — the
   single-core equivalent of a TLB shootdown, and the seam SMP needs
   anyway.
4. **Identity-mapped DMA window** — the honest answer to "devices
   speak physical, processes speak virtual" without inventing an
   IOMMU: one window, identity-mapped, where drivers place queues
   and PRP buffers. Real console kernels carve similar windows.
5. **SELF: refuse the bits that are not public** — two public
   sources agree on the container layout and *disagree* on the entry
   `flags` bit meanings. PDG accepts `flags == 0` (plain) and
   refuses anything else with the exact value in the error — a
   message a future contributor can act on.
6. **The TLS model is PDG's own, stated as such** — FS = TLS block
   base, block-relative TPOFF64. Self-consistent, enforced at both
   ends, and explicitly documented as *not* claimed to be Sony's TCB
   until a real dump verifies it.
7. **`ProcessExit` as an architectural escape** — `exit` is not a
   fault; the interpreter now propagates it as a distinct `Fault`
   variant the session maps to an outcome, so clean exits stop
   masquerading as crashes.

---

## 3. Libraries used and why

| Crate | Role | Why |
|-------|------|-----|
| existing stack | everything | Phase 5 added **no** dependencies — the kernel layer is built from the bus, MMU, block-device, console and interpreter APIs that already existed |

That is the whole table, and it is the point: the phase reports have
tracked dependency rationale since Phase 1, and a kernel plane that
needs nothing new is evidence the earlier abstractions were drawn in
the right places.

---

## 4. Current state of emulation

What genuinely works, verified by tests (319 passing workspace-wide;
52 new in `pdg-kernel`, plus the kernel integration suite):

* A real ELF64 boots through the loader — segments mapped with
  enforced per-page permissions, `RELATIVE` and `TPOFF64` relocations
  applied (the demo's data slots hold garbage in the file; only the
  loader makes them correct), TLS readable through `fs:`.
* The process calls thirteen Orbis syscalls and every one behaves:
  console writes through `write(2)`, PFS file reads
  byte-compared against the host's view, directory streams packed in
  the FreeBSD dirent layout and byte-compared whole, anonymous
  memory mapped/protected/unmapped with the permission changes
  observable as faults, event queues registered, triggered by the
  host, and drained by the guest.
* The flagship: the guest installs an IDT (`lidt` through a genuine
  10-byte pseudo-descriptor), unmasks interrupts (`sti`), programs an
  MSI-X entry (address `0xFEE00000`, vector `0x40`), doorbells an
  NVMe write — the completion hook raises the vector, the pump
  delivers it through the guest's gate, the handler stores its
  marker and `iretq`s, and the driver verifies both the interrupt
  *and* the data before proceeding. Exit code 0, `PHASE5 KERNEL OK`
  on the console, and the integration suite asserts every detail
  down to the SSD's 512-byte read/write counters.
* The PFS round-trips: builder → image → mount → the same files,
  nested directories included; malformed superblocks, encrypted and
  signed modes, truncated images, EOF reads and wrong types are all
  refused with precise errors.
* `pdg-boot --demo-kernel` prints the honest per-stage report; the
  workspace is `clippy -D warnings` and `cargo fmt --check` clean.

What does **not** work yet (the honest list):

* **No dynlib/PRX layer** — `sys_dynlib_*` are refused with `ENOSYS`
  and logged by name; module loading, NID resolution and
  `sceKernel*` exports are Phase-6 work. The object layer
  (flags/mutexes/semaphores) exists and is tested, but nothing binds
  the NID names onto it yet.
* **`fstat` unimplemented** (see §2). **No `stat` family at all.**
* **Single-threaded processes** — no `thr_create`, no scheduler
  integration; mutex "contention" reports would-block rather than
  parking.
* **`kevent` is non-blocking** — no timed waits, no blocking drains;
  `EV_DELETE`/`EV_ENABLE` and friends are `EINVAL`.
* **TLS is the documented PDG model**, not a dump-verified Orbis TCB
  (§2); no dynamic TLS, no `DT_NEEDED` processing (the loader
  records them; nothing resolves them).
* **SELF is container-level only** — encrypted/signed/compressed
  segments refused; the per-entry host-address mapping a full SELF
  loader uses is simplified to whole-image parsing.
* **PFS gaps** — signed images, encrypted images and `PFSC`
  compressed files refused; second-level indirection refused; the
  filesystem is read-only by design.
* **Interrupt gaps** — no guest-visible PBA; masked vectors latch but
  do not deliver; one in-flight delivery at a time (the honest
  single-core shape).
* **`getdirentries` `basep`** reports entries-served, not a byte
  cookie; the demo passes NULL (documented in the syscall).
* **One process** — no fork/exec, no process table; the session is a
  single-process machine.
* **Carry-over from earlier phases** — GPU depth/stencil/blending,
  async NVMe batching, network, HRTF, adaptive-trigger wire layout
  (all tracked in the Phase-3/4 reports) remain open.

---

## 5. Known issues and risks

1. **The fd table is a Vec** — `EMFILE` never fires because the table
   grows; a real kernel caps it. Cosmetic today, worth fixing when
   guest code leaks fds meaningfully.
2. **`mprotect` cannot upgrade a `PROT_NONE` reservation** — there
   are no PTEs to rewrite; upgrading requires installing pages. Real
   thread-stack patterns (`mmap PROT_NONE` then `mprotect RW`) will
   hit this; the fix (map pages on upgrade) is small but was not
   needed by any verified path this phase.
3. **The kevent poll loop is guest-driven** — the session triggers
   after the first drain attempt; a real kernel would hook device
   sources into the queue directly. The seam
   (`trigger_equeue_event`) exists for Phase 6.
4. **Console output is character-lossy for non-UTF-8** — the device
   drops NUL bytes by design (Phase 1); binary output through
   `write(2)` would need a raw console path.
5. **The demo ELF is linked at fixed addresses** — ASLR-shaped
   `ET_DYN` loading exists in the loader (bias parameter, tested),
   but the demo is `ET_EXEC` and never exercises it.

---

## 6. Next phase plan (Phase 6 — Frontend)

1. **Dynlib/NID layer** — `sys_dynlib_get_list`/`get_info`/`dlsym`
   over a real module table; bind `sceKernel*` NIDs onto the object
   layer; `DT_NEEDED` resolution through the PFS.
2. **Launcher GUI** — the egui frontend: game grid, metadata,
   settings, BIOS picker, save states (replacing
   `HeadlessLauncher`).
3. **Threads** — `thr_create` over the SMP machine; blocking waits
   on the object layer.
4. **`fstat` + the stat family** — once the amd64 `struct stat`
   layout is verified against a dump.
5. **Release engineering** — packaging, crash reporting, the
   performance pass over the whole stack.

---

*Phase 5 complete: the machine boots programs, and those programs ask
it for things by name — files, memory, events, interrupts — and every
request is served or refused for a reason. Stopping per the phase
gate until "Continue to Phase 6".*
