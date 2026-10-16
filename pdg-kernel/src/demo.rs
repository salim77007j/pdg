//! The Phase-5 demo: a real ELF64 program the emulated CPU runs.
//!
//! `kernel_demo_elf()` assembles the executable with typed emitters — the
//! Phase-4 discipline: every instruction is one of the interpreter's
//! verified forms, and the integration suite disassembles the whole text
//! through iced-x86 to prove it. The program drives the entire kernel
//! plane from inside the guest:
//!
//! ```text
//! stage 0   write(1, ...)            — console through the syscall layer
//! stage 1   getpid                   — the HLE pid
//! stage 2   open("/app/data.txt")    — PFS lookup through file syscalls
//! stage 3   read                     — file contents into guest memory
//! stage 4   lseek(SEEK_END/SET)      — size query and rewind
//! stage 5   close
//! stage 6   getdirentries("/app")    — FreeBSD dirent stream over PFS
//! stage 7   mmap + write/read        — anonymous memory, page perms live
//! stage 8   mprotect R→RW→R          — permission changes observable
//! stage 9   munmap                   — the page really goes away
//! stage 10  kqueue + kevent         — event queue: register, host
//!                                     triggers, guest drains and checks
//! stage 11  R_X86_64_RELATIVE        — the loader's relocation really ran
//! stage 12  R_X86_64_TPOFF64 + fs:   — TLS addressing through FS
//! stage 13  LIDT + STI + MSI-X +     — the flagship: the guest installs an
//!           NVMe WRITE/READ            IDT, programs the MSI-X vector,
//!                                     doorbells an NVMe write; the
//!                                     completion raises the interrupt,
//!                                     the handler runs, `iretq`s, and
//!                                     the driver reads the sector back
//!                                     through PRP DMA
//! final     write "PHASE5 KERNEL OK" and exit(0)
//! ```
//!
//! The stage pattern is the Phase-4 one: every stage stores 1 or 0 into
//! its status qword; the host reads them after the run and asserts.

use crate::idt::IdtGate;
use crate::loader::rt;
use crate::loader::{ElfBuilder, ExportSym, Rela, SharedBuilder};
use crate::msix::MSI_ADDR_MAGIC;
use crate::orbis::{self, dirent};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// Guest-visible addresses of the demo program.
pub mod layout {
    /// Executable segment (RX).
    pub const TEXT_VA: u64 = 0x0040_0000;
    /// Data segment (RW).
    pub const DATA_VA: u64 = 0x0041_0000;
    /// TLS template segment (RW).
    pub const TLS_VA: u64 = 0x0042_0000;
    /// Top of the process stack (grows down from here).
    pub const STACK_TOP: u64 = 0x0070_0000;
    /// Stack mapping length.
    pub const STACK_LEN: u64 = 0x1_0000;

    /// The identity-mapped DMA window (virtual == physical) the NVMe
    /// queues and PRP buffers live in.
    pub const DMA_BASE: u64 = 0x1000_0000;
    /// DMA window length (16 MiB).
    pub const DMA_LEN: u64 = 0x0100_0000;
    /// Admin submission queue.
    pub const ASQ: u64 = DMA_BASE + 0x1_000;
    /// Admin completion queue.
    pub const ACQ: u64 = DMA_BASE + 0x2_000;
    /// I/O completion queue.
    pub const IO_CQ: u64 = DMA_BASE + 0x3_000;
    /// I/O submission queue.
    pub const IO_SQ: u64 = DMA_BASE + 0x4_000;
    /// IDENTIFY response buffer.
    pub const IDENT: u64 = DMA_BASE + 0x5_000;
    /// WRITE source / READ target A.
    pub const DATA_A: u64 = DMA_BASE + 0x6_000;
    /// READ landing buffer B.
    pub const DATA_B: u64 = DMA_BASE + 0x7_000;

    /// NVMe register window.
    pub const NVME: u64 = 0xE000_0000;
    /// MSI-X-shaped vector table window.
    pub const MSIX: u64 = 0xE300_0000;
    /// Boot console window.
    pub const CONSOLE: u64 = 0xF000_0000;

    /// The pid the session assigns this process.
    pub const DEMO_PID: u64 = 0x1337;
    /// The IDT vector the NVMe interrupt uses.
    pub const IRQ_VECTOR: u8 = 0x40;

    // ---- .data slots ----------------------------------------------------

    /// "/app/data.txt"
    pub const PATH_DATA: u64 = DATA_VA;
    /// "/app"
    pub const PATH_APP: u64 = DATA_VA + 0x100;
    /// "PHASE5 KERNEL OK\n"
    pub const MSG: u64 = DATA_VA + 0x200;
    /// read() landing buffer.
    pub const READ_BUF: u64 = DATA_VA + 0x300;
    /// getdirentries() landing buffer.
    pub const DENT_BUF: u64 = DATA_VA + 0x400;
    /// kevent changelist (one event).
    pub const KEV_CHANGE: u64 = DATA_VA + 0x600;
    /// kevent eventlist (one event).
    pub const KEV_OUT: u64 = DATA_VA + 0x640;
    /// Expected dirent stream prefix (host-computed).
    pub const EXPECT_DENTS: u64 = DATA_VA + 0x680;
    /// Stage status qwords ([`ST_COUNT`] of them).
    pub const STATUS: u64 = DATA_VA + 0x700;
    /// RELATIVE-relocated pointer slot.
    pub const RELPTR_SLOT: u64 = DATA_VA + 0x780;
    /// TPOFF64-relocated slot.
    pub const TPOFF_SLOT: u64 = DATA_VA + 0x788;
    /// qword the IRQ handler stores.
    pub const INTR_STATUS: u64 = DATA_VA + 0x790;
    /// Saved kqueue handle.
    pub const KQ_SLOT: u64 = DATA_VA + 0x798;
    /// Saved mmap address.
    pub const MAPADDR_SLOT: u64 = DATA_VA + 0x7A0;
    /// LIDT pseudo-descriptor {limit u16, base u64}.
    pub const IDTR_DESC: u64 = DATA_VA + 0x7B0;
    /// The interrupt descriptor table (gates at `vector * 16`).
    pub const IDT: u64 = DATA_VA + 0x1000;

    // ---- stage indices ---------------------------------------------------

    /// write(2) to the console.
    pub const ST_WRITE: u64 = 0;
    /// getpid.
    pub const ST_GETPID: u64 = 1;
    /// open.
    pub const ST_OPEN: u64 = 2;
    /// read + contents.
    pub const ST_READ: u64 = 3;
    /// lseek END/SET.
    pub const ST_SEEK: u64 = 4;
    /// close.
    pub const ST_CLOSE: u64 = 5;
    /// getdirentries.
    pub const ST_DIRENTS: u64 = 6;
    /// mmap + pattern roundtrip.
    pub const ST_MMAP: u64 = 7;
    /// mprotect cycles.
    pub const ST_MPROTECT: u64 = 8;
    /// munmap.
    pub const ST_MUNMAP: u64 = 9;
    /// kqueue/kevent.
    pub const ST_KEVENT: u64 = 10;
    /// RELATIVE relocation.
    pub const ST_RELOC_REL: u64 = 11;
    /// TPOFF64 relocation + fs access.
    pub const ST_RELOC_TLS: u64 = 12;
    /// NVMe over MSI-X interrupts.
    pub const ST_NVME_IRQ: u64 = 13;
    /// Number of stages.
    pub const ST_COUNT: u64 = 14;

    /// The file the demo reads (content below must stay in sync).
    pub const DATA_FILE: &str = "/app/data.txt";
    /// Content of the demo file.
    pub const FILE_BYTES: &[u8; 16] = b"Hello,PDG!123456";
    /// The console banner the program prints on success.
    pub const BANNER: &str = "PHASE5 KERNEL OK\n";
    /// Event identity the host triggers.
    pub const DEMO_IDENT: u64 = 0x600D;
    /// Event payload the host triggers.
    pub const DEMO_EVENT_DATA: u64 = 0x0A11_CE5E_ED00;
}

/// The 8-byte marker written to the SSD and read back ("PDGSSDIO" as a
/// little-endian qword — the Phase-4 marker, reused).
const MARKER_QWORD: u64 = u64::from_le_bytes(*b"PDGSSDIO");
/// The IDENTIFY serial prefix ("PDG-SSD-").
const SN_QWORD: u64 = u64::from_le_bytes(*b"PDG-SSD-");
/// The pattern the mmap stage round-trips.
const MMAP_PATTERN: u64 = 0xDEAD_BEEF_CAFE_1234;
/// The TLS template's second qword (read through `fs:[tpoff]`).
const TLS_QWORD: u64 = 0x0BAD_C0DE_5EED_FACE;

// ---------------------------------------------------------------------------
// Emitters (fixed encodings; the integration test re-verifies them
// through the disassembler)
// ---------------------------------------------------------------------------

/// The full documented emitter set; some forms are kept for future demo
/// stages (the Phase-4 payload used the same allowance).
#[allow(dead_code)]
mod em {
    /// `mov rax, imm64`
    pub fn rax_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x48, 0xB8]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov rcx, imm64`
    pub fn rcx_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x48, 0xB9]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov rdi, imm64`
    pub fn rdi_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x48, 0xBF]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov rsi, imm64`
    pub fn rsi_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x48, 0xBE]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov rdx, imm64`
    pub fn rdx_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x48, 0xBA]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov r10, imm64`
    pub fn r10_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x49, 0xBA]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov r8, imm64`
    pub fn r8_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x49, 0xB8]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov r9, imm64`
    pub fn r9_i64(c: &mut Vec<u8>, v: u64) {
        c.extend_from_slice(&[0x49, 0xB9]);
        c.extend_from_slice(&v.to_le_bytes());
    }
    /// `mov rax, qword [abs]` (moffs load)
    pub fn rax_abs(c: &mut Vec<u8>, addr: u64) {
        c.extend_from_slice(&[0x48, 0xA1]);
        c.extend_from_slice(&addr.to_le_bytes());
    }
    /// `mov qword [abs], rax` (moffs store)
    pub fn abs_rax(c: &mut Vec<u8>, addr: u64) {
        c.extend_from_slice(&[0x48, 0xA3]);
        c.extend_from_slice(&addr.to_le_bytes());
    }
    /// `mov rdi, rax`
    pub fn rdi_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x89, 0xC7]);
    }
    /// `mov qword [rax], rcx`
    pub fn mem_rax_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x89, 0x08]);
    }
    /// `mov rbx, qword [rax]`
    pub fn rbx_mem_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x8B, 0x18]);
    }
    /// `mov rbx, qword fs:[rax]`
    pub fn rbx_fs_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x64, 0x48, 0x8B, 0x18]);
    }
    /// `cmp rax, rcx`
    pub fn cmp_rax_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x39, 0xC8]);
    }
    /// `cmp rbx, rcx`
    pub fn cmp_rbx_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x39, 0xCB]);
    }
    /// `cmp al, imm8`
    pub fn cmp_al(c: &mut Vec<u8>, v: u8) {
        c.extend_from_slice(&[0x3C, v]);
    }
    /// `test al, al`
    pub fn test_al(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x84, 0xC0]);
    }
    /// `test rax, rax`
    pub fn test_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x85, 0xC0]);
    }
    /// `xor eax, eax`
    pub fn xor_eax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x31, 0xC0]);
    }
    /// `push rax` / `pop rax`
    pub fn push_rax(c: &mut Vec<u8>) {
        c.push(0x50);
    }
    pub fn pop_rax(c: &mut Vec<u8>) {
        c.push(0x58);
    }
    /// `jc rel8` / `je rel8` / `jne rel8` / `jl rel8`
    pub fn jc(c: &mut Vec<u8>, d: i8) {
        c.extend_from_slice(&[0x72, d as u8]);
    }
    pub fn je(c: &mut Vec<u8>, d: i8) {
        c.extend_from_slice(&[0x74, d as u8]);
    }
    pub fn jne(c: &mut Vec<u8>, d: i8) {
        c.extend_from_slice(&[0x75, d as u8]);
    }
    pub fn jl(c: &mut Vec<u8>, d: i8) {
        c.extend_from_slice(&[0x7C, d as u8]);
    }
    /// `jmp rel8`
    pub fn jmp(c: &mut Vec<u8>, d: i8) {
        c.extend_from_slice(&[0xEB, d as u8]);
    }
    /// `syscall`
    pub fn syscall(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x0F, 0x05]);
    }
    /// `lidt [rax]`
    pub fn lidt_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x0F, 0x01, 0x18]);
    }
    /// `sti`
    pub fn sti(c: &mut Vec<u8>) {
        c.push(0xFB);
    }
    /// `iretq`
    pub fn iretq(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0xCF]);
    }
    /// `mov rcx, qword fs:[rax]` (Phase 6: cross-module TLS reads)
    pub fn rcx_fs_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x64, 0x48, 0x8B, 0x08]);
    }
    /// `mov qword fs:[rax], rcx` (Phase 6: cross-module TLS writes)
    pub fn fs_rax_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x64, 0x48, 0x89, 0x08]);
    }
    /// `mov rax, rcx`
    pub fn rax_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x89, 0xC8]);
    }
    /// `mov rbx, rcx`
    pub fn rbx_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x89, 0xCB]);
    }
    /// `add rcx, rdi`
    pub fn add_rcx_rdi(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x01, 0xF9]);
    }
    /// `inc rax`
    pub fn inc_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0xFF, 0xC0]);
    }
    /// `ret`
    pub fn ret(c: &mut Vec<u8>) {
        c.push(0xC3);
    }
    /// `call rax` (Phase 6: indirect calls through dlsym results and
    /// relocated GOT slots)
    pub fn call_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0xFF, 0xD0]);
    }
    /// `lea rax, [rip + disp32]` — the position-independent way shared
    /// modules find their own data (the displacement is computed at
    /// build time from link-time addresses, so any load bias works).
    pub fn lea_rax_rip(c: &mut Vec<u8>, target: u64, end: u64) {
        let disp = i32::try_from(target as i64 - end as i64).expect("rip-relative fits i32");
        c.extend_from_slice(&[0x48, 0x8D, 0x05]);
        c.extend_from_slice(&disp.to_le_bytes());
    }
    /// `mov rcx, [rax]`
    pub fn rcx_mem_rax(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x8B, 0x08]);
    }
    /// `mov rbx, fs:[rcx]`
    pub fn rbx_fs_rcx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x64, 0x48, 0x8B, 0x19]);
    }
    /// `mov fs:[rcx], rbx`
    pub fn fs_rcx_rbx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x64, 0x48, 0x89, 0x19]);
    }
    /// `mov rax, rbx`
    pub fn rax_rbx(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x89, 0xD8]);
    }
    /// `add rbx, rdi`
    pub fn add_rbx_rdi(c: &mut Vec<u8>) {
        c.extend_from_slice(&[0x48, 0x01, 0xFB]);
    }
}

/// Emit one syscall (number + up to six arguments, RAX + RDI/RSI/RDX/
/// R10/R8/R9 — the FreeBSD convention).
fn sys(c: &mut Vec<u8>, number: u64, args: &[u64]) {
    em::rax_i64(c, number);
    for (i, a) in args.iter().enumerate() {
        match i {
            0 => em::rdi_i64(c, *a),
            1 => em::rsi_i64(c, *a),
            2 => em::rdx_i64(c, *a),
            3 => em::r10_i64(c, *a),
            4 => em::r8_i64(c, *a),
            _ => em::r9_i64(c, *a),
        }
    }
    em::syscall(c);
}

/// A stage under construction: conditional checks are emitted as jumps to
/// the shared FAIL stub, patched when [`Stage::finish`] writes the pass /
/// fail tail.
struct Stage {
    code: Vec<u8>,
    /// Byte positions of the rel8 placeholders to patch to FAIL.
    fails: Vec<usize>,
}

impl Stage {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            fails: Vec::new(),
        }
    }

    /// Emit a six-byte conditional jump to FAIL (the rel32 form — stages
    /// like the NVMe driver are far larger than rel8 reaches). `op` is the
    /// short-jump opcode byte; the rel32 encoding is `0F 8x`.
    fn jcc_fail(&mut self, op: u8) {
        self.code.push(0x0F);
        self.code.push(0x80 | (op & 0x0F));
        self.code.extend_from_slice(&[0, 0, 0, 0]);
        self.fails.push(self.code.len() - 4);
    }

    /// `jc` (the error convention: CF set).
    fn fail_on_carry(&mut self) {
        self.jcc_fail(0x72);
    }

    /// Compare RAX against `v`; fail when unequal.
    fn rax_eq(&mut self, v: u64) {
        em::rcx_i64(&mut self.code, v);
        em::cmp_rax_rcx(&mut self.code);
        self.jcc_fail(0x75); // jne
    }

    /// Fail when RAX is zero (a null address / handle).
    fn rax_nonzero(&mut self) {
        em::rcx_i64(&mut self.code, 0);
        em::cmp_rax_rcx(&mut self.code);
        self.jcc_fail(0x74); // je
    }

    /// Compare RAX against `v`; fail when strictly less.
    fn rax_ge(&mut self, v: u64) {
        em::rcx_i64(&mut self.code, v);
        em::cmp_rax_rcx(&mut self.code);
        self.jcc_fail(0x7C); // jl
    }

    /// Compare the qword at `addr` against `v`; fail when unequal.
    fn mem_eq(&mut self, addr: u64, v: u64) {
        em::rax_abs(&mut self.code, addr);
        em::rcx_i64(&mut self.code, v);
        em::cmp_rax_rcx(&mut self.code);
        self.jcc_fail(0x75);
    }

    /// Write the PASS / FAIL tail storing 1 / 0 into `status` and patch
    /// every recorded jump.
    ///
    /// (The fail block starts at the `xor eax, eax`: a Phase-6 stage that
    /// failed exposed a latent Phase-5 bug here — the jump targets used
    /// to skip the xor, storing the live RAX instead of 0. It never fired
    /// before because every Phase-5 stage passed.)
    fn finish(mut self, status: u64) -> Vec<u8> {
        // PASS: mov rax, 1; mov [abs], rax; jmp END
        em::rax_i64(&mut self.code, 1);
        em::abs_rax(&mut self.code, status);
        em::jmp(&mut self.code, 12);
        // FAIL: xor eax, eax; mov [abs], rax   (12 bytes)
        let fail_start = self.code.len();
        em::xor_eax(&mut self.code);
        em::abs_rax(&mut self.code, status);
        let fail_end = self.code.len();
        debug_assert_eq!(fail_end, fail_start + 12); // xor(2) + store(10)
        for pos in &self.fails {
            let rel = (fail_start as i64) - (*pos as i64 + 4);
            let rel = i32::try_from(rel).expect("stage fail jump fits rel32");
            self.code[*pos..*pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
        self.code
    }
}

/// Poll the byte at `addr` until it equals `v` (backward loop; 14 bytes).
fn poll_eq(c: &mut Vec<u8>, addr: u64, v: u8) {
    em::rax_abs(c, addr);
    em::cmp_al(c, v);
    em::jne(c, -14);
}

/// Poll the bit `bit` of the byte at `addr` until set (14 bytes).
fn poll_bit(c: &mut Vec<u8>, addr: u64, bit: u8) {
    em::rax_abs(c, addr);
    c.extend_from_slice(&[0xA8, bit]); // test al, imm8
    em::je(c, -14);
}

/// `mov rax, imm64; mov qword [abs], rax`.
fn store_imm64(c: &mut Vec<u8>, addr: u64, v: u64) {
    em::rax_i64(c, v);
    em::abs_rax(c, addr);
}

// ---------------------------------------------------------------------------
// The program
// ---------------------------------------------------------------------------

/// Build the demo executable (a genuine ELF64 image).
pub fn kernel_demo_elf() -> Vec<u8> {
    use layout as l;
    use orbis::nr;

    let mut text: Vec<u8> = Vec::with_capacity(0x1000);

    // ---- the interrupt handler, pinned at TEXT_VA + 0 ------------------
    // push rax; mov rax, 1; mov [INTR_STATUS], rax; pop rax; iretq
    em::push_rax(&mut text);
    em::rax_i64(&mut text, 1);
    em::abs_rax(&mut text, l::INTR_STATUS);
    em::pop_rax(&mut text);
    em::iretq(&mut text);
    debug_assert_eq!(text.len(), 24);
    let handler_va = l::TEXT_VA; // the gate points here

    // ---- main entry, pinned at TEXT_VA + 0x100 ---------------------------
    while text.len() < 0x100 {
        text.push(0x90); // nop padding (verified form)
    }

    // stage 0: write(1, MSG, 4) == 4 — a console probe; the full banner
    // prints at the very end (the survived-everything signal).
    let mut s = Stage::new();
    sys(&mut s.code, nr::WRITE, &[1, l::MSG, 4]);
    s.fail_on_carry();
    s.rax_eq(4);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_WRITE));

    // stage 1: getpid == DEMO_PID -----------------------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::GETPID, &[]);
    s.fail_on_carry();
    s.rax_eq(l::DEMO_PID);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_GETPID));

    // stage 2: open("/app/data.txt", O_RDONLY) >= 3 ------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::OPEN, &[l::PATH_DATA, 0]);
    s.fail_on_carry();
    s.rax_ge(3);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_OPEN));

    // stage 3: read(fd, READ_BUF, 16) == 16, contents match ----------------
    let mut s = Stage::new();
    sys(
        &mut s.code,
        nr::READ,
        &[3, l::READ_BUF, l::FILE_BYTES.len() as u64],
    );
    s.fail_on_carry();
    s.rax_eq(l::FILE_BYTES.len() as u64);
    let head = u64::from_le_bytes(l::FILE_BYTES[..8].try_into().unwrap());
    s.mem_eq(l::READ_BUF, head);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_READ));

    // stage 4: lseek(fd, 0, SEEK_END) == 16; lseek(fd, 0, SEEK_SET) == 0 ---
    let mut s = Stage::new();
    sys(&mut s.code, nr::LSEEK, &[3, 0, 2]);
    s.fail_on_carry();
    s.rax_eq(l::FILE_BYTES.len() as u64);
    sys(&mut s.code, nr::LSEEK, &[3, 0, 0]);
    s.fail_on_carry();
    s.rax_eq(0);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_SEEK));

    // stage 5: close(fd) == 0 ----------------------------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::CLOSE, &[3]);
    s.fail_on_carry();
    s.rax_eq(0);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_CLOSE));

    // stage 6: open("/app"); getdirentries(fd, DENT_BUF, 512, 0) ----------
    let mut s = Stage::new();
    sys(&mut s.code, nr::OPEN, &[l::PATH_APP, 0]);
    s.fail_on_carry();
    s.rax_ge(3);
    // fd: first free slot after close(3) — read it from the open result:
    // (the slot is 3 again; keep it simple and deterministic)
    let expected_len = expected_dirents().len();
    let mut expected = expected_dirents();
    expected.resize(64, 0);
    sys(&mut s.code, nr::GETDIRENTRIES, &[3, l::DENT_BUF, 512, 0]);
    s.fail_on_carry();
    s.rax_eq(expected_len as u64);
    for i in 0..8u64 {
        let off = i * 8;
        let v = u64::from_le_bytes(
            expected[off as usize..(off + 8) as usize]
                .try_into()
                .unwrap(),
        );
        s.mem_eq(l::DENT_BUF + off, v);
    }
    sys(&mut s.code, nr::CLOSE, &[3]);
    s.fail_on_carry();
    s.rax_eq(0);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_DIRENTS));

    // stage 7: mmap anonymous RW, write pattern, read back ----------------
    let mut s = Stage::new();
    sys(
        &mut s.code,
        nr::MMAP,
        &[
            0,
            0x1000,
            (orbis::prot::PROT_READ | orbis::prot::PROT_WRITE) as u64,
            orbis::mmap_flags::MAP_PRIVATE | orbis::mmap_flags::MAP_ANON,
        ],
    );
    s.fail_on_carry();
    s.rax_nonzero();
    // Save the address, then round-trip a pattern through it.
    em::abs_rax(&mut s.code, l::MAPADDR_SLOT);
    em::rcx_i64(&mut s.code, MMAP_PATTERN);
    em::mem_rax_rcx(&mut s.code); // [rax] = pattern
    em::rbx_mem_rax(&mut s.code); // rbx = [rax]
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_MMAP));

    // stage 8: mprotect(addr, 0x1000, R) == 0; back to RW == 0 -------------
    let mut s = Stage::new();
    em::rax_abs(&mut s.code, l::MAPADDR_SLOT);
    em::rdi_rax(&mut s.code);
    em::rax_i64(&mut s.code, nr::MPROTECT);
    em::rsi_i64(&mut s.code, 0x1000);
    em::rdx_i64(&mut s.code, orbis::prot::PROT_READ as u64);
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    em::rax_abs(&mut s.code, l::MAPADDR_SLOT);
    em::rdi_rax(&mut s.code);
    em::rax_i64(&mut s.code, nr::MPROTECT);
    em::rsi_i64(&mut s.code, 0x1000);
    em::rdx_i64(
        &mut s.code,
        (orbis::prot::PROT_READ | orbis::prot::PROT_WRITE) as u64,
    );
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_MPROTECT));

    // stage 9: munmap(addr, 0x1000) == 0 ------------------------------------
    let mut s = Stage::new();
    em::rax_abs(&mut s.code, l::MAPADDR_SLOT);
    em::rdi_rax(&mut s.code);
    em::rax_i64(&mut s.code, nr::MUNMAP);
    em::rsi_i64(&mut s.code, 0x1000);
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_MUNMAP));

    // stage 10: kqueue; kevent register; drain loop; check the event -------
    let mut s = Stage::new();
    sys(&mut s.code, nr::KQUEUE, &[]);
    s.fail_on_carry();
    s.rax_nonzero();
    em::abs_rax(&mut s.code, l::KQ_SLOT);
    // register: kevent(kq, KEV_CHANGE, 1, 0, 0, 0)
    em::rax_abs(&mut s.code, l::KQ_SLOT);
    em::rdi_rax(&mut s.code);
    em::rax_i64(&mut s.code, nr::KEVENT);
    em::rsi_i64(&mut s.code, l::KEV_CHANGE);
    em::rdx_i64(&mut s.code, 1);
    em::r10_i64(&mut s.code, 0);
    em::r8_i64(&mut s.code, 0);
    em::r9_i64(&mut s.code, 0);
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    // drain loop: kevent(kq, 0, 0, KEV_OUT, 1, 0) until it returns 1
    let loop_start = s.code.len();
    em::rax_abs(&mut s.code, l::KQ_SLOT);
    em::rdi_rax(&mut s.code);
    em::rax_i64(&mut s.code, nr::KEVENT);
    em::rsi_i64(&mut s.code, 0);
    em::rdx_i64(&mut s.code, 0);
    em::r10_i64(&mut s.code, l::KEV_OUT);
    em::r8_i64(&mut s.code, 1);
    em::r9_i64(&mut s.code, 0);
    em::syscall(&mut s.code);
    em::test_rax(&mut s.code);
    let body = s.code.len() - loop_start;
    em::je(&mut s.code, -((body + 2) as i8)); // retry until an event lands
    s.fail_on_carry(); // a failed drain is an error
    // the event identity and payload
    s.mem_eq(l::KEV_OUT, l::DEMO_IDENT);
    s.mem_eq(l::KEV_OUT + 16, l::DEMO_EVENT_DATA);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_KEVENT));

    // stage 11: the RELATIVE relocation wrote the message address ---------
    let mut s = Stage::new();
    s.mem_eq(l::RELPTR_SLOT, l::MSG);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_RELOC_REL));

    // stage 12: TPOFF64 + fs: addressing reads the TLS template -----------
    let mut s = Stage::new();
    em::rax_abs(&mut s.code, l::TPOFF_SLOT);
    em::rbx_fs_rax(&mut s.code); // rbx = fs:[tpoff]
    em::rcx_i64(&mut s.code, TLS_QWORD);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_RELOC_TLS));

    // stage 13: the flagship — IDT + MSI-X + NVMe + interrupt --------------
    let mut s = Stage::new();
    {
        // Emit into a local buffer; the two qword checks record their fail
        // jumps manually (the Stage borrow checker cannot alias `s.code`
        // across them).
        let mut c: Vec<u8> = Vec::new();
        let mut fails: Vec<usize> = Vec::new();
        let check = |c: &mut Vec<u8>, fails: &mut Vec<usize>, addr: u64, v: u64| {
            em::rax_abs(c, addr);
            em::rcx_i64(c, v);
            em::cmp_rax_rcx(c);
            c.extend_from_slice(&[0x0F, 0x85, 0, 0, 0, 0]); // jne rel32
            fails.push(c.len() - 4);
        };
        // Install the IDT and unmask interrupts.
        em::rax_i64(&mut c, l::IDTR_DESC);
        em::lidt_rax(&mut c);
        em::sti(&mut c);
        // Program MSI-X entry 0: address = the APIC magic, data = vector.
        store_imm64(&mut c, l::MSIX, u64::from(MSI_ADDR_MAGIC));
        store_imm64(&mut c, l::MSIX + 8, u64::from(l::IRQ_VECTOR));
        // Enable the controller (the Phase-4 register sequence).
        store_imm64(&mut c, l::NVME + nv::REG_AQA, (7 << 24) | (7 << 16));
        store_imm64(&mut c, l::NVME + nv::REG_ASQ, l::ASQ);
        store_imm64(&mut c, l::NVME + nv::REG_ACQ, l::ACQ);
        store_imm64(&mut c, l::NVME + nv::REG_CC, 1);
        poll_bit(&mut c, l::NVME + nv::REG_CSTS, 1);
        // IDENTIFY (admin slot 0) and check the DMA'd serial.
        store_imm64(&mut c, l::ASQ, 0x0000_0000_0000_0006);
        store_imm64(&mut c, l::ASQ + 16, l::IDENT);
        store_imm64(&mut c, l::ASQ + 40, 0);
        store_imm64(&mut c, l::NVME + nv::DB_SQ0, 1);
        poll_bit(&mut c, l::ACQ + 12, 1);
        check(&mut c, &mut fails, l::IDENT + 4, SN_QWORD);
        // Create the I/O queue pair (admin slots 1 and 2).
        store_imm64(&mut c, l::ASQ + 64, 0x0000_0000_0000_0005);
        store_imm64(&mut c, l::ASQ + 64 + 16, l::IO_CQ);
        store_imm64(&mut c, l::ASQ + 64 + 40, 0x0001_0003);
        store_imm64(&mut c, l::ASQ + 64 + 44, 1);
        store_imm64(&mut c, l::NVME + nv::DB_SQ0, 2);
        poll_bit(&mut c, l::ACQ + 16 + 12, 1);
        store_imm64(&mut c, l::ASQ + 128, 0x0000_0000_0000_0001);
        store_imm64(&mut c, l::ASQ + 128 + 16, l::IO_SQ);
        store_imm64(&mut c, l::ASQ + 128 + 40, 0x0001_0003);
        store_imm64(&mut c, l::ASQ + 128 + 44, 0x0001_0001);
        store_imm64(&mut c, l::NVME + nv::DB_SQ0, 3);
        poll_bit(&mut c, l::ACQ + 32 + 12, 1);
        // WRITE the marker through PRP DMA (I/O SQ slot 0).
        store_imm64(&mut c, l::DATA_A, MARKER_QWORD);
        store_imm64(&mut c, l::IO_SQ, 0x0000_0001_0000_0001);
        store_imm64(&mut c, l::IO_SQ + 16, l::DATA_A);
        store_imm64(&mut c, l::IO_SQ + 40, 4);
        store_imm64(&mut c, l::IO_SQ + 48, 0);
        store_imm64(&mut c, l::NVME + nv::DB_SQ1, 1);
        // The interrupt lands here: poll the handler's qword first —
        // proving delivery — then the completion phase tag.
        poll_eq(&mut c, l::INTR_STATUS, 1);
        poll_bit(&mut c, l::IO_CQ + 12, 1);
        // READ it back (I/O SQ slot 1) and byte-compare the marker.
        store_imm64(&mut c, l::IO_SQ + 64, 0x0000_0001_0000_0002);
        store_imm64(&mut c, l::IO_SQ + 64 + 16, l::DATA_B);
        store_imm64(&mut c, l::IO_SQ + 64 + 40, 4);
        store_imm64(&mut c, l::IO_SQ + 64 + 48, 0);
        store_imm64(&mut c, l::NVME + nv::DB_SQ1, 2);
        poll_bit(&mut c, l::IO_CQ + 16 + 12, 1);
        check(&mut c, &mut fails, l::DATA_B, MARKER_QWORD);
        s.code.extend(c);
        s.fails.extend(fails);
    }
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_NVME_IRQ));

    // ---- final: banner + exit(0) ------------------------------------------
    let mut c = Vec::new();
    sys(&mut c, nr::WRITE, &[1, l::MSG, l::BANNER.len() as u64]);
    sys(&mut c, nr::EXIT, &[0]);
    text.extend_from_slice(&c);

    // ---- data --------------------------------------------------------------
    let mut data = vec![0u8; 0x1420];
    let put = |data: &mut Vec<u8>, off: usize, bytes: &[u8]| {
        data[off..off + bytes.len()].copy_from_slice(bytes);
    };
    let d = |va: u64| (va - l::DATA_VA) as usize;
    put(&mut data, d(l::PATH_DATA), b"/app/data.txt\0");
    put(&mut data, d(l::PATH_APP), b"/app\0");
    put(&mut data, d(l::MSG), l::BANNER.as_bytes());
    // kevent changelist: {ident, filter = EVFILT_READ (-1), flags = EV_ADD}.
    data[d(l::KEV_CHANGE)..d(l::KEV_CHANGE) + 8].copy_from_slice(&l::DEMO_IDENT.to_le_bytes());
    data[d(l::KEV_CHANGE) + 8..d(l::KEV_CHANGE) + 10]
        .copy_from_slice(&orbis::ev::EVFILT_READ.to_le_bytes());
    data[d(l::KEV_CHANGE) + 10..d(l::KEV_CHANGE) + 12]
        .copy_from_slice(&orbis::ev::EV_ADD.to_le_bytes());
    // expected dirent stream, padded to the compared 64 bytes
    let mut expect = expected_dirents();
    expect.resize(64, 0);
    put(&mut data, d(l::EXPECT_DENTS), &expect);
    // the relocation slots start as garbage: only the loader makes them right
    put(
        &mut data,
        d(l::RELPTR_SLOT),
        &0xEE_EE_EE_EE_EE_EE_EE_EEu64.to_le_bytes(),
    );
    put(
        &mut data,
        d(l::TPOFF_SLOT),
        &0xEE_EE_EE_EE_EE_EE_EE_EEu64.to_le_bytes(),
    );
    // LIDT pseudo-descriptor: {limit = 0x40*16+15, base = IDT}
    data[d(l::IDTR_DESC)..d(l::IDTR_DESC) + 2]
        .copy_from_slice(&((l::IRQ_VECTOR as u16) * 16 + 15).to_le_bytes());
    data[d(l::IDTR_DESC) + 2..d(l::IDTR_DESC) + 10].copy_from_slice(&l::IDT.to_le_bytes());
    // IDT gate for the NVMe vector.
    let gate = IdtGate::interrupt(handler_va).to_bytes();
    put(&mut data, d(l::IDT) + l::IRQ_VECTOR as usize * 16, &gate);

    // ---- assemble -----------------------------------------------------------
    let mut b = ElfBuilder::new(l::TEXT_VA, l::DATA_VA, l::TEXT_VA + 0x100);
    b.text = text;
    b.data = data;
    b.tls = Some((
        l::TLS_VA,
        [vec![0x5A; 8], TLS_QWORD.to_le_bytes().to_vec()].concat(),
    ));
    b.relas = vec![
        Rela {
            r_offset: l::RELPTR_SLOT,
            r_type: rt::X86_64_RELATIVE,
            r_sym: 0,
            r_addend: l::MSG as i64,
        },
        Rela {
            r_offset: l::TPOFF_SLOT,
            r_type: rt::X86_64_TPOFF64,
            r_sym: 0,
            r_addend: 8,
        },
    ];
    b.build()
}

// ---------------------------------------------------------------------------
// The Phase-6 runtime demo: modules, threads, TLS, umtx, stat
// ---------------------------------------------------------------------------

/// Guest-visible addresses of the Phase-6 runtime demo (separate from the
/// Phase-5 layout so both programs stay loadable side by side).
pub mod layout6 {
    /// Executable segment (RX) — the main image is `ET_DYN`, loaded at
    /// bias 0 through the dynlib layer.
    pub const TEXT_VA: u64 = 0x0040_0000;
    /// Data segment (RW).
    pub const DATA_VA: u64 = 0x0041_0000;
    /// The dynlib TLS arena base (aggregated template; the main thread's
    /// FS points here).
    pub const TLS_ARENA: u64 = 0x0042_0000;
    /// Top of the main thread's stack.
    pub const STACK_TOP: u64 = 0x0070_0000;
    /// Main stack mapping length.
    pub const STACK_LEN: u64 = 0x1_0000;
    /// Top of the worker thread's stack (mapped just below the main one).
    pub const WORKER_STACK_TOP: u64 = 0x006F_0000;
    /// Worker stack mapping length.
    pub const WORKER_STACK_LEN: u64 = 0x1_0000;
    /// Worker entry, pinned inside the main text.
    pub const WORKER_ENTRY: u64 = TEXT_VA + 0x1000;

    // --- data slots (offsets from DATA_VA) ---
    /// "/app0/data.txt\\0"
    pub const PATH_DATA: u64 = DATA_VA;
    /// "/app0\\0"
    pub const PATH_APP: u64 = DATA_VA + 0x100;
    /// "/app0/sce_module/libpdgx.prx\\0"
    pub const PATH_PRX: u64 = DATA_VA + 0x180;
    /// The success banner.
    pub const MSG: u64 = DATA_VA + 0x200;
    /// `struct stat` buffer (120 bytes).
    pub const STAT_BUF: u64 = DATA_VA + 0x300;
    /// Module-handle list (4 × u32) and the count.
    pub const LIST: u64 = DATA_VA + 0x400;
    /// The `get_list` count out.
    pub const LIST_COUNT: u64 = DATA_VA + 0x420;
    /// `OrbisKernelModuleInfo` buffer (0x160).
    pub const INFO: u64 = DATA_VA + 0x480;
    /// "pdgx_add\\0"
    pub const NAME_ADD: u64 = DATA_VA + 0x600;
    /// "pdgx_version\\0"
    pub const NAME_VER: u64 = DATA_VA + 0x640;
    /// "pdgy_value\\0"
    pub const NAME_PDY: u64 = DATA_VA + 0x680;
    /// dlsym result slot.
    pub const DLSYM_OUT: u64 = DATA_VA + 0x6C0;
    /// Stage statuses (16 qwords).
    pub const STATUS: u64 = DATA_VA + 0x700;
    /// JUMP_SLOT slot for `pdgx_add`.
    pub const GOT_ADD: u64 = DATA_VA + 0x780;
    /// GLOB_DAT slot for `pdgx_version`.
    pub const GOT_VER: u64 = DATA_VA + 0x788;
    /// TPOFF64 slot for the main module's TLS variable.
    pub const MAIN_TLS_PTR: u64 = DATA_VA + 0x790;
    /// The umtx-protected counter.
    pub const SHARED: u64 = DATA_VA + 0x798;
    /// The worker's "I am at the lock" flag (a umtx WAIT word).
    pub const TRYING: u64 = DATA_VA + 0x7A0;
    /// The worker's completion flag (join channel).
    pub const DONE: u64 = DATA_VA + 0x7A8;
    /// `struct umutex` (m_owner u32 at +0).
    pub const M: u64 = DATA_VA + 0x7B0;
    /// thr_create's tid out.
    pub const TID_OUT: u64 = DATA_VA + 0x7B8;
    /// The worker's `ucontext_t` (rip@0xB0, rsp@0xC8, rdi@0x18).
    pub const UCTX: u64 = DATA_VA + 0x7C0;
    /// Where the worker stores its observed (old) TLS value.
    pub const WORKER_TLS_OLD: u64 = DATA_VA + 0x8C0;
    /// The worker's thr_exit state pointer (also its join address).
    pub const STATE_PTR: u64 = DATA_VA + 0x8D0;

    // --- stages ---
    /// Console probe.
    pub const ST_WRITE: u64 = 0;
    /// getpid.
    pub const ST_GETPID: u64 = 1;
    /// dynlib_get_list.
    pub const ST_DYNLIST: u64 = 2;
    /// dynlib_get_info.
    pub const ST_DYNINFO: u64 = 3;
    /// dlsym + indirect call.
    pub const ST_DLSYM: u64 = 4;
    /// JUMP_SLOT call through the GOT.
    pub const ST_LINKCALL: u64 = 5;
    /// GLOB_DAT data + depth-2 dlsym.
    pub const ST_GLOBDAT: u64 = 6;
    /// fstat.
    pub const ST_FSTAT: u64 = 7;
    /// stat.
    pub const ST_STAT: u64 = 8;
    /// thr_self.
    pub const ST_THRSELF: u64 = 9;
    /// main TLS read through fs:.
    pub const ST_TLSMAIN: u64 = 10;
    /// thr_create (main side).
    pub const ST_WORKER_CREATE: u64 = 11;
    /// thr_self == 2 (worker side).
    pub const ST_WORKER_ID: u64 = 12;
    /// worker TLS bump (worker side).
    pub const ST_WORKER_TLS: u64 = 13;
    /// umtx choreography completes (main side).
    pub const ST_UMTX: u64 = 14;
    /// main TLS unchanged after the worker ran.
    pub const ST_TLSMAIN_AFTER: u64 = 15;
    /// Number of stages.
    pub const ST_COUNT: u64 = 16;

    /// The demo pid.
    pub const DEMO_PID: u64 = 0x1337;
    /// The success banner.
    pub const BANNER: &str = "PHASE6 RUNTIME OK\n";
    /// The data file content (10 bytes; fstat pins its size).
    pub const FILE_BYTES: &[u8; 10] = b"Hello,PDG!";
}

/// The main module's TLS variable initial value.
const MAIN_TLS_INIT: u64 = 0x0BAD_C0DE_5EED_FACE;
/// The PRX's TLS variable initial value.
const PDGX_TLS_INIT: u64 = 0x0AAA_BBBB_CCCC_DDDD;
/// `pdgx_add`'s constant.
const PDGX_ADD_K: u64 = 0x1357;
/// `pdgx_version`'s data value.
const PDGX_VERSION: u64 = 0xDEAD_BEEF;
/// `pdgy_value`'s data value (the depth-2 module).
const PDGY_VALUE: u64 = 0x9988_7766_5544_3322;

/// Assemble the Phase-6 main executable: an `ET_DYN` image with a
/// `DT_NEEDED` on `libpdgx.prx`, a `JUMP_SLOT` for `pdgx_add`, a
/// `GLOB_DAT` for `pdgx_version`, a `TPOFF64` slot for its TLS variable,
/// every stage's code, and the worker routine.
pub fn runtime_demo_elf() -> Vec<u8> {
    use layout6 as l;
    use orbis::nr;

    let mut data: Vec<u8> = vec![0u8; 0x900];
    // Data-slot addresses are absolute (DATA_VA + offset); `put`
    // converts back to image offsets.
    let put = |d: &mut Vec<u8>, at: u64, bytes: &[u8]| {
        let off = (at - l::DATA_VA) as usize;
        d[off..off + bytes.len()].copy_from_slice(bytes);
    };
    put(&mut data, l::PATH_DATA, b"/app0/data.txt\0");
    put(&mut data, l::PATH_APP, b"/app0\0");
    put(&mut data, l::PATH_PRX, b"/app0/sce_module/libpdgx.prx\0");
    put(&mut data, l::MSG, l::BANNER.as_bytes());
    put(&mut data, l::NAME_ADD, b"pdgx_add\0");
    put(&mut data, l::NAME_VER, b"pdgx_version\0");
    put(&mut data, l::NAME_PDY, b"pdgy_value\0");
    put(&mut data, l::NAME_VER + 0x20, b"pdgx_tls_bump\0");
    // The ucontext: entry, stack and one argument for the worker.
    put(
        &mut data,
        l::UCTX + crate::thread::ucontext::MC_RIP,
        &l::WORKER_ENTRY.to_le_bytes(),
    );
    put(
        &mut data,
        l::UCTX + crate::thread::ucontext::MC_RSP,
        &l::WORKER_STACK_TOP.to_le_bytes(),
    );
    put(
        &mut data,
        l::UCTX + crate::thread::ucontext::MC_RDI,
        &0x5A5A_5A5Au64.to_le_bytes(),
    );
    put(
        &mut data,
        l::UCTX + crate::thread::ucontext::MC_RFLAGS,
        &0x202u64.to_le_bytes(),
    );

    let mut text: Vec<u8> = Vec::with_capacity(0x800);

    // ---- stages 0-1: console probe and pid ------------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::WRITE, &[1, l::MSG, 4]);
    s.fail_on_carry();
    s.rax_eq(4);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_WRITE));

    let mut s = Stage::new();
    sys(&mut s.code, nr::GETPID, &[]);
    s.fail_on_carry();
    s.rax_eq(l::DEMO_PID);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_GETPID));

    // ---- stage 2: dynlib_get_list → [1, 2, 3], count 3 -----------------
    let mut s = Stage::new();
    sys(
        &mut s.code,
        nr::DYNLIB_GET_LIST,
        &[l::LIST, 8, l::LIST_COUNT],
    );
    s.fail_on_carry();
    s.rax_eq(0);
    s.mem_eq(l::LIST_COUNT, 3);
    // The list is three u32 handles read as qwords: [main, libpdgx] then
    // [libpdgy, zero padding].
    s.mem_eq(l::LIST, 1 | (2 << 32));
    s.mem_eq(l::LIST + 8, 3);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_DYNLIST));

    // ---- stage 3: dynlib_get_info(libpdgx) → name + segments -------------
    let mut s = Stage::new();
    // The caller-initialized size prefix.
    em::rax_i64(&mut s.code, orbis::module_info::SIZE as u64);
    em::abs_rax(&mut s.code, l::INFO);
    sys(&mut s.code, nr::DYNLIB_GET_INFO, &[2, l::INFO]);
    s.fail_on_carry();
    s.rax_eq(0);
    // The name field's first qword is "libpdgx\0".
    let name_qword = u64::from_le_bytes(*b"libpdgx\0");
    s.mem_eq(l::INFO + orbis::module_info::NAME as u64, name_qword);
    // segmentCount == 2 and the first segment is mapped (nonzero addr).
    s.mem_eq(l::INFO + orbis::module_info::SEGMENT_COUNT as u64, 2);
    em::rax_abs(&mut s.code, l::INFO + orbis::module_info::SEGMENTS as u64);
    s.rax_nonzero();
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_DYNINFO));

    // ---- stage 4: dlsym(libpdgx, "pdgx_add") and the indirect call ------
    let mut s = Stage::new();
    sys(
        &mut s.code,
        nr::DYNLIB_DLSYM,
        &[2, l::NAME_ADD, l::DLSYM_OUT],
    );
    s.fail_on_carry();
    s.rax_eq(0);
    em::rax_abs(&mut s.code, l::DLSYM_OUT);
    s.rax_nonzero();
    // call pdgx_add(0x22) → 0x22 + 0x1357
    em::rdi_i64(&mut s.code, 0x22);
    em::call_rax(&mut s.code);
    em::rcx_i64(&mut s.code, 0x22 + PDGX_ADD_K);
    em::cmp_rax_rcx(&mut s.code);
    s.jcc_fail(0x75); // jne
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_DLSYM));

    // ---- stage 5: the JUMP_SLOT call through the relocated GOT ----------
    let mut s = Stage::new();
    em::rax_abs(&mut s.code, l::GOT_ADD);
    s.rax_nonzero();
    em::rdi_i64(&mut s.code, 0x33);
    em::call_rax(&mut s.code);
    em::rcx_i64(&mut s.code, 0x33 + PDGX_ADD_K);
    em::cmp_rax_rcx(&mut s.code);
    s.jcc_fail(0x75); // jne
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_LINKCALL));

    // ---- stage 6: GLOB_DAT data + depth-2 dlsym -------------------------
    let mut s = Stage::new();
    // pdgx_version through the main GOT slot.
    em::rax_abs(&mut s.code, l::GOT_VER);
    s.rax_nonzero();
    em::rcx_i64(&mut s.code, 0);
    em::cmp_rax_rcx(&mut s.code);
    s.jcc_fail(0x74); // je → a null pointer fails
    // *got == PDGX_VERSION
    em::rbx_mem_rax(&mut s.code);
    em::rcx_i64(&mut s.code, PDGX_VERSION);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    // dlsym(libpdgy, "pdgy_value") → read 0x9988_7766_5544_3322
    sys(
        &mut s.code,
        nr::DYNLIB_DLSYM,
        &[3, l::NAME_PDY, l::DLSYM_OUT],
    );
    s.fail_on_carry();
    em::rax_abs(&mut s.code, l::DLSYM_OUT);
    s.rax_nonzero();
    em::rbx_mem_rax(&mut s.code);
    em::rcx_i64(&mut s.code, PDGY_VALUE);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_GLOBDAT));

    // ---- stage 7: fstat("/app0/data.txt") → size 10, S_IFREG -------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::OPEN, &[l::PATH_DATA, 0]);
    s.fail_on_carry();
    s.rax_ge(3);
    // fstat(fd, STAT_BUF): rdi = the fd the open returned.
    em::rdi_rax(&mut s.code);
    em::rsi_i64(&mut s.code, l::STAT_BUF);
    em::rax_i64(&mut s.code, nr::FSTAT);
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    s.mem_eq(
        l::STAT_BUF + orbis::stat::ST_SIZE as u64,
        l::FILE_BYTES.len() as u64,
    );
    s.mem_eq(
        l::STAT_BUF + orbis::stat::ST_MODE as u64,
        u64::from(orbis::stat::S_IFREG | 0o644) | (1 << 16),
    );
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_FSTAT));

    // ---- stage 8: stat("/app0/sce_module/libpdgx.prx") → a real ELF ---
    let prx_len = runtime_demo_prx().len() as u64;
    let mut s = Stage::new();
    em::rdi_i64(&mut s.code, l::PATH_PRX);
    em::rsi_i64(&mut s.code, l::STAT_BUF);
    em::rax_i64(&mut s.code, nr::STAT);
    em::syscall(&mut s.code);
    s.fail_on_carry();
    s.rax_eq(0);
    s.mem_eq(l::STAT_BUF + orbis::stat::ST_SIZE as u64, prx_len);
    s.mem_eq(
        l::STAT_BUF + orbis::stat::ST_MODE as u64,
        u64::from(orbis::stat::S_IFREG | 0o644) | (1 << 16),
    );
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_STAT));

    // ---- stage 9: thr_self → 1 ------------------------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::THR_SELF, &[l::TID_OUT]);
    s.fail_on_carry();
    s.rax_eq(0);
    s.mem_eq(l::TID_OUT, 1);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_THRSELF));

    // ---- stage 10: main TLS read through fs: ----------------------------
    let mut s = Stage::new();
    em::rax_abs(&mut s.code, l::MAIN_TLS_PTR);
    em::rbx_fs_rax(&mut s.code);
    em::rcx_i64(&mut s.code, MAIN_TLS_INIT);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_TLSMAIN));

    // ---- stage 11: thr_create + take the mutex + WAIT(trying) -----------
    // (the stage completes after the worker wakes main; the finish runs
    // then)
    let mut s = Stage::new();
    sys(&mut s.code, nr::THR_CREATE, &[l::UCTX, l::TID_OUT, 0]);
    s.fail_on_carry();
    s.rax_eq(0);
    s.mem_eq(l::TID_OUT, 2);
    // Take the umutex, then park on the worker's "trying" flag.
    sys(
        &mut s.code,
        nr::UMTX_OP,
        &[l::M, orbis::umtx::MUTEX_LOCK as u64, 0, 0, 0],
    );
    s.fail_on_carry();
    sys(
        &mut s.code,
        nr::UMTX_OP,
        &[l::TRYING, orbis::umtx::WAIT as u64, 0, 0, 0],
    );
    s.fail_on_carry();
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_WORKER_CREATE));

    // ---- stage 14 (runs after the wake): shared == 2, main TLS intact --
    let mut s = Stage::new();
    // shared += 1
    em::rax_abs(&mut s.code, l::SHARED);
    em::inc_rax(&mut s.code);
    em::abs_rax(&mut s.code, l::SHARED);
    // Hand the mutex to the waiting worker (or release it).
    sys(
        &mut s.code,
        nr::UMTX_OP,
        &[l::M, orbis::umtx::MUTEX_UNLOCK as u64, 0, 0, 0],
    );
    s.fail_on_carry();
    // Join: park until the worker sets done.
    sys(
        &mut s.code,
        nr::UMTX_OP,
        &[l::DONE, orbis::umtx::WAIT as u64, 0, 0, 0],
    );
    s.fail_on_carry();
    // The critical section really ran twice under the mutex.
    s.mem_eq(l::SHARED, 2);
    // The mutex is unowned again.
    s.mem_eq(l::M, 0);
    // Main's TLS block is its own: the worker's bumps did not touch it.
    em::rax_abs(&mut s.code, l::MAIN_TLS_PTR);
    em::rbx_fs_rax(&mut s.code);
    em::rcx_i64(&mut s.code, MAIN_TLS_INIT);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    // And the worker reported the PRX's TLS initial value.
    s.mem_eq(l::WORKER_TLS_OLD, PDGX_TLS_INIT);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_UMTX));

    // ---- stage 15: banner + exit ----------------------------------------
    let mut s = Stage::new();
    sys(&mut s.code, nr::WRITE, &[1, l::MSG, l::BANNER.len() as u64]);
    s.fail_on_carry();
    s.rax_eq(l::BANNER.len() as u64);
    // (TLS-after is proven inside ST_UMTX's tail; mirror it here for the
    // host's per-stage report.)
    em::rax_abs(&mut s.code, l::MAIN_TLS_PTR);
    em::rbx_fs_rax(&mut s.code);
    em::rcx_i64(&mut s.code, MAIN_TLS_INIT);
    em::cmp_rbx_rcx(&mut s.code);
    s.jcc_fail(0x75);
    text.extend_from_slice(&s.finish(l::STATUS + 8 * l::ST_TLSMAIN_AFTER));

    // ---- final: exit(0) ---------------------------------------------------
    sys(&mut text, nr::EXIT, &[0]);
    // (exit never returns; padding is unreachable)

    // ---- the worker, pinned at WORKER_ENTRY -------------------------------
    while text.len() < (l::WORKER_ENTRY - l::TEXT_VA) as usize {
        text.push(0x90);
    }
    debug_assert_eq!(text.len(), (l::WORKER_ENTRY - l::TEXT_VA) as usize);
    let mut w = Stage::new();
    // thr_self → 2 → ST_WORKER_ID
    sys(&mut w.code, nr::THR_SELF, &[l::TID_OUT]);
    w.fail_on_carry();
    w.rax_eq(0);
    w.mem_eq(l::TID_OUT, 2);
    text.extend_from_slice(&w.finish(l::STATUS + 8 * l::ST_WORKER_ID));

    // The PRX TLS bump: pdgx_tls_bump(5) returns the OLD value (the
    // template's), proving the worker's own copy of the block.
    let mut w = Stage::new();
    sys(
        &mut w.code,
        nr::DYNLIB_DLSYM,
        &[2, l::NAME_VER + 0x20, l::DLSYM_OUT], // "pdgx_tls_bump"
    );
    w.fail_on_carry();
    em::rax_abs(&mut w.code, l::DLSYM_OUT);
    w.rax_nonzero();
    em::rdi_i64(&mut w.code, 5);
    em::call_rax(&mut w.code);
    em::rcx_i64(&mut w.code, PDGX_TLS_INIT);
    em::cmp_rax_rcx(&mut w.code);
    w.jcc_fail(0x75);
    // Save the observed old value for the host-side assertion.
    em::abs_rax(&mut w.code, l::WORKER_TLS_OLD);
    text.extend_from_slice(&w.finish(l::STATUS + 8 * l::ST_WORKER_TLS));

    // The choreography: trying = 1; WAKE(trying); MUTEX_LOCK (parks);
    // shared += 1; MUTEX_UNLOCK; done = 1; WAKE(done); thr_exit.
    let mut w = Stage::new();
    store_imm64(&mut w.code, l::TRYING, 1);
    sys(
        &mut w.code,
        nr::UMTX_OP,
        &[l::TRYING, orbis::umtx::WAKE as u64, 0, 0, 0],
    );
    w.fail_on_carry();
    sys(
        &mut w.code,
        nr::UMTX_OP,
        &[l::M, orbis::umtx::MUTEX_LOCK as u64, 0, 0, 0],
    );
    w.fail_on_carry();
    em::rax_abs(&mut w.code, l::SHARED);
    em::inc_rax(&mut w.code);
    em::abs_rax(&mut w.code, l::SHARED);
    sys(
        &mut w.code,
        nr::UMTX_OP,
        &[l::M, orbis::umtx::MUTEX_UNLOCK as u64, 0, 0, 0],
    );
    w.fail_on_carry();
    store_imm64(&mut w.code, l::DONE, 1);
    sys(
        &mut w.code,
        nr::UMTX_OP,
        &[l::DONE, orbis::umtx::WAKE as u64, 0, 0, 0],
    );
    w.fail_on_carry();
    sys(&mut w.code, nr::THR_EXIT, &[l::STATE_PTR]);
    // (thr_exit does not return; the fail tail is unreachable)
    text.extend_from_slice(&w.finish(l::STATUS + 8 * l::ST_WORKER_ID)); // tail unused

    // ---- assemble the shared object --------------------------------------
    let mut m = SharedBuilder::new(l::TEXT_VA, l::DATA_VA, l::TEXT_VA);
    m.text = text;
    m.data = data;
    m.tls = Some(MAIN_TLS_INIT.to_le_bytes().to_vec());
    m.imports = vec!["pdgx_add".to_owned(), "pdgx_version".to_owned()];
    m.needed = vec!["libpdgx.prx".to_owned()];
    let add_idx = m.sym_index("pdgx_add").unwrap();
    let ver_idx = m.sym_index("pdgx_version").unwrap();
    m.relas = vec![
        Rela {
            r_offset: l::GOT_ADD,
            r_type: rt::X86_64_JUMP_SLOT,
            r_sym: add_idx,
            r_addend: 0,
        },
        Rela {
            r_offset: l::GOT_VER,
            r_type: rt::X86_64_GLOB_DAT,
            r_sym: ver_idx,
            r_addend: 0,
        },
        Rela {
            r_offset: l::MAIN_TLS_PTR,
            r_type: rt::X86_64_TPOFF64,
            r_sym: 0,
            r_addend: 0, // the main module's TLS block is at offset 0
        },
    ];
    m.build()
}

/// `libpdgx.prx`: exports `pdgx_add` (rdi + K), the data symbol
/// `pdgx_version`, and `pdgx_tls_bump` (adds rdi to the PRX's own TLS
/// variable, returns the old value); needs `libpdgy.prx` and relocations
/// for both.
pub fn runtime_demo_prx() -> Vec<u8> {
    // Link-time layout inside the PRX.
    const P_TEXT: u64 = 0;
    const P_DATA: u64 = 0x1_0000;
    // Data slots (offsets from P_DATA).
    const D_VERSION: u64 = 0x00;
    const D_PDY_GOT: u64 = 0x08;
    const D_TLS_PTR: u64 = 0x10;
    // Link-time address of pdgx_tls_bump (for the RIP-relative lea).
    const BUMP_VA: u64 = P_TEXT + 0x20;
    // pdgx_add: lea rax, [rdi + K]; ret
    let add_fn = [
        0x48,
        0x8D,
        0x87,
        (PDGX_ADD_K & 0xFF) as u8,
        ((PDGX_ADD_K >> 8) & 0xFF) as u8,
        0x00,
        0x00,
        0xC3,
    ];
    // pdgx_tls_bump(rdi): position-independent — the module finds its
    // own TLS-pointer slot through a RIP-relative lea (the displacement
    // is a build-time constant, so the code works at any load bias).
    //   lea rax, [rip + &D_TLS_PTR]  ; this module's slot
    //   mov rcx, [rax]               ; the TPOFF the loader wrote
    //   mov rbx, fs:[rcx]            ; the old value
    //   mov rax, rbx                 ; return old
    //   add rbx, rdi
    //   mov fs:[rcx], rbx            ; store the new value
    //   ret
    let mut bump = Vec::new();
    let tls_ptr_link = P_DATA + D_TLS_PTR;
    let lea_end = BUMP_VA + bump.len() as u64 + 7; // after the 7-byte lea
    em::lea_rax_rip(&mut bump, tls_ptr_link, lea_end);
    em::rcx_mem_rax(&mut bump);
    em::rbx_fs_rcx(&mut bump);
    em::rax_rbx(&mut bump);
    em::add_rbx_rdi(&mut bump);
    em::fs_rcx_rbx(&mut bump);
    em::ret(&mut bump);

    let mut text = Vec::new();
    text.extend_from_slice(&add_fn);
    while text.len() < 0x20 {
        text.push(0x90);
    }
    text.extend_from_slice(&bump);
    let add_sym = P_TEXT;
    let bump_sym = P_TEXT + 0x20;

    let mut data = vec![0u8; 0x20];
    data[D_VERSION as usize..D_VERSION as usize + 8].copy_from_slice(&PDGX_VERSION.to_le_bytes());

    let mut p = SharedBuilder::new(P_TEXT, P_DATA, P_TEXT);
    p.text = text;
    p.data = data;
    p.tls = Some(PDGX_TLS_INIT.to_le_bytes().to_vec());
    p.imports = vec!["pdgy_value".to_owned()];
    p.needed = vec!["libpdgy.prx".to_owned()];
    let pdy_idx = p.sym_index("pdgy_value").unwrap();
    p.exports = vec![
        ExportSym::func("pdgx_add", add_sym, add_fn.len() as u64),
        ExportSym::object("pdgx_version", P_DATA + D_VERSION, 8),
        ExportSym::func("pdgx_tls_bump", bump_sym, bump.len() as u64),
    ];
    p.relas = vec![
        Rela {
            r_offset: P_DATA + D_PDY_GOT,
            r_type: rt::X86_64_GLOB_DAT,
            r_sym: pdy_idx,
            r_addend: 0,
        },
        Rela {
            r_offset: P_DATA + D_TLS_PTR,
            r_type: rt::X86_64_TPOFF64,
            r_sym: 0,
            r_addend: 0, // the PRX's TLS variable is at its block offset 0
        },
    ];
    p.build()
}

/// `libpdgy.prx`: the depth-2 dependency; exports the data symbol
/// `pdgy_value`.
pub fn runtime_demo_prx2() -> Vec<u8> {
    const P_TEXT: u64 = 0;
    const P_DATA: u64 = 0x1_0000;
    let mut p = SharedBuilder::new(P_TEXT, P_DATA, P_TEXT);
    p.text = vec![0xC3]; // ret (a module with a body)
    p.data = PDGY_VALUE.to_le_bytes().to_vec();
    p.exports = vec![ExportSym::object("pdgy_value", P_DATA, 8)];
    p.build()
}

/// The PRX's TLS template bytes (tests pin the thread-copy semantics).
pub fn runtime_demo_prx_tls_probe() -> [u8; 8] {
    PDGX_TLS_INIT.to_le_bytes()
}

/// The Phase-6 demo PFS: the data file plus both modules at the
/// documented `/app0/sce_module` directory.
pub fn runtime_demo_pfs_image() -> Vec<u8> {
    use crate::pfs::PfsBuilder;
    let mut b = PfsBuilder::new();
    b.add_file("/app0/data.txt", layout6::FILE_BYTES.to_vec());
    b.add_file("/app0/sce_module/libpdgx.prx", runtime_demo_prx());
    b.add_file("/app0/sce_module/libpdgy.prx", runtime_demo_prx2());
    b.build().expect("runtime demo pfs builds")
}

/// The dirent stream the kernel must produce for `/app` (the demo's
/// expected bytes; computed with the same encoder the syscall uses).
fn expected_dirents() -> Vec<u8> {
    let mut out = Vec::new();
    orbis::encode_dirent(&mut out, 1, dirent::DT_DIR, b".");
    orbis::encode_dirent(&mut out, 0, dirent::DT_DIR, b"..");
    orbis::encode_dirent(&mut out, 2, dirent::DT_REG, b"data.txt");
    orbis::encode_dirent(&mut out, 3, dirent::DT_DIR, b"nested");
    out
}

/// NVMe register offsets the driver touches (from `pdg_io::nvme::regs`).
mod nv {
    pub const REG_AQA: u64 = 0x24;
    pub const REG_ASQ: u64 = 0x28;
    pub const REG_ACQ: u64 = 0x2C;
    pub const REG_CC: u64 = 0x14;
    pub const REG_CSTS: u64 = 0x1C;
    pub const DB_SQ0: u64 = 0x1000;
    pub const DB_SQ1: u64 = 0x1008;
}

/// Build the PFS image the demo expects (the files it opens).
pub fn demo_pfs_image() -> Vec<u8> {
    use crate::pfs::PfsBuilder;
    let mut b = PfsBuilder::new();
    b.add_file(layout::DATA_FILE, layout::FILE_BYTES.to_vec());
    b.add_file("/app/nested/deep.bin", vec![0u8; 8192]);
    b.build().expect("demo pfs builds")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elf_parses_and_has_expected_shape() {
        let img = kernel_demo_elf();
        let elf = crate::loader::ElfImage::parse(&img).expect("parse");
        assert_eq!(elf.e_entry, layout::TEXT_VA + 0x100);
        assert_eq!(elf.relas.len(), 2);
        assert!(elf.tls().is_some());
        assert!(
            elf.phdrs
                .iter()
                .any(|p| p.p_type == crate::loader::pt::DYNAMIC)
        );
    }

    #[test]
    fn demo_text_uses_only_verified_instruction_forms() {
        // Disassemble the whole text segment through iced-x86 (the same
        // discipline the Phase-4 payload test established).
        use iced_x86::{Decoder, DecoderOptions, Mnemonic};
        let img = kernel_demo_elf();
        let elf = crate::loader::ElfImage::parse(&img).unwrap();
        let text = &elf.phdrs[0];
        let code = &elf.bytes[text.p_offset as usize..(text.p_offset + text.p_filesz) as usize];
        let mut decoder = Decoder::with_ip(64, code, layout::TEXT_VA, DecoderOptions::NONE);
        let allowed = std::collections::HashSet::from([
            Mnemonic::Mov,
            Mnemonic::Push,
            Mnemonic::Pop,
            Mnemonic::Cmp,
            Mnemonic::Test,
            Mnemonic::Xor,
            Mnemonic::Je,
            Mnemonic::Jne,
            Mnemonic::Jb,
            Mnemonic::Jl,
            Mnemonic::Jmp,
            Mnemonic::Nop,
            Mnemonic::Syscall,
            Mnemonic::Lidt,
            Mnemonic::Sti,
            Mnemonic::Iretq,
        ]);
        let mut count = 0usize;
        while decoder.can_decode() {
            let instr = decoder.decode();
            assert!(
                allowed.contains(&instr.mnemonic()),
                "unexpected instruction {:?} at RIP {:#x}",
                instr.mnemonic(),
                instr.ip()
            );
            count += 1;
        }
        assert!(count > 150, "the demo is substantial: {count} instructions");
    }

    #[test]
    fn expected_dirents_match_the_public_layout() {
        let e = expected_dirents();
        assert_eq!(e.len(), 60);
        // "." : fileno 1, reclen 12, type DT_DIR(4), namlen 1
        assert_eq!(u32::from_le_bytes(e[0..4].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(e[4..6].try_into().unwrap()), 12);
        assert_eq!(e[6], dirent::DT_DIR);
        assert_eq!(e[7], 1);
        assert_eq!(&e[8..9], b".");
    }

    #[test]
    fn demo_pfs_roundtrips() {
        let img = demo_pfs_image();
        let mut pfs = crate::pfs::Pfs::mount(Box::new(crate::pfs::MemoryBlockDevice::new(img)))
            .expect("mount");
        assert_eq!(
            pfs.read_file(layout::DATA_FILE).unwrap(),
            layout::FILE_BYTES
        );
    }

    #[test]
    fn errno_values_are_the_public_freebsd_ones() {
        use crate::orbis::errno;
        assert_eq!(errno::ENOENT, 2);
        assert_eq!(errno::EBADF, 9);
        assert_eq!(errno::EINVAL, 22);
        assert_eq!(errno::ENOSYS, 78);
    }

    #[test]
    fn runtime_demo_uses_only_verified_instruction_forms() {
        // The Phase-6 discipline: the whole main text disassembles into the
        // documented set (indirect calls for dlsym/GOT targets, fs-relative
        // TLS access, inc for the shared counter).
        use iced_x86::{Decoder, DecoderOptions, Mnemonic};
        let img = runtime_demo_elf();
        let elf = crate::loader::ElfImage::parse(&img).unwrap();
        let text = &elf.phdrs[0];
        let code = &elf.bytes[text.p_offset as usize..(text.p_offset + text.p_filesz) as usize];
        let mut decoder = Decoder::with_ip(64, code, layout6::TEXT_VA, DecoderOptions::NONE);
        let allowed = std::collections::HashSet::from([
            Mnemonic::Mov,
            Mnemonic::Push,
            Mnemonic::Pop,
            Mnemonic::Cmp,
            Mnemonic::Test,
            Mnemonic::Xor,
            Mnemonic::Add,
            Mnemonic::Inc,
            Mnemonic::Je,
            Mnemonic::Jne,
            Mnemonic::Jb,
            Mnemonic::Jl,
            Mnemonic::Jmp,
            Mnemonic::Nop,
            Mnemonic::Syscall,
            Mnemonic::Call,
            Mnemonic::Ret,
        ]);
        let mut count = 0usize;
        while decoder.can_decode() {
            let instr = decoder.decode();
            assert!(
                allowed.contains(&instr.mnemonic()),
                "unexpected instruction {:?} at RIP {:#x}",
                instr.mnemonic(),
                instr.ip()
            );
            count += 1;
        }
        assert!(
            count > 120,
            "the runtime demo is substantial: {count} instructions"
        );
    }

    #[test]
    fn runtime_demo_prx_shapes_parse() {
        // Both PRXes parse as ET_DYN shared objects with the expected
        // exports and needed entries; the depth-2 module has none.
        let elf = crate::loader::ElfImage::parse(&runtime_demo_prx()).unwrap();
        assert_eq!(elf.e_type, crate::loader::ET_DYN);
        assert_eq!(elf.dynstr_name(elf.needed[0]), "libpdgy.prx");
        assert!(elf.sym_addr("pdgx_add").is_some());
        assert!(elf.sym_addr("pdgx_tls_bump").is_some());
        assert!(elf.tls().is_some());
        let elf2 = crate::loader::ElfImage::parse(&runtime_demo_prx2()).unwrap();
        assert_eq!(elf2.e_type, crate::loader::ET_DYN);
        assert!(elf2.needed.is_empty());
        assert!(elf2.sym_addr("pdgy_value").is_some());
        // The main image carries its needed closure and imports.
        let m = crate::loader::ElfImage::parse(&runtime_demo_elf()).unwrap();
        assert_eq!(m.e_type, crate::loader::ET_DYN);
        assert_eq!(m.dynstr_name(m.needed[0]), "libpdgx.prx");
        assert_eq!(m.sym_name(1), "pdgx_add");
        assert!(m.tls().is_some());
    }

    #[test]
    fn runtime_demo_pfs_roundtrips() {
        let img = runtime_demo_pfs_image();
        let mut pfs =
            crate::pfs::Pfs::mount(Box::new(crate::pfs::MemoryBlockDevice::new(img))).unwrap();
        assert_eq!(pfs.size_of("/app0/data.txt").unwrap(), 10);
        let prx = pfs.read_file("/app0/sce_module/libpdgx.prx").unwrap();
        assert_eq!(prx.len(), runtime_demo_prx().len());
        let prx2 = pfs.read_file("/app0/sce_module/libpdgy.prx").unwrap();
        assert_eq!(prx2.len(), runtime_demo_prx2().len());
    }
}
