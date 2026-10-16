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
use crate::loader::{ElfBuilder, Rela};
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
    fn finish(mut self, status: u64) -> Vec<u8> {
        // PASS: mov rax, 1; mov [abs], rax; jmp END
        em::rax_i64(&mut self.code, 1);
        em::abs_rax(&mut self.code, status);
        em::jmp(&mut self.code, 12);
        // FAIL: xor eax, eax; mov [abs], rax   (12 bytes)
        let fail_start = self.code.len() + 2; // after the xor
        em::xor_eax(&mut self.code);
        em::abs_rax(&mut self.code, status);
        let fail_end = self.code.len();
        debug_assert_eq!(fail_end, fail_start + 10);
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
}
