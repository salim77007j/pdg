//! The Phase-4 demo payload: a hand-assembled x86-64 guest driver that
//! drives the whole I/O stack through MMIO — the same tier-1 pattern the
//! GPU phase used (see `pdg-tests/tests/gpu.rs`).
//!
//! The driver:
//!
//! 1. enables the NVMe controller (AQA/ASQ/ACQ + CC.EN, polls CSTS.RDY),
//! 2. submits IDENTIFY and checks the serial number in guest RAM,
//! 3. creates an I/O queue pair, WRITEs a marker LBA and READs it back,
//! 4. reads the DualSense window (buttons + stick) the host pre-set,
//! 5. hydrates an audio block and polls the voice counter,
//! 6. prints `PHASE4 IO OK` to the console and `hlt`s.
//!
//! Every stage stores 1 (pass) or 0 (fail) into the results area so the
//! host can verify each subsystem independently.
//!
//! Only fixed-width, absolute-addressing instruction forms are emitted
//! (`mov r64, imm64`, the 64-bit moffs load/store, `push`/`pop`, `cmp`,
//! `test al`, short jumps, `hlt`) — all verified instructions of the
//! Phase-2 interpreter.

/// Guest memory layout of the demo (flat, physical == virtual).
pub mod layout {
    /// Driver code base.
    pub const CODE: u64 = 0x0010_0000;
    /// Per-stage results (six qwords: 1 = pass).
    pub const RESULTS: u64 = 0x0010_1000;
    /// Admin submission queue (8 entries).
    pub const ASQ: u64 = 0x0020_0000;
    /// Admin completion queue.
    pub const ACQ: u64 = 0x0020_1000;
    /// I/O completion queue.
    pub const IO_CQ: u64 = 0x0030_0000;
    /// I/O submission queue.
    pub const IO_SQ: u64 = 0x0031_0000;
    /// IDENTIFY landing buffer (4 KiB).
    pub const IDENT: u64 = 0x0040_0000;
    /// Data buffer A.
    pub const DATA_A: u64 = 0x0050_0000;
    /// Data buffer B.
    pub const DATA_B: u64 = 0x0051_0000;
    /// Stack top.
    pub const STACK: u64 = 0x0080_0000;

    /// Result index: controller enable.
    pub const ST_ENABLE: u64 = 0;
    /// Result index: IDENTIFY serial check.
    pub const ST_IDENTIFY: u64 = 1;
    /// Result index: I/O queue creation.
    pub const ST_QUEUES: u64 = 2;
    /// Result index: write/read round-trip.
    pub const ST_IO: u64 = 3;
    /// Result index: pad read.
    pub const ST_PAD: u64 = 4;
    /// Result index: audio hydration.
    pub const ST_AUDIO: u64 = 5;
}

/// MMIO bases the payload targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemoMmio {
    /// NVMe register window.
    pub nvme: u64,
    /// DualSense pad window.
    pub pad: u64,
    /// Audio window.
    pub audio: u64,
    /// Console window.
    pub console: u64,
}

/// The serial-number prefix the driver compares after IDENTIFY
/// ("PDG-SSD-" as a little-endian qword).
const SN_QWORD: u64 = u64::from_le_bytes(*b"PDG-SSD-");
/// The 8-byte marker the driver writes to the SSD and reads back
/// ("PDGSSDIO" as a little-endian qword).
const MARKER_QWORD: u64 = u64::from_le_bytes(*b"PDGSSDIO");
/// The CROSS button bit (see [`crate::input::button`]).
const CROSS: u64 = 1; // input::button::CROSS

// ---------------------------------------------------------------------------
// Emitters (fixed encodings).
// ---------------------------------------------------------------------------

fn emit(code: &mut Vec<u8>, bytes: &[u8]) {
    code.extend_from_slice(bytes);
}

/// `mov rax, imm64`
fn mov_rax_imm64(code: &mut Vec<u8>, v: u64) {
    emit(code, &[0x48, 0xB8]);
    emit(code, &v.to_le_bytes());
}

/// `mov qword [abs], rax` (64-bit moffs store)
fn mov_abs_rax(code: &mut Vec<u8>, addr: u64) {
    emit(code, &[0x48, 0xA3]);
    emit(code, &addr.to_le_bytes());
}

/// `mov rax, qword [abs]` (64-bit moffs load)
fn mov_rax_abs(code: &mut Vec<u8>, addr: u64) {
    emit(code, &[0x48, 0xA1]);
    emit(code, &addr.to_le_bytes());
}

/// `test al, imm8`
fn test_al(code: &mut Vec<u8>, v: u8) {
    emit(code, &[0xA8, v]);
}

/// `test al, al` — zero iff al is zero (the NUL check).
fn test_al_al(code: &mut Vec<u8>) {
    emit(code, &[0x84, 0xC0]);
}

/// `cmp al, imm8`
fn cmp_al(code: &mut Vec<u8>, v: u8) {
    emit(code, &[0x3C, v]);
}

/// `jz rel8` / `jne rel8` / `je rel8`
fn jz(code: &mut Vec<u8>, d: i8) {
    emit(code, &[0x74, d as u8]);
}
fn jne(code: &mut Vec<u8>, d: i8) {
    emit(code, &[0x75, d as u8]);
}

/// `mov rax, imm64; mov qword [abs], rax`
fn store_imm64(code: &mut Vec<u8>, addr: u64, v: u64) {
    mov_rax_imm64(code, v);
    mov_abs_rax(code, addr);
}

/// Poll a bit at `addr` until set (14-byte spin loop, backward jump).
fn poll_bit(code: &mut Vec<u8>, addr: u64, bit: u8) {
    mov_rax_abs(code, addr);
    test_al(code, bit);
    jz(code, -14);
}

/// Poll a byte at `addr` until it equals `v`.
fn poll_eq(code: &mut Vec<u8>, addr: u64, v: u8) {
    mov_rax_abs(code, addr);
    cmp_al(code, v);
    jne(code, -14);
}

/// Store 1 into the result slot if the qword at `addr` equals `v`, else 0.
fn check_qword(code: &mut Vec<u8>, addr: u64, v: u64, status: u64) {
    mov_rax_abs(code, addr); // 10
    emit(code, &[0x50]); // push rax
    mov_rax_imm64(code, v); // 10
    emit(code, &[0x59]); // pop rcx
    emit(code, &[0x48, 0x39, 0xC1]); // cmp rcx, rax
    mov_rax_imm64(code, 1); // 10
    jz(code, 10); // je over the FAIL immediate
    mov_rax_imm64(code, 0); // 10
    mov_abs_rax(code, status);
}

/// Store 1 into `status` unconditionally (a stage completed).
fn pass_status(code: &mut Vec<u8>, status: u64) {
    store_imm64(code, status, 1);
}

// ---------------------------------------------------------------------------
// NVMe register offsets used by the driver (from `regs`).
// ---------------------------------------------------------------------------
mod nv {
    pub const REG_AQA: u64 = 0x24;
    pub const REG_ASQ: u64 = 0x28;
    pub const REG_ACQ: u64 = 0x2C;
    pub const REG_CC: u64 = 0x14;
    pub const REG_CSTS: u64 = 0x1C;
    pub const DB_SQ0: u64 = 0x1000; // admin SQ tail
    pub const DB_SQ1: u64 = 0x1008; // queue 1 SQ tail
}

/// Build the demo payload for the given MMIO layout.
#[must_use]
pub fn io_demo_payload(mmio: &DemoMmio) -> Vec<u8> {
    use layout as l;
    let mut c = Vec::with_capacity(1024);

    // -- Stage 0: enable the controller ------------------------------------
    // AQA: 8-entry admin queues (fields hold 7).
    store_imm64(&mut c, mmio.nvme + nv::REG_AQA, (7 << 24) | (7 << 16));
    store_imm64(&mut c, mmio.nvme + nv::REG_ASQ, l::ASQ);
    store_imm64(&mut c, mmio.nvme + nv::REG_ACQ, l::ACQ);
    store_imm64(&mut c, mmio.nvme + nv::REG_CC, 1); // CC.EN
    poll_bit(&mut c, mmio.nvme + nv::REG_CSTS, 1); // CSTS.RDY
    pass_status(&mut c, l::RESULTS + 8 * l::ST_ENABLE);

    // -- Stage 1: IDENTIFY the controller ----------------------------------
    // SQE slot 0: cdw0 = IDENTIFY(0x06), nsid = 0.
    store_imm64(&mut c, l::ASQ, 0x0000_0000_0000_0006);
    store_imm64(&mut c, l::ASQ + 16, l::IDENT); // PRP1
    store_imm64(&mut c, l::ASQ + 40, 0); // CDW10: CNS 0 (controller)
    store_imm64(&mut c, mmio.nvme + nv::DB_SQ0, 1); // ring the doorbell
    poll_bit(&mut c, l::ACQ + 12, 1); // CQE slot 0, dword 3, phase
    check_qword(
        &mut c,
        l::IDENT + 4,
        SN_QWORD,
        l::RESULTS + 8 * l::ST_IDENTIFY,
    );

    // -- Stage 2: create the I/O queue pair --------------------------------
    // Slot 1: CREATE I/O COMPLETION QUEUE (0x05), qid 1, 4 entries at IO_CQ.
    store_imm64(&mut c, l::ASQ + 64, 0x0000_0000_0000_0005);
    store_imm64(&mut c, l::ASQ + 64 + 16, l::IO_CQ); // PRP1 = CQ base
    store_imm64(&mut c, l::ASQ + 64 + 40, 0x0001_0003); // CDW10: 3 | qid 1
    store_imm64(&mut c, l::ASQ + 64 + 44, 1); // CDW11: PC=1
    store_imm64(&mut c, mmio.nvme + nv::DB_SQ0, 2);
    poll_bit(&mut c, l::ACQ + 16 + 12, 1); // CQE slot 1 phase
    // Slot 2: CREATE I/O SUBMISSION QUEUE (0x01), qid 1, CQ 1, at IO_SQ.
    store_imm64(&mut c, l::ASQ + 128, 0x0000_0000_0000_0001);
    store_imm64(&mut c, l::ASQ + 128 + 16, l::IO_SQ); // PRP1 = SQ base
    store_imm64(&mut c, l::ASQ + 128 + 40, 0x0001_0003); // CDW10: 3 | qid 1
    store_imm64(&mut c, l::ASQ + 128 + 44, 0x0001_0001); // CDW11: PC=1 | CQID 1
    store_imm64(&mut c, mmio.nvme + nv::DB_SQ0, 3);
    poll_bit(&mut c, l::ACQ + 32 + 12, 1); // CQE slot 2 phase
    pass_status(&mut c, l::RESULTS + 8 * l::ST_QUEUES);

    // -- Stage 3: WRITE the marker, READ it back ---------------------------
    // Seed the data buffer with "PDGSSDIO".
    store_imm64(&mut c, l::DATA_A, MARKER_QWORD);
    // WRITE: IO SQE slot 0 — cdw0 = 0x01 (write), nsid = 1.
    store_imm64(&mut c, l::IO_SQ, 0x0000_0001_0000_0001);
    store_imm64(&mut c, l::IO_SQ + 16, l::DATA_A); // PRP1
    store_imm64(&mut c, l::IO_SQ + 40, 4); // CDW10: SLBA = 4
    store_imm64(&mut c, l::IO_SQ + 48, 0); // CDW12: NLB-1 = 0 (one sector)
    store_imm64(&mut c, mmio.nvme + nv::DB_SQ1, 1);
    poll_bit(&mut c, l::IO_CQ + 12, 1); // CQE slot 0 phase
    // READ: IO SQE slot 1, landing at DATA_B.
    store_imm64(&mut c, l::IO_SQ + 64, 0x0000_0001_0000_0002);
    store_imm64(&mut c, l::IO_SQ + 64 + 16, l::DATA_B); // PRP1
    store_imm64(&mut c, l::IO_SQ + 64 + 40, 4); // SLBA = 4
    store_imm64(&mut c, l::IO_SQ + 64 + 48, 0); // one sector
    store_imm64(&mut c, mmio.nvme + nv::DB_SQ1, 2);
    poll_bit(&mut c, l::IO_CQ + 16 + 12, 1); // CQE slot 1 phase
    check_qword(&mut c, l::DATA_B, MARKER_QWORD, l::RESULTS + 8 * l::ST_IO);

    // -- Stage 4: read the pad ---------------------------------------------
    // The host pre-set: CROSS held, left stick X = 255. A 64-bit read at
    // the pad window returns the register at that exact offset.
    check_qword(&mut c, mmio.pad + 0x10, CROSS, l::RESULTS + 8 * l::ST_PAD);
    check_qword(&mut c, mmio.pad + 0x18, 0xFF, l::RESULTS + 8 * l::ST_PAD);

    // -- Stage 5: hydrate an audio block ------------------------------------
    // Three mono f32 samples: 0.25, 0.5, -0.75.
    store_imm64(&mut c, l::DATA_A, 0x3F00_0000_3E80_0000); // [0.25, 0.5]
    store_imm64(&mut c, l::DATA_A + 8, 0x0000_0000_BF40_0000); // [-0.75, 0]
    store_imm64(&mut c, mmio.audio + 0x20, l::DATA_A); // BLOCK_BASE
    store_imm64(&mut c, mmio.audio + 0x24, 12); // BLOCK_LEN
    store_imm64(&mut c, mmio.audio + 0x30, 1); // HYDRATE
    poll_eq(&mut c, mmio.audio + 0x08, 1); // VOICES == 1
    check_qword(&mut c, mmio.audio + 0x0C, 3, l::RESULTS + 8 * l::ST_AUDIO); // HYDRATED

    // -- Final: print and halt ----------------------------------------------
    // The string rides at the end of the blob. Two imm64 loads (20 bytes)
    // plus the copy loop (17 bytes: 2+2+2+2+3+3+2+1) precede it.
    let string_addr = l::CODE + c.len() as u64 + 20 + 17;
    emit(&mut c, &[0x48, 0xBE]); // mov rsi, imm64
    emit(&mut c, &string_addr.to_le_bytes());
    emit(&mut c, &[0x48, 0xBF]); // mov rdi, imm64
    emit(&mut c, &mmio.console.to_le_bytes());
    emit(&mut c, &[0x8A, 0x06]); // L: mov al, [rsi]
    test_al_al(&mut c); // zero iff the byte was NUL
    jz(&mut c, 10); // jz done (skips 10 bytes of loop body)
    emit(&mut c, &[0x88, 0x07]); // mov [rdi], al
    emit(&mut c, &[0x48, 0xFF, 0xC6]); // inc rsi
    emit(&mut c, &[0x48, 0xFF, 0xC7]); // inc rdi
    emit(&mut c, &[0xEB, 0xF0]); // jmp L (rel8 = -16: back to mov al,[rsi])
    emit(&mut c, &[0xF4]); // done: hlt

    // The string itself.
    emit(&mut c, b"PHASE4 IO OK\0");
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_is_nonempty_and_ends_with_hlt_and_string() {
        let mmio = DemoMmio {
            nvme: 0xE000_0000,
            pad: 0xE100_0000,
            audio: 0xE200_0000,
            console: 0xF000_0000,
        };
        let code = io_demo_payload(&mmio);
        assert!(
            code.len() > 400,
            "the driver is substantial: {}",
            code.len()
        );
        // The very last byte before the string is `hlt`.
        let tail = &code[code.len() - 14..];
        assert_eq!(&tail[..1], &[0xF4]);
        assert_eq!(&code[code.len() - 13..], b"PHASE4 IO OK\0");
    }
}
