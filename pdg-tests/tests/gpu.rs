//! GPU subsystem integration tests.
//!
//! Three tiers, matching how the machine is built:
//!
//! 1. **Command processor** (no host GPU): ring MMIO driven by real
//!    emulated x86-64 code — the CPU programs the ring, rings the
//!    doorbell, the host pumps, the CPU then polls the fence in guest
//!    RAM. Verified through `NullGpu`.
//! 2. **Shader path**: hand-assembled GCN programs lifted to SPIR-V and
//!    translated to WGSL through naga (no host GPU needed).
//! 3. **End-to-end rendering** (lavapipe or any adapter): a full draw —
//!    guest PM4 ring with context/SH state, a GCN vertex shader that
//!    fetches vertices through the descriptor-table contract, offscreen
//!    render through wgpu, readback into guest RAM, EOP fence. The test
//!    verifies every pixel. Skips honestly (with a reason printed) when
//!    no adapter exists.

use pdg_cpu::{Interpreter, MemoryAccess, RunOutcome, state::CpuState};
use pdg_gpu::gcn::decode::{Encoder, decode_program};
use pdg_gpu::gcn::enc::Source;
use pdg_gpu::gcn::lifter::{LiftInputs, Stage as LiftStage, VertexFetchDecl, VertexFormat, lift};
use pdg_gpu::gcn::naga_fe::spirv_to_wgsl;
use pdg_gpu::gcn::opcodes as ops;
use pdg_gpu::ir::GpuCommand;
use pdg_gpu::pm4::Header;
use pdg_gpu::pm4::cp::CommandProcessor;
use pdg_gpu::pm4::opcode::ItOpcode;
use pdg_gpu::ring::{GnmRingDevice, regs as ring_regs};
use pdg_memory::bus::{AccessWidth, Bus, MmioDevice};

/// Guest layout for the integration tests.
mod layout {
    /// Ring buffer (PM4 words).
    pub const RING: u64 = 0x10000;
    /// Descriptor table (T# at +0x10 dwords).
    pub const TABLE: u64 = 0x20000;
    /// Vertex data (3 × 16-byte f32x4 records).
    pub const VERTICES: u64 = 0x30000;
    /// 64×64 RGBA8 color target.
    pub const TARGET: u64 = 0x40000;
    /// EOP fence (u64).
    pub const FENCE: u64 = 0x50000;
    /// Vertex shader program.
    pub const VS_PGM: u64 = 0x60000;
    /// GPU ring MMIO window.
    ///
    /// Outside guest RAM on purpose: the bus decodes RAM before MMIO, so
    /// an in-RAM base would silently shadow the device (the CPU-driven
    /// tier-1 test caught exactly that).
    pub const GPU_MMIO: u64 = 0x4000_0000;
}

fn t3(opcode: ItOpcode, payload: &[u32]) -> Vec<u32> {
    let mut v = vec![Header::type3(opcode as u8, payload.len() as u16).raw()];
    v.extend_from_slice(payload);
    v
}

fn set_sh(offset: u32, values: &[u32]) -> Vec<u32> {
    let mut payload = vec![offset];
    payload.extend_from_slice(values);
    t3(ItOpcode::SetShReg, &payload)
}

fn set_ctx(offset: u32, values: &[u32]) -> Vec<u32> {
    let mut payload = vec![offset];
    payload.extend_from_slice(values);
    t3(ItOpcode::SetContextReg, &payload)
}

fn write_words(bus: &Bus, at: u64, words: &[u32]) {
    for (i, w) in words.iter().enumerate() {
        bus.write(AccessWidth::W32, at + (i as u64) * 4, u64::from(*w))
            .expect("guest write");
    }
}

/// The GCN vertex shader used by the e2e draw, per the Phase-3 mini-GNM
/// contract: vertex id in SGPR 6, descriptor table at SGPR pair 0, the
/// `T#` at table + 0x10 dwords, position through `buffer_load_dwordx4`
/// `idxen`, color as constants exported through `param0`.
fn vertex_shader() -> Vec<u32> {
    let mut e = Encoder::new();
    // s_load_dwordx4 s[4:7], s[0:1], 0x10 — the T# (elided by the
    // lifter, resolved by the backend).
    e.smem_load(ops::smem::S_LOAD_DWORDX4, 4, 0, 0x10);
    // Wait for the (elided) scalar read.
    e.sopp(ops::sopp::S_WAITCNT, 0);
    // v0 = vertex id (s6 = the declared vertex-id SGPR).
    e.vop1(ops::vop1::V_MOV_B32, 0, Source::Sgpr(6));
    // v[1:4] = buffer_load_dwordx4 v0, s[4:7], 0 idxen — the position.
    e.mubuf(
        ops::mubuf::BUFFER_LOAD_DWORDX4,
        1,
        0,
        4,
        0xFF,
        0,
        false,
        true,
    );
    e.sopp(ops::sopp::S_WAITCNT, 0);
    // Color: constant (0.5, 1.0, 0.25, 1.0) into v[5:8].
    e.vop1(
        ops::vop1::V_MOV_B32,
        5,
        Source::Const(pdg_gpu::gcn::enc::InlineConst::F32(0.5)),
    );
    e.vop1(
        ops::vop1::V_MOV_B32,
        6,
        Source::Const(pdg_gpu::gcn::enc::InlineConst::F32(1.0)),
    );
    e.vop1(ops::vop1::V_MOV_B32, 7, Source::Literal(0.25f32.to_bits()));
    e.vop1(
        ops::vop1::V_MOV_B32,
        8,
        Source::Const(pdg_gpu::gcn::enc::InlineConst::F32(1.0)),
    );
    // exp pos0 v[1:4] — position.
    e.exp(12, [true; 4], false, [1, 2, 3, 4]);
    // exp param0 v[5:8] done — color varying, end of program.
    e.exp(0, [true; 4], true, [5, 6, 7, 8]);
    e.sopp(ops::sopp::S_ENDPGM, 0);
    e.build()
}

/// The full PM4 stream for the draw.
fn draw_stream() -> Vec<u32> {
    let mut stream = Vec::new();
    // VS program + user data (SH space: PGM_LO_VS at 0x48, USER_DATA at
    // 0x4C — verified kernel formula projections).
    stream.extend(set_sh(0x48, &[(layout::VS_PGM >> 8) as u32, 0]));
    stream.extend(set_sh(
        0x4C,
        &[layout::TABLE as u32, (layout::TABLE >> 32) as u32],
    ));
    // Draw state: TRILIST, linear 64×64 8_8_8_8_UNORM target 0.
    stream.extend(set_ctx(0x2242, &[4])); // VGT_PRIMITIVE_TYPE = TRILIST
    stream.extend(set_ctx(0x0318, &[layout::TARGET as u32])); // CB_COLOR0_BASE
    stream.extend(set_ctx(0x0319, &[0])); // PITCH.TILE_MAX = 0 (64 px)
    stream.extend(set_ctx(0x031A, &[7])); // SLICE.TILE_MAX = 7 (64 rows)
    stream.extend(set_ctx(0x031C, &[10 << 2])); // INFO: FORMAT=8_8_8_8, UNORM
    stream.extend(set_ctx(0x031D, &[0])); // ATTRIB: TILE_MODE_INDEX = LINEAR
    // The draw: three vertices.
    stream.extend(t3(ItOpcode::DrawIndexAuto, &[3, 0]));
    // EOP fence: 64-bit immediate into guest RAM.
    stream.extend(t3(
        ItOpcode::ReleaseMem,
        &[0, 2 << 29, layout::FENCE as u32, 0, 0xDEF0, 0x1234_5678],
    ));
    stream
}

/// Set up guest memory: descriptors, vertices, shader.
fn setup_guest(bus: &Bus) {
    // Vertex data: a fullscreen triangle in NDC (three f32x4 records).
    let verts: [f32; 12] = [
        -1.0, -1.0, 0.0, 1.0, // v0
        3.0, -1.0, 0.0, 1.0, // v1
        -1.0, 3.0, 0.0, 1.0, // v2
    ];
    for (i, f) in verts.iter().enumerate() {
        bus.write(
            AccessWidth::W32,
            layout::VERTICES + (i as u64) * 4,
            f.to_bits() as u64,
        )
        .expect("vertex write");
    }
    // T# at table + 0x10 dwords: buffer resource (public layout):
    // dword0 = base[31:0], dword1 = base[39:32] | stride<<16,
    // dword2 = num_records, dword3 = format word (data_format<<15 |
    // num_format<<12).
    let t_at = layout::TABLE + 0x10 * 4;
    write_words(
        bus,
        t_at,
        &[
            layout::VERTICES as u32,
            ((layout::VERTICES >> 32) as u32 & 0xFF) | (16 << 16),
            3,
            (14 << 15) | (7 << 12),
        ],
    );
    // The shader program.
    write_words(bus, layout::VS_PGM, &vertex_shader());
}

// ---------------------------------------------------------------------------
// Tier 1: command processor driven by emulated CPU code.
// ---------------------------------------------------------------------------

/// Assemble the tiny x86-64 driver: program the ring, ring the doorbell,
/// then spin on the fence.
fn ring_driver_code() -> Vec<u8> {
    // A minimal hand-assembled sequence using port-style MMIO through
    // the CPU's flat memory:
    //   mov rdi, GPU_MMIO
    //   mov [rdi+RING_BASE], RING
    //   mov [rdi+RING_SIZE], len
    //   mov [rdi+RING_WPTR], len
    //   mov [rdi+RING_DOORBELL], 1
    //   ; spin: mov eax, [FENCE]; test eax,eax; jz spin
    let mut code = Vec::new();
    let push = |code: &mut Vec<u8>, b: &[u8]| code.extend_from_slice(b);
    // mov rdi, imm64 (48 BF + 8 bytes)
    push(&mut code, &[0x48, 0xBF]);
    code.extend_from_slice(&layout::GPU_MMIO.to_le_bytes());
    // mov rax, imm64 (48 B8)
    push(&mut code, &[0x48, 0xB8]);
    code.extend_from_slice(&layout::RING.to_le_bytes());
    // mov [rdi+0], rax
    push(&mut code, &[0x48, 0x89, 0x07]);
    // mov eax, len (B8 + 4)
    push(&mut code, &[0xB8]);
    let len: u32 = 0; // patched below by the caller
    code.extend_from_slice(&len.to_le_bytes());
    // mov [rdi+4], eax
    push(&mut code, &[0x89, 0x47, 0x04]);
    // mov [rdi+8], eax
    push(&mut code, &[0x89, 0x47, 0x08]);
    // mov dword [rdi+0xC], 1 (C7 47 0C 01 00 00 00)
    push(&mut code, &[0xC7, 0x47, 0x0C, 0x01, 0x00, 0x00, 0x00]);
    // spin: mov eax, [FENCE] (A1 + 8 bytes, 64-bit: 48 A1)
    let spin_at = code.len();
    push(&mut code, &[0x48, 0xA1]);
    code.extend_from_slice(&layout::FENCE.to_le_bytes());
    // test rax, rax (48 85 C0)
    push(&mut code, &[0x48, 0x85, 0xC0]);
    // jz spin (74 + rel8) — back to the load
    let here = code.len() + 2;
    let rel = (spin_at as i32 - here as i32) as i8 as u8;
    push(&mut code, &[0x74, rel]);
    // hlt — the driver parks once the fence is visible (returning is
    // pointless: the initial stack holds no return address).
    push(&mut code, &[0xF4]);
    code
}

/// Patch the ring word count into the driver.
fn ring_driver(len_words: u32) -> Vec<u8> {
    let mut code = ring_driver_code();
    let len_pos = code
        .windows(4)
        .position(|w| w[0] == 0xB8 && w[1] == 0 && w[2] == 0 && w[3] == 0)
        .map(|p| p + 1)
        .expect("length immediate");
    code[len_pos..len_pos + 4].copy_from_slice(&len_words.to_le_bytes());
    code
}

#[test]
fn cpu_drives_ring_and_polls_fence() {
    let mut bus = Bus::new(1 << 20).expect("bus");
    let stream = draw_stream();
    write_words(&bus, layout::RING, &stream);
    setup_guest(&bus);

    let ring = GnmRingDevice::new(CommandProcessor::new(), Box::new(pdg_gpu::NullGpu)).shared();
    bus.attach_mmio(layout::GPU_MMIO, ring_regs::FOOTPRINT, ring.clone());

    // Place the driver in guest RAM and run it on the interpreter.
    let driver = ring_driver(stream.len() as u32);
    for (i, b) in driver.iter().enumerate() {
        bus.write(AccessWidth::W8, 0x1000 + i as u64, u64::from(*b))
            .expect("code write");
    }
    // The interpreter drives the CPU; PDG's flat guest memory maps the
    // whole RAM so the driver's absolute addresses work as-is.
    struct BusMem<'a>(&'a Bus);
    impl MemoryAccess for BusMem<'_> {
        fn read(&mut self, width: u8, addr: u64) -> Result<u64, pdg_cpu::AccessError> {
            let w = match width {
                1 => AccessWidth::W8,
                2 => AccessWidth::W16,
                4 => AccessWidth::W32,
                _ => AccessWidth::W64,
            };
            self.0
                .read(w, addr)
                .map_err(|_| pdg_cpu::AccessError::PageFault { addr, code: 0 })
        }
        fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), pdg_cpu::AccessError> {
            let w = match width {
                1 => AccessWidth::W8,
                2 => AccessWidth::W16,
                4 => AccessWidth::W32,
                _ => AccessWidth::W64,
            };
            self.0
                .write(w, addr, value)
                .map_err(|_| pdg_cpu::AccessError::PageFault { addr, code: 0 })
        }
        fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, pdg_cpu::AccessError> {
            for (i, b) in out.iter_mut().enumerate() {
                *b = self.0.read(AccessWidth::W8, addr + i as u64).unwrap_or(0) as u8;
            }
            Ok(out.len())
        }
        fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), pdg_cpu::AccessError> {
            for (i, b) in out.iter_mut().enumerate() {
                *b = self.0.read(AccessWidth::W8, addr + i as u64).unwrap_or(0) as u8;
            }
            Ok(())
        }
        fn write_buf(&mut self, addr: u64, data: &[u8]) -> Result<(), pdg_cpu::AccessError> {
            for (i, b) in data.iter().enumerate() {
                let _ = self
                    .0
                    .write(AccessWidth::W8, addr + i as u64, u64::from(*b));
            }
            Ok(())
        }
    }
    let mut interp = Interpreter::new(CpuState::new(0x1000, 0x80000));
    // Run until the CPU hits the fence spin (budget-limited), pump, then
    // let it finish the poll loop.
    let mut outcome = interp.run(&mut BusMem(&bus), 200);
    let mut total = 200;
    while bus.read(AccessWidth::W32, layout::FENCE).unwrap() == 0 {
        assert!(ring.has_pending(), "doorbell must be pending");
        ring.pump(&bus).expect("pump");
        outcome = interp.run(&mut BusMem(&bus), 200);
        total += 200;
        if total > 10_000 {
            panic!("CPU never observed the fence");
        }
    }
    // The driver parks (`hlt`) once the fence is visible — the final
    // outcome must be a clean halt, never a fault.
    assert!(
        matches!(outcome, RunOutcome::Halted { .. }),
        "driver did not halt cleanly: {outcome:?}"
    );
    // The NullGpu records no pixels but writes the fence.
    assert_eq!(bus.read(AccessWidth::W32, layout::FENCE).unwrap(), 0xDEF0);
    assert_eq!(
        bus.read(AccessWidth::W32, layout::FENCE + 4).unwrap(),
        0x1234_5678
    );
}

// ---------------------------------------------------------------------------
// Tier 2: shader path (no host GPU).
// ---------------------------------------------------------------------------

#[test]
fn vertex_shader_lifts_to_wgsl() {
    let isa = vertex_shader();
    let decoded = decode_program(&isa).expect("decode");
    let inputs = LiftInputs {
        vertex_id_sgpr: Some(6),
        descriptor_tables: vec![0],
        vertex_fetches: vec![VertexFetchDecl {
            table_sbase: 0,
            sload_offset_dwords: 0x10,
            format: VertexFormat::F32x4,
        }],
        ps_inputs: vec![(0, 2, 4)],
    };
    let lifted = lift(&decoded, LiftStage::Vertex, &inputs).expect("lift");
    assert_eq!(lifted.manifest.fetches.len(), 1);
    let wgsl = spirv_to_wgsl(&lifted.spirv).expect("wgsl");
    assert!(wgsl.contains("pdg_main"), "entry preserved: {wgsl}");
    assert!(wgsl.contains("vertex_index"), "vertex id wired: {wgsl}");
}

// ---------------------------------------------------------------------------
// Tier 3: end-to-end rendering (adapter required).
// ---------------------------------------------------------------------------

#[test]
fn end_to_end_triangle_renders_into_guest_ram() {
    let Ok(gpu) = pdg_gpu::WgpuBackend::new() else {
        eprintln!("SKIP: no host GPU adapter (set VK_ICD_FILENAMES to lavapipe)");
        return;
    };
    let mut bus = Bus::new(1 << 20).expect("bus");
    let stream = draw_stream();
    write_words(&bus, layout::RING, &stream);
    setup_guest(&bus);

    let ring = GnmRingDevice::new(CommandProcessor::new(), Box::new(gpu)).shared();
    bus.attach_mmio(layout::GPU_MMIO, ring_regs::FOOTPRINT, ring.clone());
    // Drive the ring from the host side (the CPU-driven variant is
    // covered by the tier-1 test).
    ring.write(ring_regs::RING_BASE, AccessWidth::W32, layout::RING);
    ring.write(ring_regs::RING_SIZE, AccessWidth::W32, stream.len() as u64);
    ring.write(ring_regs::RING_WPTR, AccessWidth::W32, stream.len() as u64);
    ring.write(ring_regs::RING_DOORBELL, AccessWidth::W32, 1);
    ring.pump(&bus).expect("render");

    // The fence landed.
    assert_eq!(bus.read(AccessWidth::W32, layout::FENCE).unwrap(), 0xDEF0);
    assert_eq!(
        bus.read(AccessWidth::W32, layout::FENCE + 4).unwrap(),
        0x1234_5678
    );

    // Every pixel is the fullscreen triangle's color: (0.5, 1.0, 0.25, 1)
    // in RGBA8 = (128, 255, 64, 255) with one-LSB tolerance.
    let mut mismatches = 0usize;
    for y in 0..64u64 {
        for x in 0..64u64 {
            let px = layout::TARGET + (y * 64 + x) * 4;
            let r = bus.read(AccessWidth::W8, px).unwrap() as i32;
            let g = bus.read(AccessWidth::W8, px + 1).unwrap() as i32;
            let b = bus.read(AccessWidth::W8, px + 2).unwrap() as i32;
            let a = bus.read(AccessWidth::W8, px + 3).unwrap() as i32;
            let ok = (r - 128).abs() <= 1
                && (g - 255).abs() <= 1
                && (b - 64).abs() <= 1
                && (a - 255).abs() <= 1;
            if !ok {
                mismatches += 1;
                if mismatches < 4 {
                    eprintln!("pixel ({x},{y}) = ({r},{g},{b},{a})");
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} pixels wrong out of 4096");
    let _ = GpuCommand::Nop;
}
