//! Dump the lifted WGSL of the Phase-3 test vertex shader (dev tool).

use pdg_gpu::gcn::decode::{Encoder, decode_program};
use pdg_gpu::gcn::enc::Source;
use pdg_gpu::gcn::lifter::{LiftInputs, Stage as LiftStage, VertexFetchDecl, VertexFormat, lift};
use pdg_gpu::gcn::naga_fe::spirv_to_wgsl;
use pdg_gpu::gcn::opcodes as ops;

fn main() {
    let mut e = Encoder::new();
    e.smem_load(ops::smem::S_LOAD_DWORDX4, 4, 0, 0x10);
    e.sopp(ops::sopp::S_WAITCNT, 0);
    e.vop1(ops::vop1::V_MOV_B32, 0, Source::Sgpr(6));
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
    e.exp(12, [true; 4], false, [1, 2, 3, 4]);
    e.exp(0, [true; 4], true, [5, 6, 7, 8]);
    e.sopp(ops::sopp::S_ENDPGM, 0);
    let isa = e.build();
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
    let wgsl = spirv_to_wgsl(&lifted.spirv).expect("wgsl");
    println!("{wgsl}");
}
