//! SPIR-V → WGSL through naga.
//!
//! The last hop of the shader path: the lifter's SPIR-V is parsed and
//! validated by naga (the same engine behind wgpu), then emitted as
//! WGSL the backend compiles into pipelines. Validation failures are
//! precise errors — a lifted shader that does not round-trip is a bug,
//! not a warning.

use thiserror::Error;

/// A translation/validation failure.
#[derive(Debug, Clone, Error)]
pub enum NagaError {
    /// naga could not parse the SPIR-V.
    #[error("naga spv-in: {0}")]
    Parse(String),
    /// naga validated the module and found problems.
    #[error("naga validation: {0}")]
    Validation(String),
    /// naga could not emit WGSL.
    #[error("naga wgsl-out: {0}")]
    Emit(String),
}

/// Validate SPIR-V through naga and return the WGSL text.
pub fn spirv_to_wgsl(spirv: &[u32]) -> Result<String, NagaError> {
    // PDG's lifted modules keep SPIR-V's logical addressing; do not let
    // the front end flip coordinate spaces behind our back.
    let naga_options = naga::front::spv::Options {
        adjust_coordinate_space: false,
        ..Default::default()
    };
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = naga::front::spv::parse_u8_slice(&bytes, &naga_options)
        .map_err(|e| NagaError::Parse(format!("{e:?}")))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| NagaError::Validation(format!("{e:?}")))?;
    let mut flags = naga::back::wgsl::WriterFlags::empty();
    flags.set(naga::back::wgsl::WriterFlags::EXPLICIT_TYPES, true);
    let text = naga::back::wgsl::write_string(&module, &info, flags)
        .map_err(|e| NagaError::Emit(format!("{e:?}")))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcn::decode::{Encoder, decode_program};
    use crate::gcn::enc::{InlineConst, Source};
    use crate::gcn::opcodes as ops;

    #[test]
    fn const_color_shader_roundtrips() {
        let mut e = Encoder::new();
        e.vop1(
            ops::vop1::V_MOV_B32,
            0,
            Source::Const(InlineConst::F32(0.5)),
        );
        e.vop1(
            ops::vop1::V_MOV_B32,
            1,
            Source::Const(InlineConst::F32(1.0)),
        );
        e.vop1(ops::vop1::V_MOV_B32, 2, Source::Literal(0.25f32.to_bits()));
        e.vop1(
            ops::vop1::V_MOV_B32,
            3,
            Source::Const(InlineConst::F32(1.0)),
        );
        e.exp(32, [true; 4], true, [0, 1, 2, 3]);
        e.sopp(ops::sopp::S_ENDPGM, 0);
        let prog = decode_program(&e.build()).unwrap();
        let lifted = super::super::lifter::lift(
            &prog,
            super::super::lifter::Stage::Fragment,
            &super::super::lifter::LiftInputs::default(),
        )
        .unwrap();
        let wgsl = spirv_to_wgsl(&lifted.spirv).expect("valid SPIR-V");
        assert!(wgsl.contains("pdg_main"), "entry point preserved: {wgsl}");
        assert!(wgsl.contains("0.5f"), "color lane preserved: {wgsl}");
        assert!(wgsl.contains("0.25f"), "color lane preserved: {wgsl}");
    }

    #[test]
    fn garbage_is_rejected() {
        let err = spirv_to_wgsl(&[0xDEAD_BEEF, 0]).unwrap_err();
        assert!(matches!(err, NagaError::Parse(_)));
    }
}

#[cfg(test)]
mod spirv_dump {
    #[test]
    fn const_color_module_parses_back() {
        use crate::gcn::decode::{Encoder, decode_program};
        use crate::gcn::enc::Source;
        use crate::gcn::lifter::{LiftInputs, Stage, lift};
        use crate::gcn::opcodes as ops;
        let mut e = Encoder::new();
        e.vop1(
            ops::vop1::V_MOV_B32,
            0,
            Source::Const(crate::gcn::enc::InlineConst::F32(0.5)),
        );
        e.vop1(
            ops::vop1::V_MOV_B32,
            1,
            Source::Const(crate::gcn::enc::InlineConst::F32(1.0)),
        );
        e.vop1(ops::vop1::V_MOV_B32, 2, Source::Literal(0.25f32.to_bits()));
        e.vop1(
            ops::vop1::V_MOV_B32,
            3,
            Source::Const(crate::gcn::enc::InlineConst::F32(1.0)),
        );
        e.exp(32, [true; 4], true, [0, 1, 2, 3]);
        e.sopp(ops::sopp::S_ENDPGM, 0);
        let prog = decode_program(&e.build()).unwrap();
        let lifted = lift(&prog, Stage::Fragment, &LiftInputs::default()).unwrap();
        struct Keep;
        impl rspirv::binary::Consumer for Keep {
            fn initialize(&mut self) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }
            fn finalize(&mut self) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }
            fn consume_header(
                &mut self,
                _header: rspirv::dr::ModuleHeader,
            ) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }
            fn consume_instruction(
                &mut self,
                inst: rspirv::dr::Instruction,
            ) -> rspirv::binary::ParseAction {
                use rspirv::binary::Disassemble;
                println!("{}", inst.disassemble());
                rspirv::binary::ParseAction::Continue
            }
        }
        let mut keep = Keep;
        rspirv::binary::parse_words(&lifted.spirv, &mut keep).unwrap();
        // The assembled module parses back cleanly (roundtrip through
        // rspirv's own decoder).
    }
}
