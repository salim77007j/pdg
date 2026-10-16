//! GCN → SPIR-V shader lifter.
//!
//! Compiles the decoded instruction subset to SPIR-V modules (through
//! `rspirv`) that naga can translate to WGSL for the wgpu backend.
//!
//! ## Scope (honest)
//!
//! The Phase-3 lifter compiles **straight-line shaders**: the SALU/VALU
//! arithmetic subset, the SMEM descriptor-table load (elided — see the
//! contract below), `MUBUF` vertex fetches with `idxen`, and `EXP`
//! exports. Branches (`s_branch`, `s_cbranch_*`) are refused with a
//! precise error; loop/condition lifting is Phase-4 work (it needs
//! per-block SPIR-V construction and is tracked in the report).
//!
//! ## The descriptor-resolution contract
//!
//! GCN shaders address guest memory through *buffer resource
//! descriptors* (`T#`) that the driver loads into SGPR quads from
//! descriptor tables. WGSL has no raw pointer arithmetic, so the lifter
//! and the backend split the job the way modern recompilers do:
//!
//! * The lifter tracks SGPR dataflow. An `s_load_dwordx4` from a
//!   declared descriptor-table pair, whose destination feeds an `MUBUF`
//!   `srsrc`, becomes a **manifest entry**: (table pair, s_load offset,
//!   binding slot, declared format). The load itself is elided — its
//!   value is never usable as data in the subset.
//! * The backend reads the same guest memory the driver would (through
//!   the CP's user-data registers), resolves the `T#`, copies the
//!   referenced guest range into a wgpu buffer at the manifest slot,
//!   and provides the record `stride` through a uniform.
//! * The lifted shader computes every shader-visible address exactly as
//!   the guest did: `record = vb[index * stride/16 + imm/16]`, `index`
//!   coming from the declared vertex-id SGPR (mapped to
//!   `@builtin(vertex_index)`).
//!
//! The backend **verifies** at draw time that the resolved `T#` matches
//! the fetch's declared specialization; mismatches are precise errors.
//!
//! ## Registers
//!
//! SGPRs and VGPRs become typed SSA values (`f32` / `u32` / `i32`) with
//! lazy bitcasts. Reading an uninitialized register (other than the
//! declared vertex-id SGPR) is a precise error — the lifter never
//! invents zero-initialization.

use super::decode::{Inst, SmemOffset};
use super::enc::{InlineConst, Source};
use super::opcodes;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, FunctionControl, MemoryModel, StorageClass,
    Word,
};
use thiserror::Error;

/// Shader stage being lifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Vertex stage.
    Vertex,
    /// Fragment (pixel) stage.
    Fragment,
}

/// A declared vertex-buffer format specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    /// Four f32s per record (`32_32_32_32_FLOAT`).
    F32x4,
    /// Four u32s per record (`32_32_32_32_UINT`).
    U32x4,
    /// Four unorm u8s per record (`8_8_8_8_UNORM`).
    Unorm8x4,
}

impl VertexFormat {
    /// Record size in bytes (all declared formats are 16).
    pub const RECORD_BYTES: u32 = 16;
}

/// A declared vertex fetch fed by a descriptor-table load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFetchDecl {
    /// SGPR pair that holds the descriptor-table pointer.
    pub table_sbase: u8,
    /// `s_load_dwordx4` offset (dwords) that fetches the `T#`.
    pub sload_offset_dwords: u32,
    /// Format specialization (verified by the backend against the `T#`).
    pub format: VertexFormat,
}

/// Lifter inputs: the stage contract the backend honors.
#[derive(Debug, Clone, Default)]
pub struct LiftInputs {
    /// SGPR that receives the vertex index (VS only); the backend maps
    /// it to `@builtin(vertex_index)`.
    pub vertex_id_sgpr: Option<u8>,
    /// SGPR pairs readable as descriptor tables; their values are
    /// resolved host-side, so loads from them that feed fetches are
    /// elided.
    pub descriptor_tables: Vec<u8>,
    /// Vertex fetch declarations (VS only).
    pub vertex_fetches: Vec<VertexFetchDecl>,
    /// PS input map: `(location, first VGPR, components)` — the hardware
    /// pre-interpolated values the shader expects in those VGPRs.
    pub ps_inputs: Vec<(u32, u8, u8)>,
}

/// A resolved fetch in the compiled shader's manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchManifestEntry {
    /// The declared fetch.
    pub decl: VertexFetchDecl,
    /// Binding slot the backend binds the resolved buffer at.
    pub slot: u32,
}

/// What the backend must bind for a lifted shader.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    /// Vertex fetches; binding = `1 + index`, uniform word pair = `2 * index`.
    pub fetches: Vec<FetchManifestEntry>,
}

/// The lift result.
#[derive(Debug)]
pub struct LiftedShader {
    /// SPIR-V words.
    pub spirv: Vec<u32>,
    /// The manifest.
    pub manifest: Manifest,
}

/// Lifter failure.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum LiftError {
    /// An instruction outside the supported subset.
    #[error("unsupported instruction {name} at index {index}: {why}")]
    Unsupported {
        /// Instruction description.
        name: String,
        /// Instruction index in the program.
        index: usize,
        /// Why it is not liftable.
        why: &'static str,
    },
    /// A register was read before any write reached it.
    #[error("{reg} read before write at index {index}")]
    Uninitialized {
        /// Register description.
        reg: String,
        /// Instruction index in the program.
        index: usize,
    },
    /// The program is malformed for the stage contract.
    #[error("bad shader: {why}")]
    Bad {
        /// What is wrong.
        why: String,
    },
}

/// Binding of the fetch-parameter uniform buffer (`[stride, count]`
/// pairs, two words per fetch).
pub const FETCH_UNIFORM_BINDING: u32 = 0;
/// First binding used for vertex-buffer storage.
pub const FETCH_STORAGE_BASE: u32 = 1;

/// Value type carried in an SGPR/VGPR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    /// IEEE binary32.
    F32,
    /// Unsigned 32-bit.
    U32,
    /// Signed 32-bit.
    I32,
}

/// A typed SSA value.
#[derive(Debug, Clone, Copy)]
struct Val {
    id: Word,
    ty: Ty,
}

type R<T> = Result<T, LiftError>;

/// Scalar comparison kinds (SCC producers).
enum IcmpOp {
    /// ==
    Eq,
    /// !=
    Ne,
    /// > (unsigned)
    GtU,
    /// >= (unsigned)
    GeU,
    /// < (unsigned)
    LtU,
    /// <= (unsigned)
    LeU,
}

struct FnCtx<'a> {
    b: Builder,
    sgpr: Vec<Option<Val>>,
    vgpr: Vec<Option<Val>>,
    /// SGPR indices holding resolved descriptors (fetch index).
    descriptor_regs: Vec<(u8, usize)>,
    scc: Option<Word>,
    t_f32: Word,
    t_u32: Word,
    t_i32: Word,
    t_v4f: Word,
    t_v4u: Word,
    t_bool: Word,
    glsl: Word,
    out_var: Word,
    pos_member: usize,
    color_member: usize,
    vertex_id: Option<Word>,
    fetch_vars: Vec<Word>,
    fetch_uniform: Option<Word>,
    inputs: &'a LiftInputs,
    fetches: Vec<FetchManifestEntry>,
}

macro_rules! op {
    ($self:expr, $call:expr) => {
        $call.map_err(|e| LiftError::Bad {
            why: format!("rspirv: {e:?}"),
        })
    };
}

impl FnCtx<'_> {
    fn type_of(&self, ty: Ty) -> Word {
        match ty {
            Ty::F32 => self.t_f32,
            Ty::U32 => self.t_u32,
            Ty::I32 => self.t_i32,
        }
    }

    fn const_word(&mut self, ty: Ty, bits: u32) -> Val {
        let t = self.type_of(ty);
        let id = self.b.constant_bit32(t, bits);
        Val { id, ty }
    }

    fn inline_const(&mut self, c: InlineConst, ty: Ty) -> Val {
        match c {
            InlineConst::Int(v) => self.const_word(ty, v as u32),
            InlineConst::F32(v) => self.const_word(ty, v.to_bits()),
            InlineConst::F64(v) => self.const_word(ty, (v as f32).to_bits()),
        }
    }

    fn coerce(&mut self, v: Val, to: Ty) -> R<Val> {
        if v.ty == to {
            return Ok(v);
        }
        let target = self.type_of(to);
        let id = op!(self, self.b.bitcast(target, None, v.id))?;
        Ok(Val { id, ty: to })
    }

    fn vgpr_value(&self, v: u8, index: usize) -> R<Val> {
        self.vgpr
            .get(usize::from(v))
            .and_then(|x| *x)
            .ok_or_else(|| LiftError::Uninitialized {
                reg: format!("v{v}"),
                index,
            })
    }

    fn sgpr_value(&self, v: u8, index: usize) -> R<Val> {
        if self.inputs.vertex_id_sgpr == Some(v) {
            return Ok(Val {
                id: self.vertex_id.expect("vertex id set for VS"),
                ty: Ty::U32,
            });
        }
        if self.descriptor_regs.iter().any(|(r, _)| *r == v) {
            return Err(LiftError::Unsupported {
                name: format!("s{v} (descriptor)"),
                index,
                why: "T# descriptor registers cannot be read as data in the lifted subset",
            });
        }
        self.sgpr
            .get(usize::from(v))
            .and_then(|x| *x)
            .ok_or_else(|| LiftError::Uninitialized {
                reg: format!("s{v}"),
                index,
            })
    }

    fn src(&mut self, s: &Source, index: usize, ty: Ty) -> R<Val> {
        match s {
            Source::Vgpr(v) => {
                let v = self.vgpr_value(*v, index)?;
                self.coerce(v, ty)
            }
            Source::Sgpr(v) => {
                let v = self.sgpr_value(*v, index)?;
                self.coerce(v, ty)
            }
            Source::Const(c) => Ok(self.inline_const(*c, ty)),
            Source::Literal(v) => Ok(self.const_word(ty, *v)),
            Source::Special(sp) => Err(LiftError::Unsupported {
                name: format!("special source {sp}"),
                index,
                why: "special registers are not liftable yet",
            }),
            Source::Ttmp(t) => Err(LiftError::Unsupported {
                name: format!("ttmp{t}"),
                index,
                why: "trap registers are not liftable",
            }),
        }
    }

    fn set_sgpr(&mut self, v: u8, value: Val) {
        let idx = usize::from(v);
        if self.sgpr.len() <= idx {
            self.sgpr.resize(idx + 1, None);
        }
        self.sgpr[idx] = Some(value);
    }

    fn set_vgpr(&mut self, v: u8, value: Val) {
        let idx = usize::from(v);
        if self.vgpr.len() <= idx {
            self.vgpr.resize(idx + 1, None);
        }
        self.vgpr[idx] = Some(value);
    }

    fn mark_descriptor(&mut self, v: u8, fetch_idx: usize) {
        self.descriptor_regs.push((v, fetch_idx));
    }

    fn descriptor_at(&self, v: u8) -> Option<usize> {
        self.descriptor_regs
            .iter()
            .find(|(r, _)| *r == v)
            .map(|(_, i)| *i)
    }

    // Typed op helpers (all map builder errors into LiftError).

    fn iadd(&mut self, a: Val, c: Val) -> R<Val> {
        let a = self.coerce(a, Ty::U32)?;
        let c = self.coerce(c, Ty::U32)?;
        let id = op!(self, self.b.i_add(self.t_u32, None, a.id, c.id))?;
        Ok(Val { id, ty: Ty::U32 })
    }

    fn fadd(&mut self, a: Val, c: Val) -> R<Val> {
        let a = self.coerce(a, Ty::F32)?;
        let c = self.coerce(c, Ty::F32)?;
        let id = op!(self, self.b.f_add(self.t_f32, None, a.id, c.id))?;
        Ok(Val { id, ty: Ty::F32 })
    }

    fn fmul(&mut self, a: Val, c: Val) -> R<Val> {
        let a = self.coerce(a, Ty::F32)?;
        let c = self.coerce(c, Ty::F32)?;
        let id = op!(self, self.b.f_mul(self.t_f32, None, a.id, c.id))?;
        Ok(Val { id, ty: Ty::F32 })
    }

    fn icmp(&mut self, cmp: IcmpOp, a: Val, c: Val) -> R<Word> {
        let a = self.coerce(a, Ty::U32)?;
        let c = self.coerce(c, Ty::U32)?;
        let r = match cmp {
            IcmpOp::Eq => self.b.i_equal(self.t_bool, None, a.id, c.id),
            IcmpOp::Ne => self.b.i_not_equal(self.t_bool, None, a.id, c.id),
            IcmpOp::GtU => self.b.u_greater_than(self.t_bool, None, a.id, c.id),
            IcmpOp::GeU => self.b.u_greater_than_equal(self.t_bool, None, a.id, c.id),
            IcmpOp::LtU => self.b.u_less_than(self.t_bool, None, a.id, c.id),
            IcmpOp::LeU => self.b.u_less_than_equal(self.t_bool, None, a.id, c.id),
        };
        op!(self, r)
    }

    fn x_of(&mut self, vec4: Word, lane: u32) -> Val {
        let id = self
            .b
            .composite_extract(self.t_f32, None, vec4, vec![lane])
            .expect("extract in-block");
        Val { id, ty: Ty::F32 }
    }

    /// Load `stride` for fetch `i` from the fetch buffer (word `2*i`).
    fn fetch_stride(&mut self, i: usize) -> R<Word> {
        let var = self.fetch_uniform.expect("fetch uniform exists");
        let t_ptr = self
            .b
            .type_pointer(None, StorageClass::StorageBuffer, self.t_u32);
        let word = self.b.constant_bit32(self.t_u32, (i * 2) as u32);
        let ptr = op!(self, self.b.access_chain(t_ptr, None, var, vec![word]))?;
        op!(self, self.b.load(self.t_u32, None, ptr, None, vec![]))
    }

    fn umin(&mut self, a: Val, c: Val) -> R<Val> {
        let a = self.coerce(a, Ty::U32)?;
        let c = self.coerce(c, Ty::U32)?;
        let id = op!(
            self,
            self.b.ext_inst(
                self.t_u32,
                None,
                self.glsl,
                rspirv::spirv::GLOp::UMin as u32,
                vec![Operand::IdRef(a.id), Operand::IdRef(c.id)],
            )
        )?;
        Ok(Val { id, ty: Ty::U32 })
    }

    fn umax(&mut self, a: Val, c: Val) -> R<Val> {
        let a = self.coerce(a, Ty::U32)?;
        let c = self.coerce(c, Ty::U32)?;
        let id = op!(
            self,
            self.b.ext_inst(
                self.t_u32,
                None,
                self.glsl,
                rspirv::spirv::GLOp::UMax as u32,
                vec![Operand::IdRef(a.id), Operand::IdRef(c.id)],
            )
        )?;
        Ok(Val { id, ty: Ty::U32 })
    }
}

/// Lift a decoded program to SPIR-V.
pub fn lift(program: &[(usize, Inst)], stage: Stage, inputs: &LiftInputs) -> R<LiftedShader> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let glsl = b.ext_inst_import("GLSL.std.450");

    // Types.
    let t_void = b.type_void();
    let t_f32 = b.type_float(32);
    let t_u32 = b.type_int(32, 0);
    let t_i32 = b.type_int(32, 1);
    let t_bool = b.type_bool();
    let t_v4f = b.type_vector(t_f32, 4);
    let t_v4u = b.type_vector(t_u32, 4);

    // Output struct: vertex gets [position, color]; fragment just [color].
    let (t_out, pos_member, color_member) = match stage {
        Stage::Vertex => {
            let t = b.type_struct(vec![t_v4f, t_v4f]);
            b.member_decorate(
                t,
                0,
                Decoration::BuiltIn,
                vec![Operand::BuiltIn(BuiltIn::Position)],
            );
            b.member_decorate(t, 1, Decoration::Location, vec![Operand::LiteralBit32(0)]);
            (t, 0usize, 1usize)
        }
        Stage::Fragment => {
            let t = b.type_struct(vec![t_v4f]);
            b.member_decorate(t, 0, Decoration::Location, vec![Operand::LiteralBit32(0)]);
            (t, usize::MAX, 0usize)
        }
    };

    // Fetch manifest and bindings.
    let mut fetches = Vec::new();
    for (i, decl) in inputs.vertex_fetches.iter().enumerate() {
        fetches.push(FetchManifestEntry {
            decl: *decl,
            slot: FETCH_STORAGE_BASE + i as u32,
        });
    }

    // Fetch parameter buffer: read-only storage `array<u32>` of
    // `[stride, count]` per fetch. Storage (not Uniform) on purpose:
    // WGSL requires 16-byte array strides in the `uniform` address space,
    // and the packed u32 pairs would fail naga validation there
    // (storage keeps the 4-byte stride and the same load code).
    let fetch_uniform = if !fetches.is_empty() {
        let count = (fetches.len() * 2) as u32;
        let len = b.constant_bit32(t_u32, count);
        let t_arr = b.type_array(t_u32, len);
        b.decorate(t_arr, Decoration::Block, vec![]);
        let t_ptr = b.type_pointer(None, StorageClass::StorageBuffer, t_arr);
        let var = b.variable(t_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(var, Decoration::NonWritable, vec![]);
        b.decorate(
            var,
            Decoration::DescriptorSet,
            vec![Operand::LiteralBit32(0)],
        );
        b.decorate(
            var,
            Decoration::Binding,
            vec![Operand::LiteralBit32(FETCH_UNIFORM_BINDING)],
        );
        Some(var)
    } else {
        None
    };

    // Vertex-buffer storage bindings: Block { array<vec4<u32>> }.
    let mut fetch_vars = Vec::new();
    for entry in &fetches {
        let t_arr = b.type_runtime_array(t_v4u);
        let t_strct = b.type_struct(vec![t_arr]);
        b.decorate(t_strct, Decoration::Block, vec![]);
        b.member_decorate(
            t_strct,
            0,
            Decoration::Offset,
            vec![Operand::LiteralBit32(0)],
        );
        let t_ptr = b.type_pointer(None, StorageClass::StorageBuffer, t_strct);
        let var = b.variable(t_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(var, Decoration::NonWritable, vec![]);
        b.decorate(
            var,
            Decoration::DescriptorSet,
            vec![Operand::LiteralBit32(0)],
        );
        b.decorate(
            var,
            Decoration::Binding,
            vec![Operand::LiteralBit32(entry.slot)],
        );
        fetch_vars.push(var);
    }

    // Vertex input: @builtin(vertex_index).
    let vid_var = if stage == Stage::Vertex {
        let t_ptr = b.type_pointer(None, StorageClass::Input, t_u32);
        let var = b.variable(t_ptr, None, StorageClass::Input, None);
        b.decorate(
            var,
            Decoration::BuiltIn,
            vec![Operand::BuiltIn(BuiltIn::VertexIndex)],
        );
        Some(var)
    } else {
        None
    };

    // Fragment inputs: one vec4<f32> per declared PS input. Fragment
    // stage only: a vertex shader *produces* interpolants (the `exp
    // paramN` path writes the output members), so the vertex lift must
    // not declare an input side — wgpu rejects inputs the pipeline's
    // vertex-buffer state cannot provide.
    let mut ps_in_vars = Vec::new();
    if stage == Stage::Fragment {
        for (loc, _vgpr, _comps) in &inputs.ps_inputs {
            let t_ptr = b.type_pointer(None, StorageClass::Input, t_v4f);
            let var = b.variable(t_ptr, None, StorageClass::Input, None);
            b.decorate(var, Decoration::Location, vec![Operand::LiteralBit32(*loc)]);
            ps_in_vars.push(var);
        }
    }

    // Output variable (module level: non-Function storage classes must
    // not be declared inside a function body).
    let t_out_ptr = b.type_pointer(None, StorageClass::Output, t_out);
    let out_var = b.variable(t_out_ptr, None, StorageClass::Output, None);

    // Function: `void pdg_main(void)`.
    let t_fn = b.type_function(t_void, Vec::<Word>::new());
    let entry = b
        .begin_function(t_void, None, FunctionControl::NONE, t_fn)
        .map_err(|e| LiftError::Bad {
            why: format!("{e:?}"),
        })?;
    b.begin_block(None).map_err(|e| LiftError::Bad {
        why: format!("{e:?}"),
    })?;

    let vertex_id = match vid_var {
        Some(var) => Some(
            b.load(t_u32, None, var, None, vec![])
                .map_err(|e| LiftError::Bad {
                    why: format!("{e:?}"),
                })?,
        ),
        None => None,
    };

    let mut ctx = FnCtx {
        b,
        sgpr: Vec::new(),
        vgpr: Vec::new(),
        descriptor_regs: Vec::new(),
        scc: None,
        t_f32,
        t_u32,
        t_i32,
        t_v4f,
        t_v4u,
        t_bool,
        glsl,
        out_var,
        pos_member,
        color_member,
        vertex_id,
        fetch_vars,
        fetch_uniform,
        inputs,
        fetches: fetches.clone(),
    };

    // Pre-load PS inputs into their declared VGPRs (four components).
    // Fragment stage only — see the `ps_in_vars` construction above.
    if stage == Stage::Fragment {
        for (i, (_loc, vgpr, comps)) in inputs.ps_inputs.iter().enumerate() {
            if *comps != 4 {
                return Err(LiftError::Bad {
                    why: format!("PS input with {comps} components (only 4 is supported)"),
                });
            }
            let var = ps_in_vars[i];
            let v = ctx
                .b
                .load(t_v4f, None, var, None, vec![])
                .map_err(|e| LiftError::Bad {
                    why: format!("{e:?}"),
                })?;
            ctx.set_vgpr(*vgpr, Val { id: v, ty: Ty::F32 });
            let x1 = ctx.x_of(v, 1);
            let x2 = ctx.x_of(v, 2);
            let x3 = ctx.x_of(v, 3);
            ctx.set_vgpr(vgpr + 1, x1);
            ctx.set_vgpr(vgpr + 2, x2);
            ctx.set_vgpr(vgpr + 3, x3);
        }
    }

    // Default-initialize unwritten outputs so the module stays valid
    // when a shader exports only part of the interface (the EXP path
    // writes over what it produced; the guest sees its own exports).
    let zero_f = ctx.b.constant_bit32(t_f32, 0.0f32.to_bits());
    let zero_v4 = ctx.b.constant_composite(t_v4f, vec![zero_f; 4]);
    let t_member_ptr = ctx.b.type_pointer(None, StorageClass::Output, t_v4f);
    if pos_member != usize::MAX {
        let idx = ctx.b.constant_bit32(t_u32, pos_member as u32);
        let p_ptr = op!(
            ctx,
            ctx.b
                .access_chain(t_member_ptr, None, ctx.out_var, vec![idx])
        )?;
        op!(ctx, ctx.b.store(p_ptr, zero_v4, None, vec![]))?;
    }
    let c_idx = ctx.b.constant_bit32(t_u32, color_member as u32);
    let c_ptr = op!(
        ctx,
        ctx.b
            .access_chain(t_member_ptr, None, ctx.out_var, vec![c_idx])
    )?;
    op!(ctx, ctx.b.store(c_ptr, zero_v4, None, vec![]))?;

    // Compile the straight-line program.
    for (index, inst) in program {
        compile_inst(&mut ctx, inst, *index, stage)?;
    }

    ctx.b.ret().map_err(|e| LiftError::Bad {
        why: format!("{e:?}"),
    })?;
    ctx.b.end_function().map_err(|e| LiftError::Bad {
        why: format!("{e:?}"),
    })?;

    let mut interface = vec![out_var];
    if let Some(v) = vid_var {
        interface.push(v);
    }
    interface.extend(ps_in_vars);

    let exec_model = match stage {
        Stage::Vertex => rspirv::spirv::ExecutionModel::Vertex,
        Stage::Fragment => rspirv::spirv::ExecutionModel::Fragment,
    };
    ctx.b.entry_point(exec_model, entry, "pdg_main", interface);

    let module = ctx.b.module();
    let spirv = rspirv::binary::Assemble::assemble(&module);

    Ok(LiftedShader {
        spirv,
        manifest: Manifest { fetches },
    })
}

/// Compile one instruction into the entry block.
fn compile_inst(ctx: &mut FnCtx<'_>, inst: &Inst, index: usize, stage: Stage) -> R<()> {
    match inst {
        Inst::Sop1 { op, sdst, src0 } if *op == opcodes::sop1::S_MOV_B32 => {
            let Source::Sgpr(d) = sdst else {
                return Err(LiftError::Unsupported {
                    name: "s_mov_b32".into(),
                    index,
                    why: "only SGPR destinations are liftable",
                });
            };
            let v = match src0 {
                Source::Sgpr(s) => ctx.sgpr_value(*s, index)?,
                Source::Const(c) => ctx.inline_const(*c, Ty::U32),
                Source::Literal(l) => ctx.const_word(Ty::U32, *l),
                other => {
                    return Err(LiftError::Unsupported {
                        name: format!("s_mov_b32 {other:?}"),
                        index,
                        why: "source type outside the liftable set",
                    });
                }
            };
            ctx.set_sgpr(*d, v);
            Ok(())
        }
        Inst::Sop1 { op, .. } => Err(LiftError::Unsupported {
            name: format!("sop1:{op:#04x}"),
            index,
            why: "only s_mov_b32 is liftable",
        }),
        Inst::Sop2 {
            op,
            sdst,
            src0,
            src1,
        } => {
            let Source::Sgpr(d) = sdst else {
                return Err(LiftError::Unsupported {
                    name: format!("sop2:{op:#04x}"),
                    index,
                    why: "only SGPR destinations are liftable",
                });
            };
            let a = ctx.src(src0, index, Ty::U32)?;
            let c = ctx.src(src1, index, Ty::U32)?;
            let r = match *op {
                opcodes::sop2::S_ADD_U32 | opcodes::sop2::S_ADD_I32 => ctx.iadd(a, c)?,
                opcodes::sop2::S_SUB_U32 | opcodes::sop2::S_SUB_I32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.i_sub(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_AND_B32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.bitwise_and(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_OR_B32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.bitwise_or(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_XOR_B32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.bitwise_xor(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_MIN_U32 => ctx.umin(a, c)?,
                opcodes::sop2::S_MAX_U32 => ctx.umax(a, c)?,
                opcodes::sop2::S_LSHL_B32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.shift_left_logical(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_LSHR_B32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(ctx, ctx.b.shift_right_logical(ctx.t_u32, None, a.id, c.id))?;
                    Val { id, ty: Ty::U32 }
                }
                opcodes::sop2::S_ASHR_I32 => {
                    let a = ctx.coerce(a, Ty::U32)?;
                    let c = ctx.coerce(c, Ty::U32)?;
                    let id = op!(
                        ctx,
                        ctx.b.shift_right_arithmetic(ctx.t_u32, None, a.id, c.id)
                    )?;
                    Val { id, ty: Ty::U32 }
                }
                other => {
                    return Err(LiftError::Unsupported {
                        name: format!(
                            "sop2:{other:#04x} ({})",
                            opcodes::sop2_name(other).unwrap_or("?")
                        ),
                        index,
                        why: "SOP2 opcode outside the liftable set",
                    });
                }
            };
            ctx.set_sgpr(*d, r);
            Ok(())
        }
        Inst::Sopc { op, src0, src1 } => {
            let a = ctx.src(src0, index, Ty::U32)?;
            let c = ctx.src(src1, index, Ty::U32)?;
            let r = match *op {
                opcodes::sopc::S_CMP_EQ_U32 => ctx.icmp(IcmpOp::Eq, a, c)?,
                opcodes::sopc::S_CMP_LG_U32 => ctx.icmp(IcmpOp::Ne, a, c)?,
                opcodes::sopc::S_CMP_GT_U32 => ctx.icmp(IcmpOp::GtU, a, c)?,
                opcodes::sopc::S_CMP_GE_U32 => ctx.icmp(IcmpOp::GeU, a, c)?,
                opcodes::sopc::S_CMP_LT_U32 => ctx.icmp(IcmpOp::LtU, a, c)?,
                opcodes::sopc::S_CMP_LE_U32 => ctx.icmp(IcmpOp::LeU, a, c)?,
                other => {
                    return Err(LiftError::Unsupported {
                        name: format!("sopc:{other:#04x}"),
                        index,
                        why: "SOPC opcode outside the liftable set",
                    });
                }
            };
            ctx.scc = Some(r);
            Ok(())
        }
        Inst::Sopk { op, sdst, simm16 } => {
            let Source::Sgpr(d) = sdst else {
                return Err(LiftError::Unsupported {
                    name: "sopk".into(),
                    index,
                    why: "only SGPR destinations are liftable",
                });
            };
            match *op {
                opcodes::sopk::S_MOVK_I32 => {
                    let v = ctx.const_word(Ty::U32, *simm16 as u16 as u32);
                    ctx.set_sgpr(*d, v);
                    Ok(())
                }
                other => Err(LiftError::Unsupported {
                    name: format!("sopk:{other:#04x}"),
                    index,
                    why: "only s_movk_i32 is liftable",
                }),
            }
        }
        Inst::Sopp { op, .. } => match *op {
            opcodes::sopp::S_NOP | opcodes::sopp::S_WAITCNT | opcodes::sopp::S_BARRIER => Ok(()),
            opcodes::sopp::S_ENDPGM => Ok(()), // straight-line: end of program
            other => Err(LiftError::Unsupported {
                name: format!(
                    "sopp:{other:#04x} ({})",
                    opcodes::sopp_name(other).unwrap_or("?")
                ),
                index,
                why: "control flow is not liftable yet (Phase 4)",
            }),
        },
        Inst::Smem {
            op,
            sdst,
            sbase,
            offset,
            ..
        } => {
            let Source::Sgpr(d) = sdst else {
                return Err(LiftError::Unsupported {
                    name: format!("smem:{op:#04x}"),
                    index,
                    why: "SMEM destination must be an SGPR",
                });
            };
            let Source::Sgpr(base) = sbase else {
                return Err(LiftError::Unsupported {
                    name: format!("smem:{op:#04x}"),
                    index,
                    why: "SMEM base must be an SGPR pair",
                });
            };
            let SmemOffset::Imm(off) = offset else {
                return Err(LiftError::Unsupported {
                    name: format!("smem:{op:#04x}"),
                    index,
                    why: "SGPR-offset SMEM loads are not liftable yet",
                });
            };
            let Some(fetch_idx) = ctx
                .fetches
                .iter()
                .position(|e| e.decl.table_sbase == *base && e.decl.sload_offset_dwords == *off)
            else {
                return Err(LiftError::Unsupported {
                    name: format!("smem:{op:#04x} @s[{base}]+{off:#x}"),
                    index,
                    why: "SMEM load that is not a declared descriptor-table fetch",
                });
            };
            ctx.mark_descriptor(*d, fetch_idx);
            Ok(())
        }
        Inst::Vop1 { op, vdst, src0 } => match *op {
            opcodes::vop1::V_MOV_B32 => {
                let v = match src0 {
                    Source::Vgpr(s) => ctx.vgpr_value(*s, index)?,
                    Source::Sgpr(s) => ctx.sgpr_value(*s, index)?,
                    Source::Const(c) => ctx.inline_const(*c, Ty::F32),
                    Source::Literal(l) => ctx.const_word(Ty::F32, *l),
                    other => {
                        return Err(LiftError::Unsupported {
                            name: format!("v_mov_b32 {other:?}"),
                            index,
                            why: "source type outside the liftable set",
                        });
                    }
                };
                ctx.set_vgpr(*vdst, v);
                Ok(())
            }
            opcodes::vop1::V_CVT_F32_I32 => {
                let a = ctx.src(src0, index, Ty::I32)?;
                let id = op!(ctx, ctx.b.convert_s_to_f(ctx.t_f32, None, a.id))?;
                ctx.set_vgpr(*vdst, Val { id, ty: Ty::F32 });
                Ok(())
            }
            opcodes::vop1::V_CVT_I32_F32 => {
                let a = ctx.src(src0, index, Ty::F32)?;
                let id = op!(ctx, ctx.b.convert_f_to_s(ctx.t_i32, None, a.id))?;
                ctx.set_vgpr(*vdst, Val { id, ty: Ty::I32 });
                Ok(())
            }
            opcodes::vop1::V_NOT_B32 => {
                let a = ctx.src(src0, index, Ty::U32)?;
                let id = op!(ctx, ctx.b.not(ctx.t_u32, None, a.id))?;
                ctx.set_vgpr(*vdst, Val { id, ty: Ty::U32 });
                Ok(())
            }
            other => Err(LiftError::Unsupported {
                name: format!("vop1:{other:#04x}"),
                index,
                why: "VOP1 opcode outside the liftable set",
            }),
        },
        Inst::Vop2 {
            op,
            vdst,
            src0,
            src1,
        } => {
            match *op {
                opcodes::vop2::V_ADD_F32 => {
                    let a = ctx.src(src0, index, Ty::F32)?;
                    let c = ctx.src(src1, index, Ty::F32)?;
                    let r = ctx.fadd(a, c)?;
                    ctx.set_vgpr(*vdst, r);
                }
                opcodes::vop2::V_MUL_F32 | opcodes::vop2::V_MAC_F32 => {
                    let a = ctx.src(src0, index, Ty::F32)?;
                    let c = ctx.src(src1, index, Ty::F32)?;
                    let r = ctx.fmul(a, c)?;
                    // v_mac_f32 accumulates into the destination
                    // (destructive dst); plain mul keeps the semantics.
                    if *op == opcodes::vop2::V_MAC_F32 {
                        let dst = ctx.vgpr_value(*vdst, index)?;
                        let r = ctx.fadd(r, dst)?;
                        ctx.set_vgpr(*vdst, r);
                    } else {
                        ctx.set_vgpr(*vdst, r);
                    }
                }
                other => {
                    let a = ctx.src(src0, index, Ty::U32)?;
                    let c = ctx.src(src1, index, Ty::U32)?;
                    let r = match other {
                        opcodes::vop2::V_ADD_U32 => ctx.iadd(a, c)?,
                        opcodes::vop2::V_SUB_U32 => {
                            let id = op!(ctx, ctx.b.i_sub(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_AND_B32 => {
                            let id = op!(ctx, ctx.b.bitwise_and(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_OR_B32 => {
                            let id = op!(ctx, ctx.b.bitwise_or(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_XOR_B32 => {
                            let id = op!(ctx, ctx.b.bitwise_xor(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_MIN_U32 => ctx.umin(a, c)?,
                        opcodes::vop2::V_MAX_U32 => ctx.umax(a, c)?,
                        opcodes::vop2::V_LSHLREV_B32 => {
                            let id =
                                op!(ctx, ctx.b.shift_left_logical(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_LSHRREV_B32 => {
                            let id =
                                op!(ctx, ctx.b.shift_right_logical(ctx.t_u32, None, a.id, c.id))?;
                            Val { id, ty: Ty::U32 }
                        }
                        opcodes::vop2::V_ASHRREV_I32 => {
                            let id = op!(
                                ctx,
                                ctx.b.shift_right_arithmetic(ctx.t_u32, None, a.id, c.id)
                            )?;
                            Val { id, ty: Ty::U32 }
                        }
                        _ => {
                            return Err(LiftError::Unsupported {
                                name: format!("vop2:{other:#04x}"),
                                index,
                                why: "VOP2 opcode outside the liftable set",
                            });
                        }
                    };
                    ctx.set_vgpr(*vdst, r);
                }
            }
            Ok(())
        }
        Inst::Vop3 {
            op,
            vdst,
            src0,
            src1,
            src2,
            ..
        } => match *op {
            opcodes::vop3::V_MAD_F32 | opcodes::vop3::V_FMA_F32 => {
                let a = ctx.src(src0, index, Ty::F32)?;
                let c = ctx.src(src1, index, Ty::F32)?;
                let d = ctx.src(src2, index, Ty::F32)?;
                let m = ctx.fmul(a, c)?;
                let r = ctx.fadd(m, d)?;
                ctx.set_vgpr(*vdst, r);
                Ok(())
            }
            _ => Err(LiftError::Unsupported {
                name: format!("vop3:{op:#06x}"),
                index,
                why: "VOP3 opcode outside the liftable set (e64 VOP2 forms arrive with Phase 4)",
            }),
        },
        Inst::Mubuf {
            op,
            vdata,
            vaddr,
            srsrc,
            offset,
            idxen,
            ..
        } => {
            if !*idxen {
                return Err(LiftError::Unsupported {
                    name: "buffer_load".into(),
                    index,
                    why: "MUBUF without idxen is not liftable yet",
                });
            }
            let Source::Sgpr(desc) = srsrc else {
                return Err(LiftError::Unsupported {
                    name: "buffer_load".into(),
                    index,
                    why: "srsrc must be an SGPR quad",
                });
            };
            let Some(fetch_idx) = ctx.descriptor_at(*desc) else {
                return Err(LiftError::Uninitialized {
                    reg: format!("s{desc} (descriptor)"),
                    index,
                });
            };
            let entry = ctx.fetches[fetch_idx];
            if *op != opcodes::mubuf::BUFFER_LOAD_DWORDX4 {
                return Err(LiftError::Unsupported {
                    name: format!("mubuf:{op:#04x}"),
                    index,
                    why: "only buffer_load_dwordx4 is liftable (16-byte records)",
                });
            }
            if u32::from(*offset) % VertexFormat::RECORD_BYTES != 0 {
                return Err(LiftError::Unsupported {
                    name: "buffer_load_dwordx4".into(),
                    index,
                    why: "immediate offset not 16-byte aligned for the declared format",
                });
            }

            // record = vb[(index * stride + imm) / 16].
            let vdata = *vdata;
            let idx = ctx.vgpr_value(*vaddr, index)?;
            let idx = ctx.coerce(idx, Ty::U32)?;
            let stride = ctx.fetch_stride(fetch_idx)?;
            let four = ctx.b.constant_bit32(ctx.t_u32, 4);
            let stride16 = op!(
                ctx,
                ctx.b.shift_right_logical(ctx.t_u32, None, stride, four)
            )?;
            let base = op!(ctx, ctx.b.i_mul(ctx.t_u32, None, idx.id, stride16))?;
            let imm = ctx
                .b
                .constant_bit32(ctx.t_u32, u32::from(*offset) / VertexFormat::RECORD_BYTES);
            let rec = op!(ctx, ctx.b.i_add(ctx.t_u32, None, base, imm))?;

            let fetch_var = ctx.fetch_vars[fetch_idx];
            let t_elem_ptr = ctx
                .b
                .type_pointer(None, StorageClass::StorageBuffer, ctx.t_v4u);
            let zero = ctx.b.constant_bit32(ctx.t_u32, 0);
            let rec_ptr = op!(
                ctx,
                ctx.b
                    .access_chain(t_elem_ptr, None, fetch_var, vec![zero, rec])
            )?;
            let raw = op!(ctx, ctx.b.load(ctx.t_v4u, None, rec_ptr, None, vec![]))?;

            match entry.decl.format {
                VertexFormat::F32x4 => {
                    let vec = op!(ctx, ctx.b.bitcast(ctx.t_v4f, None, raw))?;
                    for lane in 0..4 {
                        let id = op!(
                            ctx,
                            ctx.b.composite_extract(ctx.t_f32, None, vec, vec![lane])
                        )?;
                        ctx.set_vgpr(vdata + lane as u8, Val { id, ty: Ty::F32 });
                    }
                }
                VertexFormat::U32x4 => {
                    for lane in 0..4 {
                        let id = op!(
                            ctx,
                            ctx.b.composite_extract(ctx.t_u32, None, raw, vec![lane])
                        )?;
                        ctx.set_vgpr(vdata + lane as u8, Val { id, ty: Ty::U32 });
                    }
                }
                VertexFormat::Unorm8x4 => {
                    let ff = ctx.b.constant_bit32(ctx.t_f32, (1.0f32 / 255.0).to_bits());
                    for lane in 0..4 {
                        let w = op!(
                            ctx,
                            ctx.b.composite_extract(ctx.t_u32, None, raw, vec![lane])
                        )?;
                        let sh = ctx.b.constant_bit32(ctx.t_u32, (3 - lane) * 8);
                        let sv = op!(ctx, ctx.b.shift_right_logical(ctx.t_u32, None, w, sh))?;
                        let mask255 = ctx.b.constant_bit32(ctx.t_u32, 0xFF);
                        let and = op!(ctx, ctx.b.bitwise_and(ctx.t_u32, None, sv, mask255))?;
                        let f = op!(ctx, ctx.b.convert_u_to_f(ctx.t_f32, None, and))?;
                        let r = op!(ctx, ctx.b.f_mul(ctx.t_f32, None, f, ff))?;
                        ctx.set_vgpr(vdata + lane as u8, Val { id: r, ty: Ty::F32 });
                    }
                }
            }
            Ok(())
        }
        Inst::Exp { tgt, en, src, .. } => {
            let mut lanes = Vec::with_capacity(4);
            for (i, s) in src.iter().enumerate() {
                if en[i] {
                    let v = ctx.vgpr_value(*s, index)?;
                    let v = ctx.coerce(v, Ty::F32)?;
                    lanes.push(v.id);
                } else {
                    lanes.push(ctx.b.constant_bit32(ctx.t_f32, 0.0f32.to_bits()));
                }
            }
            let value = op!(ctx, ctx.b.composite_construct(ctx.t_v4f, None, lanes))?;

            let member = match (*tgt, stage) {
                (12, Stage::Vertex) => ctx.pos_member, // pos0 → builtin(position)
                (0, Stage::Vertex) => ctx.color_member, // param0 → location(0)
                (32..=63, Stage::Fragment) => ctx.color_member, // mrt0 → location(0)
                other => {
                    return Err(LiftError::Unsupported {
                        name: format!("exp tgt {other:?}"),
                        index,
                        why: "export target outside the liftable set",
                    });
                }
            };
            let idx = ctx.b.constant_bit32(ctx.t_u32, member as u32);
            let t_ptr = ctx.b.type_pointer(None, StorageClass::Output, ctx.t_v4f);
            let ptr = op!(ctx, ctx.b.access_chain(t_ptr, None, ctx.out_var, vec![idx]))?;
            op!(ctx, ctx.b.store(ptr, value, None, vec![]))?;
            Ok(())
        }
        Inst::Ds { op, .. } => Err(LiftError::Unsupported {
            name: format!("ds:{op:#04x}"),
            index,
            why: "LDS accesses arrive with compute lifting (Phase 4)",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode::{Encoder, decode_program};
    use super::super::enc::Source;
    use super::super::opcodes as ops;
    use super::*;

    #[test]
    fn pixel_const_color_lifts() {
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
        let lifted = lift(
            &prog,
            Stage::Fragment,
            &LiftInputs {
                vertex_id_sgpr: None,
                descriptor_tables: vec![],
                vertex_fetches: vec![],
                ps_inputs: vec![],
            },
        )
        .unwrap();
        assert!(!lifted.spirv.is_empty());
        assert!(lifted.manifest.fetches.is_empty());
        // SPIR-V magic + version 1.3 header.
        assert_eq!(lifted.spirv[0], 0x0723_0203);
    }

    #[test]
    fn uninitialized_read_is_refused() {
        let mut e = Encoder::new();
        e.vop2(ops::vop2::V_ADD_F32, 0, Source::Vgpr(9), Source::Vgpr(8));
        let prog = decode_program(&e.build()).unwrap();
        let err = lift(&prog, Stage::Fragment, &LiftInputs::default()).unwrap_err();
        assert!(matches!(err, LiftError::Uninitialized { .. }));
        assert!(err.to_string().contains("v9"));
    }

    #[test]
    fn branch_is_refused_honestly() {
        let mut e = Encoder::new();
        e.sopp(ops::sopp::S_BRANCH, 4);
        let prog = decode_program(&e.build()).unwrap();
        let err = lift(&prog, Stage::Fragment, &LiftInputs::default()).unwrap_err();
        assert!(err.to_string().contains("control flow"));
    }

    #[test]
    fn undeclared_smem_is_refused() {
        let mut e = Encoder::new();
        e.smem_load(ops::smem::S_LOAD_DWORDX4, 4, 8, 0x10);
        let prog = decode_program(&e.build()).unwrap();
        let err = lift(&prog, Stage::Fragment, &LiftInputs::default()).unwrap_err();
        assert!(err.to_string().contains("descriptor-table"));
    }

    #[test]
    fn declared_smem_feeding_fetch_elides() {
        let mut e = Encoder::new();
        e.smem_load(ops::smem::S_LOAD_DWORDX4, 4, 0, 0x10);
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
        e.exp(12, [true, true, true, true], false, [1, 2, 3, 4]);
        e.sopp(ops::sopp::S_ENDPGM, 0);
        let prog = decode_program(&e.build()).unwrap();
        let inputs = LiftInputs {
            vertex_id_sgpr: Some(6),
            descriptor_tables: vec![0],
            vertex_fetches: vec![VertexFetchDecl {
                table_sbase: 0,
                sload_offset_dwords: 0x10,
                format: VertexFormat::F32x4,
            }],
            ps_inputs: vec![],
        };
        let lifted = lift(&prog, Stage::Vertex, &inputs).unwrap();
        assert_eq!(lifted.manifest.fetches.len(), 1);
        assert_eq!(lifted.manifest.fetches[0].slot, FETCH_STORAGE_BASE);
    }
}
