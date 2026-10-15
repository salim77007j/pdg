//! Lift guest x86-64 instructions to cranelift IR.
//!
//! Tier-1 scope: the integer heart real code spends its time in — data
//! movement, the ALU with architecturally exact flags (mirroring the
//! interpreter formula-for-formula), shifts, the stack, control flow with
//! **loop formation** (a backward `jcc` becomes a loop in the IR, so
//! cranelift's register allocator keeps loop-carried guest registers in
//! host registers), CMOVcc and the move-extends. Everything else exits
//! with [`exit::FALLBACK`] and the dispatcher single-steps it in the
//! interpreter — nothing silently no-ops.
//!
//! Block shape: because a block ends at its first control instruction, the
//! only possible internal branch is the final `jcc` jumping backward —
//! exactly one loop back-edge per block, budget-checked on every iteration
//! so a guest infinite loop still honours `run(n)`.

use cranelift_codegen::ir::{
    AbiParam, Block, InstBuilder, MemFlagsData, Signature, UserFuncName, condcodes::IntCC, types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

use crate::env::{self, JitEnv, TLB_ENTRIES};

use pdg_cpu::state::{decode_reg, rflags};

/// Maximum instructions in one block.
const BLOCK_INSTR_CAP: usize = 512;

/// Compile errors (never guest-visible; the dispatcher falls back).
#[derive(Debug, thiserror::Error)]
pub enum LiftError {
    #[error("cranelift: {0}")]
    Cranelift(String),
    #[error("guest decode failed at {rip:#x}")]
    Decode { rip: u64 },
}

/// Build the JIT module with the PDG helper symbols registered.
pub fn make_module() -> Result<JITModule, LiftError> {
    let mut flags = settings::builder();
    flags
        .set("opt_level", "speed")
        .map_err(|e| LiftError::Cranelift(e.to_string()))?;
    // Surface verifier diagnostics with full detail.
    flags
        .set("enable_verifier", "true")
        .map_err(|e| LiftError::Cranelift(e.to_string()))?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| LiftError::Cranelift(e.to_string()))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flags))
        .map_err(|e| LiftError::Cranelift(e.to_string()))?;
    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    {
        jb.symbol("pdgjit_tlb_load", env::pdgjit_tlb_load as *const u8);
        jb.symbol("pdgjit_tlb_store", env::pdgjit_tlb_store as *const u8);
        jb.symbol("pdgjit_read8", env::pdgjit_read8 as *const u8);
        jb.symbol("pdgjit_read16", env::pdgjit_read16 as *const u8);
        jb.symbol("pdgjit_read32", env::pdgjit_read32 as *const u8);
        jb.symbol("pdgjit_read64", env::pdgjit_read64 as *const u8);
        jb.symbol("pdgjit_write8", env::pdgjit_write8 as *const u8);
        jb.symbol("pdgjit_write16", env::pdgjit_write16 as *const u8);
        jb.symbol("pdgjit_write32", env::pdgjit_write32 as *const u8);
        jb.symbol("pdgjit_write64", env::pdgjit_write64 as *const u8);
        jb.symbol(
            "pdgjit_smc_invalidate",
            env::pdgjit_smc_invalidate as *const u8,
        );
    }
    Ok(JITModule::new(jb))
}

/// Helper function ids referenced from the IR.
pub(crate) struct HelperIds {
    tlb_load: cranelift_module::FuncId,
    tlb_store: cranelift_module::FuncId,
    read: [cranelift_module::FuncId; 4],
    write: [cranelift_module::FuncId; 4],
    smc: cranelift_module::FuncId,
}

/// Declare the helper imports (public for the dispatcher).
pub(crate) fn make_helper_ids(module: &mut JITModule) -> HelperIds {
    let conv = module.isa().default_call_conv();
    let import = |module: &mut JITModule, name: &str, sig: &Signature| {
        module
            .declare_function(name, Linkage::Import, sig)
            .expect("declare helper")
    };
    let mut tlb_sig = Signature::new(conv);
    tlb_sig.params.push(AbiParam::new(types::I64));
    tlb_sig.params.push(AbiParam::new(types::I64));
    tlb_sig.params.push(AbiParam::new(types::I64));
    tlb_sig.returns.push(AbiParam::new(types::I64));
    let mut read_sig = Signature::new(conv);
    read_sig.params.push(AbiParam::new(types::I64));
    read_sig.params.push(AbiParam::new(types::I64));
    read_sig.params.push(AbiParam::new(types::I64));
    read_sig.returns.push(AbiParam::new(types::I64));
    let mut write_sig = Signature::new(conv);
    write_sig.params.push(AbiParam::new(types::I64));
    write_sig.params.push(AbiParam::new(types::I64));
    write_sig.params.push(AbiParam::new(types::I64));
    write_sig.params.push(AbiParam::new(types::I64));
    let mut smc_sig = Signature::new(conv);
    smc_sig.params.push(AbiParam::new(types::I64));
    smc_sig.params.push(AbiParam::new(types::I64));
    smc_sig.params.push(AbiParam::new(types::I64));

    let names = |read: bool| -> [&'static str; 4] {
        if read {
            [
                "pdgjit_read8",
                "pdgjit_read16",
                "pdgjit_read32",
                "pdgjit_read64",
            ]
        } else {
            [
                "pdgjit_write8",
                "pdgjit_write16",
                "pdgjit_write32",
                "pdgjit_write64",
            ]
        }
    };
    let rn = names(true);
    let wn = names(false);
    HelperIds {
        tlb_load: import(module, "pdgjit_tlb_load", &tlb_sig),
        tlb_store: import(module, "pdgjit_tlb_store", &tlb_sig),
        read: [
            import(module, rn[0], &read_sig),
            import(module, rn[1], &read_sig),
            import(module, rn[2], &read_sig),
            import(module, rn[3], &read_sig),
        ],
        write: [
            import(module, wn[0], &write_sig),
            import(module, wn[1], &write_sig),
            import(module, wn[2], &write_sig),
            import(module, wn[3], &write_sig),
        ],
        smc: import(module, "pdgjit_smc_invalidate", &smc_sig),
    }
}

// Field offsets into the #[repr(C)] JitEnv.
const OFF_GPR: i32 = core::mem::offset_of!(JitEnv, gpr) as i32;
const OFF_RIP: i32 = core::mem::offset_of!(JitEnv, rip) as i32;
const OFF_RFLAGS: i32 = core::mem::offset_of!(JitEnv, rflags) as i32;
const OFF_EXIT_CODE: i32 = core::mem::offset_of!(JitEnv, exit_code) as i32;
const OFF_BUDGET: i32 = core::mem::offset_of!(JitEnv, budget) as i32;
const OFF_TLB_TAGS: i32 = core::mem::offset_of!(JitEnv, tlb_tags) as i32;
const OFF_TLB_PTRS: i32 = core::mem::offset_of!(JitEnv, tlb_ptrs) as i32;
const OFF_BLOOM: i32 = core::mem::offset_of!(JitEnv, smc_bloom) as i32;

const M: MemFlagsData = MemFlagsData::new().with_notrap();

/// Operand width in bytes for register/memory operands.
fn op_w(i: &Instruction, n: u32) -> u32 {
    match i.op_kind(n) {
        OpKind::Register => decode_reg(i.op_register(n)).map(|(_, s, _)| s).unwrap_or(8),
        OpKind::Memory => i.memory_size().size() as u32,
        _ => 8,
    }
}

fn mask_of(w: u32) -> i64 {
    match w {
        1 => 0xFF,
        2 => 0xFFFF,
        4 => 0xFFFF_FFFF,
        _ => -1,
    }
}

#[derive(Clone, Copy)]
struct Wide {
    w: u32,
}

impl Wide {
    fn of(w: u32) -> Self {
        Self { w: w.clamp(1, 8) }
    }
    fn bits(self) -> i64 {
        i64::from(self.w * 8)
    }
    fn mask(self) -> i64 {
        mask_of(self.w)
    }
    fn idx(self) -> usize {
        // log2 of the byte width: 1→0, 2→1, 4→2, 8→3.
        (self.w as usize).trailing_zeros() as usize
    }
}

/// Decode one block starting at `entry`. The block ends at the first
/// control instruction (or the caps). Also reports the single possible
/// back-edge target (a backward `jcc`) for loop formation.
pub fn decode_block(
    mem: &mut dyn pdg_cpu::MemoryAccess,
    entry: u64,
) -> Result<(Vec<Instruction>, Option<u64>), LiftError> {
    let mut window = [0u8; 16];
    let mut rip = entry;
    let mut instrs = Vec::new();
    while instrs.len() < BLOCK_INSTR_CAP {
        let n = mem
            .fetch(rip, &mut window)
            .map_err(|_| LiftError::Decode { rip })?;
        let mut d = Decoder::with_ip(64, &window[..n], rip, DecoderOptions::NONE);
        let i = d.decode();
        if i.is_invalid() {
            return Err(LiftError::Decode { rip });
        }
        let next = i.next_ip();
        let is_jcc = is_jcc(i.mnemonic());
        instrs.push(i);
        if is_jcc || is_block_end(i.mnemonic()) {
            break;
        }
        rip = next;
    }
    // The single possible back-edge: the last instruction is a jcc whose
    // target lies inside the block.
    let back_edge = instrs.last().and_then(|last| {
        if is_jcc(last.mnemonic()) {
            let t = last.near_branch_target();
            if t >= entry && t <= last.ip() {
                Some(t)
            } else {
                None
            }
        } else {
            None
        }
    });
    Ok((instrs, back_edge))
}

fn is_jcc(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Jno | Jb | Jae | Je | Jne | Jbe | Ja | Js | Jns | Jp | Jnp | Jl | Jge | Jle | Jg
    )
}

fn is_block_end(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Jmp | Mnemonic::Call | Mnemonic::Ret | Mnemonic::Hlt | Mnemonic::Syscall
    )
}

/// Lift and compile one block; returns the finalized context.
pub fn lift_block(
    module: &mut JITModule,
    ids: &HelperIds,
    entry: u64,
    instrs: &[Instruction],
    back_edge: Option<u64>,
    name_seq: u32,
) -> Result<cranelift_codegen::Context, LiftError> {
    let mut ctx = module.make_context();
    let conv = module.isa().default_call_conv();
    let mut sig = Signature::new(conv);
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));
    ctx.func =
        cranelift_codegen::ir::Function::with_name_signature(UserFuncName::user(0, name_seq), sig);

    let tlb_load = module.declare_func_in_func(ids.tlb_load, &mut ctx.func);
    let tlb_store = module.declare_func_in_func(ids.tlb_store, &mut ctx.func);
    let read_ref: Vec<_> = ids
        .read
        .iter()
        .map(|&id| module.declare_func_in_func(id, &mut ctx.func))
        .collect();
    let write_ref: Vec<_> = ids
        .write
        .iter()
        .map(|&id| module.declare_func_in_func(id, &mut ctx.func))
        .collect();
    let smc_ref = module.declare_func_in_func(ids.smc, &mut ctx.func);

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    let mut g = [Variable::from_u32(0); 16];
    for v in g.iter_mut() {
        *v = b.declare_var(types::I64);
    }
    let f_cf = b.declare_var(types::I8);
    let f_pf = b.declare_var(types::I8);
    let f_zf = b.declare_var(types::I8);
    let f_sf = b.declare_var(types::I8);
    let f_of = b.declare_var(types::I8);
    let f_af = b.declare_var(types::I8);
    let v_exit_rip = b.declare_var(types::I64);
    let v_exit_code = b.declare_var(types::I64);

    // --- entry block: budget check -----------------------------------------
    let entry_block = b.create_block();
    b.append_block_params_for_function_params(entry_block);
    let env_ptr = b.block_params(entry_block)[0];
    b.switch_to_block(entry_block);
    b.seal_block(entry_block);
    let budget = b.ins().load(types::I64, M, env_ptr, OFF_BUDGET);
    let nb = b.ins().iadd_imm_s(budget, -(instrs.len() as i64));
    b.ins().store(M, nb, env_ptr, OFF_BUDGET);
    let neg = b.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, nb, 0);
    let body = b.create_block();
    let budget_out = b.create_block();
    b.ins().brif(neg, budget_out, &[], body, &[]);
    b.switch_to_block(budget_out);
    b.seal_block(budget_out);
    // Budget exit: nothing has executed; env.rip is still the entry.
    let budget_code = b.ins().iconst(types::I64, env::exit::BUDGET);
    b.def_var(v_exit_code, budget_code);
    let epilogue = b.create_block();
    b.ins().jump(epilogue, &[]);
    b.switch_to_block(body);
    // (body sealed later, after the back-edge is known)

    // --- loads of guest state ------------------------------------------------
    for (i, &v) in g.iter().enumerate() {
        let x = b
            .ins()
            .load(types::I64, M, env_ptr, OFF_GPR + (i as i32) * 8);
        b.def_var(v, x);
    }
    let rf = b.ins().load(types::I64, M, env_ptr, OFF_RFLAGS);
    let zero8 = b.ins().iconst(types::I8, 0);
    let one8 = b.ins().iconst(types::I8, 1);
    for (var, bitv) in [
        (f_cf, rflags::CF as i64),
        (f_pf, rflags::PF as i64),
        (f_zf, rflags::ZF as i64),
        (f_sf, rflags::SF as i64),
        (f_of, rflags::OF as i64),
        (f_af, rflags::AF as i64),
    ] {
        let s = b.ins().ushr_imm_s(rf, i64::from(bitv.trailing_zeros()));
        let bit = b.ins().band_imm_s(s, 1);
        let c = b.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
        let fv = b.ins().select(c, one8, zero8);
        b.def_var(var, fv);
    }

    // --- the lifter ------------------------------------------------------------
    let mut l = Lifter {
        b,
        env_ptr,
        g,
        cf: f_cf,
        pf: f_pf,
        zf: f_zf,
        sf: f_sf,
        of: f_of,
        af: f_af,
        v_exit_rip,
        v_exit_code,
        tlb_load,
        tlb_store,
        read_ref,
        write_ref,
        smc_ref,
        epilogue,
        loop_header: None,
        loop_cost: 0,
        terminated: false,
    };

    // Loop formation: pre-create the header block for the back-edge target;
    // the linear lift splits there.
    if let Some(target) = back_edge {
        let header = l.b.create_block();
        l.loop_cost = instrs.iter().filter(|i| i.ip() >= target).count();
        l.loop_header = Some((target, header));
    }

    for instr in instrs {
        if let Some((target, header)) = l.loop_header {
            if instr.ip() == target && target != entry {
                // End the current block and continue in the header block.
                l.b.ins().jump(header, &[]);
                l.b.switch_to_block(header);
                l.terminated = false;
            }
        }
        if l.lift(instr)? {
            // The block ended (unconditional exit/fallback/halt).
            break;
        }
    }

    // Fallthrough past the last instruction: continue at its next_ip —
    // unless the block already ended with an unconditional exit.
    if !l.terminated {
        if let Some(last) = instrs.last() {
            l.exit_continue(last.next_ip());
        }
    }

    l.b.seal_all_blocks();

    // --- epilogue -------------------------------------------------------------
    l.b.switch_to_block(epilogue);
    for (i, &v) in l.g.iter().enumerate() {
        let x = l.b.use_var(v);
        l.b.ins().store(M, x, env_ptr, OFF_GPR + (i as i32) * 8);
    }
    let ex_rip = l.b.use_var(v_exit_rip);
    l.b.ins().store(M, ex_rip, env_ptr, OFF_RIP);
    let ex_code = l.b.use_var(v_exit_code);

    let one = l.b.ins().iconst(types::I64, 1);
    let zero = l.b.ins().iconst(types::I64, 0);
    let mut rf = l.b.ins().iconst(types::I64, rflags::RESERVED_1 as i64);
    for (v, bitval) in [
        (f_cf, rflags::CF as i64),
        (f_pf, rflags::PF as i64),
        (f_zf, rflags::ZF as i64),
        (f_sf, rflags::SF as i64),
        (f_of, rflags::OF as i64),
        (f_af, rflags::AF as i64),
    ] {
        let fv = l.b.use_var(v);
        let c = l.b.ins().icmp_imm_s(IntCC::NotEqual, fv, 0);
        let bitv = l.b.ins().select(c, one, zero);
        let shifted =
            l.b.ins()
                .ishl_imm_s(bitv, i64::from(bitval.trailing_zeros()));
        rf = l.b.ins().bor(rf, shifted);
    }
    l.b.ins().store(M, rf, env_ptr, OFF_RFLAGS);
    l.b.ins().return_(&[ex_code]);

    l.b.finalize(module.isa().frontend_config());
    Ok(ctx)
}

// ---------------------------------------------------------------------------
// the lifter
// ---------------------------------------------------------------------------

struct Lifter<'f> {
    b: FunctionBuilder<'f>,
    env_ptr: cranelift_codegen::ir::Value,
    g: [Variable; 16],
    cf: Variable,
    pf: Variable,
    zf: Variable,
    sf: Variable,
    of: Variable,
    af: Variable,
    v_exit_rip: Variable,
    v_exit_code: Variable,
    tlb_load: cranelift_codegen::ir::FuncRef,
    tlb_store: cranelift_codegen::ir::FuncRef,
    read_ref: Vec<cranelift_codegen::ir::FuncRef>,
    write_ref: Vec<cranelift_codegen::ir::FuncRef>,
    smc_ref: cranelift_codegen::ir::FuncRef,
    epilogue: Block,
    /// Back-edge target + its header block (loop formation).
    loop_header: Option<(u64, Block)>,
    /// Instructions per loop iteration (budget re-check cost).
    loop_cost: usize,
    /// Whether the current cranelift block already has its terminator.
    terminated: bool,
}

impl<'f> Lifter<'f> {
    fn iconst(&mut self, v: i64) -> cranelift_codegen::ir::Value {
        self.b.ins().iconst(types::I64, v)
    }

    /// Logical NOT of an i8 icmp boolean (0/1 domain): `bnot` would produce
    /// 0xFE/0xFF, which any `brif`/`icmp ne 0` reads as *true*.
    fn bnot1(&mut self, v: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let one = self.b.ins().iconst(types::I8, 1);
        self.b.ins().bxor(v, one)
    }

    fn exit_continue(&mut self, rip: u64) {
        let code = self.iconst(env::exit::CONTINUE);
        let r = self.iconst(rip as i64);
        self.b.def_var(self.v_exit_code, code);
        self.b.def_var(self.v_exit_rip, r);
        let epi = self.epilogue;
        self.b.ins().jump(epi, &[]);
        self.terminated = true;
    }

    fn exit_with(&mut self, code: i64, rip: cranelift_codegen::ir::Value) {
        let c = self.iconst(code);
        self.b.def_var(self.v_exit_code, c);
        self.b.def_var(self.v_exit_rip, rip);
        let epi = self.epilogue;
        self.b.ins().jump(epi, &[]);
        self.terminated = true;
    }

    /// After a helper call: exit carrying the helper's own rip/exit code.
    fn check_helper_exit(&mut self) {
        let code = self
            .b
            .ins()
            .load(types::I64, M, self.env_ptr, OFF_EXIT_CODE);
        let bad = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, code, env::exit::CONTINUE);
        let trap = self.b.create_block();
        let cont = self.b.create_block();
        self.b.ins().brif(bad, trap, &[], cont, &[]);
        self.b.switch_to_block(trap);
        self.b.seal_block(trap);
        let code2 = self
            .b
            .ins()
            .load(types::I64, M, self.env_ptr, OFF_EXIT_CODE);
        let rip2 = self.b.ins().load(types::I64, M, self.env_ptr, OFF_RIP);
        self.b.def_var(self.v_exit_code, code2);
        self.b.def_var(self.v_exit_rip, rip2);
        let epi = self.epilogue;
        self.b.ins().jump(epi, &[]);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);
    }

    // ----- register file -----------------------------------------------------

    fn reg_var(&self, r: Register) -> Option<Variable> {
        decode_reg(r).map(|(idx, _, _)| self.g[idx])
    }

    /// Full 64-bit value of a GPR operand (lane views masked).
    fn read_gpr_op(&mut self, i: &Instruction, n: u32) -> cranelift_codegen::ir::Value {
        let r = i.op_register(n);
        let v = self.b.use_var(self.reg_var(r).expect("GPR operand"));
        match r {
            Register::AL
            | Register::CL
            | Register::DL
            | Register::BL
            | Register::SPL
            | Register::BPL
            | Register::SIL
            | Register::DIL
            | Register::R8L
            | Register::R9L
            | Register::R10L
            | Register::R11L
            | Register::R12L
            | Register::R13L
            | Register::R14L
            | Register::R15L => self.b.ins().band_imm_s(v, 0xFF),
            Register::AH | Register::CH | Register::DH | Register::BH => {
                let s = self.b.ins().ushr_imm_s(v, 8);
                self.b.ins().band_imm_s(s, 0xFF)
            }
            Register::AX
            | Register::CX
            | Register::DX
            | Register::BX
            | Register::SP
            | Register::BP
            | Register::SI
            | Register::DI
            | Register::R8W
            | Register::R9W
            | Register::R10W
            | Register::R11W
            | Register::R12W
            | Register::R13W
            | Register::R14W
            | Register::R15W => self.b.ins().band_imm_s(v, 0xFFFF),
            Register::EAX
            | Register::ECX
            | Register::EDX
            | Register::EBX
            | Register::ESP
            | Register::EBP
            | Register::ESI
            | Register::EDI
            | Register::R8D
            | Register::R9D
            | Register::R10D
            | Register::R11D
            | Register::R12D
            | Register::R13D
            | Register::R14D
            | Register::R15D => self.b.ins().band_imm_s(v, 0xFFFF_FFFF),
            _ => v,
        }
    }

    /// x86-64 register-write rules at width `w` (bytes).
    fn write_gpr_op(&mut self, r: Register, value: cranelift_codegen::ir::Value, w: u32) {
        let var = self.reg_var(r).expect("GPR dst");
        match w {
            8 => self.b.def_var(var, value),
            4 => {
                let z = self.b.ins().band_imm_s(value, 0xFFFF_FFFF);
                self.b.def_var(var, z);
            }
            2 => {
                let old = self.b.use_var(var);
                let keep = self.b.ins().band_imm_s(old, !0xFFFFi64);
                let v = self.b.ins().band_imm_s(value, 0xFFFF);
                let nv = self.b.ins().bor(keep, v);
                self.b.def_var(var, nv);
            }
            _ => {
                let (_, _, high) = decode_reg(r).expect("GPR");
                let old = self.b.use_var(var);
                if high {
                    let keep = self.b.ins().band_imm_s(old, !0xFF00i64);
                    let low = self.b.ins().band_imm_s(value, 0xFF);
                    let s = self.b.ins().ishl_imm_s(low, 8);
                    let nv = self.b.ins().bor(keep, s);
                    self.b.def_var(var, nv);
                } else {
                    let keep = self.b.ins().band_imm_s(old, !0xFFi64);
                    let v = self.b.ins().band_imm_s(value, 0xFF);
                    let nv = self.b.ins().bor(keep, v);
                    self.b.def_var(var, nv);
                }
            }
        }
    }

    // ----- flags (interpreter formulas, exactly) -------------------------------

    fn bool_to_i8(&mut self, v: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        // `icmp`/`bnot` results are already I8 in cranelift IR; a zero-op
        // `uextend.i8` would be invalid IR, so this is the identity.
        v
    }

    fn flags_add(
        &mut self,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
        carry: cranelift_codegen::ir::Value,
        result: cranelift_codegen::ir::Value,
        wide: Wide,
    ) {
        let bits = wide.bits();
        let sign = self.iconst(1i64 << (bits - 1));
        let mask = self.iconst(wide.mask());
        let cf = if bits == 64 {
            // (a+b) wrapped, or (a+b)+c wrapped.
            let t = self.b.ins().iadd(a, b);
            let t2 = self.b.ins().iadd(t, carry);
            let c1 = self.b.ins().icmp(IntCC::UnsignedLessThan, t, a);
            let c2 = self.b.ins().icmp(IntCC::UnsignedLessThan, t2, t);
            self.b.ins().bor(c1, c2)
        } else {
            let sum = self.b.ins().iadd(a, b);
            let sum = self.b.ins().iadd(sum, carry);
            let bit = self.b.ins().ushr_imm_s(sum, bits);
            let carried = self.b.ins().band_imm_s(bit, 1);
            self.b.ins().icmp_imm_s(IntCC::NotEqual, carried, 0)
        };
        let res = self.b.ins().band(result, mask);
        let zf = self.b.ins().icmp_imm_s(IntCC::Equal, res, 0);
        let sf_test = self.b.ins().band(res, sign);
        let sf = self.b.ins().icmp_imm_s(IntCC::NotEqual, sf_test, 0);
        // OF (add): operands with the SAME sign and a flipped result sign:
        // !(a^b) & (a^res) & sign — the interpreter's exact formula.
        let ab = self.b.ins().bxor(a, b);
        let not_ab = self.b.ins().bnot(ab);
        let a_xor_r = self.b.ins().bxor(a, res);
        let both = self.b.ins().band(not_ab, a_xor_r);
        let of_test = self.b.ins().band(both, sign);
        let of = self.b.ins().icmp_imm_s(IntCC::NotEqual, of_test, 0);
        let abr = self.b.ins().bxor(ab, res);
        let af_test = self.b.ins().band_imm_s(abr, 0x10);
        let af = self.b.ins().icmp_imm_s(IntCC::NotEqual, af_test, 0);
        let p = self.parity(res);
        let t = self.bool_to_i8(cf);
        self.b.def_var(self.cf, t);
        let t = self.bool_to_i8(zf);
        self.b.def_var(self.zf, t);
        let t = self.bool_to_i8(sf);
        self.b.def_var(self.sf, t);
        let t = self.bool_to_i8(of);
        self.b.def_var(self.of, t);
        let t = self.bool_to_i8(af);
        self.b.def_var(self.af, t);
        self.b.def_var(self.pf, p);
    }

    fn flags_sub(
        &mut self,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
        borrow: cranelift_codegen::ir::Value,
        result: cranelift_codegen::ir::Value,
        wide: Wide,
    ) {
        let bits = wide.bits();
        let sign = self.iconst(1i64 << (bits - 1));
        let mask = self.iconst(wide.mask());
        let b_gt = self.b.ins().icmp(IntCC::UnsignedGreaterThan, b, a);
        let b_eq = self.b.ins().icmp(IntCC::Equal, b, a);
        let br = self.b.ins().icmp_imm_s(IntCC::NotEqual, borrow, 0);
        let both = self.b.ins().band(b_eq, br);
        let cf = self.b.ins().bor(b_gt, both);
        let res = self.b.ins().band(result, mask);
        let zf = self.b.ins().icmp_imm_s(IntCC::Equal, res, 0);
        let sf_test = self.b.ins().band(res, sign);
        let sf = self.b.ins().icmp_imm_s(IntCC::NotEqual, sf_test, 0);
        let ab = self.b.ins().bxor(a, b);
        let ar = self.b.ins().bxor(a, res);
        let both2 = self.b.ins().band(ab, ar);
        let of_test = self.b.ins().band(both2, sign);
        let of = self.b.ins().icmp_imm_s(IntCC::NotEqual, of_test, 0);
        let abr = self.b.ins().bxor(ab, res);
        let af_test = self.b.ins().band_imm_s(abr, 0x10);
        let af = self.b.ins().icmp_imm_s(IntCC::NotEqual, af_test, 0);
        let p = self.parity(res);
        let t = self.bool_to_i8(cf);
        self.b.def_var(self.cf, t);
        let t = self.bool_to_i8(zf);
        self.b.def_var(self.zf, t);
        let t = self.bool_to_i8(sf);
        self.b.def_var(self.sf, t);
        let t = self.bool_to_i8(of);
        self.b.def_var(self.of, t);
        let t = self.bool_to_i8(af);
        self.b.def_var(self.af, t);
        self.b.def_var(self.pf, p);
    }

    fn flags_logic(&mut self, result: cranelift_codegen::ir::Value, wide: Wide) {
        let bits = wide.bits();
        let sign = self.iconst(1i64 << (bits - 1));
        let mask = self.iconst(wide.mask());
        let res = self.b.ins().band(result, mask);
        let zf = self.b.ins().icmp_imm_s(IntCC::Equal, res, 0);
        let sf_test = self.b.ins().band(res, sign);
        let sf = self.b.ins().icmp_imm_s(IntCC::NotEqual, sf_test, 0);
        let p = self.parity(res);
        let zero8 = self.b.ins().iconst(types::I8, 0);
        self.b.def_var(self.cf, zero8);
        self.b.def_var(self.of, zero8);
        self.b.def_var(self.af, zero8);
        let t = self.bool_to_i8(zf);
        self.b.def_var(self.zf, t);
        let t = self.bool_to_i8(sf);
        self.b.def_var(self.sf, t);
        self.b.def_var(self.pf, p);
    }

    /// Even-parity of the low byte (matches the interpreter's parity()).
    fn parity(&mut self, v: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        // Fold in the I64 domain; only the final predicate is an I8 icmp.
        let b = self.b.ins().band_imm_s(v, 0xFF);
        let s4 = self.b.ins().ushr_imm_s(b, 4);
        let x1 = self.b.ins().bxor(b, s4);
        let s2 = self.b.ins().ushr_imm_s(x1, 2);
        let x2 = self.b.ins().bxor(x1, s2);
        let s1 = self.b.ins().ushr_imm_s(x2, 1);
        let x3 = self.b.ins().bxor(x2, s1);
        let lsb = self.b.ins().band_imm_s(x3, 1);
        let odd = self.b.ins().icmp_imm_s(IntCC::NotEqual, lsb, 0);
        // PF is set ⟺ even parity ⟺ NOT odd.
        self.bnot1(odd)
    }

    // ----- memory ---------------------------------------------------------------

    fn ea(&mut self, i: &Instruction) -> cranelift_codegen::ir::Value {
        let mut addr = self.iconst(i.memory_displacement64() as i64);
        let base = i.memory_base();
        if base != Register::None && base != Register::RIP {
            let bv = self.b.use_var(self.reg_var(base).expect("GPR base"));
            addr = self.b.ins().iadd(addr, bv);
        }
        let index = i.memory_index();
        if index != Register::None {
            let iv = self.b.use_var(self.reg_var(index).expect("GPR index"));
            let scaled = self.b.ins().imul_imm_s(iv, i.memory_index_scale() as i64);
            addr = self.b.ins().iadd(addr, scaled);
        }
        addr
    }

    /// TLB probe: (tag, host-page-base) slot values for `va`.
    fn tlb_slot(
        &mut self,
        va: cranelift_codegen::ir::Value,
    ) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
        let tag = self.b.ins().ushr_imm_s(va, 12);
        let idx = self.b.ins().band_imm_s(tag, (TLB_ENTRIES - 1) as i64);
        let times8 = self.b.ins().imul_imm_s(idx, 8);
        let tags_base = self
            .b
            .ins()
            .iadd_imm_s(self.env_ptr, i64::from(OFF_TLB_TAGS));
        let t_addr = self.b.ins().iadd(tags_base, times8);
        let t = self.b.ins().load(types::I64, M, t_addr, 0);
        let ptrs_base = self
            .b
            .ins()
            .iadd_imm_s(self.env_ptr, i64::from(OFF_TLB_PTRS));
        let p_addr = self.b.ins().iadd(ptrs_base, times8);
        let p = self.b.ins().load(types::I64, M, p_addr, 0);
        (t, p)
    }

    fn load_width(
        &mut self,
        w: u8,
        host: cranelift_codegen::ir::Value,
    ) -> cranelift_codegen::ir::Value {
        let ty = match w {
            1 => types::I8,
            2 => types::I16,
            4 => types::I32,
            _ => types::I64,
        };
        let v = self.b.ins().load(ty, M, host, 0);
        if w == 8 {
            v
        } else {
            self.b.ins().uextend(types::I64, v)
        }
    }

    fn store_width(
        &mut self,
        w: u8,
        host: cranelift_codegen::ir::Value,
        v: cranelift_codegen::ir::Value,
    ) {
        let ty = match w {
            1 => types::I8,
            2 => types::I16,
            4 => types::I32,
            _ => types::I64,
        };
        let rv = if w == 8 {
            v
        } else {
            self.b.ins().ireduce(ty, v)
        };
        self.b.ins().store(M, rv, host, 0);
    }

    /// Inline-TLB load of `w` bytes at `va`.
    fn mem_load(
        &mut self,
        w: u8,
        va: cranelift_codegen::ir::Value,
        rip: u64,
    ) -> cranelift_codegen::ir::Value {
        let (t, p) = self.tlb_slot(va);
        let vtag = self.b.ins().ushr_imm_s(va, 12);
        let hit = self.b.ins().icmp(IntCC::Equal, t, vtag);
        let p_nz = self.b.ins().icmp_imm_s(IntCC::NotEqual, p, 0);
        let can_fast = self.b.ins().band(hit, p_nz);
        let off = self.b.ins().band_imm_s(va, 0xFFF);
        let limit = self.iconst(0x1000 - i64::from(w));
        let cross = self.b.ins().icmp(IntCC::UnsignedGreaterThan, off, limit);
        let not_cross = self.bnot1(cross);
        let fast_ok = self.b.ins().band(can_fast, not_cross);

        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let join = self.b.create_block();
        self.b.append_block_param(join, types::I64);
        self.b.ins().brif(fast_ok, fast, &[], slow, &[]);

        self.b.switch_to_block(fast);
        self.b.seal_block(fast);
        let host = self.b.ins().iadd(p, off);
        let v = self.load_width(w, host);
        self.b.ins().jump(join, &[v.into()]);

        self.b.switch_to_block(slow);
        self.b.seal_block(slow);
        let rrip = self.iconst(rip as i64);
        let fill = self.b.ins().call(self.tlb_load, &[self.env_ptr, rrip, va]);
        let base2 = self.b.inst_results(fill)[0];
        self.check_helper_exit();
        let ok2 = self.b.ins().icmp_imm_s(IntCC::NotEqual, base2, 0);
        let direct = self.b.create_block();
        let helper = self.b.create_block();
        self.b.ins().brif(ok2, direct, &[], helper, &[]);

        self.b.switch_to_block(direct);
        self.b.seal_block(direct);
        let host2 = self.b.ins().iadd(base2, off);
        let v2 = self.load_width(w, host2);
        self.b.ins().jump(join, &[v2.into()]);

        // Not host-addressable (MMIO/ROM): the read helper.
        self.b.switch_to_block(helper);
        self.b.seal_block(helper);
        let readfn = self.read_ref[Wide::of(w as u32).idx()];
        let call = self.b.ins().call(readfn, &[self.env_ptr, rrip, va]);
        let v3 = self.b.inst_results(call)[0];
        self.check_helper_exit();
        self.b.ins().jump(join, &[v3.into()]);

        self.b.switch_to_block(join);
        self.b.seal_block(join);
        self.b.block_params(join)[0]
    }

    /// Inline-TLB store with the SMC bloom check in front.
    fn mem_store(
        &mut self,
        w: u8,
        va: cranelift_codegen::ir::Value,
        v: cranelift_codegen::ir::Value,
        rip: u64,
    ) {
        // ---- SMC bloom over guest code pages ------------------------------
        let page = self.b.ins().ushr_imm_s(va, 12);
        let p7 = self.b.ins().ushr_imm_s(page, 7);
        let p13 = self.b.ins().ushr_imm_s(page, 13);
        let x1 = self.b.ins().bxor(page, p7);
        let mixed = self.b.ins().bxor(x1, p13);
        let hashed = self
            .b
            .ins()
            .imul_imm_s(mixed, 0x7C15_F4A7_9B79_E379u64 as i64);
        let h33 = self.b.ins().ushr_imm_s(hashed, 33);
        let h = self
            .b
            .ins()
            .band_imm_s(h33, (env::SMC_WORDS * 64 - 1) as i64);
        let word_idx = self.b.ins().ushr_imm_s(h, 6);
        let wtimes8 = self.b.ins().imul_imm_s(word_idx, 8);
        let bloom_base = self.b.ins().iadd_imm_s(self.env_ptr, i64::from(OFF_BLOOM));
        let waddr = self.b.ins().iadd(bloom_base, wtimes8);
        let word = self.b.ins().load(types::I64, M, waddr, 0);
        let bitpos = self.b.ins().band_imm_s(h, 63);
        let shifted = self.b.ins().ushr(word, bitpos);
        let bit = self.b.ins().band_imm_s(shifted, 1);
        let maybe = self.b.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
        let clean = self.b.create_block();
        let dirty = self.b.create_block();
        self.b.ins().brif(maybe, dirty, &[], clean, &[]);
        self.b.switch_to_block(dirty);
        self.b.seal_block(dirty);
        let rrip = self.iconst(rip as i64);
        self.b.ins().call(self.smc_ref, &[self.env_ptr, rrip, va]);
        self.exit_with(env::exit::RESTART, rrip);
        self.b.switch_to_block(clean);
        self.b.seal_block(clean);

        // ---- TLB lookup ------------------------------------------------------
        let (t, p) = self.tlb_slot(va);
        let vtag = self.b.ins().ushr_imm_s(va, 12);
        let hit = self.b.ins().icmp(IntCC::Equal, t, vtag);
        let p_nz = self.b.ins().icmp_imm_s(IntCC::NotEqual, p, 0);
        let can_fast = self.b.ins().band(hit, p_nz);
        let off = self.b.ins().band_imm_s(va, 0xFFF);
        let limit = self.iconst(0x1000 - i64::from(w));
        let cross = self.b.ins().icmp(IntCC::UnsignedGreaterThan, off, limit);
        let not_cross = self.bnot1(cross);
        let fast_ok = self.b.ins().band(can_fast, not_cross);

        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(fast_ok, fast, &[], slow, &[]);

        self.b.switch_to_block(fast);
        self.b.seal_block(fast);
        let host = self.b.ins().iadd(p, off);
        self.store_width(w, host, v);
        self.b.ins().jump(done, &[]);

        self.b.switch_to_block(slow);
        self.b.seal_block(slow);
        let rrip = self.iconst(rip as i64);
        let fill = self.b.ins().call(self.tlb_store, &[self.env_ptr, rrip, va]);
        let base2 = self.b.inst_results(fill)[0];
        self.check_helper_exit();
        let ok2 = self.b.ins().icmp_imm_s(IntCC::NotEqual, base2, 0);
        let direct = self.b.create_block();
        let helper = self.b.create_block();
        self.b.ins().brif(ok2, direct, &[], helper, &[]);

        self.b.switch_to_block(direct);
        self.b.seal_block(direct);
        let host2 = self.b.ins().iadd(base2, off);
        self.store_width(w, host2, v);
        self.b.ins().jump(done, &[]);

        // Not host-addressable (MMIO/ROM): the write helper.
        self.b.switch_to_block(helper);
        self.b.seal_block(helper);
        let writefn = self.write_ref[Wide::of(w as u32).idx()];
        self.b.ins().call(writefn, &[self.env_ptr, rrip, va, v]);
        self.check_helper_exit();
        self.b.ins().jump(done, &[]);

        self.b.switch_to_block(done);
        self.b.seal_block(done);
    }

    // ----- operands ---------------------------------------------------------------

    fn read_operand(
        &mut self,
        i: &Instruction,
        n: u32,
        rip: u64,
    ) -> Result<cranelift_codegen::ir::Value, LiftError> {
        let w = op_w(i, n);
        match i.op_kind(n) {
            OpKind::Register => Ok(self.read_gpr_op(i, n)),
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => Ok(self.iconst(i64::from(i.immediate8() as i8))),
            OpKind::Immediate16 => Ok(self.iconst(i64::from(i.immediate16() as i16))),
            OpKind::Immediate32 => Ok(self.iconst(i64::from(i.immediate32() as i32))),
            OpKind::Immediate32to64 => Ok(self.iconst(i.immediate32to64())),
            OpKind::Immediate64 => Ok(self.iconst(i.immediate64() as i64)),
            OpKind::Memory => {
                let va = self.ea(i);
                Ok(self.mem_load(w.min(8) as u8, va, rip))
            }
            _ => Err(LiftError::Decode { rip: i.ip() }),
        }
    }

    fn write_operand(
        &mut self,
        i: &Instruction,
        n: u32,
        v: cranelift_codegen::ir::Value,
        rip: u64,
    ) -> Result<(), LiftError> {
        let w = op_w(i, n);
        match i.op_kind(n) {
            OpKind::Register => {
                self.write_gpr_op(i.op_register(n), v, w);
                Ok(())
            }
            OpKind::Memory => {
                let va = self.ea(i);
                self.mem_store(w.min(8) as u8, va, v, rip);
                Ok(())
            }
            _ => Err(LiftError::Decode { rip: i.ip() }),
        }
    }

    // ----- instruction table --------------------------------------------------------

    fn lift(&mut self, i: &Instruction) -> Result<bool, LiftError> {
        use Mnemonic::*;
        self.terminated = false;
        let rip = i.ip();
        let next_ip = i.next_ip();

        // FS/GS-relative data goes through the interpreter.
        if matches!(i.segment_prefix(), Register::FS | Register::GS) {
            let r = self.iconst(rip as i64);
            self.exit_with(env::exit::FALLBACK, r);
            return Ok(true);
        }

        match i.mnemonic() {
            Nop | Endbr64 | Endbr32 => Ok(false),

            Mov => {
                let v = self.read_operand(i, 1, rip)?;
                self.write_operand(i, 0, v, rip)?;
                Ok(false)
            }
            Lea => {
                let addr = self.ea(i);
                self.write_gpr_op(i.op_register(0), addr, op_w(i, 0));
                Ok(false)
            }

            Add | Sub | And | Or | Xor | Cmp | Test | Adc | Sbb => {
                self.lift_alu(i, rip)?;
                Ok(false)
            }
            Inc | Dec => {
                self.lift_incdec(i, rip)?;
                Ok(false)
            }
            Neg | Not => {
                self.lift_unary(i, rip)?;
                Ok(false)
            }
            Shl | Sal | Shr | Sar => {
                self.lift_shift(i, rip)?;
                Ok(false)
            }

            Push => {
                let v = self.read_operand(i, 0, rip)?;
                let sp = self.b.use_var(self.g[RSP]);
                let new_sp = self.b.ins().iadd_imm_s(sp, -8);
                self.b.def_var(self.g[RSP], new_sp);
                self.mem_store(8, new_sp, v, rip);
                Ok(false)
            }
            Pop => {
                let sp = self.b.use_var(self.g[RSP]);
                let v = self.mem_load(8, sp, rip);
                let new_sp = self.b.ins().iadd_imm_s(sp, 8);
                self.b.def_var(self.g[RSP], new_sp);
                self.write_operand(i, 0, v, rip)?;
                Ok(false)
            }

            Call => {
                let target = match i.op_kind(0) {
                    OpKind::NearBranch64 | OpKind::NearBranch32 => {
                        self.iconst(i.near_branch_target() as i64)
                    }
                    _ => self.read_operand(i, 0, rip)?,
                };
                let sp = self.b.use_var(self.g[RSP]);
                let new_sp = self.b.ins().iadd_imm_s(sp, -8);
                self.b.def_var(self.g[RSP], new_sp);
                let ret = self.iconst(next_ip as i64);
                self.mem_store(8, new_sp, ret, rip);
                self.exit_with(env::exit::CONTINUE, target);
                Ok(true)
            }
            Ret => {
                let sp = self.b.use_var(self.g[RSP]);
                let v = self.mem_load(8, sp, rip);
                let mut new_sp = self.b.ins().iadd_imm_s(sp, 8);
                if let OpKind::Immediate16 = i.op_kind(0) {
                    new_sp = self.b.ins().iadd_imm_s(new_sp, i64::from(i.immediate16()));
                }
                self.b.def_var(self.g[RSP], new_sp);
                self.exit_with(env::exit::CONTINUE, v);
                Ok(true)
            }
            Jmp => {
                let target = match i.op_kind(0) {
                    OpKind::NearBranch64 | OpKind::NearBranch32 => {
                        self.iconst(i.near_branch_target() as i64)
                    }
                    _ => self.read_operand(i, 0, rip)?,
                };
                self.exit_with(env::exit::CONTINUE, target);
                Ok(true)
            }
            Jno | Jb | Jae | Je | Jne | Jbe | Ja | Js | Jns | Jp | Jnp | Jl | Jge | Jle | Jg => {
                let cond = self.cond_value(i.condition_code());
                let target = i.near_branch_target();
                if let Some((t, header)) = self.loop_header {
                    if target == t {
                        // Back-edge: budget re-check, then the loop jump.
                        let budget = self.b.ins().load(types::I64, M, self.env_ptr, OFF_BUDGET);
                        let nb = self.b.ins().iadd_imm_s(budget, -(self.loop_cost as i64));
                        self.b.ins().store(M, nb, self.env_ptr, OFF_BUDGET);
                        let neg = self.b.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, nb, 0);
                        let out = self.b.create_block();
                        let cont = self.b.create_block();
                        self.b.ins().brif(neg, out, &[], cont, &[]);
                        self.b.switch_to_block(out);
                        self.b.seal_block(out);
                        let tr = self.iconst(t as i64);
                        self.exit_with(env::exit::BUDGET, tr);
                        self.b.switch_to_block(cont);
                        self.b.seal_block(cont);
                        // Note: the budget exit above left  set,
                        // but / are fresh blocks.
                        let fallthrough = self.b.create_block();
                        self.b.ins().brif(cond, header, &[], fallthrough, &[]);
                        self.b.switch_to_block(fallthrough);
                        self.b.seal_block(fallthrough);
                        self.terminated = false;
                        return Ok(false);
                    }
                }
                // External target: exit when taken, keep the fallthrough.
                let exit_taken = self.b.create_block();
                let cont = self.b.create_block();
                self.b.ins().brif(cond, exit_taken, &[], cont, &[]);
                self.b.switch_to_block(exit_taken);
                self.b.seal_block(exit_taken);
                let tv = self.iconst(target as i64);
                self.exit_with(env::exit::CONTINUE, tv);
                self.b.switch_to_block(cont);
                self.b.seal_block(cont);
                // `decode_block` always ends a block at the jcc, so the
                // not-taken path leaves this block too (the next guest
                // instruction starts a new one). Emit the fallthrough exit
                // so `cont` gets a terminator — without it, every forward
                // conditional branch failed cranelift's verifier and fell
                // back to the interpreter forever.
                let nv = self.iconst(next_ip as i64);
                self.exit_with(env::exit::CONTINUE, nv);
                Ok(true)
            }

            Cmovno | Cmovb | Cmovae | Cmove | Cmovne | Cmovbe | Cmova | Cmovs | Cmovns | Cmovp
            | Cmovnp | Cmovl | Cmovge | Cmovle | Cmovg => {
                let cond = self.cond_value(i.condition_code());
                let src = self.read_operand(i, 1, rip)?;
                let dst = self.read_operand(i, 0, rip)?;
                let v = self.b.ins().select(cond, src, dst);
                self.write_gpr_op(i.op_register(0), v, op_w(i, 0));
                Ok(false)
            }

            Movzx | Movsx | Movsxd => {
                let src_w = match i.op_kind(1) {
                    OpKind::Memory => i.memory_size().size() as u32,
                    OpKind::Register => {
                        decode_reg(i.op_register(1)).map(|(_, s, _)| s).unwrap_or(8)
                    }
                    _ => 8,
                };
                let raw = self.read_operand(i, 1, rip)?;
                let v = if i.mnemonic() == Movzx {
                    self.b.ins().band_imm_s(raw, mask_of(src_w))
                } else {
                    let sh = self.iconst(64 - i64::from(src_w * 8));
                    let left = self.b.ins().ishl(raw, sh);
                    self.b.ins().sshr(left, sh)
                };
                self.write_gpr_op(i.op_register(0), v, op_w(i, 0));
                Ok(false)
            }

            Xchg => {
                if i.op_kind(0) == OpKind::Register && i.op_kind(1) == OpKind::Register {
                    let a = self.read_operand(i, 0, rip)?;
                    let bv = self.read_operand(i, 1, rip)?;
                    self.write_operand(i, 0, bv, rip)?;
                    self.write_operand(i, 1, a, rip)?;
                    Ok(false)
                } else {
                    // Memory xchg is implicitly locked: interpreter fallback.
                    let r = self.iconst(rip as i64);
                    self.exit_with(env::exit::FALLBACK, r);
                    Ok(true)
                }
            }

            Hlt => {
                let hv = self.iconst(rip as i64);
                self.exit_with(env::exit::HALT, hv);
                Ok(true)
            }
            Syscall => {
                let sv = self.iconst(next_ip as i64);
                self.exit_with(env::exit::SYSCALL, sv);
                Ok(false)
            }

            _ => {
                let r = self.iconst(rip as i64);
                self.exit_with(env::exit::FALLBACK, r);
                Ok(true)
            }
        }
    }

    fn lift_alu(&mut self, i: &Instruction, rip: u64) -> Result<(), LiftError> {
        use Mnemonic::*;
        let wide = Wide::of(op_w(i, 0));
        let mask = self.iconst(wide.mask());
        let ra = self.read_operand(i, 0, rip)?;
        let a = self.b.ins().band(ra, mask);
        let rb = self.read_operand(i, 1, rip)?;
        let b = self.b.ins().band(rb, mask);
        let cf0 = {
            let raw = self.b.use_var(self.cf);
            self.b.ins().band_imm_s(raw, 1)
        };
        match i.mnemonic() {
            Add | Adc => {
                let c = if i.mnemonic() == Adc {
                    cf0
                } else {
                    self.iconst(0)
                };
                let r = self.b.ins().iadd(a, b);
                let r = self.b.ins().iadd(r, c);
                self.flags_add(a, b, c, r, wide);
                self.write_operand(i, 0, r, rip)
            }
            Sub | Cmp | Sbb => {
                let c = if i.mnemonic() == Sbb {
                    cf0
                } else {
                    self.iconst(0)
                };
                let r = self.b.ins().isub(a, b);
                let r = self.b.ins().isub(r, c);
                self.flags_sub(a, b, c, r, wide);
                if i.mnemonic() != Cmp {
                    self.write_operand(i, 0, r, rip)
                } else {
                    Ok(())
                }
            }
            And | Test => {
                let r = self.b.ins().band(a, b);
                self.flags_logic(r, wide);
                if i.mnemonic() == And {
                    self.write_operand(i, 0, r, rip)
                } else {
                    Ok(())
                }
            }
            Or => {
                let r = self.b.ins().bor(a, b);
                self.flags_logic(r, wide);
                self.write_operand(i, 0, r, rip)
            }
            Xor => {
                let r = self.b.ins().bxor(a, b);
                self.flags_logic(r, wide);
                self.write_operand(i, 0, r, rip)
            }
            _ => unreachable!("lift_alu dispatch"),
        }
    }

    fn lift_incdec(&mut self, i: &Instruction, rip: u64) -> Result<(), LiftError> {
        let wide = Wide::of(op_w(i, 0));
        let mask = self.iconst(wide.mask());
        let raw = self.read_operand(i, 0, rip)?;
        let a = self.b.ins().band(raw, mask);
        let saved_cf = self.b.use_var(self.cf);
        let one = self.iconst(1);
        let zero = self.iconst(0);
        let (r, is_sub) = if i.mnemonic() == Mnemonic::Inc {
            (self.b.ins().iadd(a, one), false)
        } else {
            (self.b.ins().isub(a, one), true)
        };
        if is_sub {
            self.flags_sub(a, one, zero, r, wide);
        } else {
            self.flags_add(a, one, zero, r, wide);
        }
        self.b.def_var(self.cf, saved_cf);
        self.write_operand(i, 0, r, rip)
    }

    fn lift_unary(&mut self, i: &Instruction, rip: u64) -> Result<(), LiftError> {
        let wide = Wide::of(op_w(i, 0));
        let mask = self.iconst(wide.mask());
        let raw = self.read_operand(i, 0, rip)?;
        let a = self.b.ins().band(raw, mask);
        if i.mnemonic() == Mnemonic::Not {
            let notv = self.b.ins().bnot(a);
            let r = self.b.ins().band(notv, mask);
            self.write_operand(i, 0, r, rip)
        } else {
            let zero = self.iconst(0);
            let r = self.b.ins().isub(zero, a);
            self.flags_sub(zero, a, zero, r, wide);
            let r = self.b.ins().band(r, mask);
            self.write_operand(i, 0, r, rip)
        }
    }

    fn lift_shift(&mut self, i: &Instruction, rip: u64) -> Result<(), LiftError> {
        use Mnemonic::*;
        let wide = Wide::of(op_w(i, 0));
        let bits = wide.bits();
        let mask = self.iconst(wide.mask());
        let raw_op = self.read_operand(i, 0, rip)?;
        let a = self.b.ins().band(raw_op, mask);
        let raw = match i.op_kind(1) {
            OpKind::Immediate8 | OpKind::Immediate8to64 => self.iconst(i64::from(i.immediate8())),
            OpKind::Register => {
                let cl = self.b.use_var(self.g[RCX]);
                self.b.ins().band_imm_s(cl, 0xFF)
            }
            _ => self.iconst(0),
        };
        let count_mask = if bits == 64 { 0x3F } else { 0x1F };
        let count = self.b.ins().band_imm_s(raw, count_mask);

        // count == 0 → no flags, no writeback.
        let cont = self.b.create_block();
        let end = self.b.create_block();
        let cz = self.b.ins().icmp_imm_s(IntCC::Equal, count, 0);
        self.b.ins().brif(cz, end, &[], cont, &[]);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);

        let bits_v = self.iconst(bits);
        let zero = self.iconst(0);
        let one_v = self.iconst(1);
        let cnt_ok = self.b.ins().icmp(IntCC::UnsignedLessThan, count, bits_v);

        let (result, cf, of) = match i.mnemonic() {
            Shl | Sal => {
                let sh = self.b.ins().select(cnt_ok, count, zero);
                let shv = self.b.ins().ishl(a, sh);
                let r = self.b.ins().band(shv, mask);
                let from = self.b.ins().isub(bits_v, count);
                let a_shifted = self.b.ins().ushr(a, from);
                let cf = self.b.ins().band_imm_s(a_shifted, 1);
                let cfz = self.b.ins().select(cnt_ok, cf, zero);
                let top_sh = self.iconst(bits - 1);
                let topv = self.b.ins().ushr(r, top_sh);
                let top = self.b.ins().band_imm_s(topv, 1);
                let ofv = self.b.ins().bxor(top, cfz);
                (r, cfz, ofv)
            }
            Shr => {
                let sh = self.b.ins().select(cnt_ok, count, zero);
                let r = self.b.ins().ushr(a, sh);
                let from = self.b.ins().isub(count, one_v);
                let a_shifted = self.b.ins().ushr(a, from);
                let cf = self.b.ins().band_imm_s(a_shifted, 1);
                let cfz = self.b.ins().select(cnt_ok, cf, zero);
                let top_sh = self.iconst(bits - 1);
                let topv = self.b.ins().ushr(a, top_sh);
                let ofv = self.b.ins().band_imm_s(topv, 1);
                (r, cfz, ofv)
            }
            Sar => {
                let sx = if bits == 64 {
                    a
                } else {
                    let shamt = self.iconst(64 - bits);
                    let l = self.b.ins().ishl(a, shamt);
                    self.b.ins().sshr(l, shamt)
                };
                let sixty3 = self.iconst(63);
                let sh = self.b.ins().select(cnt_ok, count, sixty3);
                let shv = self.b.ins().sshr(sx, sh);
                let r = self.b.ins().band(shv, mask);
                let from = self.b.ins().isub(count, one_v);
                let sx_shifted = self.b.ins().sshr(sx, from);
                let cf = self.b.ins().band_imm_s(sx_shifted, 1);
                let cfz = self.b.ins().select(cnt_ok, cf, zero);
                (r, cfz, zero)
            }
            _ => unreachable!("lift_shift dispatch"),
        };

        let is_one = self.b.ins().icmp_imm_s(IntCC::Equal, count, 1);
        let of_s = self.b.ins().select(is_one, of, zero);

        self.flags_logic(result, wide);
        let cf8v = self.b.ins().icmp_imm_s(IntCC::NotEqual, cf, 0);
        self.b.def_var(self.cf, cf8v);
        let of8v = self.b.ins().icmp_imm_s(IntCC::NotEqual, of_s, 0);
        self.b.def_var(self.of, of8v);

        self.write_operand(i, 0, result, rip)?;
        self.b.ins().jump(end, &[]);
        self.b.switch_to_block(end);
        self.b.seal_block(end);
        Ok(())
    }

    /// Evaluate a condition code against the flag variables (booleans).
    fn cond_value(&mut self, cc: iced_x86::ConditionCode) -> cranelift_codegen::ir::Value {
        use iced_x86::ConditionCode as CC;
        let flag = |l: &mut Self, v: Variable| {
            let raw = l.b.use_var(v);
            l.b.ins().icmp_imm_s(IntCC::NotEqual, raw, 0)
        };
        let cf = flag(self, self.cf);
        let zf = flag(self, self.zf);
        let sf = flag(self, self.sf);
        let of = flag(self, self.of);
        let pf = flag(self, self.pf);
        let sf_xor_of = self.b.ins().bxor(sf, of);
        match cc {
            CC::o => of,
            CC::no => self.bnot1(of),
            CC::b => cf,
            CC::ae => self.bnot1(cf),
            CC::e => zf,
            CC::ne => self.bnot1(zf),
            CC::be => self.b.ins().bor(cf, zf),
            CC::a => {
                let ncf = self.bnot1(cf);
                let nzf = self.bnot1(zf);
                self.b.ins().band(ncf, nzf)
            }
            CC::s => sf,
            CC::ns => self.bnot1(sf),
            CC::p => pf,
            CC::np => self.bnot1(pf),
            CC::l => sf_xor_of,
            CC::ge => self.bnot1(sf_xor_of),
            CC::le => self.b.ins().bor(zf, sf_xor_of),
            CC::g => {
                let nzf = self.bnot1(zf);
                let nsx = self.bnot1(sf_xor_of);
                self.b.ins().band(nzf, nsx)
            }
            _ => self.bnot1(zf),
        }
    }
}

/// RSP/RCX indices (private re-import to avoid name noise).
mod gpr_idx {
    pub const RSP: usize = 4;
    pub const RCX: usize = 1;
}
use gpr_idx::{RCX, RSP};
