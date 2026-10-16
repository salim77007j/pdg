//! The interpreter: fetch → decode → execute for one logical processor.
//!
//! Decoding is delegated to `iced-x86` (fast, complete x86-64 decoding);
//! this module owns *semantics*: operand evaluation (including RIP-relative
//! addressing, FS/GS segment bases and x86-64 register-write rules),
//! architectural flags, the stack, control flow, string operations with
//! `REP` restart semantics, atomic (LOCK-prefixed) operations, exception
//! delivery through the guest IDT, and fault reporting.
//!
//! Anything outside the implemented subset raises `#UD` — nothing executes
//! as a silent no-op. The SIMD/AVX families live in [`crate::simd`] and are
//! dispatched from here.

use iced_x86::{ConditionCode, Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

use crate::{
    cpuid,
    fault::{AccessError, Fault, MemoryAccess},
    hooks::SyscallHook,
    state::{self, CpuState, mask},
};

/// Result of a `run()` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Guest executed `hlt`. `rip` is the address of the `hlt`.
    Halted {
        /// Instructions retired.
        instructions: u64,
        /// RIP at the `hlt`.
        rip: u64,
    },
    /// Instruction budget exhausted.
    MaxInstructions {
        /// Instructions retired.
        instructions: u64,
    },
    /// Architectural fault (or host error) stopped execution.
    Fault(Fault),
}

#[derive(Debug)]
pub(crate) enum Action {
    Next,
    Jump(u64),
    Halt,
    /// A `REP` string op hit its per-step iteration cap with work remaining:
    /// architectural state is restartable, RIP stays put and the outer loop
    /// re-enters the instruction (the same way a faulted-then-fixed `REP`
    /// resumes on real hardware).
    Yield,
}

/// Cap on string-op iterations per `step()` call, so one `rep movsq` cannot
/// monopolise the run loop. Registers hold the remaining work, making the
/// re-entry exact.
const REP_ITER_CAP: u64 = 4096;

/// One logical Zen 2 processor running in interpreter mode.
pub struct Interpreter {
    /// Architectural state (public for debugger/tooling access).
    pub state: CpuState,
    /// HLE kernel hook invoked on `syscall`.
    pub syscall_hook: Option<Box<dyn SyscallHook>>,
}

impl Interpreter {
    /// Create an interpreter around `state`.
    pub fn new(state: CpuState) -> Self {
        Self {
            state,
            syscall_hook: None,
        }
    }

    /// Attach an HLE `syscall` handler.
    pub fn set_syscall_hook(&mut self, hook: Box<dyn SyscallHook>) {
        self.syscall_hook = Some(hook);
    }

    /// Execute up to `max_instructions` guest instructions.
    pub fn run(&mut self, mem: &mut dyn MemoryAccess, max_instructions: u64) -> RunOutcome {
        let mut executed = 0u64;
        while executed < max_instructions {
            match self.step(mem) {
                Ok(StepEnd::Continue) => {
                    executed += 1;
                    self.state.tsc = self.state.tsc.wrapping_add(1);
                }
                Ok(StepEnd::Halted) => {
                    return RunOutcome::Halted {
                        instructions: executed + 1,
                        rip: self.state.rip,
                    };
                }
                Err(f) => {
                    if !self.try_deliver_fault(&f, mem) {
                        return RunOutcome::Fault(f);
                    }
                }
            }
        }
        RunOutcome::MaxInstructions {
            instructions: executed,
        }
    }

    fn step(&mut self, mem: &mut dyn MemoryAccess) -> Result<StepEnd, Fault> {
        let rip = self.state.rip;
        let mut window = [0u8; 16]; // max x86-64 instruction length is 15
        let n = mem
            .fetch(rip, &mut window)
            .map_err(AccessError::into_fault)?;
        let mut decoder = Decoder::with_ip(64, &window[..n], rip, DecoderOptions::NONE);
        let instr = decoder.decode();
        if instr.is_invalid() {
            // A truncated fetch window makes decodes fail; distinguish a
            // genuine #UD encoding from an instruction running off the end
            // of a mapped page by probing the next byte (Phase-1 issue #1).
            if n < window.len() {
                let mut probe = [0u8; 1];
                if let Err(e) = mem.fetch(rip + n as u64, &mut probe) {
                    return Err(e.into_fault());
                }
            }
            return Err(Fault::InvalidOpcode {
                mnemonic: "(bad encoding)".to_owned(),
                rip,
            });
        }
        let next_ip = instr.next_ip();
        match self.execute(&instr, next_ip, mem)? {
            Action::Next => self.state.rip = next_ip,
            Action::Jump(target) => self.state.rip = target,
            Action::Halt => return Ok(StepEnd::Halted),
            Action::Yield => {}
        }
        Ok(StepEnd::Continue)
    }

    // ----- operand plumbing -------------------------------------------------

    /// Operand kind of operand `n`.
    fn kind(i: &Instruction, n: u32) -> OpKind {
        i.op_kind(n)
    }

    /// Operand size in bytes (used to mask results and select access width).
    fn op_width(i: &Instruction, n: u32) -> u32 {
        match Self::kind(i, n) {
            OpKind::Register => {
                let reg = i.op_register(n);
                let (_, size, _) = crate::state::decode_reg(reg).expect("GPR operand");
                size
            }
            OpKind::Memory => i.memory_size().size() as u32,
            _ => 8,
        }
    }

    /// Effective address of operand `n` (must be a memory operand).
    ///
    /// Notes from Phase 1 kept true: iced's `memory_displacement64()` returns
    /// the *computed absolute target* (next_ip + disp) for RIP-based
    /// operands, not the raw displacement. Phase 2 adds FS/GS segment-base
    /// application (the only segments with nonzero bases in 64-bit mode).
    fn ea(&self, i: &Instruction, next_ip: u64) -> u64 {
        effective_address(i, next_ip, &self.state)
    }

    /// Read operand `n` (register, immediate or memory).
    fn read_op(
        &self,
        i: &Instruction,
        n: u32,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<u64, Fault> {
        let width = Self::op_width(i, n);
        match Self::kind(i, n) {
            OpKind::Register => Ok(self.state.reg(i.op_register(n)).expect("GPR read")),
            // Raw immediates: sign-extend from the immediate's own width so
            // consumers can mask to the operand width.
            OpKind::Immediate8 => Ok(sext_bits(i.immediate8() as u64, 8) as u64),
            OpKind::Immediate16 => Ok(sext_bits(i.immediate16() as u64, 16) as u64),
            OpKind::Immediate32 => Ok(sext_bits(i.immediate32() as u64, 32) as u64),
            // Sign-extended forms: iced's dedicated getters already did the work.
            OpKind::Immediate8to16 | OpKind::Immediate8to32 | OpKind::Immediate8to64 => {
                Ok(i.immediate8to64() as u64)
            }
            OpKind::Immediate32to64 => Ok(i.immediate32to64() as u64),
            OpKind::Immediate64 => Ok(i.immediate64()),
            OpKind::Memory => {
                let addr = self.ea(i, next_ip);
                mem.read(width as u8, addr).map_err(AccessError::into_fault)
            }
            _ => Err(Fault::InvalidOpcode {
                mnemonic: format!("{:?}/op{}", i.mnemonic(), n),
                rip: i.ip(),
            }),
        }
    }

    /// Write operand `n` (register or memory).
    fn write_op(
        &mut self,
        i: &Instruction,
        n: u32,
        next_ip: u64,
        value: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<(), Fault> {
        match Self::kind(i, n) {
            OpKind::Register => {
                self.state
                    .set_reg(i.op_register(n), value)
                    .expect("GPR write");
                Ok(())
            }
            OpKind::Memory => {
                let width = Self::op_width(i, n);
                let addr = self.ea(i, next_ip);
                mem.write(width as u8, addr, value)
                    .map_err(AccessError::into_fault)
            }
            _ => Err(Fault::InvalidOpcode {
                mnemonic: format!("{:?}/op{}", i.mnemonic(), n),
                rip: i.ip(),
            }),
        }
    }

    /// Read the shift count (imm8 or CL) without sign extension.
    fn shift_count(
        &self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<u64, Fault> {
        match Self::kind(i, 1) {
            OpKind::Immediate8 | OpKind::Immediate8to64 => Ok(i.immediate8() as u64),
            OpKind::Register => Ok(self.state.reg(i.op_register(1)).expect("GPR count") & 0xFF),
            OpKind::Memory => self.read_op(i, 1, next_ip, mem),
            _ => Ok(0),
        }
    }

    /// Branch target: near-branch immediate, register indirect, or memory
    /// indirect.
    fn branch_target(
        &self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<u64, Fault> {
        match Self::kind(i, 0) {
            OpKind::NearBranch64 | OpKind::NearBranch32 => Ok(i.near_branch_target()),
            OpKind::Register => Ok(self.state.reg(i.op_register(0)).expect("GPR branch target")),
            OpKind::Memory => self.read_op(i, 0, next_ip, mem),
            _ => Err(Fault::InvalidOpcode {
                mnemonic: format!("{:?}", i.mnemonic()),
                rip: i.ip(),
            }),
        }
    }

    fn push_u64(&mut self, value: u64, mem: &mut dyn MemoryAccess) -> Result<(), Fault> {
        self.state.gpr[state::RSP] = self.state.rsp().wrapping_sub(8);
        mem.write(8, self.state.rsp(), value)
            .map_err(AccessError::into_fault)
    }

    fn pop_u64(&mut self, mem: &mut dyn MemoryAccess) -> Result<u64, Fault> {
        let v = mem
            .read(8, self.state.rsp())
            .map_err(AccessError::into_fault)?;
        self.state.gpr[state::RSP] = self.state.rsp().wrapping_add(8);
        Ok(v)
    }

    // ----- exception delivery -------------------------------------------------

    /// Try to deliver `f` through the guest IDT. Returns true when a handler
    /// took it (execution continues at the handler).
    fn try_deliver_fault(&mut self, f: &Fault, mem: &mut dyn MemoryAccess) -> bool {
        let (vector, error_code, has_ec, cr2) = match f {
            Fault::PageFault { addr, code } => (14u8, *code, true, *addr),
            Fault::InvalidOpcode { .. } => (6, 0, false, 0),
            Fault::DivideError => (0, 0, false, 0),
            Fault::GeneralProtection { .. } => (13, 0, true, 0),
            // Host errors and unhooked syscalls are not guest-deliverable.
            _ => return false,
        };
        match self.deliver_exception(vector, error_code, has_ec, cr2, mem) {
            Ok(()) => true,
            Err(secondary) => {
                // Faulting while delivering a fault: report the cascade
                // honestly rather than looping forever.
                tracing::error!(
                    vector,
                    first = %f,
                    second = %secondary,
                    "fault during exception delivery (double-fault condition)"
                );
                false
            }
        }
    }

    /// Push the interrupt frame and vector to the IDT handler.
    fn deliver_exception(
        &mut self,
        vector: u8,
        error_code: u32,
        has_error_code: bool,
        cr2: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<(), Fault> {
        if vector == 8 {
            // Delivering #DF while already delivering #DF is a triple fault:
            // real hardware resets; PDG reports it upward.
            return Err(Fault::HostError("triple fault".to_owned()));
        }
        let idtr = self.state.idtr;
        if idtr.base == 0 {
            // No IDT installed: faults propagate to the run() caller.
            return Err(Fault::GeneralProtection {
                reason: format!("exception {vector:#x} with no IDT installed"),
            });
        }
        let gate_addr = idtr.base + u64::from(vector) * 16;
        if u64::from(vector) * 16 + 16 > u64::from(idtr.limit) + 1 {
            return Err(Fault::GeneralProtection {
                reason: format!("exception {vector:#x} outside IDT limit"),
            });
        }
        let mut gate = [0u8; 16];
        mem.read_buf(gate_addr, &mut gate)
            .map_err(AccessError::into_fault)?;
        let offset_lo = u16::from_le_bytes([gate[0], gate[1]]) as u64;
        let offset_hi = u16::from_le_bytes([gate[6], gate[7]]) as u64;
        let attr = gate[5];
        const PRESENT: u8 = 0x80;
        if attr & PRESENT == 0 {
            return Err(Fault::GeneralProtection {
                reason: format!("exception {vector:#x} gate not present"),
            });
        }
        let handler = offset_lo | (offset_hi << 16);

        // CR2 records the faulting address for #PF handlers.
        self.state.cr2 = cr2;

        // Simplified flat-model frame (no privilege change, CS is a constant
        // selector): [rflags, cs, rip, error_code?]; error code on top.
        let rflags = self.state.rflags;
        self.push_u64(rflags, mem)?;
        self.push_u64(0x08, mem)?;
        self.push_u64(self.state.rip, mem)?;
        if has_error_code {
            self.push_u64(u64::from(error_code), mem)?;
        }
        // Interrupt gates clear IF (and TF); trap gates keep IF.
        let gate_kind = attr & 0x0F;
        const INTERRUPT_GATE: u8 = 0x0E;
        if gate_kind == INTERRUPT_GATE {
            self.state.rflags &= !(state::rflags::IF | state::rflags::TF);
        } else {
            self.state.rflags &= !state::rflags::TF;
        }
        self.state.rip = handler;
        Ok(())
    }

    /// Software interrupt (`int n` / `int3` / `into`).
    fn raise_software_interrupt(
        &mut self,
        vector: u8,
        i: &Instruction,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let _ = i;
        match self.deliver_exception(vector, 0, false, 0, mem) {
            Ok(()) => Ok(Action::Jump(self.state.rip)),
            Err(f) => Err(Fault::GeneralProtection {
                reason: format!("int {vector:#x}: {f}"),
            }),
        }
    }

    /// Deliver an *external* interrupt (device-signalled) through the guest
    /// IDT, exactly as `#PF`/`#UD` delivery works: push the architectural
    /// frame, clear IF for interrupt gates, jump to the gate handler. No
    /// error code is pushed (architectural rule for external interrupts).
    ///
    /// The kernel layer calls this from its device pump — at an instruction
    /// boundary, never inside one — which is the honest single-core model
    /// of "pending until the next dispatch point". Re-entry is the guest's
    /// business: `iretq` in the handler restores the interrupted frame.
    ///
    /// Returns the fault raised while delivering (bad gate, no IDT, ...),
    /// if any.
    pub fn inject_interrupt(
        &mut self,
        vector: u8,
        mem: &mut dyn MemoryAccess,
    ) -> Result<(), Fault> {
        self.deliver_exception(vector, 0, false, 0, mem)
    }

    // ----- execution --------------------------------------------------------

    fn execute(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        match i.mnemonic() {
            // ---- SIMD / vector families: dispatched to the SIMD module ----
            m if crate::simd::is_vector_mnemonic(m) => {
                crate::simd::execute_vector(i, next_ip, &mut self.state, mem)
            }

            // ---- no-op / hint / fence families -------------------------------
            Nop | Endbr64 | Endbr32 | Pause => Ok(Action::Next),
            // Prefetches and cache-line ops are performance hints with no
            // architectural effect on guest state (PDG has no cache model).
            Prefetchnta | Prefetcht0 | Prefetcht1 | Prefetcht2 | Prefetchw | Prefetchwt1
            | Clflush | Clflushopt | Clgi | Stgi => Ok(Action::Next),
            // Memory barriers: no guest-visible state change in a
            // single-vCPU interpreter; SMP/JIT conservativeness covers them.
            Lfence | Mfence | Sfence => Ok(Action::Next),

            Hlt => Ok(Action::Halt),
            Ud2 => Err(Fault::InvalidOpcode {
                mnemonic: "ud2 (guest-requested)".to_owned(),
                rip: i.ip(),
            }),

            // ---- data movement ----------------------------------------------
            Mov => self.exec_mov(i, next_ip, mem),
            Lea => {
                let addr = self.ea(i, next_ip);
                self.write_op(i, 0, next_ip, addr, mem)?;
                Ok(Action::Next)
            }
            Movzx | Movsx | Movsxd => {
                // Source width comes from the memory operand size (or the
                // source register's own width).
                let src_w = match Self::kind(i, 1) {
                    OpKind::Memory => i.memory_size().size() as u32,
                    OpKind::Register => {
                        let (_, size, _) =
                            crate::state::decode_reg(i.op_register(1)).expect("GPR source");
                        size
                    }
                    _ => {
                        return Err(Fault::InvalidOpcode {
                            mnemonic: format!("{:?}", i.mnemonic()),
                            rip: i.ip(),
                        });
                    }
                };
                let v = self.read_op(i, 1, next_ip, mem)?;
                let ext = if i.mnemonic() == Movzx {
                    v & mask(src_w)
                } else {
                    sext_bits(v & mask(src_w), src_w * 8) as u64
                };
                self.write_op(i, 0, next_ip, ext, mem)?;
                Ok(Action::Next)
            }
            Xchg => self.exec_xchg(i, next_ip, mem),
            Bswap => {
                let w = Self::op_width(i, 0);
                let v = self.read_op(i, 0, next_ip, mem)?;
                let swapped = match w {
                    2 => ((v & 0xFF) << 8) | ((v >> 8) & 0xFF),
                    4 => (v as u32).swap_bytes() as u64,
                    _ => v.swap_bytes(),
                };
                self.write_op(i, 0, next_ip, swapped, mem)?;
                Ok(Action::Next)
            }
            Xadd => self.exec_xadd(i, next_ip, mem),
            Cmpxchg => self.exec_cmpxchg(i, next_ip, mem, false),
            Cmpxchg8b | Cmpxchg16b => self.exec_cmpxchg_wide(i, next_ip, mem),

            Push => self.exec_push(i, next_ip, mem),
            Pop => self.exec_pop(i, next_ip, mem),
            Pushf | Pushfd | Pushfq => {
                self.push_u64(self.state.rflags, mem)?;
                Ok(Action::Next)
            }
            Popf | Popfd | Popfq => {
                self.state.rflags = self.pop_u64(mem)? | crate::state::rflags::RESERVED_1;
                Ok(Action::Next)
            }
            Lahf => {
                // AH = SF ZF 0 AF 0 PF 1 CF (implicit operand, no ModRM).
                let s = &self.state;
                let ah = u64::from(s.sf()) << 7
                    | u64::from(s.zf()) << 6
                    | u64::from(s.af()) << 4
                    | u64::from(s.pf()) << 2
                    | 0x02
                    | u64::from(s.cf());
                self.state.gpr[state::RAX] =
                    (self.state.gpr[state::RAX] & !0xFF00) | ((ah & 0xFF) << 8);
                Ok(Action::Next)
            }
            Sahf => {
                let ah = ((self.state.gpr[state::RAX] >> 8) & 0xFF) as u8;
                let s = &mut self.state;
                s.rflags = (s.rflags
                    & !(state::rflags::SF
                        | state::rflags::ZF
                        | state::rflags::AF
                        | state::rflags::PF
                        | state::rflags::CF))
                    | (u64::from(ah & 0x80 != 0) * state::rflags::SF)
                    | (u64::from(ah & 0x40 != 0) * state::rflags::ZF)
                    | (u64::from(ah & 0x10 != 0) * state::rflags::AF)
                    | (u64::from(ah & 0x04 != 0) * state::rflags::PF)
                    | (u64::from(ah & 0x01 != 0) * state::rflags::CF);
                Ok(Action::Next)
            }

            // ---- ALU ----------------------------------------------------------
            Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test => self.exec_binop(i, next_ip, mem),
            Inc | Dec => self.exec_incdec(i, next_ip, mem),
            Not | Neg => self.exec_unary(i, next_ip, mem),
            Shl | Sal | Shr | Sar => self.exec_shift(i, next_ip, mem),
            Rol | Ror | Rcl | Rcr => self.exec_rotate(i, next_ip, mem),
            Shld | Shrd => self.exec_double_shift(i, next_ip, mem),

            // ---- multiplication / division ------------------------------------
            Mul | Imul | Div | Idiv => self.exec_muldiv(i, next_ip, mem),
            Cbw => {
                let al = self.state.gpr[state::RAX] & 0xFF;
                self.state.gpr[state::RAX] =
                    (self.state.gpr[state::RAX] & !0xFFFF) | (sext_bits(al, 8) as u64 & 0xFFFF);
                Ok(Action::Next)
            }
            Cwde => {
                let ax = self.state.gpr[state::RAX] & 0xFFFF;
                self.state.gpr[state::RAX] = sext_bits(ax, 16) as u64 & 0xFFFF_FFFF;
                Ok(Action::Next)
            }
            Cdqe => {
                let eax = self.state.gpr[state::RAX] & 0xFFFF_FFFF;
                self.state.gpr[state::RAX] = sext_bits(eax, 32) as u64;
                Ok(Action::Next)
            }
            Cwd => {
                let ax = self.state.gpr[state::RAX] & 0xFFFF;
                self.state.gpr[state::RDX] = if (ax as i16) < 0 { 0xFFFF } else { 0x0000 };
                Ok(Action::Next)
            }
            Cdq => {
                let eax = self.state.gpr[state::RAX] & 0xFFFF_FFFF;
                self.state.gpr[state::RDX] = if (eax as u32 as i32) < 0 {
                    0xFFFF_FFFF
                } else {
                    0
                };
                Ok(Action::Next)
            }
            Cqo => {
                let rax = self.state.gpr[state::RAX];
                self.state.gpr[state::RDX] = if (rax as i64) < 0 { u64::MAX } else { 0 };
                Ok(Action::Next)
            }

            // ---- bit operations -------------------------------------------------
            Bt | Bts | Btr | Btc => self.exec_bt(i, next_ip, mem),
            Bsf | Bsr | Popcnt | Tzcnt | Lzcnt => self.exec_bit_scan(i, next_ip, mem),

            // ---- BMI1 / BMI2 -----------------------------------------------------
            Andn => {
                // dst = (~src1) & src2
                let w = Self::op_width(i, 0);
                let m = mask(w);
                let src1 = self.read_op(i, 1, next_ip, mem)? & m;
                let src2 = self.read_op(i, 2, next_ip, mem)? & m;
                self.write_op(i, 0, next_ip, (!src1 & m) & src2, mem)?;
                self.state.flags_logic(!src1 & src2 & m, w);
                Ok(Action::Next)
            }
            Bextr => {
                // dst = (src >> (start & mask)) & ((1 << (len & mask)) - 1)
                let w = Self::op_width(i, 0);
                let bits = u64::from(w * 8);
                let m = mask(w);
                let src = self.read_op(i, 1, next_ip, mem)? & m;
                let ctrl = self.read_op(i, 2, next_ip, mem)? & m;
                let start = ctrl & 0xFF;
                let len = (ctrl >> 8) & 0xFF;
                let width_mask = if bits == 64 { 0x3F } else { 0x1F };
                let start = start & width_mask;
                let len = if bits == 64 {
                    (len & 0x3F).min(64)
                } else {
                    (len & 0x1F).min(bits)
                };
                let range_mask = if len >= bits { m } else { (1u64 << len) - 1 };
                let r = if start >= bits {
                    0
                } else {
                    (src >> start) & range_mask
                };
                self.write_op(i, 0, next_ip, r, mem)?;
                self.state.flags_logic(r, w);
                Ok(Action::Next)
            }
            Blsi | Blsmsk | Blsr => {
                let w = Self::op_width(i, 0);
                let m = mask(w);
                let src = self.read_op(i, 1, next_ip, mem)? & m;
                let r = match i.mnemonic() {
                    Blsi => src.wrapping_sub(1) & src,
                    Blsmsk => src ^ (src.wrapping_sub(1)),
                    _ => src & (src.wrapping_sub(1)),
                };
                self.write_op(i, 0, next_ip, r, mem)?;
                self.state.flags_logic(r, w);
                Ok(Action::Next)
            }
            Bzhi => {
                let w = Self::op_width(i, 0);
                let bits = u64::from(w * 8);
                let m = mask(w);
                let src = self.read_op(i, 1, next_ip, mem)? & m;
                let idx = self.read_op(i, 2, next_ip, mem)? & 0xFF;
                let r = if idx >= bits {
                    // Sets CF when the index exceeds the operand width.
                    src
                } else {
                    src & ((1u64 << idx) - 1)
                };
                self.write_op(i, 0, next_ip, r, mem)?;
                self.state.flags_logic(r, w);
                self.state.rflags = (self.state.rflags & !state::rflags::CF)
                    | (u64::from(idx >= bits) * state::rflags::CF);
                self.state.rflags &= !state::rflags::OF;
                Ok(Action::Next)
            }
            Mulx => {
                // dest_hi:dest_lo = rdx * src (flags unmodified).
                let src = self.read_op(i, 2, next_ip, mem)?;
                let prod = (self.state.gpr[state::RDX] as u128) * (src as u128);
                self.write_op(i, 0, next_ip, (prod >> 64) as u64, mem)?;
                self.write_op(i, 1, next_ip, prod as u64, mem)?;
                Ok(Action::Next)
            }
            Pdep => {
                let src = self.read_op(i, 2, next_ip, mem)?;
                let maskv = self.read_op(i, 1, next_ip, mem)?;
                let mut r = 0u64;
                let mut bit = 0u32;
                let mut m = 1u64;
                let mut s = 1u64;
                while m != 0 {
                    if maskv & m != 0 {
                        if src & s != 0 {
                            r |= m;
                        }
                        s <<= 1;
                        bit += 1;
                        if bit >= 64 {
                            break;
                        }
                    }
                    m <<= 1;
                }
                let _ = bit;
                self.write_op(i, 0, next_ip, r, mem)?;
                Ok(Action::Next)
            }
            Pext => {
                let src = self.read_op(i, 2, next_ip, mem)?;
                let maskv = self.read_op(i, 1, next_ip, mem)?;
                let mut r = 0u64;
                let mut s = 1u64;
                let mut out = 1u64;
                while s != 0 {
                    if maskv & s != 0 {
                        if src & s != 0 {
                            r |= out;
                        }
                        out <<= 1;
                    }
                    s <<= 1;
                }
                self.write_op(i, 0, next_ip, r, mem)?;
                Ok(Action::Next)
            }
            Rorx => {
                let w = Self::op_width(i, 0);
                let m = mask(w);
                let src = self.read_op(i, 1, next_ip, mem)? & m;
                let c = u64::from(i.immediate8()) & 0x3F;
                let r = src.rotate_right(c as u32);
                self.write_op(i, 0, next_ip, r, mem)?;
                Ok(Action::Next)
            }
            Sarx | Shlx | Shrx => {
                let w = Self::op_width(i, 0);
                let m = mask(w);
                let _bits = w * 8;
                let src = self.read_op(i, 1, next_ip, mem)? & m;
                let c = self.read_op(i, 2, next_ip, mem)? & 0x3F;
                // No flag effects. Shift counts are NOT masked to operand
                // width (BMI shifts use the full CL semantics).
                let r = match i.mnemonic() {
                    Shlx => (src << c) & m,
                    Shrx => src >> c,
                    _ => ((sext(src, w) >> c.min(63)) as u64) & m,
                };
                self.write_op(i, 0, next_ip, r, mem)?;
                Ok(Action::Next)
            }

            // ---- conditional moves / sets -------------------------------------
            Cmovno | Cmovb | Cmovae | Cmove | Cmovne | Cmovbe | Cmova | Cmovs | Cmovns | Cmovp
            | Cmovnp | Cmovl | Cmovge | Cmovle | Cmovg => {
                if self.cond_cc(i.condition_code()) {
                    let v = self.read_op(i, 1, next_ip, mem)?;
                    self.write_op(i, 0, next_ip, v, mem)?;
                }
                Ok(Action::Next)
            }
            Setno | Setb | Setae | Sete | Setne | Setbe | Seta | Sets | Setns | Setp | Setnp
            | Setl | Setge | Setle | Setg => {
                let v = u64::from(self.cond_cc(i.condition_code()));
                self.write_op(i, 0, next_ip, v, mem)?;
                Ok(Action::Next)
            }

            // ---- control flow -----------------------------------------------------
            Call => {
                let target = self.branch_target(i, next_ip, mem)?;
                self.push_u64(next_ip, mem)?;
                Ok(Action::Jump(target))
            }
            Ret => {
                let target = self.pop_u64(mem)?;
                if i.op_count() > 0 {
                    if let OpKind::Immediate16 = Self::kind(i, 0) {
                        let frame = i.immediate16() as u64;
                        self.state.gpr[state::RSP] = self.state.rsp().wrapping_add(frame);
                    }
                }
                Ok(Action::Jump(target))
            }
            Leave => {
                self.state.gpr[state::RSP] = self.state.gpr[state::RBP];
                let rbp = self.pop_u64(mem)?;
                self.state.gpr[state::RBP] = rbp;
                Ok(Action::Next)
            }
            Jmp => {
                let target = self.branch_target(i, next_ip, mem)?;
                Ok(Action::Jump(target))
            }
            Jno | Jb | Jae | Je | Jne | Jbe | Ja | Js | Jns | Jp | Jnp | Jl | Jge | Jle | Jg => {
                if self.cond_cc(i.condition_code()) {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }
            Loop => {
                let c = self.dec_counter(8)?;
                if c != 0 {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }
            Loope => {
                let c = self.dec_counter(8)?;
                if c != 0 && self.state.zf() {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }
            Loopne => {
                let c = self.dec_counter(8)?;
                if c != 0 && !self.state.zf() {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }
            Jrcxz => {
                if self.state.gpr[state::RCX] == 0 {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }
            Jecxz => {
                if self.state.gpr[state::RCX] & 0xFFFF_FFFF == 0 {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }

            // ---- string operations (with REP) ---------------------------------
            Movsb | Movsw | Movsd | Movsq | Stosb | Stosw | Stosd | Stosq | Lodsb | Lodsw
            | Lodsd | Lodsq | Scasb | Scasw | Scasd | Scasq | Cmpsb | Cmpsw | Cmpsd | Cmpsq => {
                self.exec_string(i, mem)
            }

            // ---- flag ops -----------------------------------------------------
            Clc => {
                self.state.rflags &= !state::rflags::CF;
                Ok(Action::Next)
            }
            Stc => {
                self.state.rflags |= state::rflags::CF;
                Ok(Action::Next)
            }
            Cmc => {
                self.state.rflags ^= state::rflags::CF;
                Ok(Action::Next)
            }
            Cld => {
                self.state.rflags &= !state::rflags::DF;
                Ok(Action::Next)
            }
            Std => {
                self.state.rflags |= state::rflags::DF;
                Ok(Action::Next)
            }
            Cli => {
                self.state.rflags &= !state::rflags::IF;
                Ok(Action::Next)
            }
            Sti => {
                self.state.rflags |= state::rflags::IF;
                Ok(Action::Next)
            }

            // ---- interrupts / exceptions --------------------------------------
            Int3 => self.raise_software_interrupt(3, i, mem),
            Int1 => self.raise_software_interrupt(1, i, mem),
            Int => {
                let v = i.immediate8();
                self.raise_software_interrupt(v, i, mem)
            }
            Into => {
                if self.state.of() {
                    self.raise_software_interrupt(4, i, mem)
                } else {
                    Ok(Action::Next)
                }
            }
            Iret | Iretq | Iretd => {
                // Flat-model frame: [rip, cs, rflags]; the handler popped any
                // error code itself (standard kernel style).
                let rip = self.pop_u64(mem)?;
                let _cs = self.pop_u64(mem)?;
                let rflags = self.pop_u64(mem)?;
                self.state.rflags = rflags | crate::state::rflags::RESERVED_1;
                Ok(Action::Jump(rip))
            }

            // ---- system / MSRs / descriptor tables ---------------------------
            Syscall => match self.syscall_hook.as_mut() {
                Some(hook) => {
                    // The `syscall` retires the moment the kernel takes
                    // over: hook faults (process exit, thread exit,
                    // umtx parking) resume execution *after* the
                    // instruction, never re-run it — a parked thread that
                    // re-executed its wait would see stale state (a
                    // handed-off mutex would read as its own).
                    self.state.rip = next_ip;
                    hook.on_syscall(&mut self.state, mem)?;
                    Ok(Action::Next)
                }
                None => Err(Fault::UnhandledSyscall { rip: i.ip() }),
            },
            Cpuid => {
                let leaf = (self.state.gpr[state::RAX] & 0xFFFF_FFFF) as u32;
                let subleaf = (self.state.gpr[state::RCX] & 0xFFFF_FFFF) as u32;
                let r = cpuid::cpuid(leaf, subleaf);
                self.set_gpr32(state::RAX, r.eax);
                self.set_gpr32(state::RBX, r.ebx);
                self.set_gpr32(state::RCX, r.ecx);
                self.set_gpr32(state::RDX, r.edx);
                Ok(Action::Next)
            }
            Rdtsc | Rdtscp => {
                let t = self.state.tsc;
                self.state.gpr[state::RAX] = t & 0xFFFF_FFFF;
                self.state.gpr[state::RDX] = t >> 32;
                if i.mnemonic() == Rdtscp {
                    // IA32_TSC_AUX not modelled: zero.
                    self.set_gpr32(state::RCX, 0);
                }
                Ok(Action::Next)
            }
            Rdmsr => {
                let id = (self.state.gpr[state::RCX] & 0xFFFF_FFFF) as u32;
                let v = self.read_msr(id)?;
                self.state.gpr[state::RAX] = v & 0xFFFF_FFFF;
                self.state.gpr[state::RDX] = v >> 32;
                Ok(Action::Next)
            }
            Wrmsr => {
                let id = (self.state.gpr[state::RCX] & 0xFFFF_FFFF) as u32;
                let v =
                    (self.state.gpr[state::RDX] << 32) | (self.state.gpr[state::RAX] & 0xFFFF_FFFF);
                self.write_msr(id, v)?;
                Ok(Action::Next)
            }
            Swapgs => {
                let kg = *self
                    .state
                    .msrs
                    .get(&crate::state::msr::KERNEL_GS_BASE)
                    .unwrap_or(&0);
                let gs = self.state.gs_base;
                self.state.gs_base = kg;
                self.state
                    .msrs
                    .insert(crate::state::msr::KERNEL_GS_BASE, gs);
                Ok(Action::Next)
            }
            Lgdt | Lidt => {
                let addr = self.ea(i, next_ip);
                let mut buf = [0u8; 10];
                mem.read_buf(addr, &mut buf)
                    .map_err(AccessError::into_fault)?;
                let limit = u16::from_le_bytes([buf[0], buf[1]]);
                let base = u64::from_le_bytes(buf[2..10].try_into().expect("8 bytes"));
                let dtr = state::DescriptorTableRegister { base, limit };
                if i.mnemonic() == Lgdt {
                    self.state.gdtr = dtr;
                } else {
                    self.state.idtr = dtr;
                }
                Ok(Action::Next)
            }
            Sgdt | Sidt => {
                let addr = self.ea(i, next_ip);
                let dtr = if i.mnemonic() == Sgdt {
                    self.state.gdtr
                } else {
                    self.state.idtr
                };
                let mut buf = [0u8; 10];
                buf[0..2].copy_from_slice(&dtr.limit.to_le_bytes());
                buf[2..10].copy_from_slice(&dtr.base.to_le_bytes());
                mem.write_buf(addr, &buf).map_err(AccessError::into_fault)?;
                Ok(Action::Next)
            }
            Invlpg => {
                let addr = self.ea(i, next_ip);
                mem.invalidate_tlb_page(addr);
                Ok(Action::Next)
            }
            Invlpga => {
                let addr = self.state.gpr[state::RAX];
                mem.invalidate_tlb_page(addr);
                Ok(Action::Next)
            }

            other => Err(Fault::InvalidOpcode {
                mnemonic: format!("{other:?}"),
                rip: i.ip(),
            }),
        }
    }

    fn set_gpr32(&mut self, idx: usize, v: u32) {
        // 32-bit writes zero-extend (x86-64 rule).
        self.state.gpr[idx] = u64::from(v);
    }

    /// Decrement the RCX-width counter (LOOP family uses full-width RCX),
    /// returning the new value.
    fn dec_counter(&mut self, _w: u32) -> Result<u64, Fault> {
        self.state.gpr[state::RCX] = self.state.gpr[state::RCX].wrapping_sub(1);
        Ok(self.state.gpr[state::RCX])
    }

    fn read_msr(&mut self, id: u32) -> Result<u64, Fault> {
        match id {
            crate::state::msr::FS_BASE => Ok(self.state.fs_base),
            crate::state::msr::GS_BASE => Ok(self.state.gs_base),
            crate::state::msr::KERNEL_GS_BASE => Ok(*self.state.msrs.get(&id).unwrap_or(&0)),
            _ => self
                .state
                .msrs
                .get(&id)
                .copied()
                .ok_or(Fault::GeneralProtection {
                    reason: format!("read of unrecognised MSR {id:#x}"),
                }),
        }
    }

    fn write_msr(&mut self, id: u32, v: u64) -> Result<(), Fault> {
        match id {
            crate::state::msr::FS_BASE => {
                self.state.fs_base = v;
                self.state.msrs.insert(id, v);
                Ok(())
            }
            crate::state::msr::GS_BASE => {
                self.state.gs_base = v;
                self.state.msrs.insert(id, v);
                Ok(())
            }
            crate::state::msr::EFER
            | crate::state::msr::STAR
            | crate::state::msr::LSTAR
            | crate::state::msr::CSTAR
            | crate::state::msr::SFMASK
            | crate::state::msr::KERNEL_GS_BASE => {
                self.state.msrs.insert(id, v);
                Ok(())
            }
            _ => Err(Fault::GeneralProtection {
                reason: format!("write to unrecognised MSR {id:#x}"),
            }),
        }
    }

    /// MOV with control-register operands; everything else takes the plain
    /// data path.
    fn exec_mov(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        // MOV CR/m forms: op0 is a CR register (r/m or reg direction below).
        let op0_reg = Self::kind(i, 0) == OpKind::Register && is_cr(i.op_register(0));
        let op1_reg = Self::kind(i, 1) == OpKind::Register && is_cr(i.op_register(1));
        if op0_reg || op1_reg {
            return self.exec_mov_cr(i, next_ip, mem);
        }
        let v = self.read_op(i, 1, next_ip, mem)?;
        self.write_op(i, 0, next_ip, v, mem)?;
        Ok(Action::Next)
    }

    fn exec_mov_cr(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let _ = next_ip;
        // Identify direction: "mov cr, gpr" (reg dest first) vs "mov gpr, cr".
        if let OpKind::Register = Self::kind(i, 0) {
            if is_cr(i.op_register(0)) {
                // mov crN, gpr
                let gpr = Self::kind(i, 1) == OpKind::Register;
                let v = if gpr {
                    self.state.gpr[gpr_index(i.op_register(1)).expect("gpr")]
                } else {
                    // CR from memory is not architecturally valid.
                    return Err(Fault::InvalidOpcode {
                        mnemonic: "mov cr, [m]".to_owned(),
                        rip: i.ip(),
                    });
                };
                let cr = i.op_register(0);
                match cr {
                    Register::CR0 => self.state.cr0 = v,
                    Register::CR2 => self.state.cr2 = v,
                    Register::CR3 => {
                        self.state.cr3 = v;
                        mem.reload_cr3(v);
                    }
                    Register::CR4 => self.state.cr4 = v,
                    _ => {
                        return Err(Fault::InvalidOpcode {
                            mnemonic: format!("mov {cr:?}, gpr"),
                            rip: i.ip(),
                        });
                    }
                }
                return Ok(Action::Next);
            }
        }
        // mov gpr, crN
        let cr = i.op_register(1);
        let v = match cr {
            Register::CR0 => self.state.cr0,
            Register::CR2 => self.state.cr2,
            Register::CR3 => self.state.cr3,
            Register::CR4 => self.state.cr4,
            _ => {
                return Err(Fault::InvalidOpcode {
                    mnemonic: format!("mov gpr, {cr:?}"),
                    rip: i.ip(),
                });
            }
        };
        self.write_op(i, 0, next_ip, v, mem)?;
        Ok(Action::Next)
    }

    fn exec_push(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let v = match Self::kind(i, 0) {
            OpKind::Register => self.state.reg(i.op_register(0)).expect("GPR push"),
            OpKind::Memory => self.read_op(i, 0, next_ip, mem)?,
            OpKind::Immediate8 => sext_bits(i.immediate8() as u64, 8) as u64,
            OpKind::Immediate16 => sext_bits(i.immediate16() as u64, 16) as u64,
            OpKind::Immediate32 => sext_bits(i.immediate32() as u64, 32) as u64,
            OpKind::Immediate8to16 | OpKind::Immediate8to32 | OpKind::Immediate8to64 => {
                i.immediate8to64() as u64
            }
            OpKind::Immediate32to64 => i.immediate32to64() as u64,
            _ => {
                return Err(Fault::InvalidOpcode {
                    mnemonic: format!("{:?}", i.mnemonic()),
                    rip: i.ip(),
                });
            }
        };
        // In 64-bit mode PUSH pushes 8 bytes except with a 66 prefix (2);
        // iced surfaces that via the operand's own width.
        let width = stack_width(i, 0);
        if width == 8 {
            self.push_u64(v, mem)?;
        } else {
            self.state.gpr[state::RSP] = self.state.rsp().wrapping_sub(2);
            mem.write(2, self.state.rsp(), v & 0xFFFF)
                .map_err(AccessError::into_fault)?;
        }
        Ok(Action::Next)
    }

    fn exec_pop(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let width = stack_width(i, 0);
        let v = if width == 8 {
            self.pop_u64(mem)?
        } else {
            let v = mem
                .read(2, self.state.rsp())
                .map_err(AccessError::into_fault)?;
            self.state.gpr[state::RSP] = self.state.rsp().wrapping_add(2);
            v
        };
        self.write_op(i, 0, next_ip, v, mem)?;
        Ok(Action::Next)
    }

    fn exec_xchg(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        // XCHG with a memory operand is implicitly LOCKed.
        if Self::kind(i, 0) == OpKind::Memory {
            let w = Self::op_width(i, 0) as u8;
            let v = self.read_op(i, 1, next_ip, mem)?;
            let addr = self.ea(i, next_ip);
            let old = mem.xchg(w, addr, v).map_err(AccessError::into_fault)?;
            self.write_op(i, 1, next_ip, old, mem)?;
            return Ok(Action::Next);
        }
        if Self::kind(i, 1) == OpKind::Memory {
            let w = Self::op_width(i, 1) as u8;
            let v = self.read_op(i, 0, next_ip, mem)?;
            let addr = self.ea(i, next_ip);
            let old = mem.xchg(w, addr, v).map_err(AccessError::into_fault)?;
            self.write_op(i, 0, next_ip, old, mem)?;
            return Ok(Action::Next);
        }
        let a = self.read_op(i, 0, next_ip, mem)?;
        let b = self.read_op(i, 1, next_ip, mem)?;
        self.write_op(i, 0, next_ip, b, mem)?;
        self.write_op(i, 1, next_ip, a, mem)?;
        Ok(Action::Next)
    }

    fn exec_xadd(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let src = self.read_op(i, 1, next_ip, mem)? & m;
        if Self::kind(i, 0) == OpKind::Memory {
            let width = w as u8;
            let addr = self.ea(i, next_ip);
            let old = mem
                .xadd(width, addr, src)
                .map_err(AccessError::into_fault)?;
            self.state
                .flags_add(old & m, src, 0, (old & m).wrapping_add(src), w);
            self.write_op(i, 1, next_ip, old, mem)?;
        } else {
            let dst = self.read_op(i, 0, next_ip, mem)? & m;
            let r = dst.wrapping_add(src);
            self.state.flags_add(dst, src, 0, r, w);
            self.write_op(i, 0, next_ip, r, mem)?;
            self.write_op(i, 1, next_ip, dst, mem)?;
        }
        Ok(Action::Next)
    }

    fn exec_cmpxchg(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
        _wide: bool,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let src = self.read_op(i, 1, next_ip, mem)? & m;
        let acc = self.state.gpr[state::RAX] & m;
        if Self::kind(i, 0) == OpKind::Memory {
            let width = w as u8;
            let addr = self.ea(i, next_ip);
            let old = mem
                .cmpxchg(width, addr, acc, src)
                .map_err(AccessError::into_fault)?;
            if old == acc {
                self.state.rflags |= state::rflags::ZF;
            } else {
                self.state.rflags &= !state::rflags::ZF;
                self.state.gpr[state::RAX] = if w == 8 {
                    old
                } else {
                    (self.state.gpr[state::RAX] & !m) | (old & m)
                };
            }
        } else {
            let dst = self.read_op(i, 0, next_ip, mem)? & m;
            if dst == acc {
                self.write_op(i, 0, next_ip, src, mem)?;
                self.state.rflags |= state::rflags::ZF;
            } else {
                self.state.rflags &= !state::rflags::ZF;
                self.state.gpr[state::RAX] = if w == 8 {
                    dst
                } else {
                    (self.state.gpr[state::RAX] & !m) | (dst & m)
                };
            }
        }
        Ok(Action::Next)
    }

    /// CMPXCHG8B / CMPXCHG16B: compare RDX:RAX with the memory destination;
    /// on match store RCX:RBX. Emulated as a non-atomic 128-bit RMW (atomic
    /// under a single vCPU; a documented SMP limitation).
    fn exec_cmpxchg_wide(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let addr = self.ea(i, next_ip);
        let lo = mem.read(8, addr).map_err(AccessError::into_fault)?;
        let hi = mem
            .read(8, addr.wrapping_add(8))
            .map_err(AccessError::into_fault)?;
        let expected =
            u128::from(self.state.gpr[state::RAX]) | (u128::from(self.state.gpr[state::RDX]) << 64);
        if (lo, hi) == (expected as u64, (expected >> 64) as u64) {
            mem.write(8, addr, self.state.gpr[state::RBX])
                .map_err(AccessError::into_fault)?;
            mem.write(8, addr.wrapping_add(8), self.state.gpr[state::RCX])
                .map_err(AccessError::into_fault)?;
            self.state.rflags |= state::rflags::ZF;
        } else {
            self.state.gpr[state::RAX] = lo;
            self.state.gpr[state::RDX] = hi;
            self.state.rflags &= !state::rflags::ZF;
        }
        Ok(Action::Next)
    }

    fn exec_binop(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let a = self.read_op(i, 0, next_ip, mem)? & m;
        let b = self.read_op(i, 1, next_ip, mem)? & m;
        let (result, writeback): (u64, bool) = match i.mnemonic() {
            Add => {
                let r = a.wrapping_add(b);
                self.state.flags_add(a, b, 0, r, w);
                (r, true)
            }
            Adc => {
                let c = u64::from(self.state.cf());
                let r = a.wrapping_add(b).wrapping_add(c);
                self.state.flags_add(a, b, c, r, w);
                (r, true)
            }
            Sub | Cmp => {
                let r = a.wrapping_sub(b);
                self.state.flags_sub(a, b, 0, r, w);
                (r, i.mnemonic() == Sub)
            }
            Sbb => {
                let c = u64::from(self.state.cf());
                let r = a.wrapping_sub(b).wrapping_sub(c);
                self.state.flags_sub(a, b, c, r, w);
                (r, true)
            }
            And | Test => {
                let r = a & b;
                self.state.flags_logic(r, w);
                (r, i.mnemonic() == And)
            }
            Or => {
                let r = a | b;
                self.state.flags_logic(r, w);
                (r, true)
            }
            Xor => {
                let r = a ^ b;
                self.state.flags_logic(r, w);
                (r, true)
            }
            _ => unreachable!("exec_binop dispatch guarantees coverage"),
        };
        if writeback {
            self.write_op(i, 0, next_ip, result, mem)?;
        }
        Ok(Action::Next)
    }

    fn exec_incdec(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let m = mask(w);
        // LOCKed INC/DEC on memory must be a *true* atomic RMW (xadd ±1);
        // a plain read-modify-write would lose updates once several vCPUs
        // share RAM (Phase-2 SMP).
        if i.has_lock_prefix() && Self::kind(i, 0) == OpKind::Memory {
            let addr = self.ea(i, next_ip);
            let delta = if i.mnemonic() == Mnemonic::Inc { 1 } else { m };
            let old = mem
                .xadd(w as u8, addr, delta)
                .map_err(AccessError::into_fault)?
                & m;
            // INC/DEC leave CF untouched.
            let saved_cf = self.state.cf();
            if i.mnemonic() == Mnemonic::Inc {
                let r = old.wrapping_add(1);
                self.state.flags_add(old, 1, 0, r, w);
            } else {
                let r = old.wrapping_sub(1);
                self.state.flags_sub(old, 1, 0, r, w);
            }
            self.state.rflags = (self.state.rflags & !state::rflags::CF)
                | (u64::from(saved_cf) * state::rflags::CF);
            return Ok(Action::Next);
        }
        let a = self.read_op(i, 0, next_ip, mem)? & m;
        // INC/DEC leave CF untouched.
        let saved_cf = self.state.cf();
        let result = if i.mnemonic() == Mnemonic::Inc {
            let r = a.wrapping_add(1);
            self.state.flags_add(a, 1, 0, r, w);
            r
        } else {
            let r = a.wrapping_sub(1);
            self.state.flags_sub(a, 1, 0, r, w);
            r
        };
        self.state.rflags =
            (self.state.rflags & !state::rflags::CF) | (u64::from(saved_cf) * state::rflags::CF);
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    fn exec_unary(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let a = self.read_op(i, 0, next_ip, mem)? & m;
        let result = if i.mnemonic() == Mnemonic::Not {
            // NOT updates no flags.
            !a & m
        } else {
            let r = 0u64.wrapping_sub(a);
            self.state.flags_sub(0, a, 0, r, w);
            r
        };
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    fn exec_shift(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let bits = u64::from(w * 8);
        let m = mask(w);
        let a = self.read_op(i, 0, next_ip, mem)? & m;
        let raw = self.shift_count(i, next_ip, mem)?;
        // Count masking matches the architecture: 0x3F for 64-bit operands,
        // 0x1F otherwise.
        let count = raw & if bits == 64 { 0x3F } else { 0x1F };

        // Shift by zero: result unchanged, no flags modified.
        if count == 0 {
            return Ok(Action::Next);
        }

        use Mnemonic::*;
        let (result, cf) = match i.mnemonic() {
            Shl | Sal => {
                if count < bits {
                    let cf = (a >> (bits - count)) & 1;
                    ((a << count) & m, cf)
                } else if count == bits {
                    // The last bit shifted out is bit 0.
                    (0, a & 1)
                } else {
                    (0, 0)
                }
            }
            Shr => {
                if count < bits {
                    let cf = (a >> (count - 1)) & 1;
                    (a >> count, cf)
                } else if count == bits {
                    let cf = (a >> (bits - 1)) & 1;
                    (0, cf)
                } else {
                    (0, 0)
                }
            }
            Sar => {
                let sa = sext(a, w);
                if count < bits {
                    let cf = ((sa >> (count - 1)) & 1) as u64;
                    (((sa >> count) as u64) & m, cf)
                } else {
                    // Shift larger than width saturates the sign.
                    let cf = ((sa >> 63) & 1) as u64;
                    if sa < 0 { (m, cf) } else { (0, cf) }
                }
            }
            _ => unreachable!("exec_shift dispatch guarantees coverage"),
        };

        // OF is defined for 1-bit shifts only; other counts leave it
        // undefined (cleared here).
        let of = match (i.mnemonic(), count) {
            (Shl | Sal, 1) => (((result >> (bits - 1)) & 1) ^ cf) != 0,
            (Shr, 1) => (a >> (bits - 1)) & 1 != 0,
            (Sar, 1) => false,
            _ => false,
        };
        self.state.flags_logic(result, w); // sets ZF/SF/PF, clears CF/OF/AF
        self.state.rflags |= cf * state::rflags::CF;
        self.state.rflags |= u64::from(of) * state::rflags::OF;
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    /// ROL/ROR/RCL/RCR.
    fn exec_rotate(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w = Self::op_width(i, 0);
        let bits = u64::from(w * 8);
        let m = mask(w);
        let a = self.read_op(i, 0, next_ip, mem)? & m;
        let raw = self.shift_count(i, next_ip, mem)?;
        let count = raw & if bits == 64 { 0x3F } else { 0x1F };
        if count == 0 {
            return Ok(Action::Next);
        }
        let cf0 = u64::from(self.state.cf());
        // sub-64 rotates need the count taken modulo the operand width.
        let count = if bits == 64 { count } else { count % bits };
        let (result, cf) = match i.mnemonic() {
            Rol => {
                let c = count % bits;
                if c == 0 {
                    // Count is a multiple of the width: only CF is written
                    // (set from the LSB that would have wrapped).
                    (a, a & 1)
                } else {
                    let r = rotl_bits(a, c, bits);
                    (r, r & 1)
                }
            }
            Ror => {
                let c = count % bits;
                if c == 0 {
                    (a, (a >> (bits - 1)) & 1)
                } else {
                    let r = rotr_bits(a, c, bits);
                    (r, (r >> (bits - 1)) & 1)
                }
            }
            Rcl | Rcr => {
                // Rotate through carry: effective width bits+1. The
                // rotation happens inside a (bits+1)-bit window; the carry
                // occupies bit `bits` of the concatenated value.
                let wide_bits = bits + 1;
                let c = count % wide_bits;
                let concat = a | (cf0 << bits);
                if c == 0 {
                    (a, cf0)
                } else if i.mnemonic() == Rcl {
                    let rotated =
                        ((concat << c) | (concat >> (wide_bits - c))) & ((1u64 << wide_bits) - 1);
                    (rotated & m, (rotated >> bits) & 1)
                } else {
                    let rotated =
                        ((concat >> c) | (concat << (wide_bits - c))) & ((1u64 << wide_bits) - 1);
                    (rotated & m, (rotated >> bits) & 1)
                }
            }
            _ => unreachable!("exec_rotate dispatch guarantees coverage"),
        };
        // OF is defined only for count == 1.
        let of = if count == 1 {
            match i.mnemonic() {
                Rol => ((result >> (bits - 1)) & 1) ^ cf != 0,
                Ror | Rcl | Rcr => ((result >> (bits - 1)) & 1) ^ (a >> (bits - 1)) & 1 != 0,
                _ => false,
            }
        } else {
            false
        };
        self.state.rflags = (self.state.rflags & !state::rflags::CF) | (cf * state::rflags::CF);
        self.state.rflags =
            (self.state.rflags & !state::rflags::OF) | (u64::from(of) * state::rflags::OF);
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    /// SHLD / SHRD.
    fn exec_double_shift(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        let w = Self::op_width(i, 0);
        let bits = u64::from(w * 8);
        let m = mask(w);
        let dst = self.read_op(i, 0, next_ip, mem)? & m;
        let src = self.read_op(i, 1, next_ip, mem)? & m;
        // The count is the THIRD operand (imm8 or CL), unlike plain shifts.
        let raw = match Self::kind(i, 2) {
            OpKind::Immediate8 | OpKind::Immediate8to64 => u64::from(i.immediate8()),
            OpKind::Register => self.state.reg(i.op_register(2)).expect("GPR count") & 0xFF,
            _ => 0,
        };
        let count = raw & if bits == 64 { 0x3F } else { 0x1F };
        if count == 0 {
            return Ok(Action::Next);
        }
        let (result, cf): (u64, u64) = if count > bits {
            // Counts larger than the width are architecturally undefined
            // for the sub-64 forms; PDG zeroes the destination.
            (0, 0)
        } else {
            let (a128, s128) = (u128::from(dst), u128::from(src));
            if i.mnemonic() == Mnemonic::Shld {
                let hi = (a128 << count) & u128::from(m);
                // src >> (bits - count) in the u128 domain; count == bits
                // contributes nothing (handled by the guard below).
                let lo = if count == bits {
                    0
                } else {
                    s128 >> (bits - count)
                };
                let r = ((hi | lo) as u64) & m;
                let cf = if count == 0 {
                    0
                } else {
                    (dst >> (bits - count)) & 1
                };
                (r, cf)
            } else {
                let lo = (s128 << (128 - bits)) >> (128 - count);
                let hi = if count == bits {
                    0u128
                } else {
                    u128::from(dst) >> count
                };
                let r = ((lo | hi) & u128::from(m)) as u64;
                let cf = if count == 0 {
                    0
                } else {
                    (dst >> (count - 1)) & 1
                };
                (r, cf)
            }
        };
        let of = count == 1 && ((result >> (bits - 1)) & 1) != ((dst >> (bits - 1)) & 1);
        self.state.flags_logic(result, w);
        self.state.rflags |= cf * state::rflags::CF;
        self.state.rflags |= u64::from(of) * state::rflags::OF;
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    /// MUL / IMUL (all forms) / DIV / IDIV (all widths).
    fn exec_muldiv(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let bits = w * 8;
        let set_mul_flags = |s: &mut CpuState, upper: u64, signed: bool, lower: u64, m: u64| {
            let significant = if signed {
                upper != sext_bits(lower, bits) as u64 & m
            } else {
                upper != 0
            };
            let sign = 1u64 << (bits - 1);
            let _ = sign;
            s.rflags &= !(state::rflags::CF | state::rflags::OF);
            if significant {
                s.rflags |= state::rflags::CF | state::rflags::OF;
            }
        };
        match i.mnemonic() {
            Mul => {
                let a = self.state.gpr[state::RAX] & m;
                let b = self.read_op(i, 0, next_ip, mem)? & m;
                let prod = a as u128 * b as u128;
                let (lo, hi) = (prod as u64 & m, (prod >> bits) as u64 & m);
                match w {
                    1 => {
                        self.state.gpr[state::RAX] =
                            (self.state.gpr[state::RAX] & !0xFFFF) | ((lo | (hi << 8)) & 0xFFFF);
                    }
                    2 => {
                        self.state.gpr[state::RAX] = (self.state.gpr[state::RAX] & !0xFFFF) | lo;
                        self.state.gpr[state::RDX] = (self.state.gpr[state::RDX] & !0xFFFF) | hi;
                    }
                    4 => {
                        self.state.gpr[state::RAX] = lo;
                        self.state.gpr[state::RDX] = hi;
                    }
                    _ => {
                        self.state.gpr[state::RAX] = lo;
                        self.state.gpr[state::RDX] = hi;
                    }
                }
                let lower = match w {
                    1 => lo & 0xFF,
                    2 => lo,
                    4 => lo,
                    _ => lo,
                };
                let upper = match w {
                    1 => hi & 0xFF,
                    2 => hi,
                    4 => hi,
                    _ => hi,
                };
                set_mul_flags(&mut self.state, upper, false, lower, m);
                Ok(Action::Next)
            }
            Imul => {
                if i.op_count() == 1 {
                    // One-operand: r/m * AL/AX/EAX/RAX → upper:lower.
                    let a = sext_bits(self.state.gpr[state::RAX] & m, bits) as i128;
                    let b = sext_bits(self.read_op(i, 0, next_ip, mem)? & m, bits) as i128;
                    let prod = a * b;
                    let lo = (prod as u64) & m;
                    let hi = ((prod >> bits) as u64) & m;
                    match w {
                        1 => {
                            self.state.gpr[state::RAX] = (self.state.gpr[state::RAX] & !0xFFFF)
                                | ((lo | (hi << 8)) & 0xFFFF);
                        }
                        2 => {
                            self.state.gpr[state::RAX] =
                                (self.state.gpr[state::RAX] & !0xFFFF) | lo;
                            self.state.gpr[state::RDX] =
                                (self.state.gpr[state::RDX] & !0xFFFF) | hi;
                        }
                        _ => {
                            self.state.gpr[state::RAX] = lo;
                            self.state.gpr[state::RDX] = hi;
                        }
                    }
                    let lower = if w == 1 { lo & 0xFF } else { lo };
                    let upper = if w == 1 { hi & 0xFF } else { hi };
                    set_mul_flags(&mut self.state, upper, true, lower, m);
                } else {
                    // Two/three-operand forms:
                    //   imul dst, src           → dst * src
                    //   imul dst, src, imm      → src * imm
                    let a = sext_bits(self.read_op(i, 0, next_ip, mem)? & m, bits) as i128;
                    let b = sext_bits(self.read_op(i, 1, next_ip, mem)? & m, bits) as i128;
                    let c = if i.op_count() == 3 {
                        match Self::kind(i, 2) {
                            OpKind::Immediate8 => sext_bits(i.immediate8() as u64, 8) as i128,
                            OpKind::Immediate16 => sext_bits(i.immediate16() as u64, 16) as i128,
                            OpKind::Immediate32 => sext_bits(i.immediate32() as u64, 32) as i128,
                            OpKind::Immediate8to64 => i.immediate8to64() as i128,
                            OpKind::Immediate32to64 => i.immediate32to64() as i128,
                            _ => sext_bits(self.read_op(i, 2, next_ip, mem)? & m, bits) as i128,
                        }
                    } else {
                        a
                    };
                    let prod = b * c;
                    let r = (prod as u64) & m;
                    let significant = prod != (sext_bits(r, bits) as i128);
                    self.state.rflags &= !(state::rflags::CF | state::rflags::OF);
                    if significant {
                        self.state.rflags |= state::rflags::CF | state::rflags::OF;
                    }
                    self.write_op(i, 0, next_ip, r, mem)?;
                }
                Ok(Action::Next)
            }
            Div => {
                let divisor = self.read_op(i, 0, next_ip, mem)? & m;
                if divisor == 0 {
                    return Err(Fault::DivideError);
                }
                let (dividend, q, r): (u128, u64, u64) = match w {
                    1 => {
                        let dividend = u128::from(self.state.gpr[state::RAX] & 0xFFFF);
                        let d = dividend / u128::from(divisor);
                        let r = dividend % u128::from(divisor);
                        (dividend, d as u64, r as u64)
                    }
                    2 => {
                        let dividend = u128::from(
                            ((self.state.gpr[state::RDX] & 0xFFFF) << 16)
                                | (self.state.gpr[state::RAX] & 0xFFFF),
                        );
                        let d = dividend / u128::from(divisor);
                        let r = dividend % u128::from(divisor);
                        (dividend, d as u64, r as u64)
                    }
                    4 => {
                        let dividend = u128::from(
                            ((self.state.gpr[state::RDX] & 0xFFFF_FFFF) << 32)
                                | (self.state.gpr[state::RAX] & 0xFFFF_FFFF),
                        );
                        let d = dividend / u128::from(divisor);
                        let r = dividend % u128::from(divisor);
                        (dividend, d as u64, r as u64)
                    }
                    _ => {
                        let dividend = (self.state.gpr[state::RDX] as u128) << 64
                            | self.state.gpr[state::RAX] as u128;
                        let d = dividend / u128::from(divisor);
                        let r = dividend % u128::from(divisor);
                        (dividend, d as u64, r as u64)
                    }
                };
                let qmax = if w == 8 { u64::MAX } else { m };
                if q > qmax {
                    return Err(Fault::DivideError);
                }
                let (q_reg, r_reg) = match w {
                    1 => {
                        // AL = quotient, AH = remainder.
                        let ax = ((r & 0xFF) << 8) | (q & 0xFF);
                        self.state.gpr[state::RAX] = (self.state.gpr[state::RAX] & !0xFFFF) | ax;
                        (0usize, 0usize)
                    }
                    2 => {
                        self.state.gpr[state::RAX] =
                            (self.state.gpr[state::RAX] & !0xFFFF) | (q & 0xFFFF);
                        self.state.gpr[state::RDX] =
                            (self.state.gpr[state::RDX] & !0xFFFF) | (r & 0xFFFF);
                        (0, 0)
                    }
                    _ => {
                        self.state.gpr[state::RAX] = q;
                        self.state.gpr[state::RDX] = r;
                        (0, 0)
                    }
                };
                let _ = (q_reg, r_reg, dividend);
                Ok(Action::Next)
            }
            Idiv => {
                let divisor = sext_bits(self.read_op(i, 0, next_ip, mem)? & m, bits);
                if divisor == 0 {
                    return Err(Fault::DivideError);
                }
                let (dividend, q, r): (i128, i64, i64) = match w {
                    1 => {
                        let dividend = (self.state.gpr[state::RAX] & 0xFFFF) as i16 as i128;
                        let d = dividend / i128::from(divisor as i16);
                        let r = dividend % i128::from(divisor as i16);
                        (dividend, d as i64, r as i64)
                    }
                    2 => {
                        let ax = (self.state.gpr[state::RAX] & 0xFFFF) as i16 as i32 as i128;
                        let dx = (self.state.gpr[state::RDX] & 0xFFFF) as i16 as i32 as i128;
                        let dividend = (dx << 16) | (ax & 0xFFFF);
                        let d = dividend / i128::from(divisor as i16);
                        let r = dividend % i128::from(divisor as i16);
                        (dividend, d as i64, r as i64)
                    }
                    4 => {
                        let eax = (self.state.gpr[state::RAX] & 0xFFFF_FFFF) as i32 as i128;
                        let edx = (self.state.gpr[state::RDX] & 0xFFFF_FFFF) as i32 as i128;
                        let dividend = (edx << 32) | (eax & 0xFFFF_FFFF);
                        let d = dividend / i128::from(divisor as i32);
                        let r = dividend % i128::from(divisor as i32);
                        (dividend, d as i64, r as i64)
                    }
                    _ => {
                        let dividend = ((self.state.gpr[state::RDX] as i128) << 64)
                            | (self.state.gpr[state::RAX] as i128);
                        let d = dividend / i128::from(divisor);
                        let r = dividend % i128::from(divisor);
                        (dividend, d as i64, r as i64)
                    }
                };
                // Range check: quotient must fit the operand width.
                let fits = match w {
                    1 => (-128i64..=127).contains(&q),
                    2 => (-32768..=32767).contains(&q),
                    4 => (-2147483648..=2147483647).contains(&q),
                    _ => true,
                };
                if !fits {
                    return Err(Fault::DivideError);
                }
                match w {
                    1 => {
                        let ax = ((r as u64 & 0xFF) << 8) | (q as u64 & 0xFF);
                        self.state.gpr[state::RAX] = (self.state.gpr[state::RAX] & !0xFFFF) | ax;
                    }
                    2 => {
                        self.state.gpr[state::RAX] =
                            (self.state.gpr[state::RAX] & !0xFFFF) | (q as u64 & 0xFFFF);
                        self.state.gpr[state::RDX] =
                            (self.state.gpr[state::RDX] & !0xFFFF) | (r as u64 & 0xFFFF);
                    }
                    _ => {
                        self.state.gpr[state::RAX] = q as u64;
                        self.state.gpr[state::RDX] = r as u64;
                    }
                }
                let _ = dividend;
                Ok(Action::Next)
            }
            _ => unreachable!("exec_muldiv dispatch guarantees coverage"),
        }
    }

    /// BT / BTS / BTR / BTC (register and memory forms).
    fn exec_bt(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let bits = u64::from(w * 8);

        // Bit index: register (unmasked) or imm8.
        let idx_raw = match Self::kind(i, 1) {
            OpKind::Immediate8 | OpKind::Immediate8to64 => i.immediate8() as u64,
            OpKind::Register => self.read_op(i, 1, next_ip, mem)?,
            _ => {
                return Err(Fault::InvalidOpcode {
                    mnemonic: format!("{:?}", i.mnemonic()),
                    rip: i.ip(),
                });
            }
        };

        let bit = match Self::kind(i, 0) {
            OpKind::Register => {
                let a = self.read_op(i, 0, next_ip, mem)? & m;
                let bit = (a >> (idx_raw % bits)) & 1;
                if i.mnemonic() != Bt {
                    let pos = idx_raw % bits;
                    let r = match i.mnemonic() {
                        Bts => a | (1 << pos),
                        Btr => a & !(1 << pos),
                        _ => a ^ (1 << pos), // Btc
                    };
                    self.write_op(i, 0, next_ip, r, mem)?;
                }
                bit
            }
            OpKind::Memory => {
                // Memory form: the index offsets the effective address by
                // (idx / bits) elements before the in-word bit position.
                let base = self.ea(i, next_ip);
                let elem = idx_raw / bits;
                let pos = idx_raw % bits;
                let addr = base.wrapping_add(elem * u64::from(w));
                let a = mem.read(w as u8, addr).map_err(AccessError::into_fault)? & m;
                let bit = (a >> pos) & 1;
                if i.mnemonic() != Bt {
                    let r = match i.mnemonic() {
                        Bts => a | (1 << pos),
                        Btr => a & !(1 << pos),
                        _ => a ^ (1 << pos),
                    };
                    mem.write(w as u8, addr, r)
                        .map_err(AccessError::into_fault)?;
                }
                bit
            }
            _ => {
                return Err(Fault::InvalidOpcode {
                    mnemonic: format!("{:?}", i.mnemonic()),
                    rip: i.ip(),
                });
            }
        };
        self.state.rflags = (self.state.rflags & !state::rflags::CF) | (bit * state::rflags::CF);
        Ok(Action::Next)
    }

    /// BSF / BSR / POPCNT / TZCNT / LZCNT.
    fn exec_bit_scan(
        &mut self,
        i: &Instruction,
        next_ip: u64,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w = Self::op_width(i, 0);
        let m = mask(w);
        let src = self.read_op(i, 1, next_ip, mem)? & m;
        let bits = w * 8;
        let r: u64 = match i.mnemonic() {
            Bsf => {
                if src == 0 {
                    self.state.rflags |= state::rflags::ZF;
                    // Destination is architecturally undefined; PDG leaves it.
                    return Ok(Action::Next);
                }
                self.state.rflags &= !state::rflags::ZF;
                src.trailing_zeros() as u64
            }
            Bsr => {
                if src == 0 {
                    self.state.rflags |= state::rflags::ZF;
                    return Ok(Action::Next);
                }
                self.state.rflags &= !state::rflags::ZF;
                (bits - 1 - src.leading_zeros() + (64 - bits)) as u64
            }
            Popcnt => {
                let c = src.count_ones() as u64;
                self.state.rflags &= !(state::rflags::CF | state::rflags::OF | state::rflags::AF);
                self.state.rflags = (self.state.rflags & !state::rflags::ZF)
                    | (u64::from(c == 0) * state::rflags::ZF);
                c
            }
            Tzcnt => {
                // CF = source is zero; ZF = result is zero.
                let c = if src == 0 {
                    bits as u64
                } else {
                    src.trailing_zeros() as u64
                };
                self.state.rflags = (self.state.rflags & !state::rflags::CF)
                    | (u64::from(src == 0) * state::rflags::CF);
                self.state.rflags = (self.state.rflags & !state::rflags::ZF)
                    | (u64::from(c == 0) * state::rflags::ZF);
                c
            }
            Lzcnt => {
                let lz = src.leading_zeros() as u64 - (64 - bits) as u64;
                let c = if src == 0 { bits as u64 } else { lz };
                self.state.rflags = (self.state.rflags & !state::rflags::CF)
                    | (u64::from(src == 0) * state::rflags::CF);
                self.state.rflags = (self.state.rflags & !state::rflags::ZF)
                    | (u64::from(c == 0) * state::rflags::ZF);
                c
            }
            _ => unreachable!("exec_bit_scan dispatch guarantees coverage"),
        };
        self.write_op(i, 0, next_ip, r, mem)?;
        Ok(Action::Next)
    }

    /// String operations with optional REP/REPE/REPNE. Iterations are
    /// restartable: RSI/RDI/RCX always hold the remaining work, so hitting
    /// the per-step cap (or a fault fixed by a handler) resumes exactly
    /// where the previous invocation stopped.
    fn exec_string(
        &mut self,
        i: &Instruction,
        mem: &mut dyn MemoryAccess,
    ) -> Result<Action, Fault> {
        use Mnemonic::*;
        let w: u32 = match i.memory_size().size() {
            1 => 1,
            2 => 2,
            4 => 4,
            _ => 8,
        };
        // NOTE: iced reports the F3 bit through BOTH has_rep_prefix() and
        // has_repe_prefix(); REPE/REPNE termination only applies to the
        // CMPS/SCAS families.
        let is_cmp_family = matches!(
            i.mnemonic(),
            Scasb | Scasw | Scasd | Scasq | Cmpsb | Cmpsw | Cmpsd | Cmpsq
        );
        let is_rep = i.has_rep_prefix() || i.has_repne_prefix();
        let mut iterations_left = if is_rep { REP_ITER_CAP } else { 1 };
        let step: i64 = if self.state.df() {
            -(w as i64)
        } else {
            w as i64
        };

        let mut zf = self.state.zf();
        while iterations_left > 0 {
            if is_rep {
                // The count register gates every repetition.
                let rcx = self.state.gpr[state::RCX];
                if rcx == 0 {
                    break;
                }
            }
            let rsi = self.state.gpr[state::RSI];
            let rdi = self.state.gpr[state::RDI];
            let mut advance = true;
            match i.mnemonic() {
                Movsb | Movsw | Movsd | Movsq => {
                    let v = mem.read(w as u8, rsi).map_err(AccessError::into_fault)?;
                    mem.write(w as u8, rdi, v)
                        .map_err(AccessError::into_fault)?;
                }
                Stosb | Stosw | Stosd | Stosq => {
                    let v = self.state.gpr[state::RAX];
                    mem.write(w as u8, rdi, v)
                        .map_err(AccessError::into_fault)?;
                }
                Lodsb | Lodsw | Lodsd | Lodsq => {
                    let v = mem.read(w as u8, rsi).map_err(AccessError::into_fault)?;
                    self.state.gpr[state::RAX] = if w == 8 {
                        v
                    } else {
                        (self.state.gpr[state::RAX] & !mask(w)) | (v & mask(w))
                    };
                }
                Scasb | Scasw | Scasd | Scasq => {
                    let a = self.state.gpr[state::RAX] & mask(w);
                    let b = mem.read(w as u8, rdi).map_err(AccessError::into_fault)?;
                    let r = a.wrapping_sub(b);
                    self.state.flags_sub(a, b, 0, r, w);
                    zf = r == 0;
                    advance = true;
                }
                Cmpsb | Cmpsw | Cmpsd | Cmpsq => {
                    let a = mem.read(w as u8, rsi).map_err(AccessError::into_fault)?;
                    let b = mem.read(w as u8, rdi).map_err(AccessError::into_fault)?;
                    let r = a.wrapping_sub(b);
                    self.state.flags_sub(a, b, 0, r, w);
                    zf = r == 0;
                }
                _ => unreachable!("exec_string dispatch guarantees coverage"),
            }
            let _ = advance;
            // Advance pointers and count.
            self.state.gpr[state::RSI] = rsi.wrapping_add(step as u64);
            self.state.gpr[state::RDI] = rdi.wrapping_add(step as u64);
            if is_rep {
                self.state.gpr[state::RCX] = self.state.gpr[state::RCX].wrapping_sub(1);
            }
            iterations_left -= 1;

            // Termination checks for cmp-like prefixes run after each
            // iteration, per the architecture.
            if is_cmp_family {
                if i.has_repne_prefix() {
                    if zf {
                        break;
                    }
                } else if !zf {
                    // F3 on CMPS/SCAS is REPE: stop when not equal.
                    break;
                }
            }
            if !is_rep {
                break;
            }
        }
        if is_rep && self.state.gpr[state::RCX] != 0 && iterations_left == 0 {
            // More work remains: re-enter the instruction on the next step.
            return Ok(Action::Yield);
        }
        Ok(Action::Next)
    }

    /// Evaluate a condition code against RFLAGS (used by Jcc/CMOVcc/SETcc).
    fn cond_cc(&self, cc: ConditionCode) -> bool {
        let s = &self.state;
        match cc {
            ConditionCode::None => false,
            ConditionCode::o => s.of(),
            ConditionCode::no => !s.of(),
            ConditionCode::b => s.cf(),
            ConditionCode::ae => !s.cf(),
            ConditionCode::e => s.zf(),
            ConditionCode::ne => !s.zf(),
            ConditionCode::be => s.cf() || s.zf(),
            ConditionCode::a => !s.cf() && !s.zf(),
            ConditionCode::s => s.sf(),
            ConditionCode::ns => !s.sf(),
            ConditionCode::p => s.pf(),
            ConditionCode::np => !s.pf(),
            ConditionCode::l => s.sf() != s.of(),
            ConditionCode::ge => s.sf() == s.of(),
            ConditionCode::le => s.zf() || s.sf() != s.of(),
            ConditionCode::g => !s.zf() && s.sf() == s.of(),
        }
    }
}

/// Effective address of `i`'s memory operand against `state` (see
/// [`Interpreter::ea`] notes: iced pre-computes RIP-relative targets;
/// FS/GS segment bases apply in 64-bit mode).
pub(crate) fn effective_address(i: &Instruction, next_ip: u64, state: &CpuState) -> u64 {
    let _ = next_ip;
    let base = i.memory_base();
    let index = i.memory_index();
    let scale = i.memory_index_scale() as u64;
    let disp = i.memory_displacement64();
    let mut addr = disp;
    if base != Register::None {
        let b = if base == Register::RIP {
            // Displacement already holds the absolute target.
            0
        } else {
            state.reg(base).expect("GPR base")
        };
        addr = addr.wrapping_add(b);
    }
    if index != Register::None {
        let v = state.reg(index).expect("GPR index");
        addr = addr.wrapping_add(v.wrapping_mul(scale));
    }
    match i.segment_prefix() {
        Register::FS => addr = addr.wrapping_add(state.fs_base),
        Register::GS => addr = addr.wrapping_add(state.gs_base),
        _ => {}
    }
    addr
}

enum StepEnd {
    Continue,
    Halted,
}

/// Is `r` one of the control registers the interpreter models?
fn is_cr(r: Register) -> bool {
    matches!(
        r,
        Register::CR0 | Register::CR2 | Register::CR3 | Register::CR4
    )
}

/// Stack-slot width of a PUSH/POP operand in 64-bit mode: 2 for 16-bit
/// operands (66 prefix shows up as the operand's own size), else 8.
fn stack_width(i: &Instruction, n: u32) -> u32 {
    match i.op_kind(n) {
        OpKind::Register => {
            let (_, size, _) = crate::state::decode_reg(i.op_register(n)).unwrap_or((0, 8, false));
            if size == 2 { 2 } else { 8 }
        }
        OpKind::Memory => {
            if i.memory_size().size() == 2 {
                2
            } else {
                8
            }
        }
        OpKind::Immediate16 | OpKind::Immediate8to16 => 2,
        _ => 8,
    }
}

/// Index of a full-width GPR register id.
fn gpr_index(r: Register) -> Option<usize> {
    use Register::*;
    Some(match r {
        RAX => 0,
        RCX => 1,
        RDX => 2,
        RBX => 3,
        RSP => 4,
        RBP => 5,
        RSI => 6,
        RDI => 7,
        R8 => 8,
        R9 => 9,
        R10 => 10,
        R11 => 11,
        R12 => 12,
        R13 => 13,
        R14 => 14,
        R15 => 15,
        _ => return Option::None,
    })
}

/// Mask for a `bits`-wide window.
#[inline]
fn bits_mask(bits: u64) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Rotate left inside a `bits`-wide window (sub-64-bit operands).
#[inline]
fn rotl_bits(v: u64, c: u64, bits: u64) -> u64 {
    ((v << c) | (v >> (bits - c))) & bits_mask(bits)
}

/// Rotate right inside a `bits`-wide window (sub-64-bit operands).
#[inline]
fn rotr_bits(v: u64, c: u64, bits: u64) -> u64 {
    ((v >> c) | (v << (bits - c))) & bits_mask(bits)
}

/// Sign-extend a value held in its low `bits` bits out to 64 bits.
#[inline]
fn sext_bits(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// Sign-extend a value that holds `w` bytes.
#[inline]
fn sext(v: u64, w: u32) -> i64 {
    match w {
        1 => v as u8 as i8 as i64,
        2 => v as u16 as i16 as i64,
        4 => v as u32 as i32 as i64,
        _ => v as i64,
    }
}
