//! The interpreter: fetch → decode → execute for one logical processor.
//!
//! Decoding is delegated to `iced-x86` (fast, complete x86-64 decoding);
//! this module owns *semantics*: operand evaluation (including RIP-relative
//! addressing and x86-64 register-write rules), architectural flags, the
//! stack, control flow, and fault reporting. Any instruction outside the
//! implemented Phase-1 subset raises `#UD` — nothing executes as a silent
//! no-op.

use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

use crate::{
    fault::{AccessError, Fault, MemoryAccess},
    hooks::SyscallHook,
    state::{CpuState, mask},
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
enum Action {
    Next,
    Jump(u64),
    Halt,
}

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
                Ok(StepEnd::Continue) => executed += 1,
                Ok(StepEnd::Halted) => {
                    return RunOutcome::Halted {
                        instructions: executed + 1,
                        rip: self.state.rip,
                    };
                }
                Err(f) => return RunOutcome::Fault(f),
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
    /// Note on RIP-relative addressing: iced's `memory_displacement64()`
    /// returns the *computed absolute target* (next_ip + disp) for
    /// RIP-based operands, not the raw displacement.
    fn ea(&self, i: &Instruction, next_ip: u64) -> u64 {
        let base = i.memory_base();
        let index = i.memory_index();
        let scale = i.memory_index_scale() as u64;
        let disp = i.memory_displacement64();
        let mut addr = disp;
        if base != Register::None {
            let b = if base == Register::RIP {
                // Displacement already holds the absolute target.
                return addr;
            } else {
                self.state.reg(base).expect("GPR base")
            };
            addr = addr.wrapping_add(b);
        }
        let _ = next_ip;
        if index != Register::None {
            let v = self.state.reg(index).expect("GPR index");
            addr = addr.wrapping_add(v.wrapping_mul(scale));
        }
        addr
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
        self.state.gpr[crate::state::RSP] = self.state.rsp().wrapping_sub(8);
        mem.write(8, self.state.rsp(), value)
            .map_err(AccessError::into_fault)
    }

    fn pop_u64(&mut self, mem: &mut dyn MemoryAccess) -> Result<u64, Fault> {
        let v = mem
            .read(8, self.state.rsp())
            .map_err(AccessError::into_fault)?;
        self.state.gpr[crate::state::RSP] = self.state.rsp().wrapping_add(8);
        Ok(v)
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
            Nop | Endbr64 | Endbr32 => Ok(Action::Next),
            Hlt => Ok(Action::Halt),

            Mov => {
                let v = self.read_op(i, 1, next_ip, mem)?;
                self.write_op(i, 0, next_ip, v, mem)?;
                Ok(Action::Next)
            }
            Lea => {
                let addr = self.ea(i, next_ip);
                self.write_op(i, 0, next_ip, addr, mem)?;
                Ok(Action::Next)
            }
            Xchg => {
                let a = self.read_op(i, 0, next_ip, mem)?;
                let b = self.read_op(i, 1, next_ip, mem)?;
                self.write_op(i, 0, next_ip, b, mem)?;
                self.write_op(i, 1, next_ip, a, mem)?;
                Ok(Action::Next)
            }

            Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test => self.exec_binop(i, next_ip, mem),
            Inc | Dec => self.exec_incdec(i, next_ip, mem),
            Not | Neg => self.exec_unary(i, next_ip, mem),
            Shl | Sal | Shr | Sar => self.exec_shift(i, next_ip, mem),

            Push => {
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
                self.push_u64(v, mem)?;
                Ok(Action::Next)
            }
            Pop => {
                let v = self.pop_u64(mem)?;
                self.write_op(i, 0, next_ip, v, mem)?;
                Ok(Action::Next)
            }
            Pushf | Pushfd | Pushfq => {
                self.push_u64(self.state.rflags, mem)?;
                Ok(Action::Next)
            }
            Popf | Popfd | Popfq => {
                self.state.rflags = self.pop_u64(mem)? | crate::state::rflags::RESERVED_1;
                Ok(Action::Next)
            }

            Call => {
                let target = self.branch_target(i, next_ip, mem)?;
                self.push_u64(next_ip, mem)?;
                Ok(Action::Jump(target))
            }
            Ret => {
                let target = self.pop_u64(mem)?;
                if i.op_count() > 0 {
                    // RET imm16: pop the frame size too.
                    if let OpKind::Immediate16 = Self::kind(i, 0) {
                        let frame = i.immediate16() as u64;
                        self.state.gpr[crate::state::RSP] = self.state.rsp().wrapping_add(frame);
                    }
                }
                Ok(Action::Jump(target))
            }
            Jmp => {
                let target = self.branch_target(i, next_ip, mem)?;
                Ok(Action::Jump(target))
            }
            Jno | Jb | Jae | Je | Jne | Jbe | Ja | Js | Jns | Jp | Jnp | Jl | Jge | Jle | Jg => {
                if self.cond(i.mnemonic()) {
                    Ok(Action::Jump(i.near_branch_target()))
                } else {
                    Ok(Action::Next)
                }
            }

            Clc => {
                self.state.rflags &= !crate::state::rflags::CF;
                Ok(Action::Next)
            }
            Stc => {
                self.state.rflags |= crate::state::rflags::CF;
                Ok(Action::Next)
            }
            Cmc => {
                self.state.rflags ^= crate::state::rflags::CF;
                Ok(Action::Next)
            }
            Cld => {
                self.state.rflags &= !crate::state::rflags::DF;
                Ok(Action::Next)
            }
            Std => {
                self.state.rflags |= crate::state::rflags::DF;
                Ok(Action::Next)
            }

            Syscall => {
                match self.syscall_hook.as_mut() {
                    Some(hook) => hook.on_syscall(&mut self.state, mem)?,
                    None => return Err(Fault::UnhandledSyscall { rip: i.ip() }),
                }
                Ok(Action::Next)
            }

            other => Err(Fault::InvalidOpcode {
                mnemonic: format!("{other:?}"),
                rip: i.ip(),
            }),
        }
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
        self.state.rflags = (self.state.rflags & !crate::state::rflags::CF)
            | (u64::from(saved_cf) * crate::state::rflags::CF);
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
        self.state.rflags |= cf * crate::state::rflags::CF;
        self.state.rflags |= u64::from(of) * crate::state::rflags::OF;
        self.write_op(i, 0, next_ip, result, mem)?;
        Ok(Action::Next)
    }

    /// Evaluate a Jcc condition.
    fn cond(&self, m: Mnemonic) -> bool {
        use Mnemonic::*;
        let s = &self.state;
        match m {
            Jno => !s.of(),
            Jb => s.cf(),
            Jae => !s.cf(),
            Je => s.zf(),
            Jne => !s.zf(),
            Jbe => s.cf() || s.zf(),
            Ja => !s.cf() && !s.zf(),
            Js => s.sf(),
            Jns => !s.sf(),
            Jp => s.pf(),
            Jnp => !s.pf(),
            Jl => s.sf() != s.of(),
            Jge => s.sf() == s.of(),
            Jle => s.zf() || s.sf() != s.of(),
            Jg => !s.zf() && s.sf() == s.of(),
            _ => false,
        }
    }
}

enum StepEnd {
    Continue,
    Halted,
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
