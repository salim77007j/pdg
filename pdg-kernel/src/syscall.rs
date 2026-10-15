//! HLE syscall dispatch.
//!
//! The dispatcher is registry-based: `register(number, handler)` installs a
//! closure that receives the guest CPU state and memory port. It implements
//! [`pdg_cpu::SyscallHook`], reading the syscall number from RAX
//! (FreeBSD/Orbis convention: number in RAX, arguments in
//! RDI/RSI/RDX/R10/R8/R9, result returned in RAX).

use std::collections::HashMap;

use pdg_cpu::{Fault, MemoryAccess, SyscallHook};
use thiserror::Error;

/// Syscall number (carried in RAX at the `syscall` instruction).
pub type SyscallNumber = u64;

/// Guest syscall argument-register convention (FreeBSD style):
/// number in RAX, arguments in RDI, RSI, RDX, R10, R8, R9; result in RAX.
pub const ARG_REGISTERS: [iced_x86::Register; 6] = [
    iced_x86::Register::RDI,
    iced_x86::Register::RSI,
    iced_x86::Register::RDX,
    iced_x86::Register::R10,
    iced_x86::Register::R8,
    iced_x86::Register::R9,
];

/// Kernel-side error for a syscall.
#[derive(Debug, Error)]
pub enum KernelError {
    /// No handler registered for this number.
    #[error("syscall {number:#x} not registered")]
    UnknownSyscall {
        /// The unknown syscall number.
        number: u64,
    },
    /// A handler existed but failed.
    #[error("syscall {number:#x} failed: {message}")]
    Failed {
        /// Syscall number.
        number: u64,
        /// Failure description.
        message: String,
    },
}

type Handler =
    Box<dyn FnMut(&mut pdg_cpu::CpuState, &mut dyn MemoryAccess) -> Result<i64, String> + Send>;

/// Registry-based syscall dispatcher.
#[derive(Default)]
pub struct SyscallDispatcher {
    handlers: HashMap<u64, Handler>,
}

impl std::fmt::Debug for SyscallDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyscallDispatcher")
            .field("registered", &self.handlers.len())
            .finish()
    }
}

impl SyscallDispatcher {
    /// Empty dispatcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) the handler for `number`.
    pub fn register<F>(&mut self, number: SyscallNumber, handler: F)
    where
        F: FnMut(&mut pdg_cpu::CpuState, &mut dyn MemoryAccess) -> Result<i64, String>
            + Send
            + 'static,
    {
        self.handlers.insert(number, Box::new(handler));
    }

    /// Number of registered syscalls.
    pub fn registered(&self) -> usize {
        self.handlers.len()
    }

    /// Dispatch `number`, writing the result into guest RAX.
    pub fn dispatch(
        &mut self,
        number: SyscallNumber,
        cpu: &mut pdg_cpu::CpuState,
        mem: &mut dyn MemoryAccess,
    ) -> Result<(), KernelError> {
        let handler = self
            .handlers
            .get_mut(&number)
            .ok_or(KernelError::UnknownSyscall { number })?;
        match handler(cpu, mem) {
            Ok(result) => {
                cpu.set_reg(iced_x86::Register::RAX, result as u64)
                    .expect("rax");
                Ok(())
            }
            Err(message) => Err(KernelError::Failed { number, message }),
        }
    }
}

impl SyscallHook for SyscallDispatcher {
    fn on_syscall(
        &mut self,
        cpu: &mut pdg_cpu::CpuState,
        mem: &mut dyn MemoryAccess,
    ) -> Result<(), Fault> {
        let number = cpu.reg(iced_x86::Register::RAX).unwrap_or(0);
        match self.dispatch(number, cpu, mem) {
            Ok(()) => Ok(()),
            Err(KernelError::UnknownSyscall { .. }) => {
                Err(Fault::UnhandledSyscall { rip: cpu.rip })
            }
            Err(KernelError::Failed { number, message }) => {
                tracing::warn!(number, %message, "guest syscall failed");
                // Syscall failure is a guest-visible error code, not a CPU
                // fault: return the conventional -1 ( errno in RAX ).
                cpu.set_reg(iced_x86::Register::RAX, u64::MAX).expect("rax");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdg_cpu::{CpuState, FlatMemory};

    #[test]
    fn registers_and_dispatches() {
        let mut disp = SyscallDispatcher::new();
        disp.register(0x99, |cpu, _| {
            let a = cpu.reg(iced_x86::Register::RDI).unwrap_or(0);
            Ok(a as i64 * 3)
        });
        let mut cpu = CpuState::new(0, 0x100);
        cpu.set_reg(iced_x86::Register::RAX, 0x99).unwrap();
        cpu.set_reg(iced_x86::Register::RDI, 7).unwrap();
        let mut mem = FlatMemory::new(0x100);
        disp.on_syscall(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.reg(iced_x86::Register::RAX), Some(21));
    }

    #[test]
    fn unknown_syscall_raises_guest_fault() {
        let mut disp = SyscallDispatcher::new();
        let mut cpu = CpuState::new(0, 0x100);
        cpu.set_reg(iced_x86::Register::RAX, 0xDEAD).unwrap();
        let mut mem = FlatMemory::new(0x100);
        assert!(matches!(
            disp.on_syscall(&mut cpu, &mut mem),
            Err(Fault::UnhandledSyscall { .. })
        ));
    }

    #[test]
    fn failing_handler_returns_minus_one() {
        let mut disp = SyscallDispatcher::new();
        disp.register(1, |_, _| Err("no such file".to_owned()));
        let mut cpu = CpuState::new(0, 0x100);
        cpu.set_reg(iced_x86::Register::RAX, 1).unwrap();
        let mut mem = FlatMemory::new(0x100);
        disp.on_syscall(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.reg(iced_x86::Register::RAX), Some(u64::MAX));
    }
}
