//! HLE hooks letting kernel code intercept guest `syscall` instructions.

use crate::{
    fault::{Fault, MemoryAccess},
    state::CpuState,
};

/// Handler invoked when the guest executes `syscall`.
///
/// Phase 1 keeps the ABI deliberately thin: the hook receives mutable CPU
/// state (to read the syscall number from RAX and write the return value) and
/// the memory port. `pdg-kernel` layers the full Orbis-style dispatcher on
/// top of this trait (see `pdg-kernel::syscall`).
pub trait SyscallHook: Send {
    /// Handle the syscall currently in flight. Return `Err` to inject a
    /// guest-visible fault.
    fn on_syscall(&mut self, cpu: &mut CpuState, mem: &mut dyn MemoryAccess) -> Result<(), Fault>;
}
