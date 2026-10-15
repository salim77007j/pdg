//! Dispatch across the SIMD sibling modules.

use iced_x86::Instruction;

use crate::{fault::Fault, state::CpuState};

use super::{V256, bridge, crypto, float, shuffle};

use crate::fault::MemoryAccess;
use crate::interpreter::Action;

/// Try each sibling family; `None` means "not mine".
pub(crate) fn try_all(
    i: &Instruction,
    next_ip: u64,
    state: &mut CpuState,
    mem: &mut dyn MemoryAccess,
    a: V256,
    b: V256,
    wide: bool,
) -> Result<Option<Action>, Fault> {
    if let Some(action) = shuffle::exec(i, next_ip, state, mem, a, b, wide)? {
        return Ok(Some(action));
    }
    if let Some(action) = float::exec(i, next_ip, state, mem, a, b, wide)? {
        return Ok(Some(action));
    }
    if let Some(action) = crypto::exec(i, next_ip, state, mem, a, b, wide)? {
        return Ok(Some(action));
    }
    if let Some(action) = bridge::exec(i, next_ip, state, mem, wide)? {
        return Ok(Some(action));
    }
    Ok(None)
}
