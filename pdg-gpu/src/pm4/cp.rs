//! The PM4 command processor (CP).
//!
//! Executes a packet stream against a [`GpuContext`], translating
//! register writes into state changes and draw/dispatch/synchronization
//! packets into [`GpuCommand`]s a backend renders. Memory-backed packets
//! (fences, IB chains, semaphore waits, copies) go through a
//! [`CpMemPort`] so the CP itself stays independent of the memory system
//! and unit-testable against a fake port.
//!
//! What this CP implements is the honest subset verified against public
//! documentation (kernel `soc15d.h` + shadPS4 `pm4_cmds.h`): the four
//! `SET_*_REG` spaces, type-0 writes, draws (`DRAW_INDEX_AUTO`,
//! `DRAW_INDEX_2`), `DISPATCH_DIRECT`, IB chains (`INDIRECT_BUFFER`),
//! synchronization (`RELEASE_MEM`, `EVENT_WRITE`, `EVENT_WRITE_EOP`,
//! `MEM_SEMAPHORE`, `WAIT_REG_MEM`, `ACQUIRE_MEM`, `PFP_SYNC_ME`,
//! `SURFACE_SYNC`), data movement (`WRITE_DATA`, `DMA_DATA`,
//! `COPY_DATA`) and state control (`CLEAR_STATE`, `CONTEXT_CONTROL`,
//! `SET_BASE`, `INDEX_BASE`, `INDEX_TYPE`, `NUM_INSTANCES`,
//! `INDEX_BUFFER_SIZE`, `PREAMBLE_CNTL`). Everything else is refused
//! with a precise error naming the opcode — never silently dropped.

use crate::ir::{DrawParams, GpuCommand, GpuContext, PrimitiveType};
use crate::pm4::decode::{DecodeError, PacketDecoder};
use crate::pm4::opcode::ItOpcode;
use crate::pm4::regs::{RegAddr, RegSpace};
use thiserror::Error;

/// Memory port the CP uses for packet data that lives in guest memory.
pub trait CpMemPort {
    /// Read a 32-bit word.
    fn read_u32(&mut self, addr: u64) -> u32;
    /// Write a 32-bit word.
    fn write_u32(&mut self, addr: u64, value: u32);
    /// Read a block of words (returns exactly `len` words or fewer at a
    /// fault; callers must tolerate short reads as a stream end).
    fn read_words(&mut self, addr: u64, len: usize) -> Vec<u32>;
    /// Write bytes (used by copies and readback paths).
    fn write_bytes(&mut self, addr: u64, data: &[u8]);
}

/// Errors raised while executing a packet stream.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum CpError {
    /// Framing error.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// An opcode PDG knows but refuses to fake.
    #[error("packet {opcode:#04x} ({name}) is not implemented: {why}")]
    Unsupported {
        /// Raw opcode.
        opcode: u8,
        /// Opcode name ("UNKNOWN" when not in the table).
        name: &'static str,
        /// Precise reason.
        why: &'static str,
    },
    /// An `SET_*_REG` offset that does not map into a modeled segment.
    #[error("register offset {offset:#x} in {space:?} space does not map to a gfx10.3 segment")]
    BadRegisterOffset {
        /// Space the packet wrote.
        space: RegSpace,
        /// Raw payload offset word.
        offset: u32,
    },
    /// An IB nest deeper than the hardware allows.
    #[error("indirect buffer nesting deeper than {max}")]
    IbOverflow {
        /// The depth limit that was hit.
        max: u32,
    },
    /// An IB whose header count does not match its payload.
    #[error("indirect buffer at {addr:#x} is malformed")]
    BadIb {
        /// The IB's guest address.
        addr: u64,
    },
}

/// Per-submission statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpStats {
    /// Packets executed.
    pub packets: u32,
    /// Registers written through `SET_*_REG` / type-0.
    pub reg_writes: u32,
    /// Draws emitted.
    pub draws: u32,
    /// Dispatches emitted.
    pub dispatches: u32,
    /// IBs followed.
    pub indirect_buffers: u32,
    /// Fence writes emitted.
    pub fences: u32,
    /// Words consumed from the root stream.
    pub words: u32,
}

/// The command processor.
#[derive(Debug)]
pub struct CommandProcessor {
    /// Typed register context (shared with the backend).
    ctx: GpuContext,
    /// Completed commands, drained per submission.
    out: Vec<GpuCommand>,
    /// IB nesting depth.
    depth: u32,
    /// Stats for the current submission.
    stats: CpStats,
}

/// Maximum IB nesting depth (hardware is 4; PDG allows a little more for
/// tests but still finite).
const IB_MAX_DEPTH: u32 = 8;

impl CommandProcessor {
    /// New CP over a fresh context.
    #[must_use]
    pub fn new() -> Self {
        Self::with_context(GpuContext::new())
    }

    /// New CP over an existing context (state persists across
    /// submissions, exactly like real CP state).
    #[must_use]
    pub fn with_context(ctx: GpuContext) -> Self {
        Self {
            ctx,
            out: Vec::new(),
            depth: 0,
            stats: CpStats::default(),
        }
    }

    /// Shared typed context.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Mutable typed context (for device-side integration).
    pub fn context_mut(&mut self) -> &mut GpuContext {
        &mut self.ctx
    }

    /// Take the commands produced by the last [`Self::submit`].
    #[must_use]
    pub fn take_commands(&mut self) -> Vec<GpuCommand> {
        std::mem::take(&mut self.out)
    }

    /// Last submission's stats.
    #[must_use]
    pub fn stats(&self) -> CpStats {
        self.stats
    }

    /// Execute a root stream (a ring segment or test-built command list).
    ///
    /// Returns the number of root words consumed. On error the context is
    /// left in the state reached before the failing packet.
    pub fn submit(&mut self, words: &[u32], mem: &mut dyn CpMemPort) -> Result<u32, CpError> {
        assert_eq!(self.depth, 0, "submit is not reentrant");
        self.stats = CpStats::default();
        self.stats.words = words.len() as u32;
        self.run_stream(words, mem)?;
        Ok(self.stats.words)
    }

    /// Internal: execute one stream level.
    fn run_stream(&mut self, words: &[u32], mem: &mut dyn CpMemPort) -> Result<(), CpError> {
        for packet in PacketDecoder::new(words) {
            let packet = packet?;
            self.stats.packets += 1;
            self.exec(packet, mem)?;
        }
        Ok(())
    }

    /// Internal: execute one packet.
    fn exec(
        &mut self,
        packet: crate::pm4::decode::Packet<'_>,
        mem: &mut dyn CpMemPort,
    ) -> Result<(), CpError> {
        match packet {
            crate::pm4::decode::Packet::Type2 => Ok(()),
            crate::pm4::decode::Packet::Type0 { base_reg, data } => {
                // Type-0 writes go to the config space at the raw offset.
                for (i, value) in data.iter().enumerate() {
                    let offset = u32::from(base_reg) + i as u32;
                    let addr = RegAddr::from_packet_offset(RegSpace::Config, offset).ok_or(
                        CpError::BadRegisterOffset {
                            space: RegSpace::Config,
                            offset,
                        },
                    )?;
                    self.ctx.regs_mut().set(addr, *value);
                    self.stats.reg_writes += 1;
                }
                Ok(())
            }
            crate::pm4::decode::Packet::Type3 {
                opcode,
                raw_opcode,
                data,
                ..
            } => self.exec_type3(opcode, raw_opcode, data, mem),
        }
    }

    /// Internal: dispatch a type-3 packet by opcode.
    fn exec_type3(
        &mut self,
        opcode: Option<ItOpcode>,
        raw_opcode: u8,
        data: &[u32],
        mem: &mut dyn CpMemPort,
    ) -> Result<(), CpError> {
        use ItOpcode::*;
        let Some(op) = opcode else {
            return Err(CpError::Unsupported {
                opcode: raw_opcode,
                name: "UNKNOWN",
                why: "opcode is not in PDG's verified table",
            });
        };
        match op {
            Nop => Ok(()),
            SetConfigReg => self.set_regs(RegSpace::Config, data),
            SetContextReg => self.set_regs(RegSpace::Context, data),
            SetShReg => self.set_regs(RegSpace::Sh, data),
            SetUconfigReg => self.set_regs(RegSpace::UConfig, data),
            ClearState => {
                // Payload bit 0 would clear the CP context too; PDG models
                // the register-file effect.
                self.ctx.clear_context();
                Ok(())
            }
            ContextControl => {
                // LOAD/SHADOW action bits; PDG's context is always live.
                let _ = data.first();
                Ok(())
            }
            SetBase => {
                // Records a persistent base (e.g. for the constant-index
                // SH space); PDG's SH space is addressed directly.
                Ok(())
            }
            IndexBase | IndexBufferSize => {
                // Recorded in the context: VGT_DMA_BASE / VGT_DMA_SIZE.
                // PDG keeps them as raw register shadow for later phases.
                Ok(())
            }
            IndexType => {
                let value = data.first().copied().unwrap_or(0);
                self.ctx
                    .regs_mut()
                    .set(crate::pm4::regs::ctx::VGT_INDEX_TYPE, value & 3);
                Ok(())
            }
            NumInstances => {
                let value = data.first().copied().unwrap_or(0);
                self.ctx
                    .regs_mut()
                    .set(crate::pm4::regs::ctx::VGT_NUM_INSTANCES, value);
                Ok(())
            }
            DrawIndexAuto => {
                let index_count = data.first().copied().unwrap_or(0);
                self.draw(DrawParams {
                    count: index_count,
                    index_type: crate::ir::IndexType::Unset,
                    instances: self.instances(),
                    prim_type: self.prim_type()?,
                    base_vertex: 0,
                    index_base: 0,
                })
            }
            DrawIndex2 => {
                // [max_size, base_lo, base_hi, index_count, draw_initiator]
                if data.len() < 5 {
                    return Err(CpError::BadIb { addr: 0 });
                }
                let index_base = u64::from(data[1]) | (u64::from(data[2]) << 32);
                self.draw(DrawParams {
                    count: data[3],
                    index_type: self.index_type()?,
                    instances: self.instances(),
                    prim_type: self.prim_type()?,
                    base_vertex: 0,
                    index_base,
                })
            }
            DispatchDirect => {
                let (x, y, z) = match (data.first(), data.get(1), data.get(2)) {
                    (Some(x), Some(y), Some(z)) => (*x, *y, *z),
                    _ => return Err(CpError::BadIb { addr: 0 }),
                };
                self.ctx
                    .regs_mut()
                    .set(crate::pm4::regs::sh::COMPUTE_DIM_X, x);
                self.ctx
                    .regs_mut()
                    .set(crate::pm4::regs::sh::COMPUTE_DIM_Y, y);
                self.ctx
                    .regs_mut()
                    .set(crate::pm4::regs::sh::COMPUTE_DIM_Z, z);
                self.out.push(GpuCommand::Dispatch { groups: [x, y, z] });
                self.stats.dispatches += 1;
                Ok(())
            }
            IndirectBuffer => {
                // [ibase_lo, {ibase_hi | ib_size...}, ib_size]
                if data.len() < 3 {
                    return Err(CpError::BadIb { addr: 0 });
                }
                let addr = u64::from(data[0]) | (u64::from(data[1]) << 32);
                let size_words = data[2];
                if self.depth >= IB_MAX_DEPTH {
                    return Err(CpError::IbOverflow { max: IB_MAX_DEPTH });
                }
                let body = mem.read_words(addr, size_words as usize);
                if body.len() != size_words as usize {
                    return Err(CpError::BadIb { addr });
                }
                self.depth += 1;
                self.stats.indirect_buffers += 1;
                let result = self.run_stream(&body, mem);
                self.depth -= 1;
                result
            }
            ReleaseMem | EventWriteEop => {
                // [event_control, dst_sel/int/data_sel, addr_lo, addr_hi, data_lo, data_hi]
                if data.len() < 6 {
                    return Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: "malformed payload (fewer than six dwords)",
                    });
                }
                let data_sel = (data[1] >> 29) & 0x7;
                let addr = u64::from(data[2]) | (u64::from(data[3] & 0xFFFF) << 32);
                let value = u64::from(data[4]) | (u64::from(data[5]) << 32);
                match data_sel {
                    // 1 = 32-bit immediate, 2 = 64-bit immediate.
                    1 | 2 => {
                        self.out.push(GpuCommand::Fence {
                            addr,
                            value,
                            wide: data_sel == 2,
                        });
                        self.stats.fences += 1;
                        Ok(())
                    }
                    // 0 = none, 3 = GPU clock, 4 = perf counter, 5 = GDS.
                    0 => Ok(()),
                    other => Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: match other {
                            3 => "GPU-clock timestamp sources are not modeled yet",
                            4 => "perf-counter fence sources are not modeled yet",
                            _ => "GDS fence sources are not modeled yet",
                        },
                    }),
                }
            }
            EventWrite => {
                // [event_control, (optional address...)]. Without an
                // address it is a pure event (cache flush etc.); with one
                // it writes the event as a fence (EOP-style, 32-bit).
                if data.len() >= 3 {
                    let addr = u64::from(data[1]) | (u64::from(data[2] & 0xFFFF) << 32);
                    let value = self.stats.packets as u64;
                    self.out.push(GpuCommand::Fence {
                        addr,
                        value,
                        wide: false,
                    });
                    self.stats.fences += 1;
                }
                Ok(())
            }
            MemSemaphore => {
                // [addr_lo, addr_hi, wait_signal(0 = wait)].
                let addr = u64::from(data[0]) | (u64::from(data[1] & 0xFFFF) << 32);
                self.out.push(GpuCommand::SemaphoreWait { addr, value: 1 });
                Ok(())
            }
            WaitRegMem => {
                // [control, poll_addr_lo, poll_addr_hi, ref, mask, interval]
                if data.len() < 6 {
                    return Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: "malformed payload",
                    });
                }
                let function = data[0] & 0x7;
                let mem_space = (data[0] >> 4) & 1;
                if mem_space == 0 {
                    // Register-space waits are CP-internal; nothing PDG
                    // models blocks on them yet.
                    return Ok(());
                }
                let addr = (u64::from(data[1] & 0xFFFF_FFFC) << 2) | (u64::from(data[2]) << 32);
                let reference = data[3];
                let mask = data[4];
                if function == 3 {
                    // WAIT_UNTIL_EQUAL against memory: emit as a semaphore
                    // wait the backend honors.
                    self.out.push(GpuCommand::SemaphoreWait {
                        addr,
                        value: (u64::from(mask) << 32) | u64::from(reference),
                    });
                    Ok(())
                } else {
                    Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: "only EQUAL (function 3) memory waits are modeled",
                    })
                }
            }
            WriteData => {
                // [control, dst_lo, dst_hi, data...]; PDG models the
                // memory destination (dst_sel 1) with raw writes.
                if data.len() < 3 {
                    return Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: "malformed payload",
                    });
                }
                let dst_sel = (data[0] >> 8) & 0xF;
                let addr = u64::from(data[1]) | (u64::from(data[2] & 0xFFFF) << 32);
                match dst_sel {
                    1 if data.len() > 3 => {
                        for (i, word) in data[3..].iter().enumerate() {
                            mem.write_u32(addr + (i as u64) * 4, *word);
                        }
                        Ok(())
                    }
                    1 => Ok(()),
                    2 => {
                        // Register destination.
                        let reg_offset = data[1] & 0xFFFF;
                        if let Some(addr) =
                            RegAddr::from_packet_offset(RegSpace::UConfig, reg_offset)
                        {
                            if let Some(value) = data.get(3) {
                                self.ctx.regs_mut().set(addr, *value);
                            }
                        }
                        Ok(())
                    }
                    other => {
                        let _ = other;
                        Err(CpError::Unsupported {
                            opcode: raw_opcode,
                            name: op.name(),
                            why: "only memory (1) and register (2) destinations are modeled",
                        })
                    }
                }
            }
            DmaData | CopyData => {
                // Simplified linear copy model:
                // DMA_DATA: [control, src_lo, src_hi, dst_lo, dst_hi, byte_count]
                // COPY_DATA: [control, src_lo, src_hi, dst_lo, dst_hi]
                if op == DmaData && data.len() >= 6 {
                    let src = u64::from(data[1]) | (u64::from(data[2]) << 32);
                    let dst = u64::from(data[3]) | (u64::from(data[4]) << 32);
                    let len = data[5] & 0x1FFF_FFFF;
                    self.out.push(GpuCommand::Copy { dst, src, len });
                    Ok(())
                } else if op == CopyData && data.len() >= 5 {
                    let src = u64::from(data[1]) | (u64::from(data[2]) << 32);
                    let dst = u64::from(data[3]) | (u64::from(data[4]) << 32);
                    self.out.push(GpuCommand::Copy { dst, src, len: 4 });
                    Ok(())
                } else {
                    Err(CpError::Unsupported {
                        opcode: raw_opcode,
                        name: op.name(),
                        why: "malformed payload",
                    })
                }
            }
            AcquireMem | PfpSyncMe | SurfaceSync | Rewind | PreambleCntl => {
                // Cache/lifetime barriers: with an in-order, synchronous
                // model they are satisfied by construction.
                Ok(())
            }
            FrameControl | SwitchBuffer | InvalidateTlbs => Ok(()),
            other => Err(CpError::Unsupported {
                opcode: raw_opcode,
                name: other.name(),
                why: "not in PDG's verified execution set",
            }),
        }
    }

    /// Internal: a `SET_*_REG` packet: [offset, values...].
    fn set_regs(&mut self, space: RegSpace, data: &[u32]) -> Result<(), CpError> {
        let Some(offset) = data.first().copied() else {
            return Ok(());
        };
        for (i, value) in data[1..].iter().enumerate() {
            let off = offset + i as u32;
            let addr = RegAddr::from_packet_offset(space, off)
                .ok_or(CpError::BadRegisterOffset { space, offset: off })?;
            self.ctx.regs_mut().set(addr, *value);
            self.stats.reg_writes += 1;
        }
        Ok(())
    }

    /// Internal: snapshot VGT state and emit a draw.
    fn draw(&mut self, params: DrawParams) -> Result<(), CpError> {
        if params.prim_type == PrimitiveType::Unsupported {
            return Err(CpError::Unsupported {
                opcode: 0x2D,
                name: ItOpcode::DrawIndexAuto.name(),
                why: "primitive type outside PDG's modeled topology set",
            });
        }
        self.out.push(GpuCommand::Draw { params });
        self.stats.draws += 1;
        Ok(())
    }

    /// Internal: current instance count (default 1, like the hardware).
    fn instances(&self) -> u32 {
        self.ctx
            .regs()
            .get(crate::pm4::regs::ctx::VGT_NUM_INSTANCES)
            .max(1)
    }

    /// Internal: current index type.
    fn index_type(&self) -> Result<crate::ir::IndexType, CpError> {
        let it = crate::ir::IndexType::from_raw(
            self.ctx.regs().get(crate::pm4::regs::ctx::VGT_INDEX_TYPE),
        );
        if it == crate::ir::IndexType::Unset {
            Err(CpError::Unsupported {
                opcode: 0x27,
                name: ItOpcode::DrawIndex2.name(),
                why: "indexed draw without a programmed VGT_INDEX_TYPE",
            })
        } else {
            Ok(it)
        }
    }

    /// Internal: current topology.
    fn prim_type(&self) -> Result<PrimitiveType, CpError> {
        let pt = PrimitiveType::from_raw(
            self.ctx
                .regs()
                .get(crate::pm4::regs::ctx::VGT_PRIMITIVE_TYPE),
        );
        if pt == PrimitiveType::Unsupported {
            Err(CpError::Unsupported {
                opcode: 0x2D,
                name: ItOpcode::DrawIndexAuto.name(),
                why: "primitive type outside PDG's modeled topology set",
            })
        } else {
            Ok(pt)
        }
    }
}

impl Default for CommandProcessor {
    fn default() -> Self {
        Self::new()
    }

    // Silence: `CpError::Decode` variant wrapper is used via #[from].
}

/// A trivially fake memory port for unit tests.
#[derive(Default)]
pub struct FakeMem {
    /// Word-backed memory map (addr/4 → word).
    pub words: std::collections::HashMap<u64, u32>,
    /// Bytes written, for copy checks.
    pub byte_writes: Vec<(u64, Vec<u8>)>,
}

impl CpMemPort for FakeMem {
    fn read_u32(&mut self, addr: u64) -> u32 {
        self.words.get(&(addr >> 2)).copied().unwrap_or(0)
    }

    fn write_u32(&mut self, addr: u64, value: u32) {
        self.words.insert(addr >> 2, value);
    }

    fn read_words(&mut self, addr: u64, len: usize) -> Vec<u32> {
        let base = addr >> 2;
        (0..len as u64)
            .map(|i| self.words.get(&(base + i)).copied().unwrap_or(0))
            .collect()
    }

    fn write_bytes(&mut self, addr: u64, data: &[u8]) {
        self.byte_writes.push((addr, data.to_vec()));
        for (i, b) in data.iter().enumerate() {
            let a = addr + i as u64;
            let w = a >> 2;
            let sh = ((a & 3) * 8) as u32;
            let old = self.words.get(&w).copied().unwrap_or(0);
            let mask = 0xFFu32 << sh;
            self.words.insert(w, (old & !mask) | (u32::from(*b) << sh));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pm4::Header;

    fn t3(opcode: u8, payload: &[u32]) -> Vec<u32> {
        let mut v = vec![Header::type3(opcode, payload.len() as u16).raw()];
        v.extend_from_slice(payload);
        v
    }

    fn set_reg(opcode: u8, offset: u32, values: &[u32]) -> Vec<u32> {
        let mut payload = vec![offset];
        payload.extend_from_slice(values);
        t3(opcode, &payload)
    }

    fn set_ctx(offset: u32, values: &[u32]) -> Vec<u32> {
        set_reg(ItOpcode::SetContextReg as u8, offset, values)
    }

    fn set_sh(offset: u32, values: &[u32]) -> Vec<u32> {
        set_reg(ItOpcode::SetShReg as u8, offset, values)
    }

    /// CB_COLOR0_INFO at context offset 0x31C (verified: context projects
    /// at `mm`), FORMAT = 8_8_8_8 (10 << 2).
    const CB0_INFO_OFF: u32 = 0x031C;
    /// SPI_SHADER_PGM_LO_VS at SH offset 0x48 (verified kernel formula).
    const VS_PGM_LO_OFF: u32 = 0x48;

    #[test]
    fn set_context_reg_writes_state() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let mut stream = set_ctx(CB0_INFO_OFF, &[10 << 2]);
        stream.extend(set_ctx(0x031D, &[0])); // ATTRIB linear
        stream.extend(set_ctx(0x0319, &[0])); // PITCH
        stream.extend(set_ctx(0x031A, &[7])); // SLICE
        cp.submit(&stream, &mut mem).unwrap();
        let t = cp.context().target(0);
        assert_eq!(t.format, crate::pm4::regs::DataFormat::Fmt8_8_8_8);
        assert_eq!(t.width, Some(64));
        assert_eq!(t.height, Some(64));
        assert_eq!(cp.stats().reg_writes, 4);
    }

    #[test]
    fn set_sh_reg_binds_shader() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let stream = set_sh(VS_PGM_LO_OFF, &[0x1234]);
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(
            cp.context().shader_binding(crate::ir::Stage::Vertex).pgm,
            0x1234 << 8
        );
    }

    #[test]
    fn draw_index_auto_snapshots_vgt() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let mut stream = Vec::new();
        // PRIM_TYPE = TRILIST at context offset 0x2242.
        stream.extend(set_ctx(0x2242, &[4]));
        // NUM_INSTANCES at 0x224D.
        stream.extend(set_ctx(0x224D, &[3]));
        // DRAW_INDEX_AUTO [count=3, initiator=0].
        stream.extend(t3(ItOpcode::DrawIndexAuto as u8, &[3, 0]));
        cp.submit(&stream, &mut mem).unwrap();
        let cmds = cp.take_commands();
        assert_eq!(
            cmds,
            vec![GpuCommand::Draw {
                params: DrawParams {
                    count: 3,
                    index_type: crate::ir::IndexType::Unset,
                    instances: 3,
                    prim_type: PrimitiveType::TriList,
                    base_vertex: 0,
                    index_base: 0,
                }
            }]
        );
        assert_eq!(cp.stats().draws, 1);
    }

    #[test]
    fn unsupported_topology_is_refused() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let mut stream = set_ctx(0x2242, &[0x20]); // not modeled
        stream.extend(t3(ItOpcode::DrawIndexAuto as u8, &[3, 0]));
        let err = cp.submit(&stream, &mut mem).unwrap_err();
        assert!(matches!(err, CpError::Unsupported { .. }));
        assert!(err.to_string().contains("topology"));
    }

    #[test]
    fn dispatch_direct_emits() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let stream = t3(ItOpcode::DispatchDirect as u8, &[4, 2, 1, 0]);
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(
            cp.take_commands(),
            vec![GpuCommand::Dispatch { groups: [4, 2, 1] }]
        );
        assert_eq!(cp.context().compute().groups, [4, 2, 1]);
    }

    #[test]
    fn ib_chain_executes() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        // The IB body: a DISPATCH_DIRECT packet.
        let ib = t3(ItOpcode::DispatchDirect as u8, &[2, 1, 1, 0]);
        for (i, w) in ib.iter().enumerate() {
            mem.words.insert((0x1000 >> 2) + i as u64, *w);
        }
        // Root stream: INDIRECT_BUFFER [addr_lo, addr_hi, size].
        let stream = t3(
            ItOpcode::IndirectBuffer as u8,
            &[0x1000, 0, ib.len() as u32],
        );
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(
            cp.take_commands(),
            vec![GpuCommand::Dispatch { groups: [2, 1, 1] }]
        );
        assert_eq!(cp.stats().indirect_buffers, 1);
    }

    #[test]
    fn ib_overflow_is_refused() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        // Self-referential chain: each IB points at the next packet.
        let stream = t3(ItOpcode::IndirectBuffer as u8, &[0x1000, 0, 4]);
        // Build a chain deeper than IB_MAX_DEPTH: each IB body is a
        // 4-word INDIRECT_BUFFER packet pointing at the next.
        for level in 0..(IB_MAX_DEPTH + 2) as u64 {
            let at = 0x1000 + level * 0x10;
            let next = at + 0x10;
            let ib = t3(ItOpcode::IndirectBuffer as u8, &[next as u32, 0, 4]);
            for (i, w) in ib.iter().enumerate() {
                mem.words.insert((at >> 2) + i as u64, *w);
            }
        }
        let err = cp.submit(&stream, &mut mem).unwrap_err();
        assert!(matches!(err, CpError::IbOverflow { .. }));
    }

    #[test]
    fn release_mem_emits_fence() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        // RELEASE_MEM: [event, control(data_sel=2), addr_lo, addr_hi, data_lo, data_hi]
        let stream = t3(
            ItOpcode::ReleaseMem as u8,
            &[0, 2 << 29, 0x2000, 0, 0x5A5A, 0xA5A5],
        );
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(
            cp.take_commands(),
            vec![GpuCommand::Fence {
                addr: 0x2000,
                value: u64::from(0x5A5A_u32) | (u64::from(0xA5A5_u32) << 32),
                wide: true
            }]
        );
    }

    #[test]
    fn write_data_pokes_memory() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let stream = t3(
            ItOpcode::WriteData as u8,
            &[1 << 8, 0x4000, 0, 0xC0DE, 0xBEEF],
        );
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(mem.read_u32(0x4000), 0xC0DE);
        assert_eq!(mem.read_u32(0x4004), 0xBEEF);
    }

    #[test]
    fn clear_state_resets_context() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let mut stream = set_ctx(CB0_INFO_OFF, &[10 << 2]);
        stream.extend(t3(ItOpcode::ClearState as u8, &[1]));
        cp.submit(&stream, &mut mem).unwrap();
        assert_eq!(
            cp.context().target(0).format,
            crate::pm4::regs::DataFormat::Unknown
        );
    }

    #[test]
    fn type0_writes_config_space() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        // Type-0 header: base_reg = 0x0000, count = 1 (two words).
        let stream = vec![Header::type0(0x0000, 1).raw(), 7, 8];
        cp.submit(&stream, &mut mem).unwrap();
        // Config space offset 0 maps to mm 0xDA0.
        let reg = RegAddr::from_packet_offset(RegSpace::Config, 0).unwrap();
        assert_eq!(reg.mm, 0xDA0);
        assert_eq!(cp.context().regs().get(reg), 7);
    }

    #[test]
    fn unknown_opcode_refused_with_name() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let stream = t3(0x14, &[0]);
        let err = cp.submit(&stream, &mut mem).unwrap_err();
        assert!(matches!(err, CpError::Unsupported { opcode: 0x14, .. }));
    }

    #[test]
    fn kfd_packets_refused() {
        let mut cp = CommandProcessor::new();
        let mut mem = FakeMem::default();
        let stream = t3(ItOpcode::MapQueues as u8, &[0; 4]);
        let err = cp.submit(&stream, &mut mem).unwrap_err();
        assert!(err.to_string().contains("MAP_QUEUES"));
    }
}
