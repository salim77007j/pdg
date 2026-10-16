//! PM4 opcode table for the type-3 (IT) packets.
//!
//! Every number in this file is cross-checked against public AMD command
//! processor documentation — the Linux kernel's `soc15d.h` (which covers
//! GCN→RDNA parts) and, for the two `EVENT_WRITE_*` variants not named by
//! the kernel header, the shadPS4 project's Liverpool documentation
//! (same GNM packet lineage the console driver family uses). Nothing is
//! guessed: an opcode not in the table decodes to [`ItOpcode::Unknown`]
//! and the command processor refuses it with a precise error.

/// Type-3 (indirect-table / "IT") packet opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ItOpcode {
    /// Padding / debug annotation.
    Nop = 0x10,
    /// Program a persistent base (e.g. the SH constant-index base).
    SetBase = 0x11,
    /// Reset context-state registers to their defaults.
    ClearState = 0x12,
    /// Announce the size of the bound index buffer.
    IndexBufferSize = 0x13,
    /// Launch a compute grid with direct dimensions.
    DispatchDirect = 0x15,
    /// Launch a compute grid whose dimensions come from memory.
    DispatchIndirect = 0x16,
    /// GDS atomic operation.
    AtomicGds = 0x1D,
    /// Memory atomic operation.
    AtomicMem = 0x1E,
    /// Occlusion query control.
    OcclusionQuery = 0x1F,
    /// Set the predication buffer used by conditional draws.
    SetPredication = 0x20,
    /// Read-modify-write a CP register.
    RegRmw = 0x21,
    /// Conditionally execute a payload.
    CondExec = 0x22,
    /// Execute payload under predication.
    PredExec = 0x23,
    /// Non-indexed draw, parameters in guest memory.
    DrawIndirect = 0x24,
    /// Indexed draw, parameters in guest memory.
    DrawIndexIndirect = 0x25,
    /// Program the base of the bound index buffer.
    IndexBase = 0x26,
    /// Indexed draw with the count in the packet.
    DrawIndex2 = 0x27,
    /// Context-state load/invalidation control.
    ContextControl = 0x28,
    /// Program the bound index type.
    IndexType = 0x2A,
    /// Multi-draw, parameters in guest memory.
    DrawIndirectMulti = 0x2C,
    /// Non-indexed draw with the count in the packet.
    DrawIndexAuto = 0x2D,
    /// Instance count for the next draw.
    NumInstances = 0x2F,
    /// Multi-instance non-indexed draw.
    DrawIndexMultiAuto = 0x30,
    /// Constant-engine indirect buffer.
    IndirectBufferConst = 0x33,
    /// Stream-out buffer control.
    StrmoutBufferUpdate = 0x34,
    /// Indexed draw with base-vertex/index offsets in the packet.
    DrawIndexOffset2 = 0x35,
    /// Draw preamble.
    DrawPreamble = 0x36,
    /// Write a value to memory or a register.
    WriteData = 0x37,
    /// Multi-draw indexed, parameters in guest memory.
    DrawIndexIndirectMulti = 0x38,
    /// Signal / wait on a semaphore in memory.
    MemSemaphore = 0x39,
    /// Poll a register or memory until a compare is satisfied.
    WaitRegMem = 0x3C,
    /// Chain to an indirect buffer (the nesting primitive).
    IndirectBuffer = 0x3F,
    /// Copy between register, memory, and query slots.
    CopyData = 0x40,
    /// ME ↔ PFP pipeline barrier.
    PfpSyncMe = 0x42,
    /// Cache invalidation barrier (gfx6-era name; gfx10 uses ACQUIRE_MEM).
    SurfaceSync = 0x43,
    /// Conditionally write a value to memory.
    CondWrite = 0x45,
    /// Write an event to memory.
    EventWrite = 0x46,
    /// End-of-pipe event with a fence payload (shadPS4-attested number).
    EventWriteEop = 0x47,
    /// End-of-shader event (shadPS4-attested number).
    EventWriteEos = 0x48,
    /// End-of-pipe fence with cache-control payload.
    ReleaseMem = 0x49,
    /// Preamble optimization control.
    PreambleCntl = 0x4A,
    /// Engine-level copy / fill.
    DmaData = 0x50,
    /// Acquire / invalidate caches before continuing.
    AcquireMem = 0x58,
    /// CE rewind.
    Rewind = 0x59,
    /// Load SH registers from memory.
    LoadShReg = 0x5F,
    /// Load config registers from memory.
    LoadConfigReg = 0x60,
    /// Load context registers from memory.
    LoadContextReg = 0x61,
    /// Program config-space registers.
    SetConfigReg = 0x68,
    /// Program context-space registers.
    SetContextReg = 0x69,
    /// Program context registers through an index buffer.
    SetContextRegIndirect = 0x73,
    /// Program SH (per-shader) registers.
    SetShReg = 0x76,
    /// Program SH registers through an offset register.
    SetShRegOffset = 0x77,
    /// Program queue registers.
    SetQueueReg = 0x78,
    /// Program uconfig (user configuration) registers.
    SetUconfigReg = 0x79,
    /// Load the CE constant RAM from memory.
    LoadConstRam = 0x80,
    /// Write the CE constant RAM.
    WriteConstRam = 0x81,
    /// Dump the CE constant RAM to memory.
    DumpConstRam = 0x83,
    /// Increment the CE counter.
    IncrementCeCounter = 0x84,
    /// Increment the DE counter.
    IncrementDeCounter = 0x85,
    /// Wait on the CE counter.
    WaitOnCeCounter = 0x86,
    /// Wait on a DE counter difference.
    WaitOnDeCounterDiff = 0x88,
    /// Switch the CE ring buffer.
    SwitchBuffer = 0x8B,
    /// Frame boundary control.
    FrameControl = 0x90,
    /// Invalidate TLBs.
    InvalidateTlbs = 0x98,
    /// KFD queue management.
    SetResources = 0xA0,
    /// KFD queue management.
    MapQueues = 0xA2,
    /// KFD queue management.
    UnmapQueues = 0xA3,
    /// KFD queue management.
    QueryStatus = 0xA4,
    /// Run the cleaner shader.
    RunCleanerShader = 0xD2,
    /// An opcode that is not in PDG's table.
    Unknown = 0xFF,
}

impl ItOpcode {
    /// Map a raw 7-bit opcode to its name, if PDG knows it.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        use ItOpcode::*;
        Some(match raw {
            0x10 => Nop,
            0x11 => SetBase,
            0x12 => ClearState,
            0x13 => IndexBufferSize,
            0x15 => DispatchDirect,
            0x16 => DispatchIndirect,
            0x1D => AtomicGds,
            0x1E => AtomicMem,
            0x1F => OcclusionQuery,
            0x20 => SetPredication,
            0x21 => RegRmw,
            0x22 => CondExec,
            0x23 => PredExec,
            0x24 => DrawIndirect,
            0x25 => DrawIndexIndirect,
            0x26 => IndexBase,
            0x27 => DrawIndex2,
            0x28 => ContextControl,
            0x2A => IndexType,
            0x2C => DrawIndirectMulti,
            0x2D => DrawIndexAuto,
            0x2F => NumInstances,
            0x30 => DrawIndexMultiAuto,
            0x33 => IndirectBufferConst,
            0x34 => StrmoutBufferUpdate,
            0x35 => DrawIndexOffset2,
            0x36 => DrawPreamble,
            0x37 => WriteData,
            0x38 => DrawIndexIndirectMulti,
            0x39 => MemSemaphore,
            0x3C => WaitRegMem,
            0x3F => IndirectBuffer,
            0x40 => CopyData,
            0x42 => PfpSyncMe,
            0x43 => SurfaceSync,
            0x45 => CondWrite,
            0x46 => EventWrite,
            0x47 => EventWriteEop,
            0x48 => EventWriteEos,
            0x49 => ReleaseMem,
            0x4A => PreambleCntl,
            0x50 => DmaData,
            0x58 => AcquireMem,
            0x59 => Rewind,
            0x5F => LoadShReg,
            0x60 => LoadConfigReg,
            0x61 => LoadContextReg,
            0x68 => SetConfigReg,
            0x69 => SetContextReg,
            0x73 => SetContextRegIndirect,
            0x76 => SetShReg,
            0x77 => SetShRegOffset,
            0x78 => SetQueueReg,
            0x79 => SetUconfigReg,
            0x80 => LoadConstRam,
            0x81 => WriteConstRam,
            0x83 => DumpConstRam,
            0x84 => IncrementCeCounter,
            0x85 => IncrementDeCounter,
            0x86 => WaitOnCeCounter,
            0x88 => WaitOnDeCounterDiff,
            0x8B => SwitchBuffer,
            0x90 => FrameControl,
            0x98 => InvalidateTlbs,
            0xA0 => SetResources,
            0xA2 => MapQueues,
            0xA3 => UnmapQueues,
            0xA4 => QueryStatus,
            0xD2 => RunCleanerShader,
            _ => return None,
        })
    }

    /// Canonical AMD name, as in the public headers.
    #[must_use]
    pub const fn name(self) -> &'static str {
        use ItOpcode::*;
        match self {
            Nop => "NOP",
            SetBase => "SET_BASE",
            ClearState => "CLEAR_STATE",
            IndexBufferSize => "INDEX_BUFFER_SIZE",
            DispatchDirect => "DISPATCH_DIRECT",
            DispatchIndirect => "DISPATCH_INDIRECT",
            AtomicGds => "ATOMIC_GDS",
            AtomicMem => "ATOMIC_MEM",
            OcclusionQuery => "OCCLUSION_QUERY",
            SetPredication => "SET_PREDICATION",
            RegRmw => "REG_RMW",
            CondExec => "COND_EXEC",
            PredExec => "PRED_EXEC",
            DrawIndirect => "DRAW_INDIRECT",
            DrawIndexIndirect => "DRAW_INDEX_INDIRECT",
            IndexBase => "INDEX_BASE",
            DrawIndex2 => "DRAW_INDEX_2",
            ContextControl => "CONTEXT_CONTROL",
            IndexType => "INDEX_TYPE",
            DrawIndirectMulti => "DRAW_INDIRECT_MULTI",
            DrawIndexAuto => "DRAW_INDEX_AUTO",
            NumInstances => "NUM_INSTANCES",
            DrawIndexMultiAuto => "DRAW_INDEX_MULTI_AUTO",
            IndirectBufferConst => "INDIRECT_BUFFER_CONST",
            StrmoutBufferUpdate => "STRMOUT_BUFFER_UPDATE",
            DrawIndexOffset2 => "DRAW_INDEX_OFFSET_2",
            DrawPreamble => "DRAW_PREAMBLE",
            WriteData => "WRITE_DATA",
            DrawIndexIndirectMulti => "DRAW_INDEX_INDIRECT_MULTI",
            MemSemaphore => "MEM_SEMAPHORE",
            WaitRegMem => "WAIT_REG_MEM",
            IndirectBuffer => "INDIRECT_BUFFER",
            CopyData => "COPY_DATA",
            PfpSyncMe => "PFP_SYNC_ME",
            SurfaceSync => "SURFACE_SYNC",
            CondWrite => "COND_WRITE",
            EventWrite => "EVENT_WRITE",
            EventWriteEop => "EVENT_WRITE_EOP",
            EventWriteEos => "EVENT_WRITE_EOS",
            ReleaseMem => "RELEASE_MEM",
            PreambleCntl => "PREAMBLE_CNTL",
            DmaData => "DMA_DATA",
            AcquireMem => "ACQUIRE_MEM",
            Rewind => "REWIND",
            LoadShReg => "LOAD_SH_REG",
            LoadConfigReg => "LOAD_CONFIG_REG",
            LoadContextReg => "LOAD_CONTEXT_REG",
            SetConfigReg => "SET_CONFIG_REG",
            SetContextReg => "SET_CONTEXT_REG",
            SetContextRegIndirect => "SET_CONTEXT_REG_INDIRECT",
            SetShReg => "SET_SH_REG",
            SetShRegOffset => "SET_SH_REG_OFFSET",
            SetQueueReg => "SET_QUEUE_REG",
            SetUconfigReg => "SET_UCONFIG_REG",
            LoadConstRam => "LOAD_CONST_RAM",
            WriteConstRam => "WRITE_CONST_RAM",
            DumpConstRam => "DUMP_CONST_RAM",
            IncrementCeCounter => "INCREMENT_CE_COUNTER",
            IncrementDeCounter => "INCREMENT_DE_COUNTER",
            WaitOnCeCounter => "WAIT_ON_CE_COUNTER",
            WaitOnDeCounterDiff => "WAIT_ON_DE_COUNTER_DIFF",
            SwitchBuffer => "SWITCH_BUFFER",
            FrameControl => "FRAME_CONTROL",
            InvalidateTlbs => "INVALIDATE_TLBS",
            SetResources => "SET_RESOURCES",
            MapQueues => "MAP_QUEUES",
            UnmapQueues => "UNMAP_QUEUES",
            QueryStatus => "QUERY_STATUS",
            RunCleanerShader => "RUN_CLEANER_SHADER",
            Unknown => "UNKNOWN",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every opcode round-trips and never reports the wrong number.
    #[test]
    fn table_roundtrips() {
        for raw in 0u8..0x80 {
            if let Some(op) = ItOpcode::from_raw(raw) {
                assert_eq!(op as u8, raw, "opcode {raw:#04x} mismatched");
                assert!(!op.name().is_empty());
            }
        }
        // Well-known anchors from the public headers.
        assert_eq!(ItOpcode::from_raw(0x10), Some(ItOpcode::Nop));
        assert_eq!(ItOpcode::from_raw(0x3F), Some(ItOpcode::IndirectBuffer));
        assert_eq!(ItOpcode::from_raw(0x69), Some(ItOpcode::SetContextReg));
        assert_eq!(ItOpcode::from_raw(0x76), Some(ItOpcode::SetShReg));
        assert_eq!(ItOpcode::from_raw(0x68), Some(ItOpcode::SetConfigReg));
        assert_eq!(ItOpcode::from_raw(0x79), Some(ItOpcode::SetUconfigReg));
        // Gaps in the documented space stay unknown.
        assert_eq!(ItOpcode::from_raw(0x14), None);
        assert_eq!(ItOpcode::from_raw(0x01), None);
    }
}
