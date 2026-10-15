//! Interrupt descriptor table helpers.
//!
//! Phase 2's interpreter delivers exceptions by reading genuine 16-byte
//! x86-64 IDT gates out of guest memory (see `pdg-cpu::interpreter`). This
//! module provides the encoding/decoding helpers the Phase-5 kernel layer
//! (and tests) use to build those gates, so the byte layout lives in one
//! audited place.

use pdg_cpu::state::DescriptorTableRegister;

/// One 16-byte x86-64 IDT gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdtGate {
    /// Handler entry point (bits 0..=31; bits 32..=63 are zero in long mode).
    pub offset: u64,
    /// Code-segment selector (0x08 in PDG's flat model).
    pub selector: u16,
    /// Gate type/attributes byte: P | DPL | type.
    pub attributes: u8,
    /// Interrupt stack table index (0 = no stack switch).
    pub ist: u8,
}

/// Present 64-bit interrupt gate (IF cleared on entry).
pub const ATTR_INTERRUPT_GATE: u8 = 0x8E;
/// Present 64-bit trap gate (IF unchanged).
pub const ATTR_TRAP_GATE: u8 = 0x8F;

impl IdtGate {
    /// A present interrupt gate to `offset` using the flat kernel CS.
    pub fn interrupt(offset: u64) -> Self {
        Self {
            offset,
            selector: 0x08,
            attributes: ATTR_INTERRUPT_GATE,
            ist: 0,
        }
    }

    /// A present trap gate to `offset`.
    pub fn trap(offset: u64) -> Self {
        Self {
            offset,
            selector: 0x08,
            attributes: ATTR_TRAP_GATE,
            ist: 0,
        }
    }

    /// Encode into the 16-byte memory representation.
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        let lo = (self.offset & 0xFFFF) as u16;
        let mid = ((self.offset >> 16) & 0xFFFF) as u16;
        b[0..2].copy_from_slice(&lo.to_le_bytes());
        b[2..4].copy_from_slice(&self.selector.to_le_bytes());
        b[4] = self.ist & 0x07;
        b[5] = self.attributes;
        b[6..8].copy_from_slice(&mid.to_le_bytes());
        // bytes 8..16: offset high (always zero in 64-bit gates)
        b
    }

    /// Decode from the 16-byte memory representation.
    pub fn from_bytes(b: &[u8; 16]) -> Self {
        let lo = u16::from_le_bytes([b[0], b[1]]) as u64;
        let mid = u16::from_le_bytes([b[6], b[7]]) as u64;
        Self {
            offset: lo | (mid << 16),
            selector: u16::from_le_bytes([b[2], b[3]]),
            attributes: b[5],
            ist: b[4] & 0x07,
        }
    }

    /// Is the gate present and usable?
    pub fn is_present(&self) -> bool {
        self.attributes & 0x80 != 0
    }
}

/// Encode an IDT gate and store it at `idtr.base + vector*16`.
pub fn install_gate(
    mem: &mut dyn pdg_cpu::MemoryAccess,
    idtr: &DescriptorTableRegister,
    vector: u8,
    gate: &IdtGate,
) -> Result<(), pdg_cpu::AccessError> {
    let addr = idtr.base + u64::from(vector) * 16;
    mem.write_buf(addr, &gate.to_bytes())
}

/// Read the IDT gate for `vector`.
pub fn read_gate(
    mem: &mut dyn pdg_cpu::MemoryAccess,
    idtr: &DescriptorTableRegister,
    vector: u8,
) -> Result<IdtGate, pdg_cpu::AccessError> {
    let addr = idtr.base + u64::from(vector) * 16;
    let mut b = [0u8; 16];
    mem.read_buf(addr, &mut b)?;
    Ok(IdtGate::from_bytes(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdg_cpu::FlatMemory;

    #[test]
    fn gate_roundtrip() {
        let g = IdtGate::interrupt(0x1234_5678);
        let b = g.to_bytes();
        assert_eq!(IdtGate::from_bytes(&b), g);
        assert!(g.is_present());
        // Offset split across the two 16-bit halves.
        assert_eq!(u16::from_le_bytes([b[0], b[1]]), 0x5678);
        assert_eq!(u16::from_le_bytes([b[6], b[7]]), 0x1234);
    }

    #[test]
    fn install_and_read_through_memory() {
        let mut mem = FlatMemory::new(0x10000);
        let idtr = DescriptorTableRegister {
            base: 0x8000,
            limit: 256 * 16 - 1,
        };
        install_gate(&mut mem, &idtr, 14, &IdtGate::interrupt(0xDEAD_BEEF)).unwrap();
        let g = read_gate(&mut mem, &idtr, 14).unwrap();
        assert_eq!(g.offset, 0xDEAD_BEEF);
        assert!(g.is_present());
    }
}
