//! Software MMU: four-level x86-64 page-table walk with a software TLB.
//!
//! Every guest virtual access is translated by walking the guest's own page
//! tables (PML4 → PDPT → PD → PT) in guest RAM — the same structures a real
//! Zen 2 core would walk. Permission semantics follow the architecture with
//! EFER.NXE=1: a write requires RW set at **every** level, and instruction
//! fetch requires NX clear at **every** level. A software TLB caches leaf
//! translations and is fully flushed on CR3 writes (per-entry invalidation
//! via INVLPG is a planned Phase 2 refinement).

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    bus::{AccessWidth, Bus},
    error::MemoryError,
    page_table::{
        FRAME_MASK, FRAME_MASK_1G, FRAME_MASK_2M, PageFlags, is_canonical, pd_index, pdpt_index,
        pml4_index, pt_index,
    },
};

/// Kind of access being translated (affects permission checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    /// Data read.
    Read,
    /// Data write.
    Write,
    /// Instruction fetch.
    Exec,
}

/// x86 page-fault error-code bits (`PFErrorCode` in architecture manuals).
pub mod fault_code {
    /// Fault was caused by a protection violation (entry present).
    pub const PRESENT: u32 = 1 << 0;
    /// Fault was caused by a write.
    pub const WRITE: u32 = 1 << 1;
    /// Fault originated from user mode (unused in Phase 1; always supervisor).
    pub const USER: u32 = 1 << 2;
    /// Fault was caused by an instruction fetch (requires EFER.NXE=1).
    pub const EXEC: u32 = 1 << 4;
}

/// Failure of a virtual→physical translation or the subsequent bus access.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AccessFault {
    /// Architectural page fault (guest must handle via its fault vector).
    #[error("page fault at {addr:#018x} (code {code:#04x})")]
    PageFault {
        /// Faulting virtual address (goes to CR2 on real hardware).
        addr: u64,
        /// Error code bits (see [`fault_code`]).
        code: u32,
    },
    /// Non-canonical virtual address (#GP on real hardware).
    #[error("non-canonical virtual address {addr:#018x}")]
    NonCanonical {
        /// The offending address.
        addr: u64,
    },
    /// The guest-physical side rejected the access.
    #[error(transparent)]
    Bus(#[from] MemoryError),
}

#[derive(Debug, Clone, Copy)]
struct TlbEntry {
    /// Guest-physical base of the page.
    base: u64,
    /// Log2 of the page size in bytes (12 / 21 / 30).
    shift: u32,
    /// Effective permissions of the path: RW allowed, exec allowed.
    writable: bool,
    executable: bool,
}

/// Software MMU rooted at a guest-physical CR3.
pub struct Mmu {
    cr3: u64,
    tlb: HashMap<u64, TlbEntry>,
    tlb_enabled: bool,
}

impl std::fmt::Debug for Mmu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mmu")
            .field("cr3", &format!("{:#x}", self.cr3))
            .field("tlb_entries", &self.tlb.len())
            .field("tlb_enabled", &self.tlb_enabled)
            .finish()
    }
}

impl Mmu {
    /// Create an MMU with page tables rooted at `cr3` (guest-physical).
    pub fn new(cr3: u64) -> Self {
        Self {
            cr3,
            tlb: HashMap::new(),
            tlb_enabled: true,
        }
    }

    /// Current CR3 value.
    pub fn cr3(&self) -> u64 {
        self.cr3
    }

    /// Load a new CR3 (flushes the whole TLB, like MOV CR3 without PCID).
    pub fn set_cr3(&mut self, cr3: u64) {
        self.cr3 = cr3;
        self.flush_tlb();
    }

    /// Flush all cached translations.
    pub fn flush_tlb(&mut self) {
        self.tlb.clear();
    }

    /// Enable/disable the software TLB (used for benchmarking the walk cost).
    pub fn set_tlb_enabled(&mut self, enabled: bool) {
        self.tlb_enabled = enabled;
        if !enabled {
            self.tlb.clear();
        }
    }

    /// True when the walk allows `kind` at `va`, given accumulated
    /// intermediate permissions. `addr` is used for fault reporting.
    #[allow(clippy::too_many_arguments)]
    fn check(
        &self,
        base: u64,
        shift: u32,
        writable: bool,
        executable: bool,
        va: u64,
        kind: AccessKind,
        page_fault: impl FnOnce(u32) -> AccessFault,
    ) -> Result<u64, AccessFault> {
        if kind == AccessKind::Write && !writable {
            return Err(page_fault(fault_code::PRESENT | fault_code::WRITE));
        }
        if kind == AccessKind::Exec && !executable {
            return Err(page_fault(fault_code::PRESENT | fault_code::EXEC));
        }
        Ok(base | (va & ((1u64 << shift) - 1)))
    }

    /// Translate `va` for access `kind`, returning the guest-physical address.
    pub fn translate(&mut self, bus: &Bus, va: u64, kind: AccessKind) -> Result<u64, AccessFault> {
        if !is_canonical(va) {
            return Err(AccessFault::NonCanonical { addr: va });
        }

        // TLB probe (key: virtual 4 KiB page number).
        if self.tlb_enabled {
            let key = va >> 12;
            if let Some(entry) = self.tlb.get(&key) {
                return self.check(
                    entry.base,
                    entry.shift,
                    entry.writable,
                    entry.executable,
                    va,
                    kind,
                    |code| AccessFault::PageFault { addr: va, code },
                );
            }
        }

        let pml4e = bus.read(AccessWidth::W64, self.cr3 + pml4_index(va) * 8)?;
        if pml4e & PageFlags::PRESENT.bits() == 0 {
            return Err(AccessFault::PageFault { addr: va, code: 0 });
        }
        let mut writable = pml4e & PageFlags::RW.bits() != 0;
        let mut executable = pml4e & PageFlags::NX.bits() == 0;

        let pdpte = bus.read(AccessWidth::W64, (pml4e & FRAME_MASK) + pdpt_index(va) * 8)?;
        if pdpte & PageFlags::PRESENT.bits() == 0 {
            return Err(AccessFault::PageFault { addr: va, code: 0 });
        }
        writable &= pdpte & PageFlags::RW.bits() != 0;
        executable &= pdpte & PageFlags::NX.bits() == 0;

        if pdpte & PageFlags::PS.bits() != 0 {
            // 1 GiB leaf (not installed by the Phase-1 builder, but the walk
            // supports it because real kernels use it).
            let base = pdpte & FRAME_MASK_1G;
            return self.check(base, 30, writable, executable, va, kind, |code| {
                AccessFault::PageFault { addr: va, code }
            });
        }

        let pde = bus.read(AccessWidth::W64, (pdpte & FRAME_MASK) + pd_index(va) * 8)?;
        if pde & PageFlags::PRESENT.bits() == 0 {
            return Err(AccessFault::PageFault { addr: va, code: 0 });
        }
        writable &= pde & PageFlags::RW.bits() != 0;
        executable &= pde & PageFlags::NX.bits() == 0;

        let (base, shift) = if pde & PageFlags::PS.bits() != 0 {
            // 2 MiB leaf.
            (pde & FRAME_MASK_2M, 21u32)
        } else {
            let pte = bus.read(AccessWidth::W64, (pde & FRAME_MASK) + pt_index(va) * 8)?;
            if pte & PageFlags::PRESENT.bits() == 0 {
                return Err(AccessFault::PageFault { addr: va, code: 0 });
            }
            writable &= pte & PageFlags::RW.bits() != 0;
            executable &= pte & PageFlags::NX.bits() == 0;
            (pte & FRAME_MASK, 12u32)
        };

        let entry = TlbEntry {
            base,
            shift,
            writable,
            executable,
        };
        if self.tlb_enabled {
            if shift == 12 {
                self.tlb.insert(va >> 12, entry);
            } else {
                // A 2 MiB leaf fills every 4 KiB slot it covers... which is
                // 512 entries; instead we cache the representative slot of
                // the current access. Slots for other 4 KiB pages inside the
                // 2 MiB span will re-walk. A hierarchical TLB lands in Phase 2.
                self.tlb.insert(va >> 12, entry);
            }
        }

        self.check(base, shift, writable, executable, va, kind, |code| {
            AccessFault::PageFault { addr: va, code }
        })
    }

    /// Read `width` bytes at virtual `va`.
    pub fn read(&mut self, bus: &Bus, width: AccessWidth, va: u64) -> Result<u64, AccessFault> {
        let pa = self.translate(bus, va, AccessKind::Read)?;
        bus.read(width, pa).map_err(AccessFault::Bus)
    }

    /// Write `width` bytes at virtual `va`.
    pub fn write(
        &mut self,
        bus: &mut Bus,
        width: AccessWidth,
        va: u64,
        value: u64,
    ) -> Result<(), AccessFault> {
        let pa = self.translate(bus, va, AccessKind::Write)?;
        bus.write(width, pa, value).map_err(AccessFault::Bus)
    }

    /// Copy `out.len()` bytes from virtual `va` (read side of `read_buf`).
    pub fn read_chunk(&mut self, bus: &Bus, va: u64, out: &mut [u8]) -> Result<(), AccessFault> {
        let pa = self.translate(bus, va, AccessKind::Read)?;
        bus.read_buf(pa, out).map_err(AccessFault::Bus)
    }

    /// Copy `src.len()` bytes to virtual `va` (write side of `write_buf`).
    pub fn write_chunk(&mut self, bus: &mut Bus, va: u64, src: &[u8]) -> Result<(), AccessFault> {
        let pa = self.translate(bus, va, AccessKind::Write)?;
        bus.write_buf(pa, src).map_err(AccessFault::Bus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_table::PageTableBuilder;

    /// 4 MiB RAM, PML4 at 0x1000, first two 2 MiB pages identity-mapped,
    /// fresh table frames from 0x100_000.
    fn fixture() -> (Bus, Mmu) {
        let mut bus = Bus::new(0x40_0000).unwrap();
        let mut builder = PageTableBuilder::new(&mut bus, 0x1000, 0x100_000);
        builder
            .map_2m(0, 0, PageFlags::PRESENT | PageFlags::RW | PageFlags::NX)
            .unwrap();
        builder
            .map_2m(
                0x20_0000,
                0x20_0000,
                PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            )
            .unwrap();
        (bus, Mmu::new(0x1000))
    }

    #[test]
    fn identity_map_translates() {
        let (mut bus, mut mmu) = fixture();
        mmu.write(&mut bus, AccessWidth::W32, 0x30_0100, 0xC0FF_EE42)
            .unwrap();
        assert_eq!(
            mmu.read(&bus, AccessWidth::W32, 0x30_0100).unwrap(),
            0xC0FF_EE42
        );
        // Physical check: identity mapping writes the same PA.
        assert_eq!(bus.read(AccessWidth::W32, 0x30_0100).unwrap(), 0xC0FF_EE42);
    }

    #[test]
    fn remapped_page_translates_to_target() {
        let (mut bus, mut mmu) = fixture();
        let mut builder = PageTableBuilder::new(&mut bus, 0x1000, 0x200_000);
        // Map VA page 0x40_0000 → PA page 0x30_0000.
        builder
            .map_4k(
                0x40_0000,
                0x30_0000,
                PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            )
            .unwrap();
        bus.write(AccessWidth::W64, 0x30_0004, 0xFEED).unwrap();
        assert_eq!(mmu.read(&bus, AccessWidth::W64, 0x40_0004).unwrap(), 0xFEED);
    }

    #[test]
    fn read_only_page_faults_on_write() {
        let (mut bus, mut mmu) = fixture();
        let mut builder = PageTableBuilder::new(&mut bus, 0x1000, 0x200_000);
        // VA page 0x60_0000 → PA page 0x30_0000, read-only.
        builder
            .map_4k(0x60_0000, 0x30_0000, PageFlags::PRESENT | PageFlags::NX)
            .unwrap();
        let fault = mmu
            .write(&mut bus, AccessWidth::W8, 0x60_0010, 1)
            .unwrap_err();
        assert_eq!(
            fault,
            AccessFault::PageFault {
                addr: 0x60_0010,
                code: fault_code::PRESENT | fault_code::WRITE
            }
        );
        // Reads still work.
        assert_eq!(mmu.read(&bus, AccessWidth::W8, 0x60_0010).unwrap(), 0);
    }

    #[test]
    fn nx_page_faults_on_exec_translation() {
        let (bus, mut mmu) = fixture();
        // The identity map from `fixture` already carries NX.
        let fault = mmu
            .translate(&bus, 0x30_0000, AccessKind::Exec)
            .unwrap_err();
        assert_eq!(
            fault,
            AccessFault::PageFault {
                addr: 0x30_0000,
                code: fault_code::PRESENT | fault_code::EXEC
            }
        );
        // Data translations succeed.
        assert!(mmu.translate(&bus, 0x30_0000, AccessKind::Read).is_ok());
    }

    #[test]
    fn unmapped_page_faults_not_present() {
        let (bus, mut mmu) = fixture();
        // VA 0x100_0000 has no mapping (outside the single 2 MiB page at 0).
        assert_eq!(
            mmu.translate(&bus, 0x100_0000, AccessKind::Read),
            Err(AccessFault::PageFault {
                addr: 0x100_0000,
                code: 0
            })
        );
    }

    #[test]
    fn non_canonical_address_rejected() {
        let (bus, mut mmu) = fixture();
        assert!(matches!(
            mmu.translate(&bus, 0x0000_8000_0000_0000, AccessKind::Read),
            Err(AccessFault::NonCanonical { .. })
        ));
    }

    #[test]
    fn tlb_cache_and_flush() {
        let (mut bus, mut mmu) = fixture();
        let mut builder = PageTableBuilder::new(&mut bus, 0x1000, 0x200_000);
        builder
            .map_4k(
                0x40_0000,
                0x30_0000,
                PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            )
            .unwrap();
        bus.write(AccessWidth::W64, 0x30_0000, 0xAAAA).unwrap();
        assert_eq!(mmu.read(&bus, AccessWidth::W64, 0x40_0000).unwrap(), 0xAAAA);
        assert_eq!(mmu.tlb.len(), 1);

        // Rewrite the PTE to point elsewhere, then force a walk via flush.
        let pdpt = bus.read(AccessWidth::W64, 0x1000).unwrap() & FRAME_MASK;
        let pd = bus.read(AccessWidth::W64, pdpt).unwrap() & FRAME_MASK;
        let pt = bus.read(AccessWidth::W64, pd + 8 * 2).unwrap() & FRAME_MASK; // pd_index(0x40_0000)==2
        bus.write(
            AccessWidth::W64,
            pt,
            0x38_0000 | PageFlags::PRESENT.bits() | PageFlags::RW.bits(),
        )
        .unwrap();
        bus.write(AccessWidth::W64, 0x38_0000, 0xBBBB).unwrap();

        // Stale TLB still hits the old page...
        assert_eq!(mmu.read(&bus, AccessWidth::W64, 0x40_0000).unwrap(), 0xAAAA);
        // ...until CR3 is reloaded.
        mmu.set_cr3(0x1000);
        assert_eq!(mmu.read(&bus, AccessWidth::W64, 0x40_0000).unwrap(), 0xBBBB);
    }

    #[test]
    fn cr3_with_empty_tables_faults() {
        let mut bus = Bus::new(0x10_0000).unwrap();
        bus.write(AccessWidth::W64, 0x1000, 0).unwrap(); // PML4[0] = not present
        let mut mmu = Mmu::new(0x1000);
        assert_eq!(
            mmu.translate(&bus, 0x1000, AccessKind::Read),
            Err(AccessFault::PageFault {
                addr: 0x1000,
                code: 0
            })
        );
    }
}
