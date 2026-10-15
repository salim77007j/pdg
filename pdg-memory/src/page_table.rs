//! x86-64 page-table construction.
//!
//! The bootstrap loader builds genuine four-level x86-64 page tables inside
//! guest RAM: PML4 → PDPT → PD → PT, with 4 KiB and 2 MiB leaves. The MMU
//! ([`crate::mmu`]) then walks these same tables, so permissions (read/write,
//! no-execute) behave architecturally rather than being emulated by policy
//! code elsewhere.

use bitflags::bitflags;

use crate::{
    bus::{AccessWidth, Bus},
    error::MemoryError,
    physical::PAGE_SIZE,
};

bitflags! {
    /// x86-64 page-table entry flags (bits as defined by the architecture).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PageFlags: u64 {
        /// Bit 0 — entry is valid.
        const PRESENT = 1 << 0;
        /// Bit 1 — writable (cleared means read-only).
        const RW = 1 << 1;
        /// Bit 2 — user-mode accessible.
        const USER = 1 << 2;
        /// Bit 5 — accessed.
        const ACCESSED = 1 << 5;
        /// Bit 6 — dirty.
        const DIRTY = 1 << 6;
        /// Bit 7 — page size (large-page leaf in PDPT/PD entries).
        const PS = 1 << 7;
        /// Bit 63 — no-execute (with EFER.NXE=1, which PDG always enables).
        const NX = 1 << 63;
    }
}

/// Mask extracting the 4 KiB frame address from a PTE (bits 12..51).
pub const FRAME_MASK: u64 = 0x000F_FFFF_FFFF_F000;
/// Mask extracting the 2 MiB large-page base from a PDE (bits 21..51).
pub const FRAME_MASK_2M: u64 = 0x000F_FFFF_FFE0_0000;
/// Mask extracting the 1 GiB large-page base from a PDPTE (bits 30..51).
pub const FRAME_MASK_1G: u64 = 0x000F_FFFF_C000_0000;

/// PML4 (level-4) index of a virtual address.
#[inline]
pub fn pml4_index(va: u64) -> u64 {
    (va >> 39) & 0x1FF
}
/// PDPT (level-3) index of a virtual address.
#[inline]
pub fn pdpt_index(va: u64) -> u64 {
    (va >> 30) & 0x1FF
}
/// PD (level-2) index of a virtual address.
#[inline]
pub fn pd_index(va: u64) -> u64 {
    (va >> 21) & 0x1FF
}
/// PT (level-1) index of a virtual address.
#[inline]
pub fn pt_index(va: u64) -> u64 {
    (va >> 12) & 0x1FF
}

/// True when the address is canonical (x86-64 48-bit canonical halves).
#[inline]
pub fn is_canonical(va: u64) -> bool {
    let top = va >> 47;
    top == 0 || top == 0x1_FFFF
}

/// Builds x86-64 page tables inside guest RAM.
///
/// `pml4_base` is the guest-physical address of the PML4 (the value that
/// will be loaded into CR3). Intermediate table frames are bump-allocated
/// from `frame_alloc_base` upward; anonymous guest RAM is zero-filled, so
/// absent entries read as "not present" as required.
pub struct PageTableBuilder<'a> {
    bus: &'a mut Bus,
    pml4_base: u64,
    next_frame: u64,
}

impl<'a> PageTableBuilder<'a> {
    /// Create a builder rooted at `pml4_base`, allocating intermediate table
    /// frames from `frame_alloc_base` upward.
    pub fn new(bus: &'a mut Bus, pml4_base: u64, frame_alloc_base: u64) -> Self {
        Self {
            bus,
            pml4_base,
            next_frame: frame_alloc_base,
        }
    }

    fn alloc_frame(&mut self) -> u64 {
        let frame = self.next_frame;
        self.next_frame += PAGE_SIZE;
        frame
    }

    /// Ensure `table[index]` points at a present table; returns its frame.
    fn ensure_table(&mut self, table_base: u64, index: u64) -> Result<u64, MemoryError> {
        let slot = table_base + index * 8;
        let entry = self.bus.read(AccessWidth::W64, slot)?;
        if entry & PageFlags::PRESENT.bits() != 0 {
            Ok(entry & FRAME_MASK)
        } else {
            let frame = self.alloc_frame();
            let flags = PageFlags::PRESENT | PageFlags::RW | PageFlags::USER;
            self.bus
                .write(AccessWidth::W64, slot, frame | flags.bits())?;
            Ok(frame)
        }
    }

    /// Map one 4 KiB page: `va` → `pa` with `flags` (PRESENT is added
    /// automatically).
    pub fn map_4k(&mut self, va: u64, pa: u64, flags: PageFlags) -> Result<(), MemoryError> {
        if va % PAGE_SIZE != 0 || pa % PAGE_SIZE != 0 {
            return Err(MemoryError::PageTable(format!(
                "4 KiB mapping requires page-aligned va/pa (va={va:#x}, pa={pa:#x})"
            )));
        }
        let pdpt = self.ensure_table(self.pml4_base, pml4_index(va))?;
        let pd = self.ensure_table(pdpt, pdpt_index(va))?;

        // A 2 MiB leaf must not already occupy this slot.
        let pd_slot = pd + pd_index(va) * 8;
        let pd_entry = self.bus.read(AccessWidth::W64, pd_slot)?;
        if pd_entry & PageFlags::PS.bits() != 0 {
            return Err(MemoryError::PageTable(format!(
                "2 MiB leaf already present at va={va:#x}; refusing to install 4 KiB page"
            )));
        }

        let pt = self.ensure_table(pd, pd_index(va))?;
        let pte = pa | (flags | PageFlags::PRESENT).bits();
        self.bus.write(AccessWidth::W64, pt + pt_index(va) * 8, pte)
    }

    /// Map one 2 MiB large page: `va` → `pa` with `flags` (PRESENT and PS are
    /// added automatically).
    pub fn map_2m(&mut self, va: u64, pa: u64, flags: PageFlags) -> Result<(), MemoryError> {
        if va % (2 * 1024 * 1024) != 0 || pa % (2 * 1024 * 1024) != 0 {
            return Err(MemoryError::PageTable(format!(
                "2 MiB mapping requires 2 MiB-aligned va/pa (va={va:#x}, pa={pa:#x})"
            )));
        }
        let pdpt = self.ensure_table(self.pml4_base, pml4_index(va))?;
        let pd = self.ensure_table(pdpt, pdpt_index(va))?;
        let entry = (pa & FRAME_MASK_2M) | (flags | PageFlags::PRESENT | PageFlags::PS).bits();
        self.bus
            .write(AccessWidth::W64, pd + pd_index(va) * 8, entry)
    }
}
