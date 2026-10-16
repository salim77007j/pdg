//! The process address space: guest page tables behind a shared MMU.
//!
//! Phase 5 builds the first *process-shaped* address space in PDG: the ELF
//! loader maps PT_LOAD segments at their virtual addresses with per-page
//! permissions, `mmap` carves anonymous regions, and the syscall layer
//! mutates the tables while the interpreter holds the memory port.
//!
//! The sharing problem is real: [`pdg_cpu::MemoryAccess`] hands the
//! interpreter `&mut` over the port, so a syscall handler cannot reach the
//! MMU through that reference. The solution is ownership + interior
//! mutability: the bus and the MMU live behind `Arc` (the bus already has
//! interior mutability — every `read`/`write` takes `&self`), and
//! [`ProcessMemory`] locks the MMU per access. The lock is uncontended on
//! the single-core session, and it lets `mprotect`/`munmap` flush the
//! software TLB at the exact instruction boundary the syscall runs at —
//! which is what a real kernel's TLB shootdown achieves.

use std::sync::{Arc, Mutex};

use pdg_memory::page_table::{FRAME_MASK, FRAME_MASK_2M};
use pdg_memory::{Bus, ConsoleDevice, Mmu, PAGE_SIZE, PageFlags, PageTableBuilder};

/// Errors from address-space management.
#[derive(Debug, thiserror::Error)]
pub enum VmError {
    /// Guest-physical allocation ran out of frames.
    #[error("out of guest-physical frames (next {next:#x}, limit {limit:#x})")]
    OutOfFrames {
        /// Next frame the allocator would hand out.
        next: u64,
        /// End of the frame pool.
        limit: u64,
    },
    /// The virtual range is already mapped.
    #[error("virtual range {start:#x}..{end:#x} overlaps an existing mapping")]
    Overlaps {
        /// Start of the requested range.
        start: u64,
        /// End of the requested range.
        end: u64,
    },
    /// The virtual range is not mapped (munmap/mprotect on nothing).
    #[error("virtual range {start:#x}..{end:#x} is not mapped")]
    NotMapped {
        /// Start of the range.
        start: u64,
        /// End of the range.
        end: u64,
    },
    /// Unaligned address or length.
    #[error("unaligned address or length (addr {addr:#x}, len {len:#x})")]
    Unaligned {
        /// Base address.
        addr: u64,
        /// Length.
        len: u64,
    },
    /// A page-table operation failed.
    #[error("page table: {0}")]
    PageTable(String),
    /// The MMU or bus reported an error.
    #[error("memory subsystem: {0}")]
    Memory(String),
}

impl From<pdg_memory::MemoryError> for VmError {
    fn from(e: pdg_memory::MemoryError) -> Self {
        VmError::Memory(e.to_string())
    }
}

/// Round up to the containing page.
pub(crate) const fn page_ceil(x: u64) -> u64 {
    (x + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// A mapped virtual region tracked by the manager.
#[derive(Debug, Clone)]
struct Region {
    /// Virtual start (page-aligned).
    start: u64,
    /// Length in bytes (page multiple).
    len: u64,
    /// Whether this region came from `mmap` (the kernel may unmap it on
    /// `munmap`; loader segments may not).
    anonymous: bool,
}

/// The process address space: page tables + a frame allocator + the region
/// bookkeeping `mmap`/`munmap`/`mprotect` operate on.
pub struct VmManager {
    /// Guest-physical bus (interior mutability, shared with [`ProcessMemory`]).
    pub bus: Arc<Bus>,
    /// MMU behind a lock so syscall handlers can flush it mid-syscall.
    mmu: Arc<Mutex<Mmu>>,
    /// Guest-physical base of this address space's PML4.
    pml4: u64,
    /// Bump allocator: next free guest-physical frame.
    next_frame: u64,
    /// End of the frame pool.
    frame_limit: u64,
    /// Virtual regions currently mapped.
    regions: Vec<Region>,
    /// Where `mmap` hands out anonymous pages (grows upward).
    mmap_cursor: u64,
    /// Hard cap for the mmap arena.
    mmap_limit: u64,
}

impl VmManager {
    /// Build an empty address space: PML4 at `pml4`, intermediate table
    /// frames and anonymous pages bumped from `frame_pool_start`, data
    /// frames capped at `frame_limit`, anonymous `mmap`s carved from
    /// `mmap_base` upward.
    ///
    /// The PML4 frame itself is `frame_pool_start` and the allocator
    /// resumes after it.
    pub fn new(
        bus: Arc<Bus>,
        pml4: u64,
        frame_pool_start: u64,
        frame_limit: u64,
        mmap_base: u64,
        mmap_limit: u64,
    ) -> Self {
        Self {
            bus,
            mmu: Arc::new(Mutex::new(Mmu::new(pml4))),
            pml4,
            next_frame: frame_pool_start + PAGE_SIZE,
            frame_limit,
            regions: Vec::new(),
            mmap_cursor: mmap_base,
            mmap_limit,
        }
    }

    /// Shared handle over the MMU (the `ProcessMemory` port uses the same).
    pub fn shared_mmu(&self) -> Arc<Mutex<Mmu>> {
        self.mmu.clone()
    }

    /// The PML4 guest-physical base (CR3 for this address space).
    pub fn pml4(&self) -> u64 {
        self.pml4
    }

    /// Allocate one zeroed 4 KiB guest-physical frame.
    pub fn alloc_frame(&mut self) -> Result<u64, VmError> {
        if self.next_frame + PAGE_SIZE > self.frame_limit {
            return Err(VmError::OutOfFrames {
                next: self.next_frame,
                limit: self.frame_limit,
            });
        }
        let frame = self.next_frame;
        self.next_frame += PAGE_SIZE;
        let zero = [0u8; PAGE_SIZE as usize];
        self.bus
            .write_buf(frame, &zero)
            .map_err(|e| VmError::Memory(e.to_string()))?;
        Ok(frame)
    }

    /// Map `[va, va+len)` page-by-page to freshly allocated frames with
    /// `flags`. `va` and `len` must be page-aligned. `anonymous` marks the
    /// region as `munmap`-able.
    ///
    /// Data frames are reserved *before* the page-table builder runs, so
    /// the builder's intermediate-table frames never collide with them.
    pub fn map_anon(
        &mut self,
        va: u64,
        len: u64,
        flags: PageFlags,
        anonymous: bool,
    ) -> Result<(), VmError> {
        if va % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
            return Err(VmError::Unaligned { addr: va, len });
        }
        self.reserve_region(va, len, anonymous)?;
        let pages = (len / PAGE_SIZE) as usize;
        let mut frames = Vec::with_capacity(pages);
        for _ in 0..pages {
            frames.push(self.alloc_frame()?);
        }
        let mut pt = PageTableBuilder::new(&self.bus, self.pml4, self.next_frame);
        for (i, pa) in frames.iter().enumerate() {
            pt.map_4k(va + i as u64 * PAGE_SIZE, *pa, flags)
                .map_err(|e| VmError::PageTable(e.to_string()))?;
        }
        self.next_frame = self.next_frame.max(pt.next_frame());
        Ok(())
    }

    /// Map one page `va → pa` with `flags` without allocating a frame.
    pub fn map_page(
        &mut self,
        va: u64,
        pa: u64,
        flags: PageFlags,
        anonymous: bool,
    ) -> Result<(), VmError> {
        if va % PAGE_SIZE != 0 || pa % PAGE_SIZE != 0 {
            return Err(VmError::Unaligned { addr: va, len: 0 });
        }
        self.reserve_region(va, PAGE_SIZE, anonymous)?;
        let mut pt = PageTableBuilder::new(&self.bus, self.pml4, self.next_frame);
        pt.map_4k(va, pa, flags)
            .map_err(|e| VmError::PageTable(e.to_string()))?;
        self.next_frame = self.next_frame.max(pt.next_frame());
        Ok(())
    }

    /// Map `[va, va+len)` **identity** (`va == pa`) with 2 MiB leaves —
    /// the DMA window (device-addressed guest RAM) and the MMIO windows.
    /// Both must be 2 MiB-aligned; the region is loader-grade (`munmap`
    /// refuses it — tearing down the DMA window mid-flight is a bug, not
    /// an operation).
    pub fn map_identity_2m(&mut self, va: u64, len: u64, flags: PageFlags) -> Result<(), VmError> {
        const M2: u64 = 2 * 1024 * 1024;
        if va % M2 != 0 || len % M2 != 0 || len == 0 {
            return Err(VmError::Unaligned { addr: va, len });
        }
        self.reserve_region(va, len, false)?;
        let mut pt = PageTableBuilder::new(&self.bus, self.pml4, self.next_frame);
        let mut off = 0u64;
        while off < len {
            pt.map_2m(va + off, va + off, flags)
                .map_err(|e| VmError::PageTable(e.to_string()))?;
            off += M2;
        }
        self.next_frame = self.next_frame.max(pt.next_frame());
        Ok(())
    }

    /// Reserve `[va, va+len)` (page-ceiled) as a tracked region with **no**
    /// page-table entries — `PROT_NONE` semantics: any access faults, but
    /// `munmap`/bookkeeping still see the mapping.
    pub fn reserve(&mut self, va: u64, len: u64) -> Result<(), VmError> {
        self.reserve_region(va, page_ceil(len.max(1)), true)
    }

    /// Reserve the region bookkeeping for `[va, va+len)`.
    fn reserve_region(&mut self, va: u64, len: u64, anonymous: bool) -> Result<(), VmError> {
        let end = va + len;
        if end <= va {
            return Err(VmError::Unaligned { addr: va, len });
        }
        for r in &self.regions {
            let r_end = r.start + r.len;
            if va < r_end && r.start < end {
                return Err(VmError::Overlaps { start: va, end });
            }
        }
        self.regions.push(Region {
            start: va,
            len,
            anonymous,
        });
        Ok(())
    }

    /// Zero the PTEs of `[va, va+len)` in the page tables and drop the
    /// region. Only `anonymous` regions may be unmapped through `munmap`;
    /// loader segments refuse with [`VmError::NotMapped`] semantics
    /// (documented: the syscall layer turns that into `EINVAL`).
    ///
    /// The software TLB is flushed afterwards — the syscall runs at an
    /// instruction boundary, so no cached translation outlives the unmap.
    pub fn unmap(&mut self, va: u64, len: u64) -> Result<(), VmError> {
        if va % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
            return Err(VmError::Unaligned { addr: va, len });
        }
        let end = va
            .checked_add(len)
            .ok_or(VmError::Unaligned { addr: va, len })?;
        let idx = self
            .regions
            .iter()
            .position(|r| r.anonymous && r.start == va && r.len >= len);
        let Some(idx) = idx else {
            return Err(VmError::NotMapped { start: va, end });
        };
        self.regions.remove(idx);
        let mut off = 0u64;
        while off < len {
            self.clear_pte(va + off)?;
            off += PAGE_SIZE;
        }
        self.flush_tlb();
        Ok(())
    }

    /// Change the permissions of the anonymous region at `va` (length must
    /// cover whole region(s)). Loader segments refuse.
    pub fn protect(&mut self, va: u64, len: u64, flags: PageFlags) -> Result<(), VmError> {
        if va % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
            return Err(VmError::Unaligned { addr: va, len });
        }
        let end = va
            .checked_add(len)
            .ok_or(VmError::Unaligned { addr: va, len })?;
        let any = self
            .regions
            .iter()
            .any(|r| r.anonymous && r.start == va && r.len == len);
        if !any {
            return Err(VmError::NotMapped { start: va, end });
        }
        let mut off = 0u64;
        while off < len {
            self.rewrite_pte(va + off, flags)?;
            off += PAGE_SIZE;
        }
        self.flush_tlb();
        Ok(())
    }

    /// Write `u64` into the page table entry of `va`, preserving the frame.
    fn rewrite_pte(&mut self, va: u64, flags: PageFlags) -> Result<(), VmError> {
        let pte_addr = self.pte_slot(va)?;
        let entry = self
            .bus
            .read(pdg_memory::AccessWidth::W64, pte_addr)
            .map_err(|e| VmError::Memory(e.to_string()))?;
        let frame = entry & FRAME_MASK;
        self.bus
            .write(
                pdg_memory::AccessWidth::W64,
                pte_addr,
                frame | (flags | PageFlags::PRESENT).bits(),
            )
            .map_err(|e| VmError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Zero the page-table entry of `va`.
    fn clear_pte(&mut self, va: u64) -> Result<(), VmError> {
        let slot = self.pte_slot(va)?;
        // Rewriting only works on a present 4 KiB entry; absent entries
        // are already clear.
        let entry = self
            .bus
            .read(pdg_memory::AccessWidth::W64, slot)
            .map_err(|e| VmError::Memory(e.to_string()))?;
        if entry & FRAME_MASK_2M == 0 {
            return Ok(());
        }
        self.bus
            .write(pdg_memory::AccessWidth::W64, slot, 0)
            .map_err(|e| VmError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Guest-physical address of the PTE for `va` (walks the live tables).
    fn pte_slot(&self, va: u64) -> Result<u64, VmError> {
        use pdg_memory::page_table::{pd_index, pdpt_index, pml4_index, pt_index};
        let w = pdg_memory::AccessWidth::W64;
        let read_entry = |addr: u64| -> Result<u64, VmError> {
            self.bus
                .read(w, addr)
                .map_err(|e| VmError::Memory(e.to_string()))
        };
        let pml4e = read_entry(self.pml4 + pml4_index(va) * 8)?;
        if pml4e & PageFlags::PRESENT.bits() == 0 {
            return Err(VmError::NotMapped { start: va, end: va });
        }
        let pdpte = read_entry((pml4e & FRAME_MASK) + pdpt_index(va) * 8)?;
        if pdpte & PageFlags::PRESENT.bits() == 0 {
            return Err(VmError::NotMapped { start: va, end: va });
        }
        let pde = read_entry((pdpte & FRAME_MASK) + pd_index(va) * 8)?;
        if pde & PageFlags::PRESENT.bits() == 0 || pde & PageFlags::PS.bits() != 0 {
            return Err(VmError::PageTable(format!("no 4 KiB PTE at va={va:#x}")));
        }
        Ok((pde & FRAME_MASK) + pt_index(va) * 8)
    }

    /// Flush the software TLB (used after table mutations).
    pub fn flush_tlb(&self) {
        self.mmu.lock().expect("mmu").flush_tlb();
    }

    /// Carve `len` bytes of anonymous address space (page-aligned up) and
    /// return its base. `PROT` semantics are applied by the caller.
    pub fn mmap_anon(&mut self, len: u64) -> Result<u64, VmError> {
        let len = page_ceil(len.max(1));
        let va = self.mmap_cursor;
        let end = va.checked_add(len).ok_or(VmError::OutOfFrames {
            next: va,
            limit: self.mmap_limit,
        })?;
        if end > self.mmap_limit {
            return Err(VmError::OutOfFrames {
                next: va,
                limit: self.mmap_limit,
            });
        }
        self.mmap_cursor = end;
        Ok(va)
    }

    /// Translate `va` through the shared MMU (guest-physical result).
    pub fn translate(&self, va: u64, write: bool) -> Result<u64, VmError> {
        let kind = if write {
            pdg_memory::AccessKind::Write
        } else {
            pdg_memory::AccessKind::Read
        };
        self.mmu
            .lock()
            .expect("mmu")
            .translate(&self.bus, va, kind)
            .map_err(|e| VmError::Memory(e.to_string()))
    }

    /// Write `bytes` at virtual `va` through the page tables.
    pub fn write_va(&self, va: u64, bytes: &[u8]) -> Result<(), VmError> {
        let mut off = 0usize;
        while off < bytes.len() {
            let page_off = ((va + off as u64) & (PAGE_SIZE - 1)) as usize;
            let chunk = (PAGE_SIZE as usize - page_off).min(bytes.len() - off);
            let pa = self.translate(va + off as u64, true)?;
            self.bus
                .write_buf(pa, &bytes[off..off + chunk])
                .map_err(|e| VmError::Memory(e.to_string()))?;
            off += chunk;
        }
        Ok(())
    }

    /// Loader write: like [`write_va`](Self::write_va) but ignoring the
    /// write-permission bit — the kernel-loader path real kernels use
    /// (supervisor stores under `CR0.WP = 0`). The page must still be
    /// present. This is how segment data and relocations land in RX
    /// pages.
    pub fn write_va_loader(&self, va: u64, bytes: &[u8]) -> Result<(), VmError> {
        let mut off = 0usize;
        while off < bytes.len() {
            let page_off = ((va + off as u64) & (PAGE_SIZE - 1)) as usize;
            let chunk = (PAGE_SIZE as usize - page_off).min(bytes.len() - off);
            // Translate as a read: present-check only, no RW enforcement.
            let pa = self.translate(va + off as u64, false)?;
            self.bus
                .write_buf(pa, &bytes[off..off + chunk])
                .map_err(|e| VmError::Memory(e.to_string()))?;
            off += chunk;
        }
        Ok(())
    }

    /// Read `out.len()` bytes from virtual `va`.
    pub fn read_va(&self, va: u64, out: &mut [u8]) -> Result<(), VmError> {
        let mut off = 0u64;
        while off < out.len() as u64 {
            let pa = self.translate(va + off, false)?;
            let page_off = (va + off) & (PAGE_SIZE - 1);
            let chunk = ((PAGE_SIZE - page_off) as usize).min(out.len() - off as usize);
            self.bus
                .read_buf(pa, &mut out[off as usize..off as usize + chunk])
                .map_err(|e| VmError::Memory(e.to_string()))?;
            off += chunk as u64;
        }
        Ok(())
    }

    /// Number of tracked regions (tests).
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// True when `[va, va+len)` is tracked as mapped (tests).
    pub fn is_mapped(&self, va: u64, len: u64) -> bool {
        self.regions
            .iter()
            .any(|r| r.start <= va && va + len <= r.start + r.len)
    }
}

/// The interpreter's memory port for a process: virtual addresses through
/// the shared MMU and bus, MMIO windows included, exactly the contract
/// `pdg-core`'s `GuestMemory` established for the boot path.
pub struct ProcessMemory {
    /// Shared bus.
    pub bus: Arc<Bus>,
    /// Shared MMU.
    pub mmu: Arc<Mutex<Mmu>>,
}

impl ProcessMemory {
    /// Build the port over the shared parts of a [`VmManager`].
    pub fn new(bus: Arc<Bus>, mmu: Arc<Mutex<Mmu>>) -> Self {
        Self { bus, mmu }
    }
}

fn map_fault(e: pdg_memory::AccessFault) -> pdg_cpu::AccessError {
    match e {
        pdg_memory::AccessFault::PageFault { addr, code } => {
            pdg_cpu::AccessError::PageFault { addr, code }
        }
        pdg_memory::AccessFault::NonCanonical { addr } => pdg_cpu::AccessError::GeneralProtection(
            format!("non-canonical guest address {addr:#x}"),
        ),
        pdg_memory::AccessFault::Bus(e) => pdg_cpu::AccessError::Host(e.to_string()),
    }
}

impl pdg_cpu::MemoryAccess for ProcessMemory {
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, pdg_cpu::AccessError> {
        let w = pdg_memory::AccessWidth::from_size(width)
            .ok_or_else(|| pdg_cpu::AccessError::Host(format!("bad access width {width}")))?;
        self.mmu
            .lock()
            .expect("mmu")
            .read(&self.bus, w, addr)
            .map_err(map_fault)
    }

    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), pdg_cpu::AccessError> {
        let w = pdg_memory::AccessWidth::from_size(width)
            .ok_or_else(|| pdg_cpu::AccessError::Host(format!("bad access width {width}")))?;
        self.mmu
            .lock()
            .expect("mmu")
            .write(&self.bus, w, addr, value)
            .map_err(map_fault)
    }

    fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), pdg_cpu::AccessError> {
        let mut off = 0usize;
        while off < out.len() {
            let page_off = ((addr + off as u64) & (PAGE_SIZE - 1)) as usize;
            let chunk = (PAGE_SIZE as usize - page_off).min(out.len() - off);
            let va = addr + off as u64;
            let pa = {
                let mut mmu = self.mmu.lock().expect("mmu");
                mmu.translate(&self.bus, va, pdg_memory::AccessKind::Read)
                    .map_err(map_fault)?
            };
            self.bus
                .read_buf(pa, &mut out[off..off + chunk])
                .map_err(|e| map_fault(pdg_memory::AccessFault::Bus(e)))?;
            off += chunk;
        }
        Ok(())
    }

    fn write_buf(&mut self, addr: u64, src: &[u8]) -> Result<(), pdg_cpu::AccessError> {
        let mut off = 0usize;
        while off < src.len() {
            let page_off = ((addr + off as u64) & (PAGE_SIZE - 1)) as usize;
            let chunk = (PAGE_SIZE as usize - page_off).min(src.len() - off);
            let va = addr + off as u64;
            let pa = {
                let mut mmu = self.mmu.lock().expect("mmu");
                mmu.translate(&self.bus, va, pdg_memory::AccessKind::Write)
                    .map_err(map_fault)?
            };
            self.bus
                .write_buf(pa, &src[off..off + chunk])
                .map_err(|e| map_fault(pdg_memory::AccessFault::Bus(e)))?;
            off += chunk;
        }
        Ok(())
    }

    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, pdg_cpu::AccessError> {
        let mut filled = 0usize;
        while filled < out.len() {
            let va = addr + filled as u64;
            let pa = {
                let mut mmu = self.mmu.lock().expect("mmu");
                match mmu.translate(&self.bus, va, pdg_memory::AccessKind::Exec) {
                    Ok(pa) => pa,
                    Err(_) if filled > 0 => break,
                    Err(e) => return Err(map_fault(e)),
                }
            };
            let page_off = (va & (PAGE_SIZE - 1)) as usize;
            let want = (out.len() - filled).min(PAGE_SIZE as usize - page_off);
            let avail = (self.bus.ram_len() as usize).saturating_sub(pa as usize);
            if avail == 0 {
                // MMIO/ROM fetch window: byte-at-a-time through the bus.
                for i in 0..want {
                    let b = self
                        .bus
                        .read(pdg_memory::AccessWidth::W8, pa + i as u64)
                        .map_err(|e| map_fault(pdg_memory::AccessFault::Bus(e)))?;
                    out[filled + i] = b as u8;
                }
                filled += want;
                continue;
            }
            let chunk = want.min(avail);
            self.bus
                .read_buf(pa, &mut out[filled..filled + chunk])
                .map_err(|e| map_fault(pdg_memory::AccessFault::Bus(e)))?;
            filled += chunk;
        }
        Ok(filled)
    }
}

/// Attach the boot console to `bus` at `base` and return the shared device.
pub fn attach_console(bus: &mut Bus, base: u64, console: Arc<ConsoleDevice>) {
    bus.attach_mmio(base, 0x1000, console);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vm(ram: usize) -> VmManager {
        let bus = Arc::new(Bus::new(ram).expect("bus"));
        VmManager::new(
            bus,
            0x0100_0000,
            0x0100_0000,
            0x0100_0000 + 0x100_000,
            0x3000_0000_0000,
            0x3000_0001_0000,
        )
    }

    #[test]
    fn map_write_read_unmap() {
        let mut vm = vm(32 * 1024 * 1024);
        let va = vm.mmap_anon(0x1000).unwrap();
        vm.map_anon(
            va,
            0x1000,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            true,
        )
        .unwrap();
        vm.write_va(va, &[1, 2, 3, 4]).unwrap();
        let mut out = [0u8; 4];
        vm.read_va(va, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        assert!(vm.is_mapped(va, 0x1000));
        vm.unmap(va, 0x1000).unwrap();
        assert!(!vm.is_mapped(va, 0x1000));
        assert!(vm.read_va(va, &mut out).is_err());
    }

    #[test]
    fn overlap_refused() {
        let mut vm = vm(32 * 1024 * 1024);
        let va = vm.mmap_anon(0x2000).unwrap();
        vm.map_anon(
            va,
            0x2000,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            true,
        )
        .unwrap();
        assert!(matches!(
            vm.map_anon(va + 0x1000, 0x1000, PageFlags::PRESENT, true),
            Err(VmError::Overlaps { .. })
        ));
    }

    #[test]
    fn protect_enforced_after_flush() {
        let mut vm = vm(32 * 1024 * 1024);
        let va = vm.mmap_anon(0x1000).unwrap();
        vm.map_anon(
            va,
            0x1000,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            true,
        )
        .unwrap();
        vm.write_va(va, &[9]).unwrap();
        // Drop the write permission; the write must now fail at translation.
        vm.protect(va, 0x1000, PageFlags::PRESENT | PageFlags::NX)
            .unwrap();
        assert!(vm.write_va(va, &[9]).is_err());
        let mut out = [0u8; 1];
        vm.read_va(va, &mut out).unwrap();
        assert_eq!(out[0], 9);
    }

    #[test]
    fn loader_region_refuses_munmap() {
        let mut vm = vm(32 * 1024 * 1024);
        let pa = vm.alloc_frame().unwrap();
        vm.map_page(0x0040_0000, pa, PageFlags::PRESENT | PageFlags::RW, false)
            .unwrap();
        assert!(matches!(
            vm.unmap(0x0040_0000, 0x1000),
            Err(VmError::NotMapped { .. })
        ));
    }

    #[test]
    fn process_memory_roundtrip() {
        use pdg_cpu::MemoryAccess;
        let mut vm = vm(32 * 1024 * 1024);
        let va = vm.mmap_anon(0x1000).unwrap();
        vm.map_anon(
            va,
            0x1000,
            PageFlags::PRESENT | PageFlags::RW | PageFlags::NX,
            true,
        )
        .unwrap();
        let bus = vm.bus.clone();
        let mmu = vm.shared_mmu();
        let mut pm = ProcessMemory::new(bus, mmu);
        pm.write(8, va + 0x40, 0xCAFE_F00D_D00D_0001).unwrap();
        assert_eq!(pm.read(8, va + 0x40).unwrap(), 0xCAFE_F00D_D00D_0001);
        // Unmapped page faults.
        assert!(pm.read(8, va + 0x1000).is_err());
    }
}
