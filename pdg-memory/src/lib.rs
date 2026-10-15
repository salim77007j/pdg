//! # pdg-memory
//!
//! Memory subsystem for the PDG PlayStation 5 emulator.
//!
//! The PS5 has 16 GiB of unified GDDR6 behind a single guest-physical address
//! space. This crate models the parts of that world the CPU, GPU and I/O
//! devices actually touch in Phase 1 of the project:
//!
//! * [`physical::PhysicalMemory`] — lazily-committed anonymous backing store
//!   for guest RAM (zero-filled, like real DRAM at power-on).
//! * [`bus::Bus`] — guest-physical address decoding: RAM, read-only ROM
//!   regions (the BIOS window), and memory-mapped I/O devices.
//! * [`mmu::Mmu`] — a software x86-64 MMU performing a genuine four-level
//!   (PML4 → PDPT → PD → PT) page-table walk over guest RAM, with permission
//!   checks (present / read-write / no-execute) and a software TLB.
//! * [`page_table::PageTableBuilder`] — constructs real x86-64 page tables in
//!   guest RAM (4 KiB and 2 MiB leaves) for the bootstrap address space.
//! * [`dma::DmaEngine`] — guest-physical DMA between any bus regions.
//!
//! Design notes and the Phase-1 guest memory map are documented in
//! `docs/ARCHITECTURE.md` at the repository root.

pub mod bus;
pub mod devices;
pub mod dma;
pub mod error;
pub mod mmu;
pub mod page_table;
pub mod physical;

pub use bus::{AccessWidth, Bus, MmioDevice, RomRegion};
pub use devices::ConsoleDevice;
pub use dma::{DmaEngine, DmaRecord};
pub use error::MemoryError;
pub use mmu::{AccessFault, AccessKind, Mmu};
pub use page_table::{PageFlags, PageTableBuilder};
pub use physical::{PAGE_SIZE, PhysicalMemory};

/// Size of the 2 MiB large page supported by x86-64 paging.
pub const LARGE_PAGE_SIZE: u64 = 512 * PAGE_SIZE;
