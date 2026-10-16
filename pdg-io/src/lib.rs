//! # pdg-io
//!
//! I/O subsystem for the PDG PlayStation 5 emulator.
//!
//! Phase 1 delivered the host-facing foundations; Phase 4 builds the
//! console's I/O contract on top of them:
//!
//! * [`block`] — a [`BlockDevice`][block::BlockDevice] trait with a
//!   host-file implementation, plus a tokio-based async positioned-I/O
//!   type (the DirectStorage-shaped path).
//! * [`nvme`] — the NVMe-shaped SSD model: a controller register file,
//!   submission/completion queues driven through doorbell MMIO, PRP DMA
//!   into guest RAM, and IDENTIFY structures — the storage stack a
//!   guest driver programs.
//! * [`dualsense`] — the DualSense pad model: full state (buttons,
//!   sticks, triggers, touch, motion), adaptive-trigger effect blocks,
//!   HID report 0x01 encode/decode, and the guest-visible
//!   [`PadDevice`][dualsense::PadDevice] MMIO window.
//! * [`input`] — the [`InputState`][input::InputState] model and the
//!   [`InputSource`][input::InputSource] trait host pad drivers
//!   implement.
//!
//! Nothing here stubs silently: drivers that do not exist yet are absent,
//! not fake.

pub mod block;
pub mod demo;
pub mod dualsense;
pub mod input;
pub mod nvme;

pub use block::{AsyncFileBlockDevice, BlockDevice, HostFileBlockDevice};
pub use dualsense::{DualSenseState, PadDevice};
pub use input::{InputSource, InputState};
pub use nvme::NvmeController;
