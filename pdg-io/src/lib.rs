//! # pdg-io
//!
//! I/O subsystem for the PDG PlayStation 5 emulator.
//!
//! Phase 1 delivers the host-facing foundations that the Phase-4 stack
//! (custom-NVMe SSD model, Kraken decompression pipeline, DualSense input,
//! network) builds on:
//!
//! * [`block`] — a [`BlockDevice`][block::BlockDevice] trait with a
//!   host-file implementation, plus a tokio-based async positioned-I/O type
//!   (the DirectStorage-shaped path).
//! * [`input`] — the [`InputState`][input::InputState] model and the
//!   [`InputSource`][input::InputSource] trait the DualSense driver will
//!   implement in Phase 4.
//!
//! Nothing here stubs silently: drivers that do not exist yet are absent,
//! not fake.

pub mod block;
pub mod input;

pub use block::{AsyncFileBlockDevice, BlockDevice, HostFileBlockDevice};
pub use input::{InputSource, InputState};
