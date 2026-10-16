//! # pdg-core
//!
//! Framework core of the PDG PlayStation 5 emulator.
//!
//! This crate owns everything that is *not* a device:
//!
//! * [`config`] — layered TOML configuration with validation.
//! * [`logging`] — `tracing` subscriber bootstrap.
//! * [`events`] — a typed publish/subscribe bus for inter-component
//!   notifications (boot events, faults, DMA completions, ...).
//! * [`clock`] — virtual clock plus a timed event scheduler.
//! * [`system`] — the [`System`] orchestrator that wires the CPU, MMU,
//!   guest-physical bus, console device and DMA engine into one machine,
//!   and implements the Phase-1 boot flow (image load → page tables → run).
//!
//! The subsystem crates (`pdg-cpu`, `pdg-memory`, ...) stay deliberately
//! independent of each other; `pdg-core` is where they meet.

pub mod clock;
pub mod config;
pub mod crash;
pub mod events;
pub mod logging;
pub mod smp;
pub mod system;

pub use clock::{Clock, Scheduler};
pub use config::{
    AudioConfig, BootConfig, CoreConfig, CpuConfig, GpuBackendKind, GpuConfig, MemoryConfig,
    PdgConfig,
};
pub use events::{Event, EventBus};
pub use smp::{CoreReport, CoreSpec, SharedGuest, SmpSystem};
pub use system::{BootReport, GuestMemory, RESET_VECTOR, ROM_BASE, System};
