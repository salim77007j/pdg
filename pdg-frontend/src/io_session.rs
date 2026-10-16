//! The Phase-4 I/O session: one flat machine exercising the whole I/O
//! stack with the emulated CPU in the driver's seat.
//!
//! The session builds the console-shaped I/O plane — an NVMe SSD over a
//! host file, the DualSense window, the Tempest-shaped audio device, and
//! the boot console — loads the hand-assembled demo driver (see
//! `pdg_io::demo`), and runs it on the interpreter with the device pump
//! between slices: exactly the doorbell/pump split the GPU ring
//! established in Phase 3. The result is an honest, end-to-end account of
//! what worked: per-stage guest statuses, the console transcript, the
//! rendered audio, and the SSD statistics.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use pdg_cpu::{AccessError, CpuState, Interpreter, MemoryAccess, RunOutcome};
use pdg_io::block::HostFileBlockDevice;
use pdg_io::demo::{self, layout};
use pdg_io::nvme::{NVME_FOOTPRINT, NvmeController};
use pdg_io::{DualSenseState, PadDevice};
use pdg_memory::bus::{AccessWidth, Bus};
use pdg_memory::devices::ConsoleDevice;

/// MMIO layout of the session (defaults mirror `PdgConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMmio {
    /// NVMe register window base.
    pub nvme: u64,
    /// Pad window base.
    pub pad: u64,
    /// Audio window base.
    pub audio: u64,
    /// Console base.
    pub console: u64,
}

impl Default for SessionMmio {
    fn default() -> Self {
        Self {
            nvme: 0xE000_0000,
            pad: 0xE100_0000,
            audio: 0xE200_0000,
            console: 0xF000_0000,
        }
    }
}

/// How the demo run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionOutcome {
    /// The driver halted cleanly (`hlt`), with the instruction count and
    /// the RIP where it parked.
    Halted { instructions: u64, rip: u64 },
    /// The driver faulted.
    Fault(String),
    /// The pump-loop budget ran out before the driver finished.
    BudgetExhausted,
}

/// Everything the session observed.
#[derive(Debug, Clone)]
pub struct SessionReport {
    /// How the driver run ended.
    pub outcome: SessionOutcome,
    /// Console transcript.
    pub console: String,
    /// Per-stage guest statuses (1 = pass), indexed by
    /// [`layout`] constants.
    pub statuses: Vec<u64>,
    /// The audio the guest hydrated, rendered to stereo i16 (the value a
    /// backend receives), or empty when the stage did not run.
    pub audio: Vec<i16>,
    /// SSD statistics observed after the run.
    pub ssd_stats: pdg_io::nvme::controller::NvmeStats,
}

/// The machine plus its devices.
pub struct IoSession {
    /// The flat guest-physical bus.
    pub bus: Bus,
    /// The SSD controller.
    pub ssd: Arc<NvmeController<HostFileBlockDevice>>,
    /// The pad window.
    pub pad: Arc<PadDevice>,
    /// The audio window (created by [`IoSession::new`] with the default
    /// 48 kHz rate).
    pub audio: Arc<pdg_audio::AudioDevice>,
    /// The boot console.
    pub console: Arc<ConsoleDevice>,
}

impl IoSession {
    /// Build the machine around `ssd_path` (created if missing) and load
    /// the demo driver into guest RAM.
    pub fn new(ssd_path: &Path) -> anyhow::Result<Self> {
        let mmio = SessionMmio::default();
        Self::with_layout(ssd_path, mmio)
    }

    /// Build with an explicit MMIO layout (tests use non-default bases).
    pub fn with_layout(ssd_path: &Path, mmio: SessionMmio) -> anyhow::Result<Self> {
        // Create the backing file if it does not exist (the user's own
        // storage image; PDG never provides content).
        if !ssd_path.exists() {
            if let Some(dir) = ssd_path.parent() {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("creating the SSD directory {}", dir.display()))?;
            }
            let size = 8 * 1024 * 1024usize;
            let f = std::fs::File::create(ssd_path)
                .with_context(|| format!("creating the SSD backing file {}", ssd_path.display()))?;
            f.set_len(size as u64)?;
        }

        let mut bus = Bus::new(8 << 20).context("guest RAM")?;
        let console = Arc::new(ConsoleDevice::new());
        bus.attach_mmio(mmio.console, 0x1000, console.clone());
        let ssd = Arc::new(NvmeController::new(
            HostFileBlockDevice::open(ssd_path)
                .with_context(|| format!("opening the SSD file {}", ssd_path.display()))?,
        ));
        bus.attach_mmio(mmio.nvme, NVME_FOOTPRINT, ssd.clone());
        let pad = PadDevice::new().shared();
        bus.attach_mmio(
            mmio.pad,
            pdg_io::dualsense::pad_regs::FOOTPRINT,
            pad.clone(),
        );
        let audio = pdg_audio::AudioDevice::new(48_000).shared();
        bus.attach_mmio(mmio.audio, pdg_audio::audio_regs::FOOTPRINT, audio.clone());

        // Load the demo driver.
        let payload = demo::io_demo_payload(&demo::DemoMmio {
            nvme: mmio.nvme,
            pad: mmio.pad,
            audio: mmio.audio,
            console: mmio.console,
        });
        bus.write_buf(layout::CODE, &payload)
            .context("placing the demo driver in guest RAM")?;
        Ok(Self {
            bus,
            ssd,
            pad,
            audio,
            console,
        })
    }

    /// Run the demo driver with the given initial pad state, servicing
    /// the SSD and audio devices between execution slices.
    pub fn run(&self, pad_state: DualSenseState) -> SessionReport {
        self.pad.set_state(pad_state);
        let mut interp = Interpreter::new(CpuState::new(layout::CODE, layout::STACK));
        let mut mem = BusMem { bus: &self.bus };

        let slice = 200u64;
        let max_slices = 200_000u64; // 40M instructions: generous for the demo
        let mut instructions = 0u64;
        let outcome = loop {
            // Service the doorbell-pumped devices, then run a slice.
            let _ = self.ssd.pump(&self.bus);
            let _ = self.audio.pump(&self.bus);
            let r = interp.run(&mut mem, slice);
            match &r {
                RunOutcome::Halted {
                    instructions: n,
                    rip,
                } => {
                    instructions += *n;
                    break SessionOutcome::Halted {
                        instructions,
                        rip: *rip,
                    };
                }
                RunOutcome::Fault(f) => break SessionOutcome::Fault(f.to_string()),
                RunOutcome::MaxInstructions { instructions: n } => {
                    instructions += *n;
                }
            }
            if instructions >= max_slices * slice {
                break SessionOutcome::BudgetExhausted;
            }
        };

        // Collect the results the driver left behind.
        let mut statuses = Vec::with_capacity(6);
        for i in 0..6u64 {
            let addr = layout::RESULTS + i * 8;
            let v = self
                .bus
                .read(AccessWidth::W64, addr)
                .map(|v| v as i64 as u64)
                .unwrap_or(0);
            statuses.push(v);
        }

        // Render whatever the guest hydrated into the recorder.
        let mut recorder = Recorder::default();
        let _ = self.audio.flush_all(&mut recorder);

        SessionReport {
            outcome,
            console: self.console.output(),
            statuses,
            audio: recorder.samples,
            ssd_stats: self.ssd.stats(),
        }
    }
}

/// A backend that records every sample it receives (tests and reports).
#[derive(Default)]
pub struct Recorder {
    /// Interleaved stereo samples.
    pub samples: Vec<i16>,
}

impl pdg_audio::AudioBackend for Recorder {
    fn name(&self) -> &'static str {
        "session-recorder"
    }
    fn init(&mut self, _rate: u32, _ch: u8) -> Result<(), pdg_audio::AudioError> {
        Ok(())
    }
    fn write_samples(&mut self, samples: &[i16]) -> Result<(), pdg_audio::AudioError> {
        self.samples.extend_from_slice(samples);
        Ok(())
    }
}

/// Flat memory over the bus: physical == virtual (the tier-1 pattern).
struct BusMem<'a> {
    bus: &'a Bus,
}

impl MemoryAccess for BusMem<'_> {
    fn read(&mut self, width: u8, addr: u64) -> Result<u64, AccessError> {
        let w = AccessWidth::from_size(width)
            .ok_or_else(|| AccessError::Host(format!("bad width {width}")))?;
        self.bus
            .read(w, addr)
            .map_err(|_| AccessError::PageFault { addr, code: 0 })
    }

    fn write(&mut self, width: u8, addr: u64, value: u64) -> Result<(), AccessError> {
        let w = AccessWidth::from_size(width)
            .ok_or_else(|| AccessError::Host(format!("bad width {width}")))?;
        self.bus
            .write(w, addr, value)
            .map_err(|_| AccessError::PageFault { addr, code: 0 })
    }

    fn fetch(&mut self, addr: u64, out: &mut [u8]) -> Result<usize, AccessError> {
        for (i, b) in out.iter_mut().enumerate() {
            *b = self
                .bus
                .read(AccessWidth::W8, addr + i as u64)
                .map_err(|_| AccessError::PageFault { addr, code: 0 })? as u8;
        }
        Ok(out.len())
    }

    fn read_buf(&mut self, addr: u64, out: &mut [u8]) -> Result<(), AccessError> {
        for (i, b) in out.iter_mut().enumerate() {
            *b = self
                .bus
                .read(AccessWidth::W8, addr + i as u64)
                .map_err(|_| AccessError::PageFault { addr, code: 0 })? as u8;
        }
        Ok(())
    }

    fn write_buf(&mut self, addr: u64, data: &[u8]) -> Result<(), AccessError> {
        for (i, b) in data.iter().enumerate() {
            self.bus
                .write(AccessWidth::W8, addr + i as u64, u64::from(*b))
                .map_err(|_| AccessError::PageFault { addr, code: 0 })?;
        }
        Ok(())
    }
}
