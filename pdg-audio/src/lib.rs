//! # pdg-audio
//!
//! Audio subsystem for the PDG PlayStation 5 emulator — the Tempest-shaped
//! mix path, Phase 4 edition.
//!
//! The PS5's Tempest Engine is a dedicated DSP; emulators generally HLE the
//! guest audio APIs instead of LLE-ing that hardware, and PDG follows that
//! strategy. The stack:
//!
//! * [`mixer`] — voice hydration, spatialization (distance attenuation +
//!   constant-power pan), saturating master mix. A real HRTF needs a
//!   licensed/verified HRIR dataset; the pan-law spatializer is what ships
//!   now, honestly labeled.
//! * [`resample`] — linear resampler with cross-block phase carry.
//! * [`device`] — the guest-visible MMIO window: PCM blocks hydrated into
//!   voices through doorbell-style latches, rendered to any
//!   [`AudioBackend`] when the host services the device.
//! * [`cpal_be`] — the host output stream (feature `backend-cpal`).
//!
//! Phase 1's [`AudioBackend`] sink and the honest [`NullAudioBackend`]
//! remain the contract every backend implements.

use thiserror::Error;

pub mod device;
pub mod mixer;
pub mod resample;

#[cfg(feature = "backend-cpal")]
pub mod cpal_be;

pub use device::{AudioDevice, audio_regs};
pub use mixer::{AudioMixer, SpatialParams};
pub use resample::LinearResampler;

#[cfg(feature = "backend-cpal")]
pub use cpal_be::CpalBackend;

/// Errors raised by audio backends.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Backend rejected the request.
    #[error("audio backend `{backend}` error: {message}")]
    Backend {
        /// Backend that raised the error.
        backend: &'static str,
        /// What went wrong.
        message: String,
    },
}

/// Audio output sink consuming interleaved 16-bit samples.
pub trait AudioBackend: Send {
    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// Prepare the sink for `sample_rate` Hz, `channels` interleaved.
    fn init(&mut self, sample_rate: u32, channels: u8) -> Result<(), AudioError>;

    /// Enqueue samples (interleaved). Blocks when the host buffer is full.
    fn write_samples(&mut self, samples: &[i16]) -> Result<(), AudioError>;
}

/// Null sink used where no host audio exists. Counts samples and drops
/// them — explicitly a placeholder, never silently "playing" audio.
#[derive(Debug, Default)]
pub struct NullAudioBackend {
    sample_count: u64,
}

impl NullAudioBackend {
    /// Create a null sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Samples consumed so far (test hook).
    pub fn samples_consumed(&self) -> u64 {
        self.sample_count
    }
}

impl AudioBackend for NullAudioBackend {
    fn name(&self) -> &'static str {
        "null-audio"
    }

    fn init(&mut self, _sample_rate: u32, _channels: u8) -> Result<(), AudioError> {
        Ok(())
    }

    fn write_samples(&mut self, samples: &[i16]) -> Result<(), AudioError> {
        self.sample_count += samples.len() as u64;
        tracing::trace!(n = samples.len(), "null-audio: dropped samples");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_counts_samples() {
        let mut audio = NullAudioBackend::new();
        audio.init(48_000, 2).unwrap();
        audio.write_samples(&[0, 100, -100, 0]).unwrap();
        assert_eq!(audio.samples_consumed(), 4);
    }
}
