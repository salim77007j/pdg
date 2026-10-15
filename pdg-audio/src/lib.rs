//! # pdg-audio
//!
//! Audio subsystem for the PDG PlayStation 5 emulator.
//!
//! The PS5's Tempest Engine is a dedicated DSP; emulators generally HLE the
//! guest audio APIs instead of LLE-ing that hardware, and PDG follows that
//! strategy. Phase 1 defines the [`AudioBackend`] sink; the cpal output and
//! the HRTF pipeline land in Phase 4.

use thiserror::Error;

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

/// Null sink used until the cpal backend exists. Counts samples and drops
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
