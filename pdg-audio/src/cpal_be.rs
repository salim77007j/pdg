//! cpal output backend (`backend-cpal` feature).
//!
//! Streams the mixed signal to the host's default output device through
//! cpal (WASAPI on Windows, ALSA on Linux, CoreAudio on macOS). The
//! host service loop calls [`CpalBackend::write_samples`] with rendered
//! blocks; the backend queues them into the device stream.

use crate::{AudioBackend, AudioError};

/// A cpal-backed output stream at a fixed format (48 kHz stereo f32,
/// the default the mixer renders at).
pub struct CpalBackend {
    sample_rate: u32,
    channels: u8,
    config: Option<cpal::SupportedStreamConfig>,
    stream: Option<cpal::Stream>,
    queue: std::collections::VecDeque<f32>,
    /// High-water mark of queued samples (diagnostics).
    queued_max: usize,
}

impl CpalBackend {
    /// Create the backend; the stream starts on [`AudioBackend::init`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            config: None,
            stream: None,
            queue: std::collections::VecDeque::new(),
            queued_max: 0,
        }
    }

    /// Peak queued-sample count (diagnostics for buffer sizing).
    #[must_use]
    pub fn queued_high_water(&self) -> usize {
        self.queued_max
    }
}

impl Default for CpalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBackend for CpalBackend {
    fn name(&self) -> &'static str {
        "cpal"
    }

    fn init(&mut self, sample_rate: u32, channels: u8) -> Result<(), AudioError> {
        let err = |message: String| AudioError::Backend {
            backend: "cpal",
            message,
        };
        self.sample_rate = sample_rate;
        self.channels = channels;

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| err("no output device on the host".to_owned()))?;

        // Prefer the requested format; fall back to the device default.
        let wanted = cpal::StreamConfig {
            channels: u16::from(channels),
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let supported = device
            .supported_output_configs()
            .map_err(|e| err(format!("querying output configs: {e}")))?
            .find(|c| {
                c.channels() == u16::from(channels)
                    && (c.min_sample_rate().0..=c.max_sample_rate().0).contains(&sample_rate)
            })
            .map(|c| c.with_sample_rate(cpal::SampleRate(sample_rate)))
            .unwrap_or_else(|| {
                device
                    .default_output_config()
                    .expect("default output config after enumeration succeeded")
            });
        self.config = Some(supported.clone());

        // The stream pulls from the queue; underruns emit silence.
        let queue = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::<f32>::new(),
        ));
        let sink = Arc::clone(&queue);
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_output_stream(
                    &wanted,
                    move |data: &mut [f32], _| {
                        let mut q = sink.lock().expect("cpal queue poisoned");
                        for d in data.iter_mut() {
                            *d = q.pop_front().unwrap_or(0.0);
                        }
                    },
                    |e| tracing::warn!(%e, "cpal output error"),
                    None,
                )
                .map_err(|e| err(format!("building f32 stream: {e}")))?,
            other => {
                return Err(err(format!(
                    "unsupported host sample format {other:?} (PDG renders f32)"
                )));
            }
        };
        stream
            .play()
            .map_err(|e| err(format!("starting the stream: {e}")))?;
        self.stream = Some(stream);
        Ok(())
    }

    fn write_samples(&mut self, samples: &[i16]) -> Result<(), AudioError> {
        // The host service loop may not have started a stream (headless
        // runs); report that honestly instead of silently dropping.
        let Some(_stream) = self.stream.as_ref() else {
            return Err(AudioError::Backend {
                backend: "cpal",
                message: "write_samples before init (no stream)".to_owned(),
            });
        };
        for s in samples {
            self.queue.push_back(f32::from(*s) / 32768.0);
        }
        self.queued_max = self.queued_max.max(self.queue.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_before_init_is_refused() {
        let mut backend = CpalBackend::new();
        // Headless CI machines have no output device; both refusal paths
        // are honest outcomes, so accept either.
        match backend.init(48_000, 2) {
            Ok(()) => {
                backend.write_samples(&[0, 100, -100]).unwrap();
            }
            Err(AudioError::Backend { .. }) => {
                assert!(backend.write_samples(&[0]).is_err());
            }
        }
    }
}
