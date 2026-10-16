//! The guest-visible audio window: PCM blocks in, mixed stream out.
//!
//! The console's audio path is guest → kernel → Tempest DSP → HDMI. PDG
//! models the contract the guest sees: PCM blocks written into guest
//! RAM and handed to the device through MMIO, hydrated as spatialized
//! voices into the mixer, and rendered to the [`AudioBackend`] when
//! the host services the device (the same doorbell/pump split as the
//! GNM ring and the NVMe queues: DMA needs the bus, MMIO writes
//! cannot).
//!
//! Register layout (dword offsets from the device base):
//!
//! | offset | access | meaning |
//! |---|---|---|
//! | `0x00` | read | configured sample rate |
//! | `0x04` | read | output channel count |
//! | `0x08` | read | voices currently hydrated |
//! | `0x0C` | read | samples hydrated so far |
//! | `0x10` | read | frames rendered so far |
//! | `0x20` | write | block base (guest-physical) |
//! | `0x24` | write | block length in bytes (multiple of 4) |
//! | `0x28` | write | pan, milli-units (-1000 = hard left) |
//! | `0x2C` | write | distance, centimeters |
//! | `0x30` | write | HYDRATE: latch and submit the block |
//! | `0x34` | write | KILL ALL voices |
//!
//! Blocks are little-endian mono f32 PCM.

use std::sync::Mutex;

use pdg_memory::bus::{AccessWidth, Bus, MmioDevice};

use crate::AudioBackend;
use crate::mixer::{AudioMixer, SpatialParams};

/// Register offsets (see the module table).
pub mod audio_regs {
    /// Configured sample rate.
    pub const RATE: u64 = 0x00;
    /// Output channel count.
    pub const CHANNELS: u64 = 0x04;
    /// Voices currently hydrated.
    pub const VOICES: u64 = 0x08;
    /// Samples hydrated so far.
    pub const HYDRATED: u64 = 0x0C;
    /// Frames rendered so far.
    pub const RENDERED: u64 = 0x10;
    /// Block base (write).
    pub const BLOCK_BASE: u64 = 0x20;
    /// Block length in bytes (write).
    pub const BLOCK_LEN: u64 = 0x24;
    /// Pan in milli-units, -1000..=1000 (write).
    pub const PAN: u64 = 0x28;
    /// Distance in centimeters (write).
    pub const DISTANCE: u64 = 0x2C;
    /// HYDRATE latch (write 1).
    pub const HYDRATE: u64 = 0x30;
    /// Kill all voices (write 1).
    pub const KILL_ALL: u64 = 0x34;
    /// Register footprint.
    pub const FOOTPRINT: u64 = 0x38;
}

struct Inner {
    mixer: AudioMixer,
    /// Latched hydrate request from the guest.
    pending: Vec<(u64, u64, SpatialParams)>,
    /// Samples hydrated since reset (guest-visible cursor).
    hydrated: u64,
    /// Latched block base (guest-physical).
    block_base: u64,
    /// Latched block length in bytes.
    block_len: u64,
    /// Latched pan (milli-units).
    pan: i32,
    /// Latched distance (centimeters).
    distance: u32,
}

/// The audio MMIO device.
pub struct AudioDevice {
    sample_rate: u32,
    inner: Mutex<Inner>,
}

impl AudioDevice {
    /// Create a device over `sample_rate` with a stereo mixer.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            inner: Mutex::new(Inner {
                mixer: AudioMixer::new(2),
                pending: Vec::new(),
                hydrated: 0,
                block_base: 0,
                block_len: 0,
                pan: 0,
                distance: 100,
            }),
        }
    }

    /// Wrap in `Arc` for bus registration.
    #[must_use]
    pub fn shared(self) -> std::sync::Arc<Self> {
        std::sync::Arc::new(self)
    }

    /// Configured sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Hydrate every latched block from guest RAM (mono f32 PCM).
    /// Returns the blocks hydrated.
    pub fn pump(&self, bus: &Bus) -> usize {
        let mut inner = self.inner.lock().expect("audio lock poisoned");
        let pending = std::mem::take(&mut inner.pending);
        let mut hydrated = 0;
        for (base, len, spatial) in pending {
            let mut raw = vec![0u8; len as usize];
            if bus.read_buf(base, &mut raw).is_err() {
                tracing::warn!(base, len, "audio block unreadable; dropped");
                continue;
            }
            let samples: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("f32 chunk")))
                .collect();
            inner.hydrated += samples.len() as u64;
            inner.mixer.hydrate(samples, spatial);
            hydrated += 1;
        }
        hydrated
    }

    /// Render up to `frames` frames and deliver them (interleaved i16)
    /// to `backend`. Returns frames delivered.
    pub fn flush(&self, frames: usize, backend: &mut dyn AudioBackend) -> usize {
        let mut inner = self.inner.lock().expect("audio lock poisoned");
        let out = inner.mixer.render(frames);
        let frames = out.len() / inner.mixer.channels();
        if frames > 0 {
            let pcm: Vec<i16> = out.iter().map(|s| (s * 32767.0) as i16).collect();
            if let Err(e) = backend.write_samples(&pcm) {
                tracing::warn!(%e, "audio backend rejected a block");
            }
        }
        frames
    }

    /// Drain every hydrated voice into `backend`, rendering exactly the
    /// frames the voices still hold.
    pub fn flush_all(&self, backend: &mut dyn AudioBackend) -> usize {
        let mut total = 0;
        loop {
            let remaining = {
                let inner = self.inner.lock().expect("audio lock poisoned");
                inner.mixer.max_remaining()
            };
            if remaining == 0 {
                break;
            }
            let n = self.flush(remaining.min(512), backend);
            total += n;
            if n == 0 {
                break;
            }
        }
        total
    }

    /// Whether a hydrate request is latched but not yet serviced.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self
            .inner
            .lock()
            .expect("audio lock poisoned")
            .pending
            .is_empty()
    }
}

impl MmioDevice for AudioDevice {
    fn name(&self) -> &'static str {
        "tempest-audio"
    }

    fn read(&self, offset: u64, _width: AccessWidth) -> u64 {
        let inner = self.inner.lock().expect("audio lock poisoned");
        match offset {
            audio_regs::RATE => u64::from(self.sample_rate),
            audio_regs::CHANNELS => 2,
            audio_regs::VOICES => inner.mixer.voice_count() as u64,
            audio_regs::HYDRATED => inner.hydrated,
            audio_regs::RENDERED => inner.mixer.frames_rendered(),
            _ => 0,
        }
    }

    fn write(&self, offset: u64, _width: AccessWidth, value: u64) {
        let mut inner = self.inner.lock().expect("audio lock poisoned");
        match offset {
            audio_regs::BLOCK_BASE => inner.block_base = value,
            audio_regs::BLOCK_LEN => inner.block_len = value,
            audio_regs::PAN => inner.pan = value as i64 as i32,
            audio_regs::DISTANCE => inner.distance = value as u32,
            audio_regs::HYDRATE if value == 1 => {
                // Latch the request; the host services it (DMA needs the bus).
                let spatial = SpatialParams {
                    distance: f32::max(inner.distance as f32 / 100.0, 0.0),
                    min_distance: 1.0,
                    max_distance: 100.0,
                    pan: (inner.pan as f32 / 1000.0).clamp(-1.0, 1.0),
                };
                let (base, len) = (inner.block_base, inner.block_len);
                inner.pending.push((base, len, spatial));
            }
            audio_regs::KILL_ALL if value == 1 => {
                let inner = &mut *inner;
                inner.pending.clear();
                // Kill all voices: re-create the mixer state.
                inner.mixer = AudioMixer::new(2);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NullAudioBackend;
    use std::sync::Arc;

    fn bus_with_audio() -> (Bus, Arc<AudioDevice>) {
        let mut bus = Bus::new(8 << 20).expect("bus");
        let audio = AudioDevice::new(48_000).shared();
        bus.attach_mmio(0xE200_0000, audio_regs::FOOTPRINT, audio.clone());
        (bus, audio)
    }

    const BASE: u64 = 0xE200_0000;

    fn w32(bus: &Bus, off: u64, v: u32) {
        bus.write(AccessWidth::W32, BASE + off, u64::from(v))
            .unwrap();
    }

    fn r32(bus: &Bus, off: u64) -> u32 {
        bus.read(AccessWidth::W32, BASE + off).unwrap() as u32
    }

    /// A backend that records every sample it receives.
    #[derive(Default)]
    struct Recorder {
        samples: Vec<i16>,
    }
    impl AudioBackend for Recorder {
        fn name(&self) -> &'static str {
            "recorder"
        }
        fn init(&mut self, _rate: u32, _ch: u8) -> Result<(), crate::AudioError> {
            Ok(())
        }
        fn write_samples(&mut self, samples: &[i16]) -> Result<(), crate::AudioError> {
            self.samples.extend_from_slice(samples);
            Ok(())
        }
    }

    fn push_block(bus: &Bus, addr: u64, samples: &[f32]) {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        bus.write_buf(addr, &bytes).unwrap();
    }

    #[test]
    fn guest_hydrates_through_mmio_and_host_pumps() {
        let (bus, audio) = bus_with_audio();
        assert_eq!(r32(&bus, audio_regs::RATE), 48_000);
        assert_eq!(r32(&bus, audio_regs::CHANNELS), 2);

        // The guest writes a block and latches HYDRATE.
        push_block(&bus, 0x500_000, &[0.25, 0.5, -0.75]);
        w32(&bus, audio_regs::BLOCK_BASE, 0x500_000u32);
        w32(&bus, audio_regs::BLOCK_LEN, 12);
        w32(&bus, audio_regs::HYDRATE, 1);
        assert!(audio.has_pending());
        assert_eq!(r32(&bus, audio_regs::VOICES), 0, "not yet pumped");

        // The host services it: one block hydrated, three samples in.
        assert_eq!(audio.pump(&bus), 1);
        assert!(!audio.has_pending());
        assert_eq!(r32(&bus, audio_regs::VOICES), 1);
        assert_eq!(r32(&bus, audio_regs::HYDRATED), 3);

        // Flush drains the voice into the backend, stereo-doubled. The
        // centered constant-power pan contributes cos(π/4) per channel.
        let center = std::f32::consts::FRAC_1_SQRT_2;
        let mut rec = Recorder::default();
        let frames = audio.flush_all(&mut rec);
        assert_eq!(frames, 3);
        assert_eq!(rec.samples.len(), 6);
        assert_eq!(rec.samples[0], (0.25 * center * 32767.0) as i16);
        assert_eq!(rec.samples[5], (-0.75 * center * 32767.0) as i16);
        assert_eq!(r32(&bus, audio_regs::VOICES), 0, "voice retired");
    }

    #[test]
    fn pan_and_distance_shape_the_mix() {
        let (bus, audio) = bus_with_audio();
        push_block(&bus, 0x500_000, &[1.0, 1.0]);
        w32(&bus, audio_regs::BLOCK_BASE, 0x500_000u32);
        w32(&bus, audio_regs::BLOCK_LEN, 8);
        // Hard left, 2 m distance (attenuation 0.5).
        w32(&bus, audio_regs::PAN, (-1000i32) as u32);
        w32(&bus, audio_regs::DISTANCE, 200);
        w32(&bus, audio_regs::HYDRATE, 1);
        audio.pump(&bus);

        let mut rec = Recorder::default();
        audio.flush_all(&mut rec);
        assert_eq!(rec.samples.len(), 4);
        let left = i32::from(rec.samples[0]);
        let right = i32::from(rec.samples[1]);
        assert!(left > 15_000, "left carries the signal: {left}");
        assert!(right < 100, "right is near-silent: {right}");
    }

    #[test]
    fn kill_all_clears_voices_and_pending() {
        let (bus, audio) = bus_with_audio();
        push_block(&bus, 0x500_000, &[0.5; 64]);
        w32(&bus, audio_regs::BLOCK_BASE, 0x500_000u32);
        w32(&bus, audio_regs::BLOCK_LEN, 256);
        w32(&bus, audio_regs::HYDRATE, 1);
        w32(&bus, audio_regs::KILL_ALL, 1);
        assert!(!audio.has_pending(), "pending cleared");
        assert_eq!(audio.pump(&bus), 0);
        assert_eq!(r32(&bus, audio_regs::VOICES), 0);
        let mut null = NullAudioBackend::new();
        assert_eq!(audio.flush_all(&mut null), 0);
    }

    #[test]
    fn unreadable_block_is_dropped_not_invented() {
        let (bus, audio) = bus_with_audio();
        w32(&bus, audio_regs::BLOCK_BASE, 0xF000_0000); // beyond RAM
        w32(&bus, audio_regs::BLOCK_LEN, 8);
        w32(&bus, audio_regs::HYDRATE, 1);
        assert_eq!(audio.pump(&bus), 0);
        assert_eq!(r32(&bus, audio_regs::VOICES), 0);
    }
}
