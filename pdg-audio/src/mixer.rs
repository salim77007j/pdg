//! Voice hydration and mixing — the Tempest-shaped core.
//!
//! The PS5's Tempest Engine hydrates hundreds of audio objects into a
//! final mix. PDG models the *shape* a guest sees: named voices with
//! per-voice gain and spatialization, hydrated (started) and killed
//! (stopped) by the guest, mixed into a master bus with saturating
//! accumulation. Everything is f32 internally; the backend converts.

use thiserror::Error;

/// Mixer errors.
#[derive(Debug, Error)]
pub enum MixerError {
    /// The voice id is not hydrated.
    #[error("voice {0} is not hydrated")]
    NoSuchVoice(u32),
    /// The voice ring is drained beyond what was written.
    #[error("voice {0} underran")]
    Underrun(u32),
}

/// Spatialization parameters for one voice (the Tempest-object posture;
/// HRTF itself lands when a verified HRIR dataset is wired in — see the
/// module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialParams {
    /// Source distance in meters.
    pub distance: f32,
    /// Minimum distance where attenuation begins.
    pub min_distance: f32,
    /// Maximum distance (full attenuation clamp).
    pub max_distance: f32,
    /// Horizontal pan: -1.0 (left) ..= 1.0 (right).
    pub pan: f32,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            distance: 1.0,
            min_distance: 1.0,
            max_distance: 100.0,
            pan: 0.0,
        }
    }
}

impl SpatialParams {
    /// Distance attenuation (clamped inverse).
    #[must_use]
    pub fn attenuation(&self) -> f32 {
        let d = self.distance.max(0.0);
        if d <= self.min_distance {
            1.0
        } else if d >= self.max_distance {
            self.min_distance / self.max_distance
        } else {
            self.min_distance / d
        }
    }

    /// Constant-power stereo pan coefficients `(left, right)`.
    #[must_use]
    pub fn pan_gains(&self) -> (f32, f32) {
        let theta = (self.pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
        (theta.cos(), theta.sin())
    }

    /// Total per-channel gains after distance + pan: `(left, right)`.
    #[must_use]
    pub fn channel_gains(&self) -> (f32, f32) {
        let att = self.attenuation();
        let (l, r) = self.pan_gains();
        (att * l, att * r)
    }
}

/// One hydrated voice: a mono sample stream with its spatial mix state.
struct Voice {
    samples: Vec<f32>,
    cursor: usize,
    gain: f32,
    spatial: SpatialParams,
    /// Set when the stream drained: the voice finishes.
    done: bool,
}

/// The mixer: hydrated voices + master gain.
pub struct AudioMixer {
    voices: std::collections::BTreeMap<u32, Voice>,
    next_id: u32,
    channels: usize,
    master_gain: f32,
    /// Samples rendered so far (per channel-frame).
    frames_rendered: u64,
}

impl AudioMixer {
    /// Create a mixer rendering `channels`-wide frames (2 for Phase 4).
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            voices: std::collections::BTreeMap::new(),
            next_id: 1,
            channels,
            master_gain: 1.0,
            frames_rendered: 0,
        }
    }

    /// Channel count per output frame.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of currently hydrated voices.
    #[must_use]
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Master gain (applied after the voice sum).
    pub fn set_master_gain(&mut self, gain: f32) {
        self.master_gain = gain.clamp(0.0, 1.0);
    }

    /// Master gain.
    #[must_use]
    pub fn master_gain(&self) -> f32 {
        self.master_gain
    }

    /// Frames rendered so far.
    #[must_use]
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Hydrate a voice from a mono stream; returns the voice id.
    pub fn hydrate(&mut self, samples: Vec<f32>, spatial: SpatialParams) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.voices.insert(
            id,
            Voice {
                samples,
                cursor: 0,
                gain: 1.0,
                spatial,
                done: false,
            },
        );
        id
    }

    /// Set a hydrated voice's gain.
    pub fn set_voice_gain(&mut self, id: u32, gain: f32) -> Result<(), MixerError> {
        let v = self
            .voices
            .get_mut(&id)
            .ok_or(MixerError::NoSuchVoice(id))?;
        v.gain = gain.clamp(0.0, 1.0);
        Ok(())
    }

    /// Kill (dehydrate) a voice.
    pub fn kill(&mut self, id: u32) -> Result<(), MixerError> {
        self.voices.remove(&id).ok_or(MixerError::NoSuchVoice(id))?;
        Ok(())
    }

    /// Frames still buffered in the longest-lived voice — the exact
    /// render budget that drains everything.
    #[must_use]
    pub fn max_remaining(&self) -> usize {
        self.voices
            .values()
            .filter(|v| !v.done)
            .map(|v| v.samples.len().saturating_sub(v.cursor))
            .max()
            .unwrap_or(0)
    }

    /// Render the next `frames` interleaved frames (saturating to
    /// [-1, 1]) and retire voices that drained.
    pub fn render(&mut self, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * self.channels];
        let mut finished: Vec<u32> = Vec::new();
        for (&id, v) in &mut self.voices {
            if v.done {
                continue;
            }
            let (gl, gr) = v.spatial.channel_gains();
            let gains = [v.gain * gl, v.gain * gr];
            for (frame, out_frame) in out.chunks_mut(self.channels).enumerate() {
                match v.samples.get(v.cursor + frame).copied() {
                    Some(s) => {
                        for (ch, o) in out_frame.iter_mut().enumerate() {
                            let g = gains.get(ch).copied().unwrap_or(gains[0]);
                            *o = (*o + s * g).clamp(-1.0, 1.0);
                        }
                    }
                    None => {
                        v.done = true;
                        break;
                    }
                }
            }
            v.cursor += frames;
            if v.cursor >= v.samples.len() {
                v.done = true;
            }
            if v.done {
                finished.push(id);
            }
        }
        for id in finished {
            self.voices.remove(&id);
        }
        if self.master_gain != 1.0 {
            for o in &mut out {
                *o = (*o * self.master_gain).clamp(-1.0, 1.0);
            }
        }
        self.frames_rendered += frames as u64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(n: f32) -> Vec<f32> {
        vec![n; 4]
    }

    #[test]
    fn single_voice_mixes_at_gain() {
        let mut mixer = AudioMixer::new(2);
        mixer.hydrate(voice(0.5), SpatialParams::default());
        let out = mixer.render(4);
        // Centered pan = -3 dB per channel (constant-power law):
        // 0.5 * cos(π/4) ≈ 0.3536 on L and R.
        assert_eq!(out, vec![0.35355338; 8]);
        assert_eq!(mixer.voice_count(), 0, "drained voice retires");
    }

    #[test]
    fn two_voices_sum_and_saturate() {
        let mut mixer = AudioMixer::new(2);
        mixer.hydrate(voice(0.9), SpatialParams::default());
        mixer.hydrate(voice(0.9), SpatialParams::default());
        let out = mixer.render(4);
        // 0.9 + 0.9 = 1.8 → saturated to 1.0.
        assert_eq!(out, vec![1.0; 8]);
    }

    #[test]
    fn pan_attenuates_channels_constant_power() {
        let sp = SpatialParams {
            pan: -1.0,
            ..SpatialParams::default()
        };
        let mut mixer = AudioMixer::new(2);
        mixer.hydrate(voice(1.0), sp);
        let out = mixer.render(2);
        assert_eq!(out[0], 1.0, "hard left: full left");
        assert!(out[1] < 0.01, "hard left: silent right");
        // Center: equal power.
        let (l, r) = SpatialParams::default().pan_gains();
        assert!((l - r).abs() < 1e-6);
        assert!((l * l + r * r - 1.0).abs() < 1e-6, "constant power");
    }

    #[test]
    fn distance_attenuates_clamped() {
        let mut sp = SpatialParams {
            distance: 0.0,
            ..SpatialParams::default()
        };
        assert_eq!(sp.attenuation(), 1.0, "inside min distance");
        sp.distance = 10.0;
        assert!((sp.attenuation() - 0.1).abs() < 1e-6);
        sp.distance = 1000.0;
        assert!((sp.attenuation() - 0.01).abs() < 1e-6, "clamped at max");
    }

    #[test]
    fn master_gain_scales_after_the_sum() {
        let mut mixer = AudioMixer::new(2);
        mixer.set_master_gain(0.5);
        mixer.hydrate(voice(1.0), SpatialParams::default());
        let out = mixer.render(2);
        // 1.0 × cos(π/4) × 0.5.
        assert_eq!(out, vec![0.35355338; 4]);
    }

    #[test]
    fn kill_and_unknown_voice() {
        let mut mixer = AudioMixer::new(2);
        let id = mixer.hydrate(voice(0.2), SpatialParams::default());
        mixer.set_voice_gain(id, 0.5).unwrap();
        mixer.kill(id).unwrap();
        assert!(mixer.kill(id).is_err());
        assert_eq!(mixer.render(2), vec![0.0; 4]);
        assert!(mixer.set_voice_gain(id, 1.0).is_err());
    }

    #[test]
    fn render_counts_frames() {
        let mut mixer = AudioMixer::new(2);
        mixer.hydrate(vec![0.1; 100], SpatialParams::default());
        mixer.render(30);
        mixer.render(30);
        assert_eq!(mixer.frames_rendered(), 60);
        assert_eq!(mixer.voice_count(), 1, "100 samples survive 60 frames");
    }
}
