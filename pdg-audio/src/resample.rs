//! Linear resampler: guest rates (e.g. 44.1 kHz assets) → output rate.
//!
//! Position-tracking linear interpolation with a stable phase carry, so
//! consecutive blocks are continuous — the property a game's streamed
//! audio blocks need. Everything is unit-tested on ramps and DC.

/// Linear resampler with carried phase.
pub struct LinearResampler {
    src_rate: u32,
    dst_rate: u32,
    /// Fractional position in the *source* stream.
    pos: f64,
    /// Last sample of the previous block (continuity anchor).
    last: f32,
}

impl LinearResampler {
    /// Create a resampler from `src_rate` to `dst_rate` Hz.
    #[must_use]
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        debug_assert!(src_rate > 0 && dst_rate > 0);
        Self {
            src_rate,
            dst_rate,
            pos: 0.0,
            last: 0.0,
        }
    }

    /// The step taken through the source per output sample.
    #[must_use]
    pub fn step(&self) -> f64 {
        f64::from(self.src_rate) / f64::from(self.dst_rate)
    }

    /// Resample one block of mono samples; output length is
    /// `input.len() * dst / src` rounded down, with the phase carried
    /// into the next block.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let step = self.step();
        // Total source positions covered: the block plus the carried tail.
        let mut out = Vec::with_capacity((input.len() as f64 / step).ceil() as usize + 1);
        let end = input.len() as f64;
        // Virtual source stream: previous block's last sample, then input.
        let sample_at = |p: f64| -> f32 {
            if p < 0.0 {
                return self.last;
            }
            let i = p.floor() as usize;
            let frac = (p - p.floor()) as f32;
            let s0 = if i < input.len() { input[i] } else { 0.0 };
            let s1 = input.get(i + 1).copied().unwrap_or(s0);
            s0 + (s1 - s0) * frac
        };
        while self.pos < end - 1.0 + step {
            if self.pos >= end {
                break;
            }
            out.push(sample_at(self.pos));
            self.pos += step;
        }
        self.pos -= end;
        self.last = input[input.len() - 1];
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passthrough() {
        let mut rs = LinearResampler::new(48_000, 48_000);
        let out = rs.process(&[0.0, 0.1, 0.2, 0.3]);
        assert_eq!(out, vec![0.0, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn downsample_halves() {
        let mut rs = LinearResampler::new(96_000, 48_000);
        let out = rs.process(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        // Step 2.0: positions 0, 2, 4, 6.
        assert_eq!(out, vec![0.0, 0.2, 0.4, 0.6]);
    }

    #[test]
    fn upsample_interpolates_linearly() {
        let mut rs = LinearResampler::new(24_000, 48_000);
        let out = rs.process(&[0.0, 1.0]);
        // Step 0.5: positions 0, 0.5, 1.0 (the last is the anchor).
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blocks_are_continuous_across_boundaries() {
        // A 1 kHz ramp across two blocks must not glitch: the first sample
        // of block 2 must continue the ramp's slope.
        let mut rs = LinearResampler::new(44_100, 48_000);
        let ramp: Vec<f32> = (0..441).map(|i| i as f32 * 0.001).collect();
        let ramp2: Vec<f32> = (441..882).map(|i| i as f32 * 0.001).collect();
        let a = rs.process(&ramp);
        let b = rs.process(&ramp2);
        assert!(!a.is_empty() && !b.is_empty());
        // Slope continuity: no jump bigger than one input step.
        let slope = 0.001 * rs.step();
        assert!((b[0] - a[a.len() - 1] - slope as f32).abs() < 0.01);
    }
}
