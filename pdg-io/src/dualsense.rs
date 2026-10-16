//! DualSense pad model (Phase 4 input).
//!
//! Three layers, honestly separated:
//!
//! 1. **The state model** — [`DualSenseState`] carries the full pad
//!    state: face buttons + dpad (hat), analog sticks (0–255, 128
//!    center), L2/R2 analog pressure, two touch points, gyroscope and
//!    accelerometer samples, battery. This is what emulation consumes.
//! 2. **The effect model** — adaptive triggers as *behavior*:
//!    [`TriggerEffect::resistance_at`] computes the trigger's
//!    resistive force for a given pull, per effect mode, so the guest
//!    experiences the same trigger physics a player would. (Wire-format
//!    output blocks exist for the host path; their byte layout follows
//!    the public DualSense effect documentation and is verified
//!    against hardware in the host-input phase, not guessed here.)
//! 3. **The guest window** — [`PadDevice`], an MMIO device exposing a
//!    sequence-numbered snapshot of the pad state the guest reads with
//!    a seqlock pattern (read seq, read state, re-read seq; retry
//!    while the seq moved). The host injects state with
//!    [`PadDevice::set_state`].
//!
//! The HID report 0x01 encoder covers the report subset whose layout is
//! unambiguous in the public documentation (report id, both button
//! bytes with the dpad hat, stick axes, analog triggers, PS/touchpad
//! buttons); touch and motion extensions are deliberately not
//! serialized until a real transport verifies their offsets.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pdg_memory::bus::{AccessWidth, MmioDevice};

use crate::input::{self, InputState};

/// D-pad hat values as carried in HID reports (clockwise from up).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dpad {
    /// No direction pressed.
    #[default]
    Neutral = 8,
    /// Up.
    Up = 0,
    /// Up + right.
    UpRight = 1,
    /// Right.
    Right = 2,
    /// Down + right.
    DownRight = 3,
    /// Down.
    Down = 4,
    /// Down + left.
    DownLeft = 5,
    /// Left.
    Left = 6,
    /// Up + left.
    UpLeft = 7,
}

impl Dpad {
    /// Hat value for the (up, right, down, left) press bits.
    #[must_use]
    pub fn from_pressed(up: bool, right: bool, down: bool, left: bool) -> Self {
        match (up, right, down, left) {
            (true, false, false, false) => Self::Up,
            (true, true, false, false) => Self::UpRight,
            (false, true, false, false) => Self::Right,
            (false, true, true, false) => Self::DownRight,
            (false, false, true, false) => Self::Down,
            (false, false, true, true) => Self::DownLeft,
            (false, false, false, true) => Self::Left,
            (true, false, false, true) => Self::UpLeft,
            _ => Self::Neutral,
        }
    }
}

/// One capacitive touch point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TouchPoint {
    /// Whether the finger is on the pad.
    pub active: bool,
    /// Tracking id (rotates per touch session).
    pub id: u8,
    /// X coordinate (0..=1919 on the DualSense pad).
    pub x: u16,
    /// Y coordinate (0..=1079).
    pub y: u16,
}

/// Full DualSense state snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DualSenseState {
    /// Digital buttons (PDG [`input::button`] bitfield; the digital
    /// L2/R2 bits reflect a hard press).
    pub buttons: u32,
    /// D-pad hat value.
    pub dpad: Dpad,
    /// Left stick `[x, y]`, 0–255, 128 centered.
    pub left_stick: [u8; 2],
    /// Right stick `[x, y]`.
    pub right_stick: [u8; 2],
    /// Left trigger pressure 0–255.
    pub l2: u8,
    /// Right trigger pressure 0–255.
    pub r2: u8,
    /// Touch points (index 0 = first finger).
    pub touch: [TouchPoint; 2],
    /// Gyroscope sample `[x, y, z]` (rad/s in the pad's sensor units —
    /// calibration lands with host-hardware verification).
    pub gyro: [i16; 3],
    /// Accelerometer sample `[x, y, z]`.
    pub accel: [i16; 3],
    /// Battery level, 0–100 percent.
    pub battery: u8,
}

impl DualSenseState {
    /// The neutral pad (everything released, sticks centered).
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            left_stick: [128, 128],
            right_stick: [128, 128],
            ..Self::default()
        }
    }
}

/// Adaptive-trigger effect modes, modeled by behavior.
///
/// `resistance_at(pull, t)` returns the resistive force the trigger
/// exerts (0.0–1.0) when pulled to fraction `pull` (0.0–1.0) at
/// time `t` (seconds, for the oscillating modes) — the physics a
/// player feels and a game's trigger-position polling observes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TriggerEffect {
    /// No adaptive resistance (the trigger's internal spring only).
    #[default]
    Off,
    /// Constant resistance after `start` fraction of the pull.
    Feedback {
        /// Pull fraction (0.0–1.0) where resistance begins.
        start: f32,
        /// Resistance magnitude (0.0–1.0).
        force: f32,
    },
    /// Weapon-style: soft until `start`, stiff until `end`, then the
    /// trigger breaks through (fires) and goes slack.
    Weapon {
        /// Pull fraction where resistance ramps up.
        start: f32,
        /// Pull fraction where the trigger breaks.
        end: f32,
        /// Resistance magnitude between start and end.
        force: f32,
    },
    /// Oscillating resistance (e.g., machine-gun recoil).
    Vibration {
        /// Resistance amplitude (0.0–1.0).
        amplitude: f32,
        /// Oscillation frequency in Hz.
        frequency: f32,
    },
}

impl TriggerEffect {
    /// Resistive force at pull fraction `pull`, time `t` seconds.
    #[must_use]
    pub fn resistance_at(&self, pull: f32, t: f32) -> f32 {
        let pull = pull.clamp(0.0, 1.0);
        match *self {
            Self::Off => 0.0,
            Self::Feedback { start, force } if pull >= start => force,
            Self::Feedback { .. } => 0.0,
            Self::Weapon { start, end, force } => {
                if pull < start {
                    0.0
                } else if pull < end {
                    // Ramp over the first 10% of the stiff zone, hold.
                    let zone = (pull - start) / (end - start).max(1e-3);
                    force * (zone / 0.1).min(1.0)
                } else {
                    // Broke through: slack.
                    0.0
                }
            }
            Self::Vibration {
                amplitude,
                frequency,
            } => {
                let phase = 2.0 * std::f32::consts::PI * frequency * t;
                (amplitude * (0.5 + 0.5 * phase.sin())).clamp(0.0, 1.0)
            }
        }
    }

    /// Wire mode byte for the host output path (public documentation:
    /// 0x00 off, 0x01 feedback, 0x02 weapon, 0x21 vibration, 0x22 sled).
    #[must_use]
    pub fn mode_byte(&self) -> u8 {
        match self {
            Self::Off => 0x00,
            Self::Feedback { .. } => 0x01,
            Self::Weapon { .. } => 0x02,
            Self::Vibration { .. } => 0x21,
        }
    }
}

/// Map a host [`InputState`] (normalized) onto a [`DualSenseState`]
/// with a stick deadzone.
///
/// Sticks pass through `[-1, 1]` → `[0, 255]` with `deadzone`
/// (0.0–0.5) around the center; triggers map `[0, 1]` → `[0, 255]`;
/// buttons map directly, and a trigger above `0.9` sets its digital
/// bit like a hard press does.
#[must_use]
pub fn map_host_input(host: &InputState, deadzone: f32) -> DualSenseState {
    let deadzone = deadzone.clamp(0.0, 0.5);
    let axis = |v: f32| -> u8 {
        let v = if v.abs() < deadzone { 0.0 } else { v };
        ((v * 127.0) + 128.0).round().clamp(0.0, 255.0) as u8
    };
    let mut buttons = host.buttons;
    if host.l2 >= 0.9 {
        buttons |= input::button::L2;
    }
    if host.r2 >= 0.9 {
        buttons |= input::button::R2;
    }
    DualSenseState {
        buttons,
        dpad: Dpad::from_pressed(
            host.buttons & input::button::DPAD_UP != 0,
            host.buttons & input::button::DPAD_RIGHT != 0,
            host.buttons & input::button::DPAD_DOWN != 0,
            host.buttons & input::button::DPAD_LEFT != 0,
        ),
        left_stick: [axis(host.left_stick[0]), axis(host.left_stick[1])],
        right_stick: [axis(host.right_stick[0]), axis(host.right_stick[1])],
        l2: (host.l2.clamp(0.0, 1.0) * 255.0).round() as u8,
        r2: (host.r2.clamp(0.0, 1.0) * 255.0).round() as u8,
        ..DualSenseState::neutral()
    }
}

/// HID input report 0x01 (the subset PDG serializes; see the module
/// docs for why touch/motion extensions are not encoded yet).
pub mod hid01 {
    /// Report identifier.
    pub const REPORT_ID: u8 = 0x01;
    /// Report size in bytes.
    pub const LEN: usize = 64;
    /// Byte 1: face buttons (bits 0-3) + dpad hat (bits 4-7).
    pub const BUTTONS0: usize = 1;
    /// Byte 2: L3/R3/L1/R1/L2/R2/create/options.
    pub const BUTTONS1: usize = 2;
    /// Byte 3: PS button, touchpad click, report counter.
    pub const BUTTONS2: usize = 3;
    /// Byte 4: left stick X (0-255).
    pub const LX: usize = 4;
    /// Byte 5: left stick Y.
    pub const LY: usize = 5;
    /// Byte 6: right stick X.
    pub const RX: usize = 6;
    /// Byte 7: right stick Y.
    pub const RY: usize = 7;
    /// Byte 8: L2 analog.
    pub const L2: usize = 8;
    /// Byte 9: R2 analog.
    pub const R2: usize = 9;
}

/// Encode the state into an HID report 0x01 (documented subset).
#[must_use]
pub fn encode_hid01(state: &DualSenseState, counter: u8) -> [u8; hid01::LEN] {
    let mut out = [0u8; hid01::LEN];
    out[0] = hid01::REPORT_ID;
    let b = state.buttons;
    // Face buttons in DualSense order: cross, circle, square, triangle.
    let mut b0 = 0u8;
    if b & input::button::CROSS != 0 {
        b0 |= 0x01;
    }
    if b & input::button::CIRCLE != 0 {
        b0 |= 0x02;
    }
    if b & input::button::SQUARE != 0 {
        b0 |= 0x04;
    }
    if b & input::button::TRIANGLE != 0 {
        b0 |= 0x08;
    }
    out[hid01::BUTTONS0] = b0 | ((state.dpad as u8 & 0x0F) << 4);
    let mut b1 = 0u8;
    if b & input::button::L3 != 0 {
        b1 |= 0x01;
    }
    if b & input::button::R3 != 0 {
        b1 |= 0x02;
    }
    if b & input::button::L1 != 0 {
        b1 |= 0x04;
    }
    if b & input::button::R1 != 0 {
        b1 |= 0x08;
    }
    if b & input::button::L2 != 0 {
        b1 |= 0x10;
    }
    if b & input::button::R2 != 0 {
        b1 |= 0x20;
    }
    if b & input::button::CREATE != 0 {
        b1 |= 0x40;
    }
    if b & input::button::OPTIONS != 0 {
        b1 |= 0x80;
    }
    out[hid01::BUTTONS1] = b1;
    out[hid01::BUTTONS2] = counter & 0x0F;
    out[hid01::LX] = state.left_stick[0];
    out[hid01::LY] = state.left_stick[1];
    out[hid01::RX] = state.right_stick[0];
    out[hid01::RY] = state.right_stick[1];
    out[hid01::L2] = state.l2;
    out[hid01::R2] = state.r2;
    out
}

/// MMIO window layout (dword offsets from the device base).
pub mod pad_regs {
    /// Magic "PDGP".
    pub const MAGIC: u64 = 0x00;
    /// Contract version.
    pub const VERSION: u64 = 0x04;
    /// Sequence number (bumped on every host state change).
    pub const SEQ: u64 = 0x08;
    /// Button bitfield.
    pub const BUTTONS: u64 = 0x10;
    /// D-pad hat value.
    pub const DPAD: u64 = 0x14;
    /// Left stick X.
    pub const LX: u64 = 0x18;
    /// Left stick Y.
    pub const LY: u64 = 0x1C;
    /// Right stick X.
    pub const RX: u64 = 0x20;
    /// Right stick Y.
    pub const RY: u64 = 0x24;
    /// L2 pressure.
    pub const L2: u64 = 0x28;
    /// R2 pressure.
    pub const R2: u64 = 0x2C;
    /// Touch 0: X with the active flag in bit 31.
    pub const T0X: u64 = 0x30;
    /// Touch 0: Y.
    pub const T0Y: u64 = 0x34;
    /// Touch 1: X with the active flag in bit 31.
    pub const T1X: u64 = 0x38;
    /// Touch 1: Y.
    pub const T1Y: u64 = 0x3C;
    /// Gyroscope `[x, y, z]`.
    pub const GYRO: u64 = 0x40;
    /// Accelerometer `[x, y, z]`.
    pub const ACCEL: u64 = 0x4C;
    /// Battery percentage.
    pub const BATTERY: u64 = 0x58;
    /// Total register footprint.
    pub const FOOTPRINT: u64 = 0x5C;
    /// Magic value: ASCII "PDGP".
    pub const MAGIC_VALUE: u32 = u32::from_le_bytes(*b"PDGP");
}

/// The guest-visible pad window (see [`pad_regs`] for the layout).
///
/// The host injects snapshots with [`PadDevice::set_state`], which
/// bumps the sequence number; the guest reads `SEQ`, reads the state
/// registers, re-reads `SEQ`, and retries if it moved — a seqlock.
pub struct PadDevice {
    state: Mutex<DualSenseState>,
    seq: AtomicU64,
}

impl PadDevice {
    /// Create a pad in the neutral state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(DualSenseState::neutral()),
            seq: AtomicU64::new(0),
        }
    }

    /// Wrap in `Arc` for bus registration.
    #[must_use]
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Host-side state injection (bumps the sequence).
    pub fn set_state(&self, state: DualSenseState) {
        self.seq.fetch_add(1, Ordering::Release);
        *self.state.lock().expect("pad lock poisoned") = state;
        self.seq.fetch_add(1, Ordering::Release);
    }

    /// Current state snapshot (host-side diagnostics/tests).
    pub fn state(&self) -> DualSenseState {
        self.state.lock().expect("pad lock poisoned").clone()
    }

    /// Current sequence number.
    pub fn seq(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }
}

impl Default for PadDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for PadDevice {
    fn name(&self) -> &'static str {
        "dualsense-pad"
    }

    fn read(&self, offset: u64, _width: AccessWidth) -> u64 {
        match offset {
            pad_regs::MAGIC => u64::from(pad_regs::MAGIC_VALUE),
            pad_regs::VERSION => 1,
            pad_regs::SEQ => self.seq.load(Ordering::Acquire),
            other => {
                let state = self.state.lock().expect("pad lock poisoned");
                match other {
                    pad_regs::BUTTONS => u64::from(state.buttons),
                    pad_regs::DPAD => u64::from(state.dpad as u8),
                    pad_regs::LX => u64::from(state.left_stick[0]),
                    pad_regs::LY => u64::from(state.left_stick[1]),
                    pad_regs::RX => u64::from(state.right_stick[0]),
                    pad_regs::RY => u64::from(state.right_stick[1]),
                    pad_regs::L2 => u64::from(state.l2),
                    pad_regs::R2 => u64::from(state.r2),
                    pad_regs::T0X => {
                        u64::from(state.touch[0].x)
                            | (u64::from(state.touch[0].active as u32) << 31)
                    }
                    pad_regs::T0Y => u64::from(state.touch[0].y),
                    pad_regs::T1X => {
                        u64::from(state.touch[1].x)
                            | (u64::from(state.touch[1].active as u32) << 31)
                    }
                    pad_regs::T1Y => u64::from(state.touch[1].y),
                    pad_regs::BATTERY => u64::from(state.battery),
                    pad_regs::GYRO..pad_regs::ACCEL => {
                        let i = ((other - pad_regs::GYRO) / 4) as usize;
                        u64::from(state.gyro.get(i).copied().unwrap_or(0) as u32)
                    }
                    pad_regs::ACCEL..pad_regs::BATTERY => {
                        let i = ((other - pad_regs::ACCEL) / 4) as usize;
                        u64::from(state.accel.get(i).copied().unwrap_or(0) as u32)
                    }
                    _ => 0,
                }
            }
        }
    }

    fn write(&self, _offset: u64, _width: AccessWidth, _value: u64) {
        // The pad window is read-only for the guest.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpad_hat_combines_directions() {
        assert_eq!(Dpad::from_pressed(true, false, false, false), Dpad::Up);
        assert_eq!(
            Dpad::from_pressed(false, true, true, false),
            Dpad::DownRight
        );
        assert_eq!(
            Dpad::from_pressed(false, false, false, false),
            Dpad::Neutral
        );
        assert_eq!(
            Dpad::from_pressed(true, true, true, true),
            Dpad::Neutral,
            "impossible diagonals stay neutral"
        );
    }

    #[test]
    fn feedback_resistance_follows_start_threshold() {
        let eff = TriggerEffect::Feedback {
            start: 0.5,
            force: 0.8,
        };
        assert_eq!(eff.resistance_at(0.0, 0.0), 0.0);
        assert_eq!(eff.resistance_at(0.49, 0.0), 0.0);
        assert_eq!(eff.resistance_at(0.5, 0.0), 0.8);
        assert_eq!(eff.resistance_at(1.0, 0.0), 0.8);
    }

    #[test]
    fn weapon_effect_breaks_through() {
        let eff = TriggerEffect::Weapon {
            start: 0.3,
            end: 0.7,
            force: 1.0,
        };
        assert_eq!(eff.resistance_at(0.1, 0.0), 0.0);
        assert!(eff.resistance_at(0.35, 0.0) > 0.0, "stiff zone engages");
        assert_eq!(eff.resistance_at(0.55, 0.0), 1.0);
        assert_eq!(eff.resistance_at(0.9, 0.0), 0.0, "broke through");
    }

    #[test]
    fn vibration_oscillates() {
        let eff = TriggerEffect::Vibration {
            amplitude: 1.0,
            frequency: 1.0,
        };
        let a = eff.resistance_at(0.5, 0.0);
        let b = eff.resistance_at(0.5, 0.25);
        let c = eff.resistance_at(0.5, 0.5);
        assert!((a - c).abs() < 1e-5, "periodic");
        assert!((b - a).abs() > 0.1, "quarter-period differs");
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn host_mapping_applies_deadzone_and_triggers() {
        let host = InputState {
            buttons: input::button::CROSS | input::button::DPAD_LEFT,
            left_stick: [0.05, -0.7], // below a 0.15 deadzone on x
            right_stick: [0.9, 0.0],
            l2: 0.5,
            r2: 1.0,
        };
        let ds = map_host_input(&host, 0.15);
        assert_eq!(ds.buttons & input::button::CROSS, input::button::CROSS);
        assert_eq!(ds.dpad, Dpad::Left);
        assert_eq!(ds.left_stick[0], 128, "inside deadzone stays centered");
        assert!(ds.left_stick[1] < 64, "y pressed down hard");
        assert!(ds.right_stick[0] > 230);
        assert_eq!(ds.l2, 128);
        assert_eq!(ds.r2, 255);
        assert_eq!(
            ds.buttons & input::button::R2,
            input::button::R2,
            "hard press bit"
        );
    }

    #[test]
    fn hid01_encodes_buttons_sticks_triggers() {
        let state = DualSenseState {
            buttons: input::button::CROSS
                | input::button::TRIANGLE
                | input::button::R1
                | input::button::OPTIONS,
            dpad: Dpad::Down,
            left_stick: [200, 100],
            right_stick: [10, 240],
            l2: 40,
            r2: 250,
            ..DualSenseState::neutral()
        };
        let rep = encode_hid01(&state, 3);
        assert_eq!(rep[0], hid01::REPORT_ID);
        // cross | triangle in low nibble; dpad Down (4) in the hat.
        assert_eq!(rep[hid01::BUTTONS0] & 0x0F, 0x01 | 0x08);
        assert_eq!(rep[hid01::BUTTONS0] >> 4, Dpad::Down as u8);
        assert_eq!(rep[hid01::BUTTONS1], 0x08 | 0x80, "R1 | options");
        assert_eq!(rep[hid01::BUTTONS2] & 0x0F, 3);
        assert_eq!(rep[hid01::LX], 200);
        assert_eq!(rep[hid01::LY], 100);
        assert_eq!(rep[hid01::RX], 10);
        assert_eq!(rep[hid01::RY], 240);
        assert_eq!(rep[hid01::L2], 40);
        assert_eq!(rep[hid01::R2], 250);
    }

    #[test]
    fn pad_device_seqlock_and_reads() {
        let pad = PadDevice::new();
        assert_eq!(
            pad.read(pad_regs::MAGIC, AccessWidth::W32),
            u64::from(pad_regs::MAGIC_VALUE)
        );
        assert_eq!(
            pad.read(pad_regs::LX, AccessWidth::W32),
            128,
            "neutral center"
        );
        let seq_before = pad.read(pad_regs::SEQ, AccessWidth::W64);
        pad.set_state(DualSenseState {
            buttons: input::button::CIRCLE,
            left_stick: [255, 0],
            l2: 200,
            touch: [
                TouchPoint {
                    active: true,
                    id: 1,
                    x: 500,
                    y: 250,
                },
                TouchPoint::default(),
            ],
            battery: 80,
            ..DualSenseState::neutral()
        });
        let seq_after = pad.read(pad_regs::SEQ, AccessWidth::W64);
        assert!(seq_after > seq_before, "seq bumped");
        assert_eq!(
            pad.read(pad_regs::BUTTONS, AccessWidth::W32),
            input::button::CIRCLE as u64
        );
        assert_eq!(pad.read(pad_regs::LX, AccessWidth::W32), 255);
        assert_eq!(pad.read(pad_regs::L2, AccessWidth::W32), 200);
        let t0x = pad.read(pad_regs::T0X, AccessWidth::W32);
        assert_eq!(t0x & 0x8000_0000, 0x8000_0000, "touch 0 active flag");
        assert_eq!(t0x & 0x7FFF, 500);
        assert_eq!(pad.read(pad_regs::BATTERY, AccessWidth::W32), 80);
        // Guest writes are ignored.
        pad.write(pad_regs::BUTTONS, AccessWidth::W32, 0xFFFF_FFFF);
        assert_eq!(
            pad.read(pad_regs::BUTTONS, AccessWidth::W32),
            input::button::CIRCLE as u64
        );
    }
}
