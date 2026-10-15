//! Input abstraction shared by all controller drivers.

/// Digital button bitmask layout (DualSense button order).
pub mod button {
    /// Cross.
    pub const CROSS: u32 = 1 << 0;
    /// Circle.
    pub const CIRCLE: u32 = 1 << 1;
    /// Square.
    pub const SQUARE: u32 = 1 << 2;
    /// Triangle.
    pub const TRIANGLE: u32 = 1 << 3;
    /// Options.
    pub const OPTIONS: u32 = 1 << 4;
    /// L1.
    pub const L1: u32 = 1 << 5;
    /// R1.
    pub const R1: u32 = 1 << 6;
    /// L2 (digital edge; analog value in [`InputState::l2`]).
    pub const L2: u32 = 1 << 7;
    /// R2 (digital edge; analog value in [`InputState::r2`]).
    pub const R2: u32 = 1 << 8;
    /// D-pad up.
    pub const DPAD_UP: u32 = 1 << 9;
    /// D-pad down.
    pub const DPAD_DOWN: u32 = 1 << 10;
    /// D-pad left.
    pub const DPAD_LEFT: u32 = 1 << 11;
    /// D-pad right.
    pub const DPAD_RIGHT: u32 = 1 << 12;
    /// L3 (stick click).
    pub const L3: u32 = 1 << 13;
    /// R3 (stick click).
    pub const R3: u32 = 1 << 14;
    /// Create.
    pub const CREATE: u32 = 1 << 15;
}

/// One controller's state, normalized to [-1.0, 1.0] axes and [0.0, 1.0]
/// triggers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    /// Digital button bitmask (see [`button`]).
    pub buttons: u32,
    /// Left stick: `[x, y]`.
    pub left_stick: [f32; 2],
    /// Right stick: `[x, y]`.
    pub right_stick: [f32; 2],
    /// Left trigger pressure.
    pub l2: f32,
    /// Right trigger pressure.
    pub r2: f32,
}

/// A polled input source. The Phase-4 DualSense driver maps host gamepads
/// (via the host gamepad stack) onto this trait.
pub trait InputSource: Send {
    /// Poll the current state.
    fn poll(&mut self) -> InputState;

    /// Driver name for diagnostics.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_defaults_neutral() {
        let s = InputState::default();
        assert_eq!(s.buttons, 0);
        assert_eq!(s.left_stick, [0.0, 0.0]);
        assert_eq!(s.l2, 0.0);
    }
}
