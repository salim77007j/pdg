//! Virtual clock and timed event scheduler.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Virtual clock ticking at `freq_hz`, anchored to host wall time.
///
/// Phase 1 runs free (guest time = host time scaled by the tick rate);
/// Phase 2 will pace the scheduler against the real host clock to keep
/// emulation speed honest.
#[derive(Debug)]
pub struct Clock {
    freq_hz: u64,
    origin: Instant,
}

impl Clock {
    /// Create a clock ticking at `freq_hz` starting now.
    pub fn new(freq_hz: u64) -> Self {
        Self {
            freq_hz,
            origin: Instant::now(),
        }
    }

    /// Virtual ticks elapsed since the clock started.
    pub fn now_ticks(&self) -> u64 {
        let nanos = self.origin.elapsed().as_nanos();
        (nanos * u64::from(self.freq_hz as u32) as u128 / 1_000_000_000u128) as u64
    }

    /// Host seconds elapsed since the clock started.
    pub fn host_elapsed_secs(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// Priority-queue scheduler firing one-shot callbacks at virtual tick
/// deadlines. Ties break in scheduling order.
pub struct Scheduler {
    events: BinaryHeap<Reverse<TimedEvent>>,
    seq: u64,
}

struct TimedEvent {
    tick: u64,
    seq: u64,
    callback: Box<dyn FnOnce() + Send>,
}

impl Eq for TimedEvent {}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.tick == other.tick && self.seq == other.seq
    }
}

impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Earlier (tick, seq) pops first.
        (self.tick, self.seq).cmp(&(other.tick, other.seq))
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("pending", &self.events.len())
            .finish()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self {
            events: BinaryHeap::new(),
            seq: 0,
        }
    }

    /// Schedule `callback` to fire at absolute virtual tick `tick`.
    pub fn schedule_at(&mut self, tick: u64, callback: Box<dyn FnOnce() + Send>) {
        self.events.push(Reverse(TimedEvent {
            tick,
            seq: self.seq,
            callback,
        }));
        self.seq += 1;
    }

    /// Schedule `callback` to fire `ticks` after the current deadline head
    /// (convenience wrapper for relative deadlines).
    pub fn schedule_after(&mut self, ticks: u64, callback: Box<dyn FnOnce() + Send>) {
        let base = self.events.peek().map_or(0, |Reverse(e)| e.tick);
        self.schedule_at(base + ticks, callback);
    }

    /// Number of pending events.
    pub fn pending(&self) -> usize {
        self.events.len()
    }

    /// Fire every event whose deadline is at or before `tick`, in order.
    pub fn run_until(&mut self, tick: u64) {
        while let Some(Reverse(event)) = self.events.peek() {
            if event.tick > tick {
                break;
            }
            let Reverse(event) = self.events.pop().expect("peek guaranteed a value");
            (event.callback)();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn fires_in_tick_order_with_stable_seq() {
        let mut sched = Scheduler::new();
        let (tx, rx) = mpsc::channel();
        sched.schedule_at(30, {
            let tx = tx.clone();
            Box::new(move || tx.send("c").unwrap())
        });
        sched.schedule_at(10, {
            let tx = tx.clone();
            Box::new(move || tx.send("a").unwrap())
        });
        sched.schedule_at(10, {
            let tx = tx.clone();
            Box::new(move || tx.send("b").unwrap())
        });
        sched.schedule_at(50, {
            let tx = tx.clone();
            Box::new(move || tx.send("d").unwrap())
        });

        sched.run_until(9);
        assert!(rx.try_recv().is_err()); // nothing yet

        sched.run_until(10);
        assert_eq!(rx.try_recv(), Ok("a"));
        assert_eq!(rx.try_recv(), Ok("b"));

        sched.run_until(u64::MAX);
        assert_eq!(rx.try_recv(), Ok("c"));
        assert_eq!(rx.try_recv(), Ok("d"));
        assert_eq!(sched.pending(), 0);
    }

    #[test]
    fn clock_monotonic() {
        let clock = Clock::new(3_500_000_000);
        let t0 = clock.now_ticks();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(clock.now_ticks() >= t0);
    }
}
