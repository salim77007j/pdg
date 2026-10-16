//! Typed publish/subscribe event bus for inter-component notifications.

use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};

/// Events emitted by the emulation components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The machine started executing at `entry`.
    BootStarted {
        /// Entry point RIP.
        entry: u64,
    },
    /// The guest executed `hlt` after `instructions` retired instructions.
    Halted {
        /// Total instructions retired.
        instructions: u64,
    },
    /// A guest-visible fault stopped execution.
    Fault {
        /// Rendered fault description.
        reason: String,
    },
    /// A DMA transfer completed.
    DmaCompleted {
        /// Transfer id.
        id: u64,
        /// Source guest-physical address.
        src: u64,
        /// Destination guest-physical address.
        dst: u64,
        /// Transfer length in bytes.
        len: u64,
    },
    /// An inter-processor interrupt was delivered (SMP machines).
    IpiSent {
        /// Destination logical core.
        target: u8,
        /// Vector delivered.
        vector: u8,
    },
    /// A storage I/O completed through the NVMe model (Phase 4).
    BlockIo {
        /// True for WRITE, false for READ.
        write: bool,
        /// First LBA of the transfer.
        lba: u64,
        /// Blocks transferred.
        blocks: u32,
    },
    /// The guest hydrated an audio block into the mixer (Phase 4).
    AudioHydrated {
        /// Samples hydrated.
        samples: u32,
    },
}

/// Broadcast bus. Subscribers receive a copy of every published event.
///
/// Phase 1 uses this for observable milestones; phases 3–5 will route GPU
/// fence signals and interrupt delivery through the same mechanism.
#[derive(Debug, Default)]
pub struct EventBus {
    subscribers: Mutex<Vec<Sender<Event>>>,
}

impl EventBus {
    /// Create an empty bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe; returns the receiving end of a fresh channel.
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = channel();
        self.subscribers
            .lock()
            .expect("event bus lock poisoned")
            .push(tx);
        rx
    }

    /// Publish an event to all live subscribers; dead ones are pruned.
    pub fn publish(&self, event: Event) {
        let mut subs = self.subscribers.lock().expect("event bus lock poisoned");
        subs.retain(|tx| tx.send(event.clone()).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribers_receive_events_in_order() {
        let bus = EventBus::new();
        let rx = bus.subscribe();
        bus.publish(Event::BootStarted { entry: 0x1000 });
        bus.publish(Event::Halted { instructions: 5 });
        assert_eq!(rx.recv().unwrap(), Event::BootStarted { entry: 0x1000 });
        assert_eq!(rx.recv().unwrap(), Event::Halted { instructions: 5 });
    }

    #[test]
    fn dead_subscribers_are_pruned() {
        let bus = EventBus::new();
        let rx = bus.subscribe();
        drop(rx);
        bus.publish(Event::Fault { reason: "x".into() });
        assert_eq!(bus.subscribers.lock().unwrap().len(), 0);
    }
}
