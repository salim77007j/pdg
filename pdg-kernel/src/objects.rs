//! Kernel synchronization objects: event flags, mutexes, semaphores and
//! event queues.
//!
//! These are the four object disciplines a console kernel hands to games.
//! The *semantics* implemented here are the publicly documented ones:
//!
//! * **Event flags** — 64-bit pattern registers with AND/OR wait modes and
//!   CLEAR_ALL / CLEAR_PAT consumption (`sceKernelEventFlag` shape,
//!   publicly documented by the homebrew community; single- and
//!   multi-wait attributes are modeled).
//! * **Mutexes** — owner-tracking, recursive, with priority notes left to
//!   the scheduler phase; `lock`/`try_lock`/`unlock` with EPERM on
//!   foreign unlock (the FreeBSD `umtx` discipline the Orbis pthread
//!   implementation builds on).
//! * **Semaphores** — counting with a max, EAGAIN on would-block and
//!   EOVERFLOW on over-post.
//! * **Event queues** — the kqueue-shaped queue Orbis `sceKernelCreateEqueue`
//!   is publicly documented to build on: a list of `(filter, ident, data)`
//!   events, `trigger` posts, `wait` drains.
//!
//! Everything is handle-indexed through [`ObjectTable`], and no object is
//! ever guessed: unknown handles and wrong-kind handles are refused with
//! precise errors.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as PlMutex, RwLock};

/// Errors from the kernel object layer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObjectError {
    /// The handle does not name any live object.
    #[error("no such kernel object (handle {handle})")]
    NoSuchObject {
        /// The unknown handle.
        handle: u64,
    },
    /// The handle names an object of a different kind.
    #[error("handle {handle} is a {actual}, not a {wanted}")]
    WrongKind {
        /// The handle.
        handle: u64,
        /// What the handle actually is.
        actual: &'static str,
        /// What the caller asked for.
        wanted: &'static str,
    },
    /// The waiter's requested pattern cannot be satisfied and blocking is
    /// impossible in the current execution model (single guest thread).
    #[error("would block waiting on handle {handle}")]
    WouldBlock {
        /// The object handle.
        handle: u64,
    },
    /// Foreign unlock of a mutex.
    #[error("mutex {handle} owned by {owner}, unlock by {caller} refused")]
    NotOwner {
        /// The mutex handle.
        handle: u64,
        /// Owning thread id.
        owner: u64,
        /// Calling thread id.
        caller: u64,
    },
    /// Semaphore post exceeds max.
    #[error("semaphore {handle} at max")]
    Overflow {
        /// The semaphore handle.
        handle: u64,
    },
    /// Event flag wait mode is invalid.
    #[error("invalid event flag wait mode {mode:#x}")]
    BadWaitMode {
        /// The raw mode bits.
        mode: u32,
    },
    /// Trigger fired on a queue that carries no matching event.
    #[error("no queued event with ident {ident}")]
    NoEvent {
        /// The looked-up event identity.
        ident: u64,
    },
}

/// The four object kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    /// 64-bit event flag.
    EventFlag,
    /// Owner-tracking recursive mutex.
    Mutex,
    /// Counting semaphore.
    Semaphore,
    /// kqueue-shaped event queue.
    Equeue,
}

impl ObjectKind {
    /// Name used in error messages.
    pub fn name(self) -> &'static str {
        match self {
            ObjectKind::EventFlag => "event flag",
            ObjectKind::Mutex => "mutex",
            ObjectKind::Semaphore => "semaphore",
            ObjectKind::Equeue => "equeue",
        }
    }
}

// ---------------------------------------------------------------------------
// Event flags
// ---------------------------------------------------------------------------

/// AND-only or OR-based matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitMode {
    /// All bits of the pattern must be set.
    And,
    /// Any bit of the pattern may be set.
    Or,
}

/// How the matched bits are consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearMode {
    /// Leave the pattern untouched.
    None,
    /// Clear every bit.
    All,
    /// Clear only the matched bits.
    Pattern,
}

/// An event flag's public knobs (attributes at creation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventFlagAttrs {
    /// Allow multiple waiters ( Orbis `EVF_ATTR_MULTI` shape ).
    pub multi_wait: bool,
}

/// A 64-bit event flag.
#[derive(Debug)]
pub struct EventFlag {
    /// Current pattern.
    pub pattern: u64,
    /// Creation attributes.
    pub attrs: EventFlagAttrs,
    /// Number of waiters currently blocked (host-side bookkeeping for the
    /// single-threaded execution model).
    pub waiters: u32,
}

impl EventFlag {
    /// A new flag with `pattern` and single-wait attributes.
    pub fn new(pattern: u64, attrs: EventFlagAttrs) -> Self {
        Self {
            pattern,
            attrs,
            waiters: 0,
        }
    }

    /// Set `bits` (OR into the pattern), returning the new pattern.
    pub fn set(&mut self, bits: u64) -> u64 {
        self.pattern |= bits;
        self.pattern
    }

    /// Clear `bits` (AND-out), returning the new pattern.
    pub fn clear(&mut self, bits: u64) -> u64 {
        self.pattern &= !bits;
        self.pattern
    }

    /// Check whether `pattern` satisfies a wait for `bits` under `mode`.
    pub fn satisfied(&self, bits: u64, mode: WaitMode) -> bool {
        match mode {
            WaitMode::And => self.pattern & bits == bits,
            WaitMode::Or => self.pattern & bits != 0,
        }
    }

    /// Consume per `clear` after a successful match; returns the pattern
    /// *before* consumption (the value a waiting thread observes).
    pub fn consume(&mut self, bits: u64, clear: ClearMode) -> u64 {
        let before = self.pattern;
        match clear {
            ClearMode::None => {}
            ClearMode::All => self.pattern = 0,
            ClearMode::Pattern => self.pattern &= !bits,
        }
        before
    }

    /// Poll-and-consume in one step; `None` when not satisfied yet.
    pub fn poll(&mut self, bits: u64, mode: WaitMode, clear: ClearMode) -> Option<u64> {
        if self.satisfied(bits, mode) {
            Some(self.consume(bits, clear))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

/// A recursive, owner-tracking mutex.
#[derive(Debug)]
pub struct KMutex {
    /// Owning thread id, 0 = unlocked.
    pub owner: u64,
    /// Recursion depth.
    pub depth: u32,
    /// Name given at creation (diagnostics).
    pub name: String,
}

impl KMutex {
    /// An unlocked mutex.
    pub fn new(name: &str) -> Self {
        Self {
            owner: 0,
            depth: 0,
            name: name.to_owned(),
        }
    }

    /// Lock: `tid == owner` recurses; unlocked locks; anything else reports
    /// [`ObjectError::WouldBlock`] (in the single-threaded guest, another
    /// thread holding the lock cannot make progress, so blocking is a
    /// deadlock — the kernel refuses honestly).
    pub fn lock(&mut self, tid: u64) -> Result<(), ObjectError> {
        if self.owner == 0 {
            self.owner = tid;
            self.depth = 1;
            Ok(())
        } else if self.owner == tid {
            self.depth = self
                .depth
                .checked_add(1)
                .ok_or(ObjectError::Overflow { handle: 0 })?;
            Ok(())
        } else {
            Err(ObjectError::WouldBlock { handle: 0 })
        }
    }

    /// Try-lock: never blocks.
    pub fn try_lock(&mut self, tid: u64) -> Result<(), ObjectError> {
        if self.owner == 0 || self.owner == tid {
            self.lock(tid)
        } else {
            Err(ObjectError::WouldBlock { handle: 0 })
        }
    }

    /// Unlock: only the owner may; returns EPERM-shaped error otherwise.
    pub fn unlock(&mut self, tid: u64) -> Result<(), ObjectError> {
        if self.owner == tid {
            self.depth -= 1;
            if self.depth == 0 {
                self.owner = 0;
            }
            Ok(())
        } else if self.owner == 0 {
            Err(ObjectError::NotOwner {
                handle: 0,
                owner: 0,
                caller: tid,
            })
        } else {
            Err(ObjectError::NotOwner {
                handle: 0,
                owner: self.owner,
                caller: tid,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Semaphore
// ---------------------------------------------------------------------------

/// A counting semaphore.
#[derive(Debug)]
pub struct Semaphore {
    /// Current count.
    pub value: u32,
    /// Maximum count.
    pub max: u32,
}

impl Semaphore {
    /// A semaphore starting at `value` with capacity `max`.
    pub fn new(value: u32, max: u32) -> Self {
        Self { value, max }
    }

    /// Wait (decrement); `None` when the count is zero (would block).
    pub fn wait(&mut self) -> Result<(), ObjectError> {
        if self.value == 0 {
            Err(ObjectError::WouldBlock { handle: 0 })
        } else {
            self.value -= 1;
            Ok(())
        }
    }

    /// Try-wait: identical to wait in this non-blocking model.
    pub fn try_wait(&mut self) -> Result<(), ObjectError> {
        self.wait()
    }

    /// Post (increment); refuses past max with EOVERFLOW semantics.
    pub fn post(&mut self) -> Result<(), ObjectError> {
        if self.value >= self.max {
            Err(ObjectError::Overflow { handle: 0 })
        } else {
            self.value += 1;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Event queue (kqueue-shaped)
// ---------------------------------------------------------------------------

/// One queued kernel event (the shape Orbis equeues expose).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KEvent {
    /// Identifying token (the Orbis event id / kernel object handle).
    pub ident: u64,
    /// Filter that raised it (EVFILT_USER-style tag).
    pub filter: i16,
    /// Filter-specific flags.
    pub fflags: u32,
    /// Payload (the u64 Orbis equeue events carry).
    pub data: u64,
}

/// An event queue: posted events wait to be drained.
#[derive(Debug, Default)]
pub struct Equeue {
    /// Posted, undrained events (FIFO).
    queue: Vec<KEvent>,
}

impl Equeue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Post an event.
    pub fn trigger(&mut self, ev: KEvent) {
        self.queue.push(ev);
    }

    /// Number of queued events.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// True when no events are queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Drain up to `out.len()` events into `out`; returns the count copied.
    pub fn drain(&mut self, out: &mut [KEvent]) -> usize {
        let n = out.len().min(self.queue.len());
        out[..n].copy_from_slice(&self.queue[..n]);
        self.queue.drain(..n);
        n
    }

    /// Drain every queued event.
    pub fn drain_all(&mut self) -> Vec<KEvent> {
        std::mem::take(&mut self.queue)
    }
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// A live kernel object.
#[derive(Debug)]
pub enum KernelObject {
    /// Event flag.
    EventFlag(EventFlag),
    /// Mutex.
    Mutex(KMutex),
    /// Semaphore.
    Semaphore(Semaphore),
    /// Event queue.
    Equeue(Equeue),
}

impl KernelObject {
    /// The kind tag.
    pub fn kind(&self) -> ObjectKind {
        match self {
            KernelObject::EventFlag(_) => ObjectKind::EventFlag,
            KernelObject::Mutex(_) => ObjectKind::Mutex,
            KernelObject::Semaphore(_) => ObjectKind::Semaphore,
            KernelObject::Equeue(_) => ObjectKind::Equeue,
        }
    }
}

/// Handle-indexed object table. Handles are nonzero, never reused while
/// their object lives, and allocated in creation order.
#[derive(Default)]
pub struct ObjectTable {
    next: u64,
    objects: HashMap<u64, KernelObject>,
}

impl std::fmt::Debug for ObjectTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectTable")
            .field("objects", &self.objects.len())
            .finish()
    }
}

impl ObjectTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `obj`, returning its handle.
    pub fn insert(&mut self, obj: KernelObject) -> u64 {
        self.next += 1;
        let h = self.next;
        self.objects.insert(h, obj);
        h
    }

    /// Remove a handle, returning the object (or an error).
    pub fn remove(&mut self, handle: u64) -> Result<KernelObject, ObjectError> {
        self.objects
            .remove(&handle)
            .ok_or(ObjectError::NoSuchObject { handle })
    }

    /// Number of live objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// True when the table is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Read access with a kind check.
    pub fn get(&self, handle: u64, kind: ObjectKind) -> Result<&KernelObject, ObjectError> {
        let obj = self
            .objects
            .get(&handle)
            .ok_or(ObjectError::NoSuchObject { handle })?;
        if obj.kind() != kind {
            return Err(ObjectError::WrongKind {
                handle,
                actual: obj.kind().name(),
                wanted: kind.name(),
            });
        }
        Ok(obj)
    }

    /// Write access with a kind check.
    pub fn get_mut(
        &mut self,
        handle: u64,
        kind: ObjectKind,
    ) -> Result<&mut KernelObject, ObjectError> {
        let obj = self
            .objects
            .get_mut(&handle)
            .ok_or(ObjectError::NoSuchObject { handle })?;
        let actual = obj.kind();
        if actual != kind {
            return Err(ObjectError::WrongKind {
                handle,
                actual: actual.name(),
                wanted: kind.name(),
            });
        }
        Ok(obj)
    }

    /// Typed event-flag accessor.
    pub fn event_flag(&mut self, handle: u64) -> Result<&mut EventFlag, ObjectError> {
        match self.get_mut(handle, ObjectKind::EventFlag)? {
            KernelObject::EventFlag(f) => Ok(f),
            _ => unreachable!("kind check passed"),
        }
    }

    /// Typed mutex accessor.
    pub fn mutex(&mut self, handle: u64) -> Result<&mut KMutex, ObjectError> {
        match self.get_mut(handle, ObjectKind::Mutex)? {
            KernelObject::Mutex(m) => Ok(m),
            _ => unreachable!("kind check passed"),
        }
    }

    /// Typed semaphore accessor.
    pub fn semaphore(&mut self, handle: u64) -> Result<&mut Semaphore, ObjectError> {
        match self.get_mut(handle, ObjectKind::Semaphore)? {
            KernelObject::Semaphore(s) => Ok(s),
            _ => unreachable!("kind check passed"),
        }
    }

    /// Typed equeue accessor.
    pub fn equeue(&mut self, handle: u64) -> Result<&mut Equeue, ObjectError> {
        match self.get_mut(handle, ObjectKind::Equeue)? {
            KernelObject::Equeue(q) => Ok(q),
            _ => unreachable!("kind check passed"),
        }
    }
}

/// Thread-safe shared object table (the syscall layer and the device pump
/// both touch queues).
pub type SharedObjectTable = Arc<RwLock<ObjectTable>>;

/// A shared mutex for the syscall layer's inner state.
pub type Shared<T> = Arc<PlMutex<T>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_flag_or_and_clear() {
        let mut f = EventFlag::new(0, EventFlagAttrs { multi_wait: false });
        assert!(!f.satisfied(0b1010, WaitMode::Or));
        f.set(0b1000);
        assert!(f.satisfied(0b1010, WaitMode::Or));
        assert!(!f.satisfied(0b1010, WaitMode::And));
        f.set(0b0010);
        assert!(f.satisfied(0b1010, WaitMode::And));
        // Consume pattern: only matched bits clear.
        let before = f.consume(0b0010, ClearMode::Pattern);
        assert_eq!(before, 0b1010);
        assert_eq!(f.pattern, 0b1000);
        let before = f.consume(0, ClearMode::All);
        assert_eq!(before, 0b1000);
        assert_eq!(f.pattern, 0);
    }

    #[test]
    fn mutex_owner_and_recursion() {
        let mut m = KMutex::new("test");
        m.lock(7).unwrap();
        m.lock(7).unwrap();
        assert!(matches!(m.lock(9), Err(ObjectError::WouldBlock { .. })));
        assert!(matches!(m.unlock(9), Err(ObjectError::NotOwner { .. })));
        m.unlock(7).unwrap();
        m.unlock(7).unwrap();
        assert_eq!(m.owner, 0);
        assert!(matches!(m.unlock(7), Err(ObjectError::NotOwner { .. })));
    }

    #[test]
    fn semaphore_counting() {
        let mut s = Semaphore::new(1, 2);
        s.wait().unwrap();
        assert!(matches!(s.try_wait(), Err(ObjectError::WouldBlock { .. })));
        s.post().unwrap();
        s.post().unwrap();
        assert!(matches!(s.post(), Err(ObjectError::Overflow { .. })));
    }

    #[test]
    fn equeue_fifo() {
        let mut q = Equeue::new();
        q.trigger(KEvent {
            ident: 1,
            filter: -1,
            fflags: 0,
            data: 10,
        });
        q.trigger(KEvent {
            ident: 2,
            filter: -1,
            fflags: 0,
            data: 20,
        });
        let mut out = [KEvent {
            ident: 0,
            filter: 0,
            fflags: 0,
            data: 0,
        }; 1];
        assert_eq!(q.drain(&mut out), 1);
        assert_eq!(out[0].data, 10);
        assert_eq!(q.len(), 1);
        assert_eq!(q.drain_all().len(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn table_kind_discipline() {
        let mut t = ObjectTable::new();
        let m = t.insert(KernelObject::Mutex(KMutex::new("m")));
        let q = t.insert(KernelObject::Equeue(Equeue::new()));
        assert_eq!(t.len(), 2);
        assert!(t.get(m, ObjectKind::Mutex).is_ok());
        assert!(matches!(
            t.get(m, ObjectKind::Equeue),
            Err(ObjectError::WrongKind { .. })
        ));
        assert!(matches!(
            t.remove(999),
            Err(ObjectError::NoSuchObject { handle: 999 })
        ));
        t.remove(m).unwrap();
        assert!(matches!(
            t.get_mut(q, ObjectKind::Mutex),
            Err(ObjectError::WrongKind { .. })
        ));
    }
}
