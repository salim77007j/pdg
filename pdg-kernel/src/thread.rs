//! Guest thread bookkeeping: the table behind `thr_create` / `thr_exit` /
//! `thr_self` and the `_umtx_op` park/wake machinery.
//!
//! The PS4/PS5 kernel derives from FreeBSD 9.2, and its thread syscalls
//! keep the FreeBSD numbers and shapes (verified against the public
//! `sys/kern/syscalls.master` of that release):
//!
//! * `thr_create(ucontext_t *ctx, long *id, int flags)` — 430
//! * `thr_exit(long *state)` — 431
//! * `thr_self(long *id)` — 432
//!
//! ## The `ucontext_t` layout (FreeBSD 9.2 amd64, public headers)
//!
//! `sigset_t uc_sigmask` (16 bytes: `_SIG_WORDS == 4`) at `+0x00`, then
//! `mcontext_t uc_mcontext` at `+0x10` whose register block PDG reads:
//! `mc_rdi` at `+0x18`, `mc_rip` at `+0xB0`, `mc_rflags` at `+0xC0`,
//! `mc_rsp` at `+0xC8`. These are the fields the kernel's
//! `create_thread` consumes; the file offsets are cited from
//! `sys/sys/ucontext.h` + `sys/amd64/include/ucontext.h` field order
//! with `__register_t` = 8 bytes.
//!
//! ## Thread ids
//!
//! PDG's documented model: the main thread is tid 1; `thr_create`
//! continues from 2 — the same shape the payload SDKs observe on real
//! firmware.

/// FreeBSD 9.2 amd64 `ucontext_t` field offsets `thr_create` consumes
/// (sources: the public 9.2 headers; see the module docs).
pub mod ucontext {
    /// `uc_mcontext.mc_rdi` — the thread's first argument.
    pub const MC_RDI: u64 = 0x18;
    /// `uc_mcontext.mc_rip` — the entry point.
    pub const MC_RIP: u64 = 0xB0;
    /// `uc_mcontext.mc_rflags` — initial flags.
    pub const MC_RFLAGS: u64 = 0xC0;
    /// `uc_mcontext.mc_rsp` — the initial stack pointer.
    pub const MC_RSP: u64 = 0xC8;
}

/// The main thread's id (documented PDG policy).
pub const MAIN_TID: u64 = 1;

/// A thread the kernel created but the session has not yet materialized
/// into an interpreter: everything `thr_create` needs to convey.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnedThread {
    /// The new thread's id.
    pub tid: u64,
    /// Entry point (`mc_rip`).
    pub rip: u64,
    /// Initial stack pointer (`mc_rsp`).
    pub rsp: u64,
    /// First argument (`mc_rdi`) — the single-argument convention PDG
    /// documents (the FreeBSD kernel copies the whole mcontext; PDG's
    /// threads receive `mc_rdi` because that is all the demo-shaped
    /// payload uses).
    pub rdi: u64,
    /// Initial RFLAGS (`mc_rflags`).
    pub rflags: u64,
    /// The new thread's TLS block base (its own copy of the aggregated
    /// template; `FS` is set to this).
    pub tls_base: u64,
}

/// What a thread is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    /// Runnable (or running).
    Runnable,
    /// Parked on a blocking primitive at the guest address `obj`
    /// (`_umtx_op` waits and mutex contention).
    Parked {
        /// The guest address being waited on.
        obj: u64,
    },
    /// Exited.
    Exited,
}

/// One thread's kernel-side metadata.
#[derive(Debug, Clone)]
pub struct ThreadMeta {
    /// Thread id.
    pub tid: u64,
    /// What it is doing.
    pub state: ThreadState,
    /// The value passed to `thr_exit` (None while alive).
    pub exit_code: Option<i64>,
}

/// The process thread table.
#[derive(Debug, Default, Clone)]
pub struct ThreadTable {
    next_tid: u64,
    threads: Vec<ThreadMeta>,
}

impl ThreadTable {
    /// A table with the main thread (tid 1) runnable.
    pub fn new() -> Self {
        Self {
            next_tid: MAIN_TID + 1,
            threads: vec![ThreadMeta {
                tid: MAIN_TID,
                state: ThreadState::Runnable,
                exit_code: None,
            }],
        }
    }

    /// The main thread's id.
    pub fn main_tid(&self) -> u64 {
        MAIN_TID
    }

    /// Register a spawned thread (from `thr_create`) and return its
    /// interpreter-creation record.
    pub fn create(
        &mut self,
        rip: u64,
        rsp: u64,
        rdi: u64,
        rflags: u64,
        tls_base: u64,
    ) -> SpawnedThread {
        let tid = self.next_tid;
        self.next_tid += 1;
        self.threads.push(ThreadMeta {
            tid,
            state: ThreadState::Runnable,
            exit_code: None,
        });
        SpawnedThread {
            tid,
            rip,
            rsp,
            rdi,
            rflags,
            tls_base,
        }
    }

    /// Look up a thread's metadata.
    pub fn get(&self, tid: u64) -> Option<&ThreadMeta> {
        self.threads.iter().find(|t| t.tid == tid)
    }

    /// Mark `tid` parked on `obj`.
    pub fn park(&mut self, tid: u64, obj: u64) -> bool {
        let Some(t) = self.threads.iter_mut().find(|t| t.tid == tid) else {
            return false;
        };
        match t.state {
            ThreadState::Runnable => {
                t.state = ThreadState::Parked { obj };
                true
            }
            _ => false,
        }
    }

    /// Wake every thread parked on `obj` (the `_umtx_op` WAKE and
    /// `thr_exit` discipline). Returns the woken tids in wait order.
    pub fn wake_all(&mut self, obj: u64) -> Vec<u64> {
        let mut woken = Vec::new();
        for t in self.threads.iter_mut() {
            if matches!(t.state, ThreadState::Parked { obj: o } if o == obj) {
                t.state = ThreadState::Runnable;
                woken.push(t.tid);
            }
        }
        woken
    }

    /// Wake the first thread parked on `obj` (the mutex handoff
    /// discipline: one waiter receives the lock). Returns its tid.
    pub fn wake_one(&mut self, obj: u64) -> Option<u64> {
        let hit = self
            .threads
            .iter()
            .position(|t| matches!(t.state, ThreadState::Parked { obj: o } if o == obj))?;
        self.threads[hit].state = ThreadState::Runnable;
        Some(self.threads[hit].tid)
    }

    /// Exit every thread at once (the process `exit` syscall: exit
    /// kills the process and all its threads).
    pub fn exit_all(&mut self) {
        for t in self.threads.iter_mut() {
            t.state = ThreadState::Exited;
            if t.exit_code.is_none() {
                t.exit_code = Some(0);
            }
        }
    }

    /// Mark `tid` exited with `code`; returns true when **no** live
    /// (runnable or parked) thread remains — the process exits.
    pub fn exit(&mut self, tid: u64, code: i64) -> bool {
        if let Some(t) = self.threads.iter_mut().find(|t| t.tid == tid) {
            t.state = ThreadState::Exited;
            t.exit_code = Some(code);
        }
        !self.threads.iter().any(|t| t.state != ThreadState::Exited)
    }

    /// Any live (runnable or parked) thread?
    pub fn any_live(&self) -> bool {
        self.threads.iter().any(|t| t.state != ThreadState::Exited)
    }

    /// Runnable tids, in creation order.
    pub fn runnable_tids(&self) -> Vec<u64> {
        self.threads
            .iter()
            .filter(|t| t.state == ThreadState::Runnable)
            .map(|t| t.tid)
            .collect()
    }

    /// Live (runnable or parked) tids, in creation order.
    pub fn live_tids(&self) -> Vec<u64> {
        self.threads
            .iter()
            .filter(|t| t.state != ThreadState::Exited)
            .map(|t| t.tid)
            .collect()
    }

    /// Is `tid` parked right now (used by the syscall hook to end the
    /// slice cleanly)?
    pub fn is_parked(&self, tid: u64) -> bool {
        self.get(tid)
            .is_some_and(|t| matches!(t.state, ThreadState::Parked { .. }))
    }

    /// The address `tid` is parked on, when it is parked.
    pub fn parked_on(&self, tid: u64) -> Option<u64> {
        self.get(tid).and_then(|t| match t.state {
            ThreadState::Parked { obj } => Some(obj),
            _ => None,
        })
    }

    /// Is `tid` exited?
    pub fn is_exited(&self, tid: u64) -> bool {
        self.get(tid)
            .is_some_and(|t| t.state == ThreadState::Exited)
    }
}

impl ThreadTable {
    /// Encode (little-endian, length-prefixed).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(&self.next_tid.to_le_bytes());
        o.extend_from_slice(&(self.threads.len() as u32).to_le_bytes());
        for t in &self.threads {
            o.extend_from_slice(&t.tid.to_le_bytes());
            match t.state {
                ThreadState::Runnable => o.extend_from_slice(&[0u8]),
                ThreadState::Parked { obj } => {
                    o.extend_from_slice(&[1u8]);
                    o.extend_from_slice(&obj.to_le_bytes());
                }
                ThreadState::Exited => o.extend_from_slice(&[2u8]),
            }
            match t.exit_code {
                Some(c) => {
                    o.extend_from_slice(&[1u8]);
                    o.extend_from_slice(&c.to_le_bytes());
                }
                None => o.extend_from_slice(&[0u8]),
            }
        }
        o
    }

    /// Decode [`ThreadTable::to_bytes`].
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        let rd_u64 = |b: &[u8], p: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let next_tid = rd_u64(b, &mut p)?;
        let n = u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        let mut threads = Vec::with_capacity(n);
        for _ in 0..n {
            let tid = rd_u64(b, &mut p)?;
            let state = match *b.get(p)? {
                0 => ThreadState::Runnable,
                1 => {
                    p += 1;
                    ThreadState::Parked {
                        obj: rd_u64(b, &mut p)?,
                    }
                }
                _ => {
                    p += 1;
                    ThreadState::Exited
                }
            };
            if matches!(state, ThreadState::Runnable) {
                p += 1;
            }
            let exit_code = if *b.get(p)? == 1 {
                p += 1;
                Some(i64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?))
            } else {
                p += 1;
                None
            };
            threads.push(ThreadMeta {
                tid,
                state,
                exit_code,
            });
        }
        Some(Self { next_tid, threads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_thread_is_tid_one_runnable() {
        let t = ThreadTable::new();
        assert_eq!(t.main_tid(), 1);
        assert_eq!(t.get(1).unwrap().state, ThreadState::Runnable);
        assert!(t.any_live());
    }

    #[test]
    fn create_park_wake_exit() {
        let mut t = ThreadTable::new();
        let sp = t.create(0x1000, 0x2000, 0x42, 0x202, 0x3000);
        assert_eq!(sp.tid, 2);
        assert_eq!(sp.rip, 0x1000);
        assert_eq!(sp.rdi, 0x42);
        assert_eq!(sp.tls_base, 0x3000);
        // Park both threads on the same object; wake_all returns both.
        assert!(t.park(1, 0x5000));
        assert!(t.park(2, 0x5000));
        assert_eq!(t.wake_all(0x5000), vec![1, 2]);
        assert!(!t.is_parked(1));
        // wake_one hands to exactly one.
        assert!(t.park(2, 0x6000));
        assert_eq!(t.wake_one(0x6000), Some(2));
        assert_eq!(t.wake_one(0x6000), None);
        // Exit: last live thread decides the process.
        assert!(!t.exit(2, 77)); // main still live
        assert_eq!(t.get(2).unwrap().exit_code, Some(77));
        assert!(t.exit(1, 0)); // nobody left
        assert!(!t.any_live());
        assert!(t.is_exited(1));
    }
}
