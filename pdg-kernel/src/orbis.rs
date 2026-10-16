//! The Orbis syscall layer: HLE translation of the PS4/PS5 kernel's
//! public syscall surface.
//!
//! The PS5 kernel ("Orbis OS") derives from FreeBSD 9.2, and its syscall
//! *numbers* follow the FreeBSD 9.2 amd64 table (verified against the
//! public `sys/kern/syscalls.master` of that release) plus a Sony-defined
//! block from 588 upward (verified against the public psdevwiki-derived
//! tables). PDG implements the subset whose behavior it can genuinely
//! back; everything else — implemented or not — is refused through the
//! same return convention, with the number and (when known) the name
//! logged. Nothing is guessed into existence.
//!
//! ## Calling convention (FreeBSD raw ABI, public)
//!
//! number in `RAX`, arguments in `RDI`, `RSI`, `RDX`, `R10`, `R8`, `R9`;
//! on success `CF = 0` and `RAX` holds the result; on failure `CF = 1`
//! and `RAX` holds a positive `errno`. This is the raw-kernel ABI the
//! Orbis `libkernel` wrappers sit on (`jc` after `syscall`).
//!
//! ## What Phase 5 implements
//!
//! * process: `exit`, `getpid`
//! * console/file I/O over the mounted read-only PFS: `read`, `write`,
//!   `open`, `close`, `lseek`, `getdirentries`
//! * memory: `mmap` (anonymous), `munmap`, `mprotect`
//! * event queues: `kqueue`, `kevent` (registration + non-blocking drain)
//!
//! ## What Phase 6 adds
//!
//! * `stat`/`fstat` (188/189) — the FreeBSD 9.2 amd64 `struct stat`
//!   layout, 120 bytes, verified against two independent public sources
//!   (the FreeBSD 9.2 headers and the OpenOrbis `OrbisKernelStat`, which
//!   agree field-for-field).
//! * threads: `thr_create` (430, the FreeBSD `ucontext_t` shape),
//!   `thr_exit` (431), `thr_self` (432) over the [`crate::thread`] table,
//!   with per-thread TLS copies of the aggregated template.
//! * `_umtx_op` (454): the FreeBSD umtx discipline — legacy `LOCK`/
//!   `UNLOCK`, `WAIT`/`WAKE`, and `MUTEX_TRYLOCK`/`MUTEX_LOCK`/
//!   `MUTEX_UNLOCK` with real parking (a would-block thread parks until
//!   another thread wakes it; mutex unlock hands the lock to one waiter,
//!   FIFO). Timed waits are refused with `ENOSYS`, logged — the honest
//!   unimplemented-primitive marker.
//! * the Sony dynlib block whose ABI shapes the public payload SDKs pin:
//!   `dlsym` (591: `handle, name, &out`), `get_list` (592: `int*, max,
//!   int*`), `get_info` (593: `handle, &info`, size-prefixed),
//!   `load_prx` (594: `name, 0, &id, 0`), `unload_prx` (595: `id, 0, 0`)
//!   — over the [`crate::dynlib`] module table. The rest of the Sony
//!   block stays refused by name, including `get_proc_param` (598) and
//!   `process_needed_and_relocate` (599): no public source pins their
//!   ABI shapes, and PDG's `load_prx` performs needed-loading and
//!   relocation eagerly instead — documented, not guessed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pdg_cpu::{Fault, SyscallHook};
use pdg_memory::MmioDevice;
use pdg_memory::bus::AccessWidth;
use pdg_memory::devices::ConsoleDevice;
use tracing::warn;

use crate::objects::{Equeue, KEvent, KernelObject, ObjectTable};
use crate::pfs::{Pfs, dirent_type};
use crate::syscall::ARG_REGISTERS;
use crate::vm::VmManager;

// ---------------------------------------------------------------------------
// Verified syscall numbers
// ---------------------------------------------------------------------------

/// Syscall numbers, verified against FreeBSD 9.2 `sys/kern/syscalls.master`
/// (the PS4/PS5 kernel's documented base) and the public Sony tables.
pub mod nr {
    /// `exit(int code)` — FreeBSD 1.
    pub const EXIT: u64 = 1;
    /// `read(int fd, void *buf, size_t len)` — FreeBSD 3.
    pub const READ: u64 = 3;
    /// `write(int fd, const void *buf, size_t len)` — FreeBSD 4.
    pub const WRITE: u64 = 4;
    /// `open(const char *path, int flags, ...)` — FreeBSD 5.
    pub const OPEN: u64 = 5;
    /// `close(int fd)` — FreeBSD 6.
    pub const CLOSE: u64 = 6;
    /// `getpid(void)` — FreeBSD 20.
    pub const GETPID: u64 = 20;
    /// `munmap(void *addr, size_t len)` — FreeBSD 73.
    pub const MUNMAP: u64 = 73;
    /// `mprotect(const void *addr, size_t len, int prot)` — FreeBSD 74.
    pub const MPROTECT: u64 = 74;
    /// `getdirentries(int fd, char *buf, size_t nbytes, off_t *basep)` —
    /// FreeBSD 196.
    pub const GETDIRENTRIES: u64 = 196;
    /// `kqueue(void)` — FreeBSD 362.
    pub const KQUEUE: u64 = 362;
    /// `kevent(int kq, const struct kevent *, int, struct kevent *, int,
    /// const struct timespec *)` — FreeBSD 363.
    pub const KEVENT: u64 = 363;
    /// `mmap(void *addr, size_t len, int prot, int flags, int fd,
    /// off_t pos)` — FreeBSD 477.
    pub const MMAP: u64 = 477;
    /// `lseek(int fd, off_t off, int whence)` — FreeBSD 478.
    pub const LSEEK: u64 = 478;
    /// `stat(const char *path, struct stat *sb)` — FreeBSD 188.
    pub const STAT: u64 = 188;
    /// `fstat(int fd, struct stat *sb)` — FreeBSD 189.
    pub const FSTAT: u64 = 189;
    /// `thr_create(ucontext_t *ctx, long *id, int flags)` — FreeBSD 430.
    pub const THR_CREATE: u64 = 430;
    /// `thr_exit(long *state)` — FreeBSD 431.
    pub const THR_EXIT: u64 = 431;
    /// `thr_self(long *id)` — FreeBSD 432.
    pub const THR_SELF: u64 = 432;
    /// `_umtx_op(void *obj, int op, u_long val, void *uaddr1,
    /// void *uaddr2)` — FreeBSD 454.
    pub const UMTX_OP: u64 = 454;
    /// `sys_dynlib_dlsym(handle, name, &out)` — Sony 591, shape pinned
    /// by the public payload SDKs.
    pub const DYNLIB_DLSYM: u64 = 591;
    /// `sys_dynlib_get_list(int *ids, int max, int *count)` — Sony 592.
    pub const DYNLIB_GET_LIST: u64 = 592;
    /// `sys_dynlib_get_info(handle, &info)` — Sony 593, size-prefixed
    /// struct.
    pub const DYNLIB_GET_INFO: u64 = 593;
    /// `sys_dynlib_load_prx(name, 0, &id, 0)` — Sony 594.
    pub const DYNLIB_LOAD_PRX: u64 = 594;
    /// `sys_dynlib_unload_prx(id, 0, 0)` — Sony 595.
    pub const DYNLIB_UNLOAD_PRX: u64 = 595;
}

/// Names of the Sony-defined syscall block (588–672), from the public
/// psdevwiki-derived tables. Phase 6 implements the dynlib subset whose
/// shapes the public payload SDKs pin (591–595); the rest is logged on
/// refusal so guests and developers see exactly what was asked for.
pub fn sony_name(nr: u64) -> Option<&'static str> {
    Some(match nr {
        588 => "mname",
        589 => "dynlib_dlopen",
        590 => "dynlib_dlclose",
        591 => "dynlib_dlsym",
        592 => "dynlib_get_list",
        593 => "dynlib_get_info",
        594 => "dynlib_load_prx",
        595 => "dynlib_unload_prx",
        596 => "dynlib_do_copy_relocations",
        597 => "dynlib_prepare_dlclose",
        598 => "dynlib_get_proc_param",
        599 => "dynlib_process_needed_and_relocate",
        600 => "sandbox_path",
        601 => "mdbg_service",
        602 => "randomized_path",
        603 => "rdup",
        604 => "dl_get_metadata",
        605 => "workaround8849",
        606 => "is_development_mode",
        607 => "get_self_auth_info",
        608 => "dynlib_get_info_ex",
        609 => "budget_getid",
        610 => "budget_get_ptype",
        611 => "get_paging_stats_of_all_threads",
        612 => "get_proc_type_info",
        613 => "get_resident_count",
        614 => "prepare_to_suspend_process",
        615 => "get_resident_fmem_count",
        616 => "thr_get_name",
        617 => "set_gpo",
        618 => "get_paging_stats_of_all_objects",
        619 => "test_debug_rwmem",
        620 => "free_stack",
        621 => "suspend_system",
        622 => "ipmimgr_call",
        623 => "get_gpo",
        624 => "get_vm_map_timestamp",
        625 => "opmc_set_hw",
        626 => "opmc_get_hw",
        627 => "get_cpu_usage_all",
        628 => "mmap_dmem",
        629 => "physhm_open",
        630 => "physhm_unlink",
        631 => "resume_internal_hdd",
        632 => "thr_suspend_ucontext",
        633 => "thr_resume_ucontext",
        634 => "thr_get_ucontext",
        635 => "thr_set_ucontext",
        636 => "set_timezone_info",
        637 => "set_phys_fmem_limit",
        638 => "utc_to_localtime",
        639 => "localtime_to_utc",
        640 => "set_uevt",
        641 => "get_cpu_usage_proc",
        642 => "get_map_statistics",
        643 => "set_chicken_switches",
        644 => "extend_page_table_pool",
        646 => "get_kernel_mem_statistics",
        647 => "get_sdk_compiled_version",
        648 => "app_state_change",
        649 => "dynlib_get_obj_member",
        650 => "budget_get_ptype_of_budget",
        651 => "prepare_to_resume_process",
        652 => "process_terminate",
        653 => "blockpool_open",
        654 => "blockpool_map",
        655 => "blockpool_unmap",
        656 => "dynlib_get_info_for_libdbg",
        657 => "blockpool_batch",
        658 => "fdatasync",
        659 => "dynlib_get_list2",
        660 => "dynlib_get_info2",
        661 => "aio_submit",
        662 => "aio_multi_delete",
        663 => "aio_multi_wait",
        664 => "aio_multi_poll",
        665 => "aio_get_data",
        666 => "aio_multi_cancel",
        667 => "get_bio_usage_all",
        668 => "aio_create",
        669 => "aio_submit_cmd",
        670 => "aio_init",
        671 => "get_page_table_stats",
        672 => "dynlib_get_list_for_libdbg",
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Public constants the handlers share with the demo builder
// ---------------------------------------------------------------------------

/// FreeBSD errno values (public).
pub mod errno {
    /// Operation not permitted.
    pub const EPERM: i32 = 1;
    /// No such file or directory.
    pub const ENOENT: i32 = 2;
    /// No such process (PDG's unknown-handle errno, documented).
    pub const ESRCH: i32 = 3;
    /// Interrupted system call.
    pub const EINTR: i32 = 4;
    /// I/O error.
    pub const EIO: i32 = 5;
    /// Resource temporarily unavailable (FreeBSD EAGAIN).
    pub const EAGAIN: i32 = 11;
    /// Exec format error.
    pub const ENOEXEC: i32 = 8;
    /// Bad file descriptor.
    pub const EBADF: i32 = 9;
    /// Out of memory.
    pub const ENOMEM: i32 = 12;
    /// Permission denied.
    pub const EACCES: i32 = 13;
    /// Bad address.
    pub const EFAULT: i32 = 14;
    /// Device busy.
    pub const EBUSY: i32 = 16;
    /// Device not configured.
    pub const ENODEV: i32 = 19;
    /// Not a directory.
    pub const ENOTDIR: i32 = 20;
    /// Is a directory.
    pub const EISDIR: i32 = 21;
    /// Invalid argument.
    pub const EINVAL: i32 = 22;
    /// Too many open files.
    pub const EMFILE: i32 = 24;
    /// Read-only filesystem.
    pub const EROFS: i32 = 30;
    /// Resource deadlock avoided (FreeBSD 45).
    pub const EDEADLK: i32 = 45;
    /// Too many levels of symbolic links (FreeBSD 62).
    pub const ELOOP: i32 = 62;
    /// Function not implemented.
    pub const ENOSYS: i32 = 78;
}

/// `open` flags (FreeBSD, public).
pub mod open_flags {
    /// Access-mode mask.
    pub const O_ACCMODE: i32 = 0x3;
}

/// `lseek` whence values (public).
pub mod whence {
    /// Seek from the start.
    pub const SEEK_SET: i64 = 0;
    /// Seek from the current position.
    pub const SEEK_CUR: i64 = 1;
    /// Seek from the end (size query).
    pub const SEEK_END: i64 = 2;
}

/// `mmap` protections (public).
pub mod prot {
    /// No access.
    pub const PROT_NONE: i32 = 0;
    /// Readable.
    pub const PROT_READ: i32 = 1;
    /// Writable.
    pub const PROT_WRITE: i32 = 2;
    /// Executable.
    pub const PROT_EXEC: i32 = 4;
}

/// `mmap` flags (FreeBSD, public).
pub mod mmap_flags {
    /// Changes are shared.
    pub const MAP_SHARED: u64 = 0x01;
    /// Changes are private.
    pub const MAP_PRIVATE: u64 = 0x02;
    /// Interpret `addr` exactly.
    pub const MAP_FIXED: u64 = 0x10;
    /// Anonymous memory (no file backing).
    pub const MAP_ANON: u64 = 0x1000;
}

/// `kevent` flags (public).
pub mod ev {
    /// Add the event to the queue.
    pub const EV_ADD: u16 = 0x0001;
    /// Clear the event after reporting (always how PDG drains).
    pub const EV_CLEAR: u16 = 0x0020;
    /// Register interest in reading (`EVFILT_READ`, public FreeBSD value).
    pub const EVFILT_READ: i16 = -1;
    /// Size of one `struct kevent` (public layout: ident, filter, flags,
    /// fflags, data, udata).
    pub const KEVENT_SIZE: usize = 32;
    /// Register only, do not block (PDG's drain semantics).
    pub const NOTE_NONE: u32 = 0;
}

/// `struct dirent` constants (FreeBSD, public).
pub mod dirent {
    /// Total fixed header before the name.
    pub const FIXED: usize = 8;
    /// Directory type.
    pub const DT_DIR: u8 = 4;
    /// Regular-file type.
    pub const DT_REG: u8 = 8;
}

/// The FreeBSD 9.2 amd64 `struct stat` layout — 120 bytes, verified
/// against two independent public sources (the FreeBSD 9.2 headers and
/// the OpenOrbis `OrbisKernelStat`, which agree field-for-field).
pub mod stat {
    /// `sizeof(struct stat)`.
    pub const SIZE: usize = 120;
    /// `st_dev` (u32).
    pub const ST_DEV: usize = 0x00;
    /// `st_ino` (u32).
    pub const ST_INO: usize = 0x04;
    /// `st_mode` (u16).
    pub const ST_MODE: usize = 0x08;
    /// `st_nlink` (u16).
    pub const ST_NLINK: usize = 0x0A;
    /// `st_uid` (u32).
    pub const ST_UID: usize = 0x0C;
    /// `st_gid` (u32).
    pub const ST_GID: usize = 0x10;
    /// `st_rdev` (u32).
    pub const ST_RDEV: usize = 0x14;
    /// `st_atim` (timespec, 16).
    pub const ST_ATIM: usize = 0x18;
    /// `st_mtim` (timespec, 16).
    pub const ST_MTIM: usize = 0x28;
    /// `st_ctim` (timespec, 16).
    pub const ST_CTIM: usize = 0x38;
    /// `st_size` (i64).
    pub const ST_SIZE: usize = 0x48;
    /// `st_blocks` (i64).
    pub const ST_BLOCKS: usize = 0x50;
    /// `st_blksize` (u32).
    pub const ST_BLKSIZE: usize = 0x58;
    /// `st_flags` (u32).
    pub const ST_FLAGS: usize = 0x5C;
    /// `st_gen` (u32).
    pub const ST_GEN: usize = 0x60;
    /// `st_lspare` (i32).
    pub const ST_LSPARE: usize = 0x64;
    /// `st_birthtim` (timespec, 16).
    pub const ST_BIRTHTIM: usize = 0x68;
    /// `S_IFMT` mask (public).
    pub const S_IFMT: u16 = 0xF000;
    /// Regular file.
    pub const S_IFREG: u16 = 0x8000;
    /// Directory.
    pub const S_IFDIR: u16 = 0x4000;
}

/// `_umtx_op` operation numbers (FreeBSD 9.2 `sys/sys/umtx.h`, public).
pub mod umtx {
    /// Legacy umtx word lock.
    pub const LOCK: i32 = 0;
    /// Legacy umtx word unlock.
    pub const UNLOCK: i32 = 1;
    /// Wait while `*obj == val` (64-bit compare).
    pub const WAIT: i32 = 2;
    /// Wake waiters on `obj`.
    pub const WAKE: i32 = 3;
    /// Try to take a `struct umutex`.
    pub const MUTEX_TRYLOCK: i32 = 4;
    /// Take a `struct umutex`, parking on contention.
    pub const MUTEX_LOCK: i32 = 5;
    /// Release a `struct umutex`, handing it to one waiter.
    pub const MUTEX_UNLOCK: i32 = 6;
}

/// The `OrbisKernelModuleInfo` layout the `sys_dynlib_get_info` syscall
/// fills (0x160 bytes; cross-checked between the public payload SDK's
/// struct offsets and the OpenOrbis SDK documentation).
pub mod module_info {
    /// `sizeof(OrbisKernelModuleInfo)`.
    pub const SIZE: usize = 0x160;
    /// Caller-initialized `size` prefix (u64).
    pub const SIZE_FIELD: usize = 0x00;
    /// Module name, 256 bytes.
    pub const NAME: usize = 0x08;
    /// `segmentInfo[4]` — 16 bytes each: `{addr u64, size u32, prot i32}`.
    pub const SEGMENTS: usize = 0x108;
    /// `segmentCount` (u32).
    pub const SEGMENT_COUNT: usize = 0x148;
    /// `fingerprint[0x14]`.
    pub const FINGERPRINT: usize = 0x14C;
}

/// Encode one FreeBSD `struct dirent` into `out`; returns the record
/// length (`d_reclen`), or 0 when it does not fit.
///
/// Layout: `d_fileno u32, d_reclen u16, d_type u8, d_namlen u8, name[]`
/// with the name padded so `d_reclen` is a multiple of 4.
pub fn encode_dirent(out: &mut Vec<u8>, fileno: u32, ty: u8, name: &[u8]) -> usize {
    let namlen = name.len();
    let reclen = dirent::FIXED + ((namlen + 1 + 3) & !3);
    if out.len() + reclen > u16::MAX as usize {
        return 0;
    }
    let start = out.len();
    out.resize(start + reclen, 0);
    out[start..start + 4].copy_from_slice(&fileno.to_le_bytes());
    out[start + 4..start + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
    out[start + 6] = ty;
    out[start + 7] = namlen as u8;
    out[start + 8..start + 8 + namlen].copy_from_slice(name);
    reclen
}

// ---------------------------------------------------------------------------
// Guest file descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FdKind {
    /// Descriptors 0–2: not backed by the PFS; `read` is EOF and `write`
    /// goes to the console (1/2) — the honest headless stdio.
    Reserved,
    File {
        pos: u64,
        size: u64,
    },
    Dir {
        /// How many dirents have already been served.
        served: usize,
    },
}

#[derive(Debug, Clone)]
pub struct GuestFd {
    /// The PFS path (or stdio name) this fd was opened against.
    pub path: String,
    /// What kind of descriptor it is.
    pub kind: FdKind,
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// Shared kernel state behind `Arc` (the SyscallHook is movable).
struct KernelState {
    vm: Mutex<VmManager>,
    pfs: Mutex<Option<Pfs>>,
    fds: Mutex<Vec<Option<GuestFd>>>,
    objects: Mutex<ObjectTable>,
    /// kevent registrations: kqueue handle → (ident, filter) list.
    regs: Mutex<HashMap<u64, Vec<(u64, i16)>>>,
    /// The process module table (Phase 6).
    dynlib: Mutex<crate::dynlib::Dynlib>,
    /// The guest thread table (Phase 6).
    threads: Mutex<crate::thread::ThreadTable>,
    /// Threads created but not yet materialized by the session scheduler.
    spawned: Mutex<Vec<crate::thread::SpawnedThread>>,
    /// The thread currently executing (set by the session before each
    /// slice; the syscall layer reads it for `thr_self`/umtx ownership).
    current_tid: AtomicU64,
    console: Arc<ConsoleDevice>,
    pid: u64,
    exit: Mutex<Option<i32>>,
    kevent_calls: AtomicU64,
    syscalls: AtomicU64,
    last_kqueue: AtomicU64,
    /// Set when the last live thread exited through `thr_exit` (the
    /// process then exits with code 0 — the documented PDG policy).
    threads_all_exited: AtomicBool,
}

/// Everything the Orbis kernel keeps outside guest RAM (session
/// snapshots; see [`OrbisKernel::snapshot_state`]).
///
/// The binary codec is a compact length-prefixed little-endian format —
/// save-state files carry it verbatim (framed by
/// `pdg_frontend`'s `SessionSnapshot` container).
#[derive(Debug, Clone)]
pub struct KernelSnapshot {
    /// The fd table.
    pub fds: Vec<Option<GuestFd>>,
    /// Kernel objects.
    pub objects: ObjectTable,
    /// kevent registrations.
    pub regs: HashMap<u64, Vec<(u64, i16)>>,
    /// The module table.
    pub dynlib: crate::dynlib::DynlibSnapshot,
    /// The thread table.
    pub threads: crate::thread::ThreadTable,
    /// The address-space bookkeeping.
    pub vm: crate::vm::VmSnapshot,
    /// The process exit code, once `exit` ran.
    pub exit: Option<i32>,
    /// The thread executing at capture time.
    pub current_tid: u64,
    /// Whether every thread had exited.
    pub threads_all_exited: bool,
}

impl KernelSnapshot {
    /// Encode every table (little-endian, length-prefixed; each
    /// variable-length section carries a `u32` byte-length header).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        let put_str = |o: &mut Vec<u8>, s: &[u8]| {
            o.extend_from_slice(&(s.len() as u32).to_le_bytes());
            o.extend_from_slice(s);
        };
        // fds
        o.extend_from_slice(&(self.fds.len() as u32).to_le_bytes());
        for fd in &self.fds {
            match fd {
                None => o.extend_from_slice(&[0u8]),
                Some(g) => {
                    let kind_tag: u8 = match &g.kind {
                        FdKind::Reserved => 0,
                        FdKind::File { .. } => 1,
                        FdKind::Dir { .. } => 2,
                    };
                    o.extend_from_slice(&[1u8, kind_tag]);
                    put_str(&mut o, g.path.as_bytes());
                    match &g.kind {
                        FdKind::Reserved => {}
                        FdKind::File { pos, size } => {
                            o.extend_from_slice(&pos.to_le_bytes());
                            o.extend_from_slice(&size.to_le_bytes());
                        }
                        FdKind::Dir { served } => {
                            o.extend_from_slice(&(*served as u64).to_le_bytes());
                        }
                    }
                }
            }
        }
        // objects
        let ob = self.objects.to_bytes();
        o.extend_from_slice(&(ob.len() as u32).to_le_bytes());
        o.extend_from_slice(&ob);
        // kevent registrations
        let mut keys: Vec<u64> = self.regs.keys().copied().collect();
        keys.sort_unstable();
        o.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for k in keys {
            let v = &self.regs[&k];
            o.extend_from_slice(&k.to_le_bytes());
            o.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for (ident, filter) in v {
                o.extend_from_slice(&ident.to_le_bytes());
                o.extend_from_slice(&filter.to_le_bytes());
            }
        }
        // dynlib
        let d = self.dynlib.to_bytes();
        o.extend_from_slice(&(d.len() as u32).to_le_bytes());
        o.extend_from_slice(&d);
        // threads
        let t = self.threads.to_bytes();
        o.extend_from_slice(&(t.len() as u32).to_le_bytes());
        o.extend_from_slice(&t);
        // vm
        let v = self.vm.to_bytes();
        o.extend_from_slice(&(v.len() as u32).to_le_bytes());
        o.extend_from_slice(&v);
        // scalars
        match self.exit {
            Some(c) => {
                o.extend_from_slice(&[1u8]);
                o.extend_from_slice(&(c as u32).to_le_bytes());
            }
            None => o.extend_from_slice(&[0u8]),
        }
        o.extend_from_slice(&self.current_tid.to_le_bytes());
        o.extend_from_slice(&[u8::from(self.threads_all_exited)]);
        o
    }

    /// Decode [`KernelSnapshot::to_bytes`] (returns `None` on any
    /// malformed input — restore refuses garbage rather than guessing).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        macro_rules! rd_u64 {
            () => {{
                let v = u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?);
                p += 8;
                v
            }};
        }
        macro_rules! rd_u32 {
            () => {{
                let v = u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?);
                p += 4;
                v
            }};
        }
        macro_rules! rd_str {
            () => {{
                let n = rd_u32!() as usize;
                let s = String::from_utf8(b.get(p..p + n)?.to_vec()).ok()?;
                p += n;
                s
            }};
        }
        macro_rules! rd_section {
            () => {{
                let n = rd_u32!() as usize;
                let sec = b.get(p..p + n)?.to_vec();
                p += n;
                sec
            }};
        }
        // fds
        let n = rd_u32!();
        let mut fds = Vec::with_capacity(n as usize);
        for _ in 0..n {
            if *b.get(p)? == 0 {
                p += 1;
                fds.push(None);
                continue;
            }
            let kind_tag = *b.get(p + 1)?;
            p += 2;
            let path = rd_str!();
            let kind = match kind_tag {
                0 => FdKind::Reserved,
                1 => {
                    let pos = rd_u64!();
                    let size = rd_u64!();
                    FdKind::File { pos, size }
                }
                _ => {
                    let served = rd_u64!() as usize;
                    FdKind::Dir { served }
                }
            };
            fds.push(Some(GuestFd { path, kind }));
        }
        // objects
        let objects = ObjectTable::from_bytes(&rd_section!())?;
        // kevent registrations
        let n = rd_u32!();
        let mut regs = HashMap::with_capacity(n as usize);
        for _ in 0..n {
            let k = rd_u64!();
            let m = rd_u32!();
            let mut v = Vec::with_capacity(m as usize);
            for _ in 0..m {
                let ident = rd_u64!();
                let filter = i16::from_le_bytes(b.get(p..p + 2)?.try_into().ok()?);
                p += 2;
                v.push((ident, filter));
            }
            regs.insert(k, v);
        }
        // dynlib
        let dynlib = crate::dynlib::DynlibSnapshot::from_bytes(&rd_section!())?;
        // threads
        let threads = crate::thread::ThreadTable::from_bytes(&rd_section!())?;
        // vm
        let vm = crate::vm::VmSnapshot::from_bytes(&rd_section!())?;
        // scalars
        let exit = if *b.get(p)? == 1 {
            p += 1;
            Some(i32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?))
        } else {
            p += 1;
            None
        };
        let current_tid = rd_u64!();
        let threads_all_exited = *b.get(p)? != 0;
        Some(Self {
            fds,
            objects,
            regs,
            dynlib,
            threads,
            vm,
            exit,
            current_tid,
            threads_all_exited,
        })
    }
}

/// The Orbis HLE kernel: a [`SyscallHook`] plus the host-side APIs the
/// session uses (mounting, triggering, reporting). Cloning shares the
/// kernel state (the session installs a clone as the CPU hook).
#[derive(Clone)]
pub struct OrbisKernel {
    state: Arc<KernelState>,
}

impl std::fmt::Debug for OrbisKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrbisKernel")
            .field("pid", &self.state.pid)
            .finish_non_exhaustive()
    }
}

impl OrbisKernel {
    /// A kernel over `vm` writing console output to `console`, with the
    /// default module table (PRX arena at `0x6000_0000_0000`, TLS arena
    /// at `0x0060_0000`).
    pub fn new(vm: VmManager, console: Arc<ConsoleDevice>, pid: u64) -> Self {
        let dynlib = crate::dynlib::Dynlib::new(0x6000_0000_0000, 0x6000_1000_0000, 0x0060_0000);
        Self::with_dynlib(vm, console, pid, dynlib)
    }

    /// A kernel over an explicit module table (sessions with their own
    /// arena layout).
    pub fn with_dynlib(
        vm: VmManager,
        console: Arc<ConsoleDevice>,
        pid: u64,
        dynlib: crate::dynlib::Dynlib,
    ) -> Self {
        Self {
            state: Arc::new(KernelState {
                vm: Mutex::new(vm),
                pfs: Mutex::new(None),
                fds: Mutex::new(vec![
                    Some(GuestFd {
                        path: "<stdin>".to_owned(),
                        kind: FdKind::Reserved,
                    }),
                    Some(GuestFd {
                        path: "<stdout>".to_owned(),
                        kind: FdKind::Reserved,
                    }),
                    Some(GuestFd {
                        path: "<stderr>".to_owned(),
                        kind: FdKind::Reserved,
                    }),
                ]),
                objects: Mutex::new(ObjectTable::new()),
                regs: Mutex::new(HashMap::new()),
                dynlib: Mutex::new(dynlib),
                threads: Mutex::new(crate::thread::ThreadTable::new()),
                spawned: Mutex::new(Vec::new()),
                current_tid: AtomicU64::new(crate::thread::MAIN_TID),
                console,
                pid,
                exit: Mutex::new(None),
                kevent_calls: AtomicU64::new(0),
                syscalls: AtomicU64::new(0),
                last_kqueue: AtomicU64::new(0),
                threads_all_exited: AtomicBool::new(false),
            }),
        }
    }

    /// Mount the read-only PFS every file syscall resolves against.
    pub fn mount_pfs(&self, pfs: Pfs) {
        *self.state.pfs.lock().expect("pfs") = Some(pfs);
    }

    /// The process exit code, once `exit` ran.
    pub fn exit_code(&self) -> Option<i32> {
        *self.state.exit.lock().expect("exit")
    }

    /// Total syscalls dispatched (diagnostics).
    pub fn syscall_count(&self) -> u64 {
        self.state.syscalls.load(Ordering::Relaxed)
    }

    /// Times `kevent` was called (the session triggers the demo event
    /// after the first drain attempt).
    pub fn kevent_call_count(&self) -> u64 {
        self.state.kevent_calls.load(Ordering::Relaxed)
    }

    /// The most recently created kqueue handle (the session's trigger
    /// target); 0 when none exists.
    pub fn last_kqueue(&self) -> u64 {
        self.state.last_kqueue.load(Ordering::Relaxed)
    }

    /// Host-side: read `out.len()` bytes of guest memory (statuses,
    /// buffers) after the run.
    pub fn read_guest(&self, va: u64, out: &mut [u8]) -> Result<(), String> {
        self.state
            .vm
            .lock()
            .expect("vm")
            .read_va(va, out)
            .map_err(|e| e.to_string())
    }

    /// Capture every kernel table (session snapshots: plain deep copies —
    /// the state is data, and cloning it beats inventing a serialization
    /// format nobody reads back).
    pub fn snapshot_state(&self) -> KernelSnapshot {
        KernelSnapshot {
            fds: self.state.fds.lock().expect("fds").clone(),
            objects: self.state.objects.lock().expect("objects").clone(),
            regs: self.state.regs.lock().expect("regs").clone(),
            dynlib: self.state.dynlib.lock().expect("dynlib").snapshot(),
            threads: self.state.threads.lock().expect("threads").clone(),
            vm: self.state.vm.lock().expect("vm").snapshot(),
            exit: *self.state.exit.lock().expect("exit"),
            current_tid: self.state.current_tid.load(Ordering::Relaxed),
            threads_all_exited: self.state.threads_all_exited.load(Ordering::Relaxed),
        }
    }

    /// Restore every kernel table over a freshly built kernel (the same
    /// constructor layout — the page tables and segment data live in
    /// guest RAM, restored with it).
    pub fn restore_state(&self, snap: &KernelSnapshot) {
        *self.state.fds.lock().expect("fds") = snap.fds.clone();
        *self.state.objects.lock().expect("objects") = snap.objects.clone();
        *self.state.regs.lock().expect("regs") = snap.regs.clone();
        self.state
            .dynlib
            .lock()
            .expect("dynlib")
            .restore(&snap.dynlib);
        *self.state.threads.lock().expect("threads") = snap.threads.clone();
        self.state.vm.lock().expect("vm").restore(&snap.vm);
        *self.state.exit.lock().expect("exit") = snap.exit;
        self.state
            .current_tid
            .store(snap.current_tid, Ordering::Relaxed);
        self.state
            .threads_all_exited
            .store(snap.threads_all_exited, Ordering::Relaxed);
    }

    /// Session scheduler hook: which thread is executing the next
    /// slice.
    pub fn set_current_thread(&self, tid: u64) {
        self.state.current_tid.store(tid, Ordering::Relaxed);
    }

    /// The shared MMU (tests construct a `ProcessMemory` against it).
    pub fn shared_mmu_for_test(&self) -> std::sync::Arc<std::sync::Mutex<pdg_memory::Mmu>> {
        self.state.vm.lock().expect("vm").shared_mmu()
    }

    /// Drain threads `thr_create` produced (the session materializes one
    /// interpreter per record).
    pub fn take_spawned_threads(&self) -> Vec<crate::thread::SpawnedThread> {
        std::mem::take(&mut *self.state.spawned.lock().expect("spawned"))
    }

    /// Which tids are runnable right now (the scheduler's run queue).
    pub fn runnable_threads(&self) -> Vec<u64> {
        self.state.threads.lock().expect("threads").runnable_tids()
    }

    /// A guest thread executed `hlt`: PDG's documented model treats it as
    /// that thread's end (parked forever); the process ends when the last
    /// thread is gone.
    pub fn thread_halt(&self, tid: u64) {
        let all_gone = self.state.threads.lock().expect("threads").exit(tid, 0);
        if all_gone {
            self.state.threads_all_exited.store(true, Ordering::Relaxed);
        }
    }

    /// Whether `tid` has exited (the scheduler retires it).
    pub fn thread_exited(&self, tid: u64) -> bool {
        self.state.threads.lock().expect("threads").is_exited(tid)
    }

    /// Has the process lost its last thread (`thr_exit` of the final
    /// live thread)?
    pub fn threads_all_exited(&self) -> bool {
        self.state.threads_all_exited.load(Ordering::Relaxed)
    }

    /// The live thread count (diagnostics, the report).
    pub fn thread_count(&self) -> usize {
        self.state
            .threads
            .lock()
            .expect("threads")
            .live_tids()
            .len()
    }

    /// Host-side: post an event to a guest event queue (the shape a real
    /// kernel's `sceKernelTriggerUserEvent`/interrupt path takes).
    pub fn trigger_equeue_event(
        &self,
        kq: u64,
        ident: u64,
        filter: i16,
        data: u64,
    ) -> Result<(), crate::objects::ObjectError> {
        let mut objects = self.state.objects.lock().expect("objects");
        objects.equeue(kq)?.trigger(KEvent {
            ident,
            filter,
            fflags: 0,
            data,
        });
        Ok(())
    }

    /// Handle one syscall (unit-testable without a CPU: the registers are
    /// passed and returned explicitly).
    fn dispatch(
        &self,
        number: u64,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        match number {
            nr::EXIT => {
                *self.state.exit.lock().expect("exit") = Some(args[0] as i32);
                // `exit` ends the whole process: every thread goes with
                // it (the documented POSIX shape).
                self.state.threads.lock().expect("threads").exit_all();
                Ok(0)
            }
            nr::READ => self.read(args, mem),
            nr::WRITE => self.write(args, mem),
            nr::OPEN => self.open(args, mem),
            nr::CLOSE => self.close(args),
            nr::GETPID => Ok(self.state.pid as i64),
            nr::MUNMAP => self.munmap(args),
            nr::MPROTECT => self.mprotect(args),
            nr::GETDIRENTRIES => self.getdirentries(args, mem),
            nr::KQUEUE => {
                let handle = self
                    .state
                    .objects
                    .lock()
                    .expect("objects")
                    .insert(KernelObject::Equeue(Equeue::new()));
                self.state.last_kqueue.store(handle, Ordering::Relaxed);
                Ok(handle as i64)
            }
            nr::KEVENT => self.kevent(args, mem),
            nr::MMAP => self.mmap(args),
            nr::LSEEK => self.lseek(args),
            nr::STAT => self.stat_syscall(args, mem),
            nr::FSTAT => self.fstat(args, mem),
            nr::THR_CREATE => self.thr_create(args, mem),
            nr::THR_EXIT => self.thr_exit(args),
            nr::THR_SELF => self.thr_self(args, mem),
            nr::UMTX_OP => self.umtx_op(args, mem),
            nr::DYNLIB_DLSYM => self.dynlib_dlsym(args, mem),
            nr::DYNLIB_GET_LIST => self.dynlib_get_list(args, mem),
            nr::DYNLIB_GET_INFO => self.dynlib_get_info(args, mem),
            nr::DYNLIB_LOAD_PRX => self.dynlib_load_prx(args, mem),
            nr::DYNLIB_UNLOAD_PRX => self.dynlib_unload_prx(args),
            other => {
                if let Some(name) = sony_name(other) {
                    warn!(
                        nr = other,
                        name, "Sony syscall refused (not implemented in Phase 5)"
                    );
                } else {
                    warn!(nr = other, "unknown syscall refused");
                }
                Err(errno::ENOSYS)
            }
        }
    }

    // ----- file I/O -------------------------------------------------------

    fn read(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (fd, buf, len) = (args[0], args[1], args[2] as usize);
        match fd {
            0 => Ok(0), // stdin: EOF, the honest headless answer
            1 | 2 => Err(errno::EBADF),
            _ => {
                let mut fds = self.state.fds.lock().expect("fds");
                let Some(slot) = fds.get_mut(fd as usize).and_then(|s| s.as_mut()) else {
                    return Err(errno::EBADF);
                };
                let FdKind::File { pos, size } = &mut slot.kind else {
                    return Err(errno::EISDIR);
                };
                let n = len.min(size.saturating_sub(*pos) as usize);
                if n == 0 {
                    return Ok(0);
                }
                let mut tmp = vec![0u8; n];
                {
                    let mut pfs = self.state.pfs.lock().expect("pfs");
                    let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
                    pfs.read_at(&slot.path, *pos, &mut tmp)
                        .map_err(|_| errno::EIO)?;
                }
                mem.write_buf(buf, &tmp).map_err(|_| errno::EFAULT)?;
                *pos += n as u64;
                Ok(n as i64)
            }
        }
    }

    fn write(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (fd, buf, len) = (args[0], args[1], args[2] as usize);
        match fd {
            1 | 2 => {
                let mut tmp = vec![0u8; len];
                mem.read_buf(buf, &mut tmp).map_err(|_| errno::EFAULT)?;
                let console = &self.state.console;
                for chunk in tmp.chunks(8) {
                    let mut v = 0u64;
                    for (i, b) in chunk.iter().enumerate() {
                        v |= u64::from(*b) << (i * 8);
                    }
                    let width = AccessWidth::from_size(chunk.len() as u8).expect("1|2|4|8");
                    console.write(0, width, v);
                }
                Ok(len as i64)
            }
            _ => Err(errno::EBADF), // the PFS is read-only by design
        }
    }

    fn open(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (path_va, flags) = (args[0], args[1] as i32);
        if flags & open_flags::O_ACCMODE != 0 {
            return Err(errno::EROFS); // the filesystem is read-only
        }
        let path = read_cstring(mem, path_va).ok_or(errno::EFAULT)?;
        let ino = {
            let mut pfs = self.state.pfs.lock().expect("pfs");
            let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
            pfs.lookup(&path).map_err(|e| match e {
                crate::pfs::PfsError::NotFound { .. } => errno::ENOENT,
                crate::pfs::PfsError::NotADirectory { .. } => errno::ENOTDIR,
                _ => errno::EIO,
            })?
        };
        let kind = if ino.is_dir() {
            FdKind::Dir { served: 0 }
        } else {
            FdKind::File {
                pos: 0,
                size: ino.size,
            }
        };
        let mut fds = self.state.fds.lock().expect("fds");
        let slot = fds.iter().position(|s| s.is_none()).unwrap_or_else(|| {
            fds.push(None);
            fds.len() - 1
        });
        fds[slot] = Some(GuestFd { path, kind });
        Ok(slot as i64)
    }

    fn close(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let fd = args[0] as usize;
        if fd <= 2 {
            // stdio is not closeable in the Phase-5 headless model.
            return Err(errno::EBADF);
        }
        let mut fds = self.state.fds.lock().expect("fds");
        match fds.get_mut(fd) {
            Some(slot) if slot.is_some() => {
                *slot = None;
                Ok(0)
            }
            _ => Err(errno::EBADF),
        }
    }

    fn lseek(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let (fd, off, wh) = (args[0], args[1] as i64, args[2] as i64);
        let mut fds = self.state.fds.lock().expect("fds");
        let Some(slot) = fds.get_mut(fd as usize).and_then(|s| s.as_mut()) else {
            return Err(errno::EBADF);
        };
        let FdKind::File { pos, size } = &mut slot.kind else {
            return Err(errno::EINVAL); // lseek on a directory
        };
        let new = match wh {
            whence::SEEK_SET => off,
            whence::SEEK_CUR => *pos as i64 + off,
            whence::SEEK_END => *size as i64 + off,
            _ => return Err(errno::EINVAL),
        };
        if new < 0 {
            return Err(errno::EINVAL);
        }
        *pos = new as u64;
        Ok(new)
    }

    fn getdirentries(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (fd, buf, nbytes, basep) = (args[0], args[1], args[2] as usize, args[3]);
        let path = {
            let fds = self.state.fds.lock().expect("fds");
            let Some(slot) = fds.get(fd as usize).and_then(|s| s.as_ref()) else {
                return Err(errno::EBADF);
            };
            if !matches!(slot.kind, FdKind::Dir { .. }) {
                return Err(errno::EINVAL);
            }
            slot.path.clone()
        };
        let entries = {
            let mut pfs = self.state.pfs.lock().expect("pfs");
            let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
            pfs.readdir(&path).map_err(|_| errno::EIO)?
        };
        let served = {
            let mut fds = self.state.fds.lock().expect("fds");
            let Some(slot) = fds.get_mut(fd as usize).and_then(|s| s.as_mut()) else {
                return Err(errno::EBADF);
            };
            let FdKind::Dir { served } = &mut slot.kind else {
                return Err(errno::EINVAL);
            };
            *served
        };
        let mut out = Vec::new();
        let mut packed = 0usize;
        for e in &entries[served.min(entries.len())..] {
            let before = out.len();
            let ty = if matches!(
                e.ty,
                dirent_type::DIRECTORY | dirent_type::PARENT | dirent_type::SELF
            ) {
                dirent::DT_DIR
            } else {
                dirent::DT_REG
            };
            let reclen = encode_dirent(&mut out, e.ino, ty, e.name.as_bytes());
            if reclen == 0 || out.len() > nbytes {
                out.truncate(before);
                break;
            }
            packed += 1;
        }
        if packed > 0 {
            let mut fds = self.state.fds.lock().expect("fds");
            if let Some(slot) = fds.get_mut(fd as usize).and_then(|s| s.as_mut()) {
                if let FdKind::Dir { served } = &mut slot.kind {
                    *served += packed;
                }
            }
        }
        let bytes = out.len();
        mem.write_buf(buf, &out).map_err(|_| errno::EFAULT)?;
        if basep != 0 {
            let total = (served + packed) as u64;
            mem.write_buf(basep, &total.to_le_bytes())
                .map_err(|_| errno::EFAULT)?;
        }
        Ok(bytes as i64)
    }

    // ----- memory ----------------------------------------------------------

    fn mmap(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let (addr, len, prot, flags) = (args[0], args[1], args[2] as i32, args[3]);
        if len == 0 || len > u32::MAX as u64 {
            return Err(errno::EINVAL);
        }
        if flags & mmap_flags::MAP_ANON == 0 {
            // File-backed mappings need the unified cache Phase 5 does not
            // have; the honest answer is ENODEV (what FreeBSD returns for
            // unmappable objects).
            return Err(errno::ENODEV);
        }
        if flags & mmap_flags::MAP_SHARED != 0 {
            return Err(errno::EINVAL); // no shared anonymous mappings yet
        }
        let mut vm = self.state.vm.lock().expect("vm");
        let va = if addr != 0 {
            if flags & mmap_flags::MAP_FIXED == 0 {
                return Err(errno::EINVAL); // hint placement unimplemented
            }
            addr
        } else {
            vm.mmap_anon(len).map_err(|_| errno::ENOMEM)?
        };
        if prot == prot::PROT_NONE {
            vm.reserve(va, len).map_err(|_| errno::ENOMEM)?;
        } else {
            vm.map_anon(va, crate::vm::page_ceil(len), prot_flags(prot), true)
                .map_err(|_| errno::ENOMEM)?;
        }
        Ok(va as i64)
    }

    fn munmap(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let (addr, len) = (args[0], args[1]);
        let mut vm = self.state.vm.lock().expect("vm");
        vm.unmap(addr, len).map_err(|_| errno::EINVAL)?;
        Ok(0)
    }

    fn mprotect(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let (addr, len, p) = (args[0], args[1], args[2] as i32);
        let mut vm = self.state.vm.lock().expect("vm");
        vm.protect(addr, len, prot_flags(p))
            .map_err(|_| errno::EINVAL)?;
        Ok(0)
    }

    // ----- event queues -----------------------------------------------------

    fn kevent(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        self.state.kevent_calls.fetch_add(1, Ordering::Relaxed);
        let (kq, changelist, nchanges, eventlist, nevents) = (
            args[0],
            args[1],
            args[2] as usize,
            args[3],
            args[4] as usize,
        );
        {
            let objects = self.state.objects.lock().expect("objects");
            objects
                .get(kq, crate::objects::ObjectKind::Equeue)
                .map_err(|_| errno::EBADF)?;
        }
        // Apply the changelist: EV_ADD registrations only.
        for i in 0..nchanges {
            let mut ev = [0u8; ev::KEVENT_SIZE];
            mem.read_buf(changelist + i as u64 * ev::KEVENT_SIZE as u64, &mut ev)
                .map_err(|_| errno::EFAULT)?;
            let ident = u64::from_le_bytes(ev[0..8].try_into().unwrap());
            let filter = i16::from_le_bytes(ev[8..10].try_into().unwrap());
            let flags = u16::from_le_bytes(ev[10..12].try_into().unwrap());
            if flags & !(ev::EV_ADD | ev::EV_CLEAR) != 0 {
                return Err(errno::EINVAL); // EV_DELETE/ONESHOT/... unimplemented
            }
            if flags & ev::EV_ADD != 0 {
                self.state
                    .regs
                    .lock()
                    .expect("regs")
                    .entry(kq)
                    .or_default()
                    .push((ident, filter));
            }
        }
        // Drain: deliver queued events that match a registration.
        let queued = {
            let mut objects = self.state.objects.lock().expect("objects");
            objects.equeue(kq).map_err(|_| errno::EBADF)?.drain_all()
        };
        let regs = self
            .state
            .regs
            .lock()
            .expect("regs")
            .get(&kq)
            .cloned()
            .unwrap_or_default();
        let mut delivered = 0usize;
        let mut keep = Vec::new();
        for e in queued {
            let wanted = regs.iter().any(|(id, f)| *id == e.ident && *f == e.filter);
            if wanted && delivered < nevents {
                let mut ev = [0u8; ev::KEVENT_SIZE];
                ev[0..8].copy_from_slice(&e.ident.to_le_bytes());
                ev[8..10].copy_from_slice(&e.filter.to_le_bytes());
                ev[10..12].copy_from_slice(&(ev::EV_CLEAR).to_le_bytes());
                ev[12..16].copy_from_slice(&e.fflags.to_le_bytes());
                ev[16..24].copy_from_slice(&e.data.to_le_bytes());
                mem.write_buf(eventlist + delivered as u64 * ev::KEVENT_SIZE as u64, &ev)
                    .map_err(|_| errno::EFAULT)?;
                delivered += 1;
            } else {
                keep.push(e);
            }
        }
        if !keep.is_empty() {
            let mut objects = self.state.objects.lock().expect("objects");
            let q = objects.equeue(kq).map_err(|_| errno::EBADF)?;
            for e in keep {
                q.trigger(e);
            }
        }
        Ok(delivered as i64)
    }

    // ----- stat family (Phase 6) ------------------------------------------

    /// Encode one FreeBSD 9.2 amd64 `struct stat` (120 bytes; layout
    /// verified against two public sources — see [`stat`]).
    fn encode_stat(&self, out: &mut [u8; stat::SIZE], ino: u32, is_dir: bool, size: u64) {
        *out = [0u8; stat::SIZE];
        let put_u32 = |b: &mut [u8; stat::SIZE], at: usize, v: u32| {
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_u16 = |b: &mut [u8; stat::SIZE], at: usize, v: u16| {
            b[at..at + 2].copy_from_slice(&v.to_le_bytes());
        };
        let put_u64 = |b: &mut [u8; stat::SIZE], at: usize, v: u64| {
            b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        };
        // st_dev = 0: the read-only PFS is not a device with a number.
        put_u32(out, stat::ST_DEV, 0);
        put_u32(out, stat::ST_INO, ino);
        // Directories 0755, files 0644 — the conventional modes.
        let mode = if is_dir {
            stat::S_IFDIR | 0o755
        } else {
            stat::S_IFREG | 0o644
        };
        put_u16(out, stat::ST_MODE, mode);
        put_u16(out, stat::ST_NLINK, 1);
        put_u32(out, stat::ST_UID, 0);
        put_u32(out, stat::ST_GID, 0);
        put_u32(out, stat::ST_RDEV, 0);
        put_u64(out, stat::ST_SIZE, size);
        put_u64(out, stat::ST_BLOCKS, size.div_ceil(512));
        let blksize = self
            .state
            .pfs
            .lock()
            .expect("pfs")
            .as_ref()
            .map_or(0x1000, |p| u64::from(p.block_size()));
        put_u32(out, stat::ST_BLKSIZE, blksize as u32);
        // The PFS carries no timestamps: the time fields read zero —
        // the honest answer for what the on-disk format keeps.
    }

    fn stat_syscall(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (path_va, sb) = (args[0], args[1]);
        let path = read_cstring(mem, path_va).ok_or(errno::EFAULT)?;
        let ino = {
            let mut pfs = self.state.pfs.lock().expect("pfs");
            let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
            pfs.lookup(&path).map_err(|e| match e {
                crate::pfs::PfsError::NotFound { .. } => errno::ENOENT,
                crate::pfs::PfsError::NotADirectory { .. } => errno::ENOTDIR,
                _ => errno::EIO,
            })?
        };
        let mut buf = [0u8; stat::SIZE];
        self.encode_stat(&mut buf, ino.ino, ino.is_dir(), ino.size);
        mem.write_buf(sb, &buf).map_err(|_| errno::EFAULT)?;
        Ok(0)
    }

    fn fstat(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (fd, sb) = (args[0], args[1]);
        let path = {
            let fds = self.state.fds.lock().expect("fds");
            let slot = fds.get(fd as usize).and_then(|s| s.as_ref());
            match slot {
                Some(s) if matches!(s.kind, FdKind::File { .. }) => s.path.clone(),
                Some(s) if matches!(s.kind, FdKind::Dir { .. }) => s.path.clone(),
                _ => return Err(errno::EBADF),
            }
        };
        let ino = {
            let mut pfs = self.state.pfs.lock().expect("pfs");
            let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
            pfs.lookup(&path).map_err(|_| errno::EIO)?
        };
        let mut buf = [0u8; stat::SIZE];
        self.encode_stat(&mut buf, ino.ino, ino.is_dir(), ino.size);
        mem.write_buf(sb, &buf).map_err(|_| errno::EFAULT)?;
        Ok(0)
    }

    // ----- threads (Phase 6) ------------------------------------------------

    /// `thr_create(ucontext_t *ctx, long *id, int flags)` — FreeBSD 430.
    fn thr_create(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (ctx, id_out, flags) = (args[0], args[1], args[2]);
        if flags != 0 {
            // THR_SUSPENDED/THR_DETACHED-shaped flags: refused until a
            // use pins their behavior (the honest subset).
            return Err(errno::EINVAL);
        }
        let mut rd =
            |off: u64| -> Result<u64, i32> { mem.read(8, ctx + off).map_err(|_| errno::EFAULT) };
        let rip = rd(crate::thread::ucontext::MC_RIP)?;
        let rsp = rd(crate::thread::ucontext::MC_RSP)?;
        let rdi = rd(crate::thread::ucontext::MC_RDI)?;
        let rflags = rd(crate::thread::ucontext::MC_RFLAGS)?;
        if rip == 0 || rsp == 0 {
            return Err(errno::EINVAL);
        }
        // The new thread's TLS block: its own copy of the aggregated
        // template, allocated through the process mmap arena.
        let spawned = {
            let mut vm = self.state.vm.lock().expect("vm");
            let dynlib = self.state.dynlib.lock().expect("dynlib");
            let total = dynlib.tls_total();
            let tls_base = if total == 0 {
                0
            } else {
                // Reserve an arena address, map the pages (RW, user,
                // no-execute), then copy the aggregated template in.
                let len = crate::vm::page_ceil(total);
                let base = vm.mmap_anon(len).map_err(|_| errno::ENOMEM)?;
                vm.map_anon(
                    base,
                    len,
                    pdg_memory::PageFlags::PRESENT
                        | pdg_memory::PageFlags::RW
                        | pdg_memory::PageFlags::USER
                        | pdg_memory::PageFlags::NX,
                    true,
                )
                .map_err(|_| errno::ENOMEM)?;
                let template = dynlib.read_tls_template(&mut vm).map_err(|_| errno::EIO)?;
                vm.write_va(base, &template).map_err(|_| errno::EIO)?;
                base
            };
            let mut threads = self.state.threads.lock().expect("threads");
            threads.create(rip, rsp, rdi, rflags, tls_base)
        };
        self.state
            .spawned
            .lock()
            .expect("spawned")
            .push(spawned.clone());
        if id_out != 0 {
            mem.write_buf(id_out, &spawned.tid.to_le_bytes())
                .map_err(|_| errno::EFAULT)?;
        }
        Ok(0)
    }

    /// `thr_exit(long *state)` — FreeBSD 431. The state pointer is the
    /// pthread join channel: waiters parked on it are woken (the same
    /// discipline libthr's join applies).
    fn thr_exit(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let state_ptr = args[0];
        let tid = self.state.current_tid.load(Ordering::Relaxed);
        let all_gone = {
            let mut threads = self.state.threads.lock().expect("threads");
            if state_ptr != 0 {
                threads.wake_all(state_ptr);
            }
            threads.exit(tid, state_ptr as i64)
        };
        if all_gone {
            self.state.threads_all_exited.store(true, Ordering::Relaxed);
        }
        Ok(0)
    }

    /// `thr_self(long *id)` — FreeBSD 432.
    fn thr_self(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let id_out = args[0];
        let tid = self.state.current_tid.load(Ordering::Relaxed);
        mem.write_buf(id_out, &tid.to_le_bytes())
            .map_err(|_| errno::EFAULT)?;
        Ok(0)
    }

    // ----- _umtx_op (Phase 6) ----------------------------------------------

    /// `_umtx_op(obj, op, val, uaddr1, uaddr2)` — FreeBSD 454. The
    /// implemented discipline: legacy word `LOCK`/`UNLOCK`, `WAIT`/`WAKE`
    /// on 64-bit words, and the umutex family with FIFO handoff on
    /// unlock. Timed waits (`uaddr2 != 0`) are refused with `ENOSYS`.
    fn umtx_op(&self, args: &[u64; 6], mem: &mut dyn pdg_cpu::MemoryAccess) -> Result<i64, i32> {
        let (obj, op, val, _uaddr1, uaddr2) = (args[0], args[1] as i32, args[2], args[3], args[4]);
        let tid = self.state.current_tid.load(Ordering::Relaxed);
        let rd_u32 = |mem: &mut dyn pdg_cpu::MemoryAccess| -> Result<u32, i32> {
            mem.read(4, obj)
                .map(|v| v as u32)
                .map_err(|_| errno::EFAULT)
        };
        let wr_u32 = |mem: &mut dyn pdg_cpu::MemoryAccess, v: u32| -> Result<(), i32> {
            mem.write(4, obj, u64::from(v)).map_err(|_| errno::EFAULT)
        };
        // A helper for "park the caller": the syscall's success return is
        // written first (RAX is set by the hook below); the thread then
        // stops until a wake.
        let park = |obj: u64| -> Result<i64, i32> {
            self.state.threads.lock().expect("threads").park(tid, obj);
            Ok(0)
        };
        match op {
            umtx::LOCK => {
                let owner = rd_u32(mem)?;
                if owner == 0 {
                    wr_u32(mem, tid as u32)?;
                    Ok(0)
                } else if owner == tid as u32 {
                    Err(errno::EDEADLK)
                } else {
                    park(obj)
                }
            }
            umtx::UNLOCK => {
                let owner = rd_u32(mem)?;
                if owner != tid as u32 {
                    return Err(errno::EPERM);
                }
                wr_u32(mem, 0)?;
                self.state.threads.lock().expect("threads").wake_all(obj);
                Ok(0)
            }
            umtx::WAIT => {
                if uaddr2 != 0 {
                    warn!("umtx WAIT: timed waits refused (not implemented)");
                    return Err(errno::ENOSYS);
                }
                // FreeBSD's do_wait: queue first, then compare; a
                // mismatched value returns without sleeping.
                let word = mem.read(8, obj).map_err(|_| errno::EFAULT)?;
                if word != val {
                    return Ok(0);
                }
                park(obj)
            }
            umtx::WAKE => {
                self.state.threads.lock().expect("threads").wake_all(obj);
                Ok(0)
            }
            umtx::MUTEX_TRYLOCK => {
                let owner = rd_u32(mem)?;
                if owner == 0 {
                    wr_u32(mem, tid as u32)?;
                    Ok(0)
                } else {
                    Err(errno::EBUSY)
                }
            }
            umtx::MUTEX_LOCK => {
                if uaddr2 != 0 {
                    warn!("umtx MUTEX_LOCK: timed locks refused (not implemented)");
                    return Err(errno::ENOSYS);
                }
                let owner = rd_u32(mem)?;
                if owner == 0 {
                    wr_u32(mem, tid as u32)?;
                    Ok(0)
                } else if owner == tid as u32 {
                    Err(errno::EDEADLK)
                } else {
                    park(obj)
                }
            }
            umtx::MUTEX_UNLOCK => {
                let owner = rd_u32(mem)?;
                if owner != tid as u32 {
                    return Err(errno::EPERM);
                }
                // FIFO handoff: the first waiter receives the lock; the
                // others stay parked for its unlock.
                let next = {
                    let mut threads = self.state.threads.lock().expect("threads");
                    threads.wake_one(obj)
                };
                match next {
                    Some(next_tid) => {
                        wr_u32(mem, next_tid as u32)?;
                    }
                    None => {
                        wr_u32(mem, 0)?;
                    }
                }
                Ok(0)
            }
            other => {
                warn!(op = other, "umtx op refused (not implemented)");
                Err(errno::ENOSYS)
            }
        }
    }

    // ----- dynlib syscalls (Phase 6) -----------------------------------------

    /// `sys_dynlib_dlsym(handle, name, &out)` — Sony 591.
    fn dynlib_dlsym(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (handle, name_va, out_va) = (args[0], args[1], args[2]);
        let name = read_cstring(mem, name_va).ok_or(errno::EFAULT)?;
        let addr = {
            let dynlib = self.state.dynlib.lock().expect("dynlib");
            dynlib.dlsym(handle, &name).map_err(|e| match e {
                crate::dynlib::DynlibError::UnknownHandle { .. } => errno::ESRCH,
                crate::dynlib::DynlibError::UnknownSymbol { .. } => errno::ENOENT,
                other => {
                    warn!(error = %other, "dlsym failed");
                    errno::EIO
                }
            })?
        };
        mem.write_buf(out_va, &addr.to_le_bytes())
            .map_err(|_| errno::EFAULT)?;
        Ok(0)
    }

    /// `sys_dynlib_get_list(int *ids, int max, int *count)` — Sony 592.
    fn dynlib_get_list(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (dest, max, count_out) = (args[0], args[1] as i64, args[2]);
        if max < 0 {
            return Err(errno::EINVAL);
        }
        let handles = {
            let dynlib = self.state.dynlib.lock().expect("dynlib");
            dynlib.module_handles()
        };
        let n = (max as usize).min(handles.len());
        for (i, h) in handles.iter().take(n).enumerate() {
            mem.write_buf(dest + (i as u64) * 4, &(*h as u32).to_le_bytes())
                .map_err(|_| errno::EFAULT)?;
        }
        // The count reports the *total* module count (the documented
        // PDG choice — a caller sizing its buffer learns the truth).
        if count_out != 0 {
            mem.write_buf(count_out, &(handles.len() as u32).to_le_bytes())
                .map_err(|_| errno::EFAULT)?;
        }
        Ok(0)
    }

    /// `sys_dynlib_get_info(handle, &info)` — Sony 593 (size-prefixed
    /// `OrbisKernelModuleInfo`).
    fn dynlib_get_info(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (handle, info_va) = (args[0], args[1]);
        let declared = mem.read(8, info_va).map_err(|_| errno::EFAULT)?;
        if declared < module_info::SIZE as u64 {
            return Err(errno::EINVAL);
        }
        let (name, segments, bias) = {
            let dynlib = self.state.dynlib.lock().expect("dynlib");
            let module = dynlib.module(handle).map_err(|e| match e {
                crate::dynlib::DynlibError::UnknownHandle { .. } => errno::ESRCH,
                other => {
                    warn!(error = %other, "get_info failed");
                    errno::EIO
                }
            })?;
            (
                module.name.clone(),
                module.image.segments.clone(),
                module.image.bias,
            )
        };
        let mut buf = [0u8; module_info::SIZE];
        // size
        buf[module_info::SIZE_FIELD..module_info::SIZE_FIELD + 8]
            .copy_from_slice(&(module_info::SIZE as u64).to_le_bytes());
        // name (NUL-terminated, truncated to the field)
        let n = name.len().min(255);
        buf[module_info::NAME..module_info::NAME + n].copy_from_slice(&name.as_bytes()[..n]);
        // segments
        let count = segments.len().min(4);
        for (i, seg) in segments.iter().take(4).enumerate() {
            let at = module_info::SEGMENTS + i * 16;
            let addr = seg.p_vaddr + bias;
            buf[at..at + 8].copy_from_slice(&addr.to_le_bytes());
            buf[at + 8..at + 12].copy_from_slice(&(seg.span() as u32).to_le_bytes());
            // prot: PF_R|PF_W|PF_X bits (the public flags).
            let mut prot = 0u32;
            if seg.p_flags & crate::loader::pf::R != 0 {
                prot |= 4;
            }
            if seg.p_flags & crate::loader::pf::W != 0 {
                prot |= 2;
            }
            if seg.p_flags & crate::loader::pf::X != 0 {
                prot |= 1;
            }
            buf[at + 12..at + 16].copy_from_slice(&prot.to_le_bytes());
        }
        buf[module_info::SEGMENT_COUNT..module_info::SEGMENT_COUNT + 4]
            .copy_from_slice(&(count as u32).to_le_bytes());
        // fingerprint: PDG modules carry none — zeros, the honest value.
        mem.write_buf(info_va, &buf).map_err(|_| errno::EFAULT)?;
        Ok(0)
    }

    /// `sys_dynlib_load_prx(name, 0, &id, 0)` — Sony 594.
    fn dynlib_load_prx(
        &self,
        args: &[u64; 6],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let (name_va, zero1, id_out, zero2) = (args[0], args[1], args[2], args[3]);
        if zero1 != 0 || zero2 != 0 {
            return Err(errno::EINVAL);
        }
        let name = read_cstring(mem, name_va).ok_or(errno::EFAULT)?;
        let handle = {
            let mut pfs = self.state.pfs.lock().expect("pfs");
            let pfs = pfs.as_mut().ok_or(errno::ENODEV)?;
            let mut source = crate::dynlib::PfsModuleSource::new(pfs);
            let mut vm = self.state.vm.lock().expect("vm");
            let mut dynlib = self.state.dynlib.lock().expect("dynlib");
            dynlib.load_prx(&name, &mut vm, &mut source)
        }
        .map_err(|e| match e {
            crate::dynlib::DynlibError::NotFound { .. } => errno::ENOENT,
            crate::dynlib::DynlibError::NotASharedObject { .. } => errno::ENOEXEC,
            crate::dynlib::DynlibError::ArenaExhausted { .. } => errno::ENOMEM,
            crate::dynlib::DynlibError::DepthExceeded { .. } => errno::ELOOP,
            other => {
                warn!(error = %other, "load_prx failed");
                errno::EIO
            }
        })?;
        if id_out != 0 {
            mem.write_buf(id_out, &(handle as u32).to_le_bytes())
                .map_err(|_| errno::EFAULT)?;
        }
        Ok(0)
    }

    /// `sys_dynlib_unload_prx(id, 0, 0)` — Sony 595.
    fn dynlib_unload_prx(&self, args: &[u64; 6]) -> Result<i64, i32> {
        let (handle, zero1, zero2) = (args[0], args[1], args[2]);
        if zero1 != 0 || zero2 != 0 {
            return Err(errno::EINVAL);
        }
        let mut vm = self.state.vm.lock().expect("vm");
        let mut dynlib = self.state.dynlib.lock().expect("dynlib");
        dynlib.unload_prx(handle, &mut vm).map_err(|e| match e {
            crate::dynlib::DynlibError::UnknownHandle { .. } => errno::ESRCH,
            crate::dynlib::DynlibError::StillNeeded { .. } => errno::EBUSY,
            other => {
                warn!(error = %other, "unload_prx failed");
                errno::EIO
            }
        })?;
        Ok(0)
    }
}

/// PROT_* → page flags.
fn prot_flags(p: i32) -> pdg_memory::PageFlags {
    use pdg_memory::PageFlags;
    let mut f = PageFlags::PRESENT | PageFlags::USER;
    if p & prot::PROT_WRITE != 0 {
        f |= PageFlags::RW;
    }
    if p & prot::PROT_EXEC == 0 {
        f |= PageFlags::NX;
    }
    f
}

/// Read a NUL-terminated guest string (max 4096 bytes).
fn read_cstring(mem: &mut dyn pdg_cpu::MemoryAccess, va: u64) -> Option<String> {
    let mut bytes = Vec::with_capacity(64);
    for i in 0..4096u64 {
        let b = mem.read(1, va + i).ok()? as u8;
        if b == 0 {
            break;
        }
        bytes.push(b);
    }
    String::from_utf8(bytes).ok()
}

impl SyscallHook for OrbisKernel {
    fn on_syscall(
        &mut self,
        cpu: &mut pdg_cpu::CpuState,
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<(), Fault> {
        self.state.syscalls.fetch_add(1, Ordering::Relaxed);
        let number = cpu.reg(iced_x86::Register::RAX).unwrap_or(0);
        let mut args = [0u64; 6];
        for (i, r) in ARG_REGISTERS.iter().enumerate() {
            args[i] = cpu.reg(*r).unwrap_or(0);
        }
        let result = self.dispatch(number, &args, mem);
        const CF: u64 = 1;
        match result {
            Ok(value) => {
                cpu.rflags &= !CF;
                cpu.set_reg(iced_x86::Register::RAX, value as u64)
                    .expect("rax");
            }
            Err(errno) => {
                cpu.rflags |= CF;
                cpu.set_reg(iced_x86::Register::RAX, (errno as u64) & 0xFFFF_FFFF)
                    .expect("rax");
            }
        }
        if number == nr::EXIT {
            return Err(Fault::ProcessExit {
                code: args[0] as i32,
            });
        }
        // Thread lifecycle: `thr_exit` ends this thread (and the process
        // when it was the last); a parked `_umtx_op` ends this slice with
        // the return value already in RAX. Neither is an error.
        let tid = self.state.current_tid.load(Ordering::Relaxed);
        if number == nr::THR_EXIT {
            let code = self
                .state
                .threads
                .lock()
                .expect("threads")
                .get(tid)
                .and_then(|t| t.exit_code)
                .unwrap_or(0);
            if self.state.threads_all_exited.load(Ordering::Relaxed) {
                return Err(Fault::ProcessExit { code: 0 });
            }
            return Err(Fault::ThreadExit { tid, code });
        }
        if let Some(obj) = self.state.threads.lock().expect("threads").parked_on(tid) {
            return Err(Fault::ThreadParked { tid, obj });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pfs::{MemoryBlockDevice, PfsBuilder};
    use crate::vm::ProcessMemory;
    use pdg_cpu::{CpuState, MemoryAccess};
    use pdg_memory::Bus;

    fn kernel_vm() -> VmManager {
        let bus = Arc::new(Bus::new(16 * 1024 * 1024).unwrap());
        VmManager::new(
            bus,
            0x0080_0000,
            0x0080_0000,
            0x0100_0000,
            0x3000_0000_0000,
            0x3000_0010_0000,
        )
    }

    fn kernel() -> (OrbisKernel, ProcessMemory) {
        let bus = Arc::new(Bus::new(16 * 1024 * 1024).unwrap());
        let console = Arc::new(ConsoleDevice::new());
        let vm = VmManager::new(
            bus.clone(),
            0x0080_0000,
            0x0080_0000,
            0x0100_0000,
            0x3000_0000_0000,
            0x3000_0001_0000,
        );
        let k = OrbisKernel::new(vm, console, 0x1337);
        let mem = ProcessMemory::new(bus, k.state.vm.lock().unwrap().shared_mmu());
        (k, mem)
    }

    fn mount_demo_pfs(k: &OrbisKernel) {
        let mut b = PfsBuilder::new();
        b.add_file("/app/data.txt", b"hello pfs");
        b.add_file("/app/nested/deep.bin", [0u8; 8192]);
        let img = b.build().unwrap();
        let pfs = Pfs::mount(Box::new(MemoryBlockDevice::new(img))).unwrap();
        k.mount_pfs(pfs);
    }

    /// Direct dispatch with synthetic args.
    fn call(
        k: &OrbisKernel,
        nr: u64,
        args: &[u64],
        mem: &mut dyn pdg_cpu::MemoryAccess,
    ) -> Result<i64, i32> {
        let mut a = [0u64; 6];
        a[..args.len().min(6)].copy_from_slice(&args[..args.len().min(6)]);
        k.dispatch(nr, &a, mem)
    }

    #[test]
    fn getpid_and_exit() {
        let (k, mut mem) = kernel();
        assert_eq!(call(&k, nr::GETPID, &[], &mut mem).unwrap(), 0x1337);
        assert!(call(&k, nr::EXIT, &[5], &mut mem).is_ok());
        assert_eq!(k.exit_code(), Some(5));
    }

    #[test]
    fn unknown_syscall_is_enosys_with_the_name_logged() {
        let (k, mut mem) = kernel();
        // 598 (dynlib_get_proc_param) is a *known* Sony name PDG still
        // refuses; 592 is implemented since Phase 6 and must not answer
        // ENOSYS.
        assert_eq!(call(&k, 598, &[], &mut mem).unwrap_err(), errno::ENOSYS);
        assert_eq!(call(&k, 0x7FF, &[], &mut mem).unwrap_err(), errno::ENOSYS);
    }

    #[test]
    fn console_write_and_read_stdin() {
        let (k, mut mem) = kernel();
        // "PDG" at a scratch guest buffer.
        let buf = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                buf,
                0x1000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        mem.write_buf(buf, b"PDG!").unwrap();
        assert_eq!(call(&k, nr::WRITE, &[1, buf, 4], &mut mem).unwrap(), 4);
        assert_eq!(k.state.console.output(), "PDG!");
        // fd 0 is EOF; fd 9 is bad.
        assert_eq!(call(&k, nr::READ, &[0, buf, 4], &mut mem).unwrap(), 0);
        assert_eq!(
            call(&k, nr::WRITE, &[9, buf, 4], &mut mem).unwrap_err(),
            errno::EBADF
        );
    }

    #[test]
    fn open_read_lseek_close_over_pfs() {
        let (k, mut mem) = kernel();
        mount_demo_pfs(&k);
        let buf = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                buf,
                0x1000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        let path = 0x3000_0000_0100u64;
        mem.write_buf(path, b"/app/data.txt\0").unwrap();
        let fd = call(&k, nr::OPEN, &[path, 0], &mut mem).unwrap();
        assert!(fd >= 3);
        assert_eq!(
            call(&k, nr::READ, &[fd as u64, buf, 9], &mut mem).unwrap(),
            9
        );
        let mut out = [0u8; 9];
        mem.read_buf(buf, &mut out).unwrap();
        assert_eq!(&out, b"hello pfs");
        // Second read: EOF.
        assert_eq!(
            call(&k, nr::READ, &[fd as u64, buf, 9], &mut mem).unwrap(),
            0
        );
        // Size via SEEK_END, rewind via SEEK_SET.
        assert_eq!(
            call(&k, nr::LSEEK, &[fd as u64, 0, 2], &mut mem).unwrap(),
            9
        );
        assert_eq!(
            call(&k, nr::LSEEK, &[fd as u64, 0, 0], &mut mem).unwrap(),
            0
        );
        assert_eq!(call(&k, nr::CLOSE, &[fd as u64], &mut mem).unwrap(), 0);
        assert_eq!(
            call(&k, nr::READ, &[fd as u64, buf, 1], &mut mem).unwrap_err(),
            errno::EBADF
        );
    }

    #[test]
    fn open_missing_file_is_enoent() {
        let (k, mut mem) = kernel();
        mount_demo_pfs(&k);
        let path = 0x3000_0000_0100u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                0x3000_0000_0000,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        mem.write_buf(path, b"/app/nope\0").unwrap();
        assert_eq!(
            call(&k, nr::OPEN, &[path, 0], &mut mem).unwrap_err(),
            errno::ENOENT
        );
    }

    #[test]
    fn getdirentries_packs_freebsd_dirents() {
        let (k, mut mem) = kernel();
        mount_demo_pfs(&k);
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        let path = base + 0x1000;
        mem.write_buf(path, b"/app\0").unwrap();
        let fd = call(&k, nr::OPEN, &[path, 0], &mut mem).unwrap();
        let buf = base + 0x800;
        let n = call(&k, nr::GETDIRENTRIES, &[fd as u64, buf, 512, 0], &mut mem).unwrap();
        assert!(n > 0);
        let mut raw = vec![0u8; n as usize];
        mem.read_buf(buf, &mut raw).unwrap();
        // Walk the records: names must include "." ".." "data.txt" "nested".
        let mut names = Vec::new();
        let mut off = 0usize;
        while off + 8 <= raw.len() {
            let reclen = u16::from_le_bytes(raw[off + 4..off + 6].try_into().unwrap()) as usize;
            let namlen = raw[off + 7] as usize;
            let name = String::from_utf8_lossy(&raw[off + 8..off + 8 + namlen]).into_owned();
            names.push(name);
            off += reclen;
        }
        assert!(names.contains(&".".to_owned()));
        assert!(names.contains(&"..".to_owned()));
        assert!(names.contains(&"data.txt".to_owned()));
        assert!(names.contains(&"nested".to_owned()));
        // Replaying returns 0: everything was served.
        assert_eq!(
            call(&k, nr::GETDIRENTRIES, &[fd as u64, buf, 512, 0], &mut mem).unwrap(),
            0
        );
    }

    #[test]
    fn mmap_munmap_mprotect_anonymous() {
        let (k, mut mem) = kernel();
        // Auto-placed RW page.
        let va = call(
            &k,
            nr::MMAP,
            &[
                0,
                0x1000,
                (prot::PROT_READ | prot::PROT_WRITE) as u64,
                mmap_flags::MAP_PRIVATE | mmap_flags::MAP_ANON,
            ],
            &mut mem,
        )
        .unwrap() as u64;
        assert!(va != 0);
        mem.write_buf(va, &[1, 2, 3]).unwrap();
        // PROT_NONE reservation faults on access.
        let none = call(
            &k,
            nr::MMAP,
            &[
                0,
                0x1000,
                prot::PROT_NONE as u64,
                mmap_flags::MAP_PRIVATE | mmap_flags::MAP_ANON,
            ],
            &mut mem,
        )
        .unwrap() as u64;
        assert!(mem.read(1, none).is_err());
        // Downgrade and upgrade.
        assert_eq!(
            call(
                &k,
                nr::MPROTECT,
                &[va, 0x1000, prot::PROT_READ as u64],
                &mut mem
            )
            .unwrap(),
            0
        );
        assert!(mem.write(1, va, 1).is_err());
        assert_eq!(
            call(
                &k,
                nr::MPROTECT,
                &[va, 0x1000, (prot::PROT_READ | prot::PROT_WRITE) as u64],
                &mut mem
            )
            .unwrap(),
            0
        );
        mem.write(1, va, 1).unwrap();
        // Unmap, then access faults.
        assert_eq!(call(&k, nr::MUNMAP, &[va, 0x1000], &mut mem).unwrap(), 0);
        assert!(mem.read(1, va).is_err());
        // File-backed and shared mappings are refused honestly.
        assert_eq!(
            call(
                &k,
                nr::MMAP,
                &[0, 0x1000, 1, mmap_flags::MAP_PRIVATE],
                &mut mem
            )
            .unwrap_err(),
            errno::ENODEV
        );
        assert_eq!(
            call(
                &k,
                nr::MMAP,
                &[0, 0x1000, 1, mmap_flags::MAP_SHARED | mmap_flags::MAP_ANON],
                &mut mem
            )
            .unwrap_err(),
            errno::EINVAL
        );
    }

    #[test]
    fn kqueue_kevent_registration_and_drain() {
        let (k, mut mem) = kernel();
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        let kq = call(&k, nr::KQUEUE, &[], &mut mem).unwrap() as u64;
        assert!(kq > 0);
        // Register (ident 7, EVFILT_READ) via the changelist.
        let change = base;
        let mut ev = [0u8; ev::KEVENT_SIZE];
        ev[0..8].copy_from_slice(&7u64.to_le_bytes());
        ev[8..10].copy_from_slice(&ev::EVFILT_READ.to_le_bytes());
        ev[10..12].copy_from_slice(&ev::EV_ADD.to_le_bytes());
        mem.write_buf(change, &ev).unwrap();
        let out = base + 0x800;
        // Nothing queued: zero events.
        assert_eq!(
            call(&k, nr::KEVENT, &[kq, change, 1, out, 4, 0], &mut mem).unwrap(),
            0
        );
        // The host posts two events: one registered, one not.
        k.trigger_equeue_event(kq, 7, ev::EVFILT_READ, 0xAAAA)
            .unwrap();
        k.trigger_equeue_event(kq, 8, ev::EVFILT_READ, 0xBBBB)
            .unwrap();
        let n = call(&k, nr::KEVENT, &[kq, 0, 0, out, 4, 0], &mut mem).unwrap();
        assert_eq!(n, 1);
        let mut got = [0u8; ev::KEVENT_SIZE];
        mem.read_buf(out, &mut got).unwrap();
        assert_eq!(u64::from_le_bytes(got[0..8].try_into().unwrap()), 7);
        assert_eq!(u64::from_le_bytes(got[16..24].try_into().unwrap()), 0xAAAA);
        // The unregistered event is still queued.
        let n2 = call(&k, nr::KEVENT, &[kq, 0, 0, out, 4, 0], &mut mem).unwrap();
        assert_eq!(n2, 0);
        // Unknown handles are EBADF.
        assert_eq!(
            call(&k, nr::KEVENT, &[999, 0, 0, out, 4, 0], &mut mem).unwrap_err(),
            errno::EBADF
        );
    }

    #[test]
    fn syscall_hook_sets_cf_and_rax() {
        let (k, mem_port) = kernel();
        let mut hook = k;
        let mut cpu = CpuState::new(0x1000, 0x2000);
        let mut mem = mem_port;
        // getpid: CF cleared, RAX = pid.
        cpu.set_reg(iced_x86::Register::RAX, nr::GETPID).unwrap();
        cpu.rflags = 0x3; // CF set beforehand
        hook.on_syscall(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.reg(iced_x86::Register::RAX), Some(0x1337));
        assert_eq!(cpu.rflags & 1, 0);
        // unknown: CF set, RAX = ENOSYS (598: a known-refused Sony name).
        cpu.set_reg(iced_x86::Register::RAX, 598).unwrap();
        hook.on_syscall(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.reg(iced_x86::Register::RAX), Some(78));
        assert_eq!(cpu.rflags & 1, 1);
    }

    // ----- Phase 6: stat / fstat -------------------------------------------

    #[test]
    fn stat_and_fstat_fill_the_verified_layout() {
        let (k, mut mem) = kernel();
        mount_demo_pfs(&k);
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        // stat("/app/data.txt") — 9 bytes ("hello pfs").
        let path = base;
        mem.write_buf(path, b"/app/data.txt\0").unwrap();
        let sb = base + 0x1000;
        assert_eq!(call(&k, nr::STAT, &[path, sb], &mut mem).unwrap(), 0);
        let mut raw = [0u8; stat::SIZE];
        mem.read_buf(sb, &mut raw).unwrap();
        let rd32 = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
        let rd16 = |at: usize| u16::from_le_bytes(raw[at..at + 2].try_into().unwrap());
        let rd64 = |at: usize| u64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
        assert_eq!(rd64(stat::ST_SIZE), 9);
        assert_eq!(rd16(stat::ST_MODE) & stat::S_IFMT, stat::S_IFREG);
        assert_eq!(rd16(stat::ST_NLINK), 1);
        assert!(rd32(stat::ST_INO) != 0);
        assert_eq!(rd64(stat::ST_BLOCKS), 1);
        // fstat on an open fd agrees.
        let fd = call(&k, nr::OPEN, &[path, 0], &mut mem).unwrap() as u64;
        let sb2 = base + 0x1800;
        assert_eq!(call(&k, nr::FSTAT, &[fd, sb2], &mut mem).unwrap(), 0);
        let mut raw2 = [0u8; stat::SIZE];
        mem.read_buf(sb2, &mut raw2).unwrap();
        assert_eq!(raw, raw2);
        // stat on a directory has S_IFDIR.
        let dirp = base + 0x100;
        mem.write_buf(dirp, b"/app\0").unwrap();
        assert_eq!(call(&k, nr::STAT, &[dirp, sb], &mut mem).unwrap(), 0);
        let mut raw = [0u8; stat::SIZE];
        mem.read_buf(sb, &mut raw).unwrap();
        let mode = u16::from_le_bytes(raw[stat::ST_MODE..stat::ST_MODE + 2].try_into().unwrap());
        assert_eq!(mode & stat::S_IFMT, stat::S_IFDIR);
        // Missing paths are ENOENT.
        mem.write_buf(dirp, b"/app/nope\0").unwrap();
        assert_eq!(
            call(&k, nr::STAT, &[dirp, sb], &mut mem).unwrap_err(),
            errno::ENOENT
        );
        assert_eq!(
            call(&k, nr::FSTAT, &[99, sb], &mut mem).unwrap_err(),
            errno::EBADF
        );
    }

    // ----- Phase 6: threads + umtx ------------------------------------------

    #[test]
    fn thr_create_self_exit_the_thread_lifecycle() {
        let (k, mut mem) = kernel();
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        // thr_self: the main thread is tid 1.
        let out = base;
        assert_eq!(call(&k, nr::THR_SELF, &[out], &mut mem).unwrap(), 0);
        let mut tid = [0u8; 8];
        mem.read_buf(out, &mut tid).unwrap();
        assert_eq!(u64::from_le_bytes(tid), crate::thread::MAIN_TID);
        // thr_create with a FreeBSD-shaped ucontext.
        let ctx = base + 0x1000;
        let put = |mem: &mut dyn pdg_cpu::MemoryAccess, at: u64, v: u64| {
            mem.write_buf(ctx + at, &v.to_le_bytes()).unwrap()
        };
        put(&mut mem, crate::thread::ucontext::MC_RIP, 0x0040_0500);
        put(&mut mem, crate::thread::ucontext::MC_RSP, 0x0070_0000);
        put(&mut mem, crate::thread::ucontext::MC_RDI, 0x42);
        put(&mut mem, crate::thread::ucontext::MC_RFLAGS, 0x202);
        let id_out = base + 0x800;
        assert_eq!(
            call(&k, nr::THR_CREATE, &[ctx, id_out, 0], &mut mem).unwrap(),
            0
        );
        let spawned = k.take_spawned_threads();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].tid, 2);
        assert_eq!(spawned[0].rip, 0x0040_0500);
        assert_eq!(spawned[0].rsp, 0x0070_0000);
        assert_eq!(spawned[0].rdi, 0x42);
        assert_eq!(spawned[0].rflags, 0x202);
        mem.read_buf(id_out, &mut tid).unwrap();
        assert_eq!(u64::from_le_bytes(tid), 2);
        assert_eq!(k.thread_count(), 2);
        // Flags are refused honestly.
        assert_eq!(
            call(&k, nr::THR_CREATE, &[ctx, id_out, 1], &mut mem).unwrap_err(),
            errno::EINVAL
        );
        // thr_exit of the spawned thread leaves the main one live.
        k.set_current_thread(2);
        assert_eq!(call(&k, nr::THR_EXIT, &[0], &mut mem).unwrap(), 0);
        assert!(!k.threads_all_exited());
        assert_eq!(k.thread_count(), 1);
        // The main thread exiting last ends the process: no live threads
        // remain.
        k.set_current_thread(1);
        assert_eq!(call(&k, nr::THR_EXIT, &[0], &mut mem).unwrap(), 0);
        assert!(k.threads_all_exited());
        assert_eq!(k.thread_count(), 0);
    }

    #[test]
    fn umtx_mutex_park_wake_and_handoff() {
        let (k, mut mem) = kernel();
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        let m = base; // struct umutex: m_owner u32 at +0.
        mem.write(4, m, 0).unwrap();
        // tid 1 takes the mutex.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 5, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert_eq!(mem.read(4, m).unwrap() as u32, 1);
        // tid 1 re-locks: EDEADLK.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 5, 0, 0, 0], &mut mem).unwrap_err(),
            errno::EDEADLK
        );
        // tid 2 (a second thread) contends: parks with success preset.
        k.set_current_thread(2);
        let _ = k.state.threads.lock().unwrap().create(0, 0, 0, 0, 0);
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 5, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert!(k.state.threads.lock().unwrap().is_parked(2));
        // tid 2's trylock while parked is still honest: EBUSY.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 4, 0, 0, 0], &mut mem).unwrap_err(),
            errno::EBUSY
        );
        // tid 1 unlocks: FIFO handoff — owner becomes 2, thread 2 runnable.
        k.set_current_thread(1);
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 6, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert_eq!(mem.read(4, m).unwrap() as u32, 2);
        assert!(!k.state.threads.lock().unwrap().is_parked(2));
        // tid 2 finishes.
        k.set_current_thread(2);
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 6, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert_eq!(mem.read(4, m).unwrap() as u32, 0);
        // Foreign unlock is EPERM.
        k.set_current_thread(1);
        assert_eq!(
            call(&k, nr::UMTX_OP, &[m, 6, 0, 0, 0], &mut mem).unwrap_err(),
            errno::EPERM
        );
    }

    #[test]
    fn umtx_wait_wake_value_semantics() {
        let (k, mut mem) = kernel();
        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        let flag = base;
        mem.write_buf(flag, &0u64.to_le_bytes()).unwrap();
        // WAIT with a mismatched value returns immediately (no park).
        assert_eq!(
            call(&k, nr::UMTX_OP, &[flag, 2, 7, 0, 0], &mut mem).unwrap(),
            0
        );
        assert!(!k.state.threads.lock().unwrap().is_parked(1));
        // WAIT with the matching value parks.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[flag, 2, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert!(k.state.threads.lock().unwrap().is_parked(1));
        // A second thread stores a new value and wakes.
        k.set_current_thread(2);
        let _ = k.state.threads.lock().unwrap().create(0, 0, 0, 0, 0);
        mem.write_buf(flag, &1u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&k, nr::UMTX_OP, &[flag, 3, 0, 0, 0], &mut mem).unwrap(),
            0
        );
        assert!(!k.state.threads.lock().unwrap().is_parked(1));
        // Timed waits are refused honestly.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[flag, 2, 1, 0, 0x5000], &mut mem).unwrap_err(),
            errno::ENOSYS
        );
        // Unknown ops are refused by number.
        assert_eq!(
            call(&k, nr::UMTX_OP, &[flag, 12, 0, 0, 0], &mut mem).unwrap_err(),
            errno::ENOSYS
        );
    }

    // ----- Phase 6: dynlib syscalls -----------------------------------------

    /// A PRX image the dynlib syscall path can load.
    fn demo_prx() -> Vec<u8> {
        use crate::loader::{ExportSym, Rela, SharedBuilder, rt};
        let mut p = SharedBuilder::new(0, 0x1_0000, 0);
        // pdgx_add: lea rax, [rdi + 0x1357]; ret
        p.text = vec![0x48, 0x8D, 0x87, 0x57, 0x13, 0x00, 0x00, 0xC3];
        p.data = vec![0xEF, 0xBE, 0xAD, 0xDE, 0, 0, 0, 0];
        p.exports = vec![
            ExportSym::func("pdgx_add", 0, 8),
            ExportSym::object("pdgx_version", 0x1_0000, 8),
        ];
        p.relas.push(Rela {
            r_offset: 0x1_0008,
            r_type: rt::X86_64_RELATIVE,
            r_sym: 0,
            r_addend: 0x1_0000,
        });
        p.build()
    }

    #[test]
    fn dynlib_syscalls_load_prx_dlsym_get_list_get_info() {
        let (k, mut mem) = kernel();
        // A PFS with the PRX at the documented module directory.
        let mut b = PfsBuilder::new();
        b.add_file("/app0/sce_module/libpdgx.prx", demo_prx());
        b.add_file("/app0/data.txt", b"hello");
        let img = b.build().unwrap();
        k.mount_pfs(Pfs::mount(Box::new(MemoryBlockDevice::new(img))).unwrap());

        let base = 0x3000_0000_0000u64;
        k.state
            .vm
            .lock()
            .unwrap()
            .map_anon(
                base,
                0x8000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        // load_prx(name, 0, &id, 0).
        let name = base;
        mem.write_buf(name, b"libpdgx.prx\0").unwrap();
        let id_out = base + 0x100;
        assert_eq!(
            call(&k, nr::DYNLIB_LOAD_PRX, &[name, 0, id_out, 0], &mut mem).unwrap(),
            0
        );
        let mut hid = [0u8; 4];
        mem.read_buf(id_out, &mut hid).unwrap();
        let handle = u32::from_le_bytes(hid) as u64;
        assert_eq!(handle, 2); // main would be 1; this PRX is the first.
        // get_list: one module, the PRX.
        let list = base + 0x200;
        let count_out = base + 0x300;
        assert_eq!(
            call(&k, nr::DYNLIB_GET_LIST, &[list, 4, count_out], &mut mem).unwrap(),
            0
        );
        let mut cnt = [0u8; 4];
        mem.read_buf(count_out, &mut cnt).unwrap();
        assert_eq!(u32::from_le_bytes(cnt), 1);
        let mut first = [0u8; 4];
        mem.read_buf(list, &mut first).unwrap();
        assert_eq!(u32::from_le_bytes(first) as u64, handle);
        // get_info: size-prefixed, the name, one RX segment with the
        // mapped address, prot R|X.
        let info = base + 0x1000;
        mem.write_buf(info, &(module_info::SIZE as u64).to_le_bytes())
            .unwrap();
        assert_eq!(
            call(&k, nr::DYNLIB_GET_INFO, &[handle, info], &mut mem).unwrap(),
            0
        );
        let mut raw = [0u8; module_info::SIZE];
        mem.read_buf(info, &mut raw).unwrap();
        let name_field =
            String::from_utf8_lossy(&raw[module_info::NAME..module_info::NAME + 7]).into_owned();
        assert_eq!(name_field, "libpdgx");
        let seg_addr = u64::from_le_bytes(
            raw[module_info::SEGMENTS..module_info::SEGMENTS + 8]
                .try_into()
                .unwrap(),
        );
        let seg_prot = u32::from_le_bytes(
            raw[module_info::SEGMENTS + 12..module_info::SEGMENTS + 16]
                .try_into()
                .unwrap(),
        );
        assert!(seg_addr != 0);
        assert_eq!(seg_prot, 5); // R(4) | X(1)
        let seg_count = u32::from_le_bytes(
            raw[module_info::SEGMENT_COUNT..module_info::SEGMENT_COUNT + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(seg_count, 2); // text + data
        // A too-small size prefix is EINVAL.
        mem.write_buf(info, &0x10u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&k, nr::DYNLIB_GET_INFO, &[handle, info], &mut mem).unwrap_err(),
            errno::EINVAL
        );
        // dlsym: pdgx_add's runtime address reads back a real `lea`
        // first byte at the mapped segment.
        let sym = base + 0x400;
        mem.write_buf(sym, b"pdgx_add\0").unwrap();
        let addr_out = base + 0x500;
        assert_eq!(
            call(&k, nr::DYNLIB_DLSYM, &[handle, sym, addr_out], &mut mem).unwrap(),
            0
        );
        let mut addr = [0u8; 8];
        mem.read_buf(addr_out, &mut addr).unwrap();
        let addr = u64::from_le_bytes(addr);
        assert_eq!(addr, seg_addr); // the export is at the text start
        let mut byte = [0u8; 1];
        mem.read_buf(addr, &mut byte).unwrap();
        assert_eq!(byte[0], 0x48); // the lea r64, [r64+disp32] opcode
        // dlsym misses are precise.
        mem.write_buf(sym, b"pdgx_nope\0").unwrap();
        assert_eq!(
            call(&k, nr::DYNLIB_DLSYM, &[handle, sym, addr_out], &mut mem).unwrap_err(),
            errno::ENOENT
        );
        assert_eq!(
            call(&k, nr::DYNLIB_DLSYM, &[99, sym, addr_out], &mut mem).unwrap_err(),
            errno::ESRCH
        );
        // unload_prx(id, 0, 0) works for an unneeded module.
        assert_eq!(
            call(&k, nr::DYNLIB_UNLOAD_PRX, &[handle, 0, 0], &mut mem).unwrap(),
            0
        );
        assert_eq!(
            call(&k, nr::DYNLIB_DLSYM, &[handle, sym, addr_out], &mut mem).unwrap_err(),
            errno::ESRCH
        );
        // The shape guards: nonzero reserved arguments are EINVAL.
        mem.write_buf(name, b"libpdgx.prx\0").unwrap();
        assert_eq!(
            call(&k, nr::DYNLIB_LOAD_PRX, &[name, 1, id_out, 0], &mut mem).unwrap_err(),
            errno::EINVAL
        );
        assert_eq!(
            call(&k, nr::DYNLIB_LOAD_PRX, &[name, 0, id_out, 1], &mut mem).unwrap_err(),
            errno::EINVAL
        );
    }

    #[test]
    fn thr_create_with_loaded_dynlib_allocates_tls() {
        // The session conditions: the main module loaded through the
        // dynlib with its needed closure, so tls_total > 0 and the
        // spawned thread must get its own copy of the aggregated block.
        let console = Arc::new(ConsoleDevice::new());
        let dynlib = crate::dynlib::Dynlib::new(0x1000_0000, 0x1200_0000, 0x0042_0000);
        let k = OrbisKernel::with_dynlib(kernel_vm(), console.clone(), 0x1337, dynlib);
        let mut pfs_b = PfsBuilder::new();
        // Reuse the demo PRX pair for a real closure with TLS.
        pfs_b.add_file(
            "/app0/sce_module/libpdgx.prx",
            crate::demo::runtime_demo_prx(),
        );
        pfs_b.add_file(
            "/app0/sce_module/libpdgy.prx",
            crate::demo::runtime_demo_prx2(),
        );
        let mut pfs = Pfs::mount(Box::new(MemoryBlockDevice::new(pfs_b.build().unwrap()))).unwrap();
        // Load the PRX directly so the table has TLS.
        {
            let mut vmg = k.state.vm.lock().unwrap();
            let mut table = k.state.dynlib.lock().unwrap();
            let mut source = crate::dynlib::PfsModuleSource::new(&mut pfs);
            table
                .load_prx("libpdgx.prx", &mut vmg, &mut source)
                .unwrap();
        }
        k.mount_pfs(pfs);

        // Scratch pages away from the mmap arena (thr_create allocates
        // the TLS block from the arena; a collision would be a false
        // failure).
        let base = 0x4000_0000_0000u64;
        {
            let mut vmg = k.state.vm.lock().unwrap();
            vmg.map_anon(
                base,
                0x2000,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .unwrap();
        }
        let (bus, mmu) = {
            let vmg = k.state.vm.lock().unwrap();
            (vmg.bus.clone(), vmg.shared_mmu())
        };
        let mut mem = ProcessMemory::new(bus, mmu);
        // A ucontext.
        let ctx = base + 0x1000;
        let put = |mem: &mut dyn pdg_cpu::MemoryAccess, at: u64, v: u64| {
            mem.write_buf(ctx + at, &v.to_le_bytes()).unwrap()
        };
        put(&mut mem, crate::thread::ucontext::MC_RIP, 0x1234);
        put(&mut mem, crate::thread::ucontext::MC_RSP, 0x5678);
        put(&mut mem, crate::thread::ucontext::MC_RDI, 0x42);
        put(&mut mem, crate::thread::ucontext::MC_RFLAGS, 0x202);
        let id_out = base + 0x800;
        let r = call(&k, nr::THR_CREATE, &[ctx, id_out, 0], &mut mem);
        assert_eq!(r, Ok(0), "thr_create failed: {r:?}");
        let spawned = k.take_spawned_threads();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].tid, 2);
        assert!(spawned[0].tls_base != 0);
        // The thread's TLS block holds a copy of the aggregated template.
        let mut t = [0u8; 16];
        mem.read_buf(spawned[0].tls_base, &mut t).unwrap();
        assert_eq!(&t[..8], &crate::demo::runtime_demo_prx_tls_probe());
    }
}
