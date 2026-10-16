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
//! `fstat` is deliberately *not* implemented: the FreeBSD 9.2 amd64
//! `struct stat` layout is the next thing to verify against a real dump,
//! and `lseek(fd, 0, SEEK_END)` serves size queries honestly. The Sony
//! 588+ block is registered by name for diagnosis and refused with
//! `ENOSYS` — the dynlib/PRX work that needs it is Phase 6, and inventing
//! those ABIs now would be the guess-this-project-refuses-to-make.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
}

/// Names of the Sony-defined syscall block (588–672), from the public
/// psdevwiki-derived tables. PDG does not implement these in Phase 5;
/// they are logged on refusal so guests and developers see exactly what
/// was asked for.
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
    /// I/O error.
    pub const EIO: i32 = 5;
    /// Bad file descriptor.
    pub const EBADF: i32 = 9;
    /// Out of memory.
    pub const ENOMEM: i32 = 12;
    /// Permission denied.
    pub const EACCES: i32 = 13;
    /// Bad address.
    pub const EFAULT: i32 = 14;
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

#[derive(Debug)]
enum FdKind {
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

#[derive(Debug)]
struct GuestFd {
    path: String,
    kind: FdKind,
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
    console: Arc<ConsoleDevice>,
    pid: u64,
    exit: Mutex<Option<i32>>,
    kevent_calls: AtomicU64,
    syscalls: AtomicU64,
    last_kqueue: AtomicU64,
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
    /// A kernel over `vm` writing console output to `console`.
    pub fn new(vm: VmManager, console: Arc<ConsoleDevice>, pid: u64) -> Self {
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
                console,
                pid,
                exit: Mutex::new(None),
                kevent_calls: AtomicU64::new(0),
                syscalls: AtomicU64::new(0),
                last_kqueue: AtomicU64::new(0),
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
        assert_eq!(call(&k, 592, &[], &mut mem).unwrap_err(), errno::ENOSYS);
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
        // unknown: CF set, RAX = ENOSYS.
        cpu.set_reg(iced_x86::Register::RAX, 592).unwrap();
        hook.on_syscall(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.reg(iced_x86::Register::RAX), Some(78));
        assert_eq!(cpu.rflags & 1, 1);
    }
}
