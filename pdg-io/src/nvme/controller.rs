//! The NVMe controller: an MMIO device the guest programs, with a
//! host-side [`NvmeController::pump`] that executes latched doorbell
//! submissions against the backing [`BlockDevice`].
//!
//! The split mirrors real hardware (and the Phase-3 GNM ring): the guest
//! driver rings doorbells through MMIO; completions are *asynchronous*,
//! so the model latches submissions at doorbell time and executes them
//! at the host's scheduling point — `pump(&bus)` — posting completions
//! (result, phase tag, status) into the guest's completion queues in
//! guest RAM. A driver that polls a CQE phase bit observes exactly the
//! hardware contract.
//!
//! Modeled surface (everything else is refused with a precise status):
//! * Controller enable/reset through CC.EN with CSTS.RDY handshake.
//! * Admin: IDENTIFY (controller + namespace), CREATE/DELETE I/O CQ/SQ,
//!   ABORT (aborts nothing — no command is ever in flight across a
//!   doorbell), GET FEATURES (Number-of-Queues only), SET FEATURES
//!   (refused: no feature changes behavior).
//! * NVM I/O: READ, WRITE (PRP-resolved DMA into guest RAM), FLUSH
//!   (backing writes are synchronous, so a no-op is the truth).

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex};

use pdg_memory::bus::{AccessWidth, Bus, MmioDevice};

use super::NAMESPACE_ID;
use super::cmd::{
    AdminOpcode, CQE_SIZE, CompletionEntry, IoOpcode, SQE_SIZE, Status, SubmissionEntry,
};
use super::identify;
use super::prp;
use super::regs::{self, CtrlConfig};
use crate::block::BlockDevice;

/// Maximum queues the controller reports through GET FEATURES
/// (Number of Queues): 8 completion + 8 submission queues.
pub const MAX_IO_QUEUES: u16 = 8;

/// I/O statistics since controller enable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeStats {
    /// READ commands completed.
    pub reads: u64,
    /// WRITE commands completed.
    pub writes: u64,
    /// Bytes DMA'd into guest RAM.
    pub read_bytes: u64,
    /// Bytes DMA'd out of guest RAM.
    pub write_bytes: u64,
    /// Completions posted.
    pub completions: u64,
}

/// One submission/completion queue pair.
struct Queue {
    /// Queue id (0 = admin pair).
    id: u16,
    /// Completion queue this SQ completes into (admin: 0).
    cq_id: u16,
    /// Guest-physical SQ base.
    sq_base: u64,
    /// SQ size in entries (1-based).
    sq_size: u32,
    /// SQ head: next entry to consume.
    sq_head: u32,
    /// SQ tail: latched from the doorbell (mod size).
    sq_tail: u32,
    /// Guest-physical CQ base.
    cq_base: u64,
    /// CQ size in entries (1-based).
    cq_size: u32,
    /// CQ: next entry slot to post into.
    cq_next: u32,
    /// CQ phase tag for the current cycle.
    cq_phase: bool,
    /// Next command identifier for this SQ.
    cid_next: u16,
}

struct Inner<D: BlockDevice> {
    dev: D,
    cc: u32,
    csts: u32,
    aqa: u32,
    asq: u64,
    acq: u64,
    enabled: bool,
    queues: BTreeMap<u16, Queue>,
    lba_count: u64,
    stats: NvmeStats,
}

/// The NVMe-shaped SSD controller.
pub struct NvmeController<D: BlockDevice> {
    inner: Mutex<Inner<D>>,
    /// Optional host hook invoked (outside the device lock) after a pump
    /// posts completions — the seam the kernel's MSI-X-shaped interrupt
    /// plumbing attaches to (Phase 5).
    #[allow(clippy::type_complexity)] // the callback shape is the API
    completion_hook: Mutex<Option<Arc<dyn Fn(u64) + Send + Sync>>>,
}

impl<D: BlockDevice> NvmeController<D> {
    /// Build a controller over `dev`, sizing the namespace from the
    /// backing device length.
    pub fn new(mut dev: D) -> Self {
        let len = dev.byte_len().unwrap_or(0);
        Self {
            inner: Mutex::new(Inner {
                dev,
                cc: 0,
                csts: 0,
                aqa: 0,
                asq: 0,
                acq: 0,
                enabled: false,
                queues: BTreeMap::new(),
                lba_count: len / super::LBA_SIZE,
                stats: NvmeStats::default(),
            }),
            completion_hook: Mutex::new(None),
        }
    }

    /// Wrap in `Arc` for bus registration.
    #[must_use]
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> NvmeStats {
        self.inner.lock().expect("nvme lock poisoned").stats
    }

    /// Namespace size in 512-byte blocks.
    pub fn lba_count(&self) -> u64 {
        self.inner.lock().expect("nvme lock poisoned").lba_count
    }

    /// Install the host hook fired (outside the device lock) after a pump
    /// posts at least one completion; the argument is the completion
    /// count of that pump pass. The kernel's interrupt plumbing uses this
    /// to raise the device's vector.
    pub fn set_completion_hook(&self, hook: Arc<dyn Fn(u64) + Send + Sync>) {
        *self
            .completion_hook
            .lock()
            .expect("nvme hook lock poisoned") = Some(hook);
    }

    /// Execute everything latched by doorbell writes since the last
    /// pump: consumes pending SQ entries and posts CQEs into guest RAM.
    ///
    /// Returns the number of completions posted.
    pub fn pump(&self, bus: &Bus) -> u64 {
        let mut inner = self.inner.lock().expect("nvme lock poisoned");
        if !inner.enabled {
            return 0;
        }
        let mut posted = 0u64;
        // BTreeMap iteration is ascending by id, so the admin pair (0) is
        // serviced before the I/O queues — queue creation completes before
        // the new queue's doorbells make sense.
        let order: Vec<u16> = inner.queues.keys().copied().collect();
        for qid in order {
            loop {
                let (pending, head) = {
                    let q = inner.queues.get(&qid).expect("queue vanished during pump");
                    // A CQ-only queue (CREATE I/O CQ done, SQ not yet) has
                    // no submission side to service.
                    if q.sq_size == 0 {
                        break;
                    }
                    ((q.sq_tail + q.sq_size - q.sq_head) % q.sq_size, q.sq_head)
                };
                if pending == 0 {
                    break;
                }
                let cid = {
                    let q = inner.queues.get_mut(&qid).expect("sq");
                    let cid = q.cid_next;
                    q.cid_next = q.cid_next.wrapping_add(1);
                    cid
                };
                let Some(cmd) = read_sqe(bus, inner.queues.get(&qid).expect("sq"), head) else {
                    // Unreadable SQ entry: post a data-transfer error rather
                    // than inventing behavior.
                    let q = inner.queues.get_mut(&qid).expect("sq");
                    post_cqe(bus, q, cid, 0, Status::DataTransferError.packed());
                    q.sq_head = (q.sq_head + 1) % q.sq_size;
                    inner.stats.completions += 1;
                    posted += 1;
                    continue;
                };
                let (result, status) = execute(&mut inner, bus, &cmd, qid == 0);
                let q = inner.queues.get_mut(&qid).expect("sq");
                post_cqe(bus, q, cid, result, status);
                q.sq_head = (q.sq_head + 1) % q.sq_size;
                inner.stats.completions += 1;
                posted += 1;
            }
        }
        drop(inner);
        if posted > 0 {
            let hook = self
                .completion_hook
                .lock()
                .expect("nvme hook lock poisoned")
                .clone();
            if let Some(fire) = hook {
                fire(posted);
            }
        }
        posted
    }
}

/// Read one SQE from guest RAM; `None` when the read fails.
fn read_sqe(bus: &Bus, q: &Queue, head: u32) -> Option<SubmissionEntry> {
    let mut raw = [0u8; SQE_SIZE];
    bus.read_buf(q.sq_base + u64::from(head) * SQE_SIZE as u64, &mut raw)
        .ok()?;
    Some(SubmissionEntry::from_bytes(&raw))
}

/// Post a completion into guest RAM and advance the queue's CQ cursor.
fn post_cqe(bus: &Bus, q: &mut Queue, cid: u16, result: u32, status_packed: u32) {
    let cqe = CompletionEntry {
        result,
        sqhd: ((q.sq_head + 1) % q.sq_size) as u16,
        sqid: q.id,
        cid,
        status_packed,
        phase: q.cq_phase,
    };
    let _ = bus.write_buf(
        q.cq_base + u64::from(q.cq_next) * CQE_SIZE as u64,
        &cqe.to_bytes(),
    );
    q.cq_next = (q.cq_next + 1) % q.cq_size;
    if q.cq_next == 0 {
        // Wrapped the completion queue: invert the phase tag.
        q.cq_phase = !q.cq_phase;
    }
}

/// Execute one command; returns (CQE result, packed status).
///
/// The queue type decides the opcode namespace — admin commands live on
/// queue 0, NVM I/O commands on the I/O queues — because the byte values
/// overlap (admin `CREATE I/O SQ` 0x01 vs NVM `WRITE` 0x01).
fn execute<D: BlockDevice>(
    inner: &mut Inner<D>,
    bus: &Bus,
    cmd: &SubmissionEntry,
    admin_queue: bool,
) -> (u32, u32) {
    if cmd.fuse() != 0 {
        return (0, Status::InvalidField.packed());
    }
    if cmd.psdt() != 0 {
        return (0, Status::InvalidField.packed());
    }
    if !admin_queue {
        // NVM command: must reference the namespace honestly.
        if cmd.nsid != NAMESPACE_ID {
            return (0, Status::InvalidNamespaceOrFormat.packed());
        }
        match IoOpcode::from(cmd.opcode()) {
            IoOpcode::Read => return (0, do_read(inner, bus, cmd)),
            IoOpcode::Write => return (0, do_write(inner, bus, cmd)),
            IoOpcode::Flush => {
                // Backing writes are synchronous positioned writes; there
                // is nothing in flight to flush.
                return (0, Status::Success.packed());
            }
            IoOpcode::Unknown(op) => {
                tracing::warn!(op, "refused unknown NVM opcode");
                return (0, Status::InvalidOpcode.packed());
            }
        }
    }
    match AdminOpcode::from(cmd.opcode()) {
        AdminOpcode::Identify => do_identify(inner, bus, cmd),
        AdminOpcode::CreateIoCq => do_create_cq(inner, cmd),
        AdminOpcode::CreateIoSq => do_create_sq(inner, cmd),
        AdminOpcode::DeleteIoCq => do_delete(inner, cmd, false),
        AdminOpcode::DeleteIoSq => do_delete(inner, cmd, true),
        AdminOpcode::Abort => {
            // No command is ever in flight across a doorbell write in this
            // model, so "abort zero commands" is the honest result.
            (0, Status::Success.packed())
        }
        AdminOpcode::GetFeatures => do_get_features(cmd),
        AdminOpcode::SetFeatures => {
            tracing::warn!(
                fid = cmd.cdw10[0] & 0xFF,
                "refused SET FEATURES (no supported features)"
            );
            (0, Status::InvalidField.packed())
        }
        AdminOpcode::GetLogPage => {
            // No log pages are populated by this model; refusing is more
            // honest than returning zeros.
            (0, Status::InvalidField.packed())
        }
        AdminOpcode::Unknown(op) => {
            tracing::warn!(op, "refused unknown admin opcode");
            (0, Status::InvalidOpcode.packed())
        }
    }
}

fn do_identify<D: BlockDevice>(
    inner: &mut Inner<D>,
    bus: &Bus,
    cmd: &SubmissionEntry,
) -> (u32, u32) {
    const LEN: u64 = 4096;
    let cns = cmd.cdw10[0] & 0xFF;
    let data = match cns {
        0 => identify::identify_controller(inner.lba_count),
        1 => identify::identify_namespace(inner.lba_count),
        other => {
            tracing::warn!(cns = other, "refused unsupported IDENTIFY CNS");
            return (0, Status::InvalidField.packed());
        }
    };
    let segs = match prp::resolve(bus, cmd.prp1, cmd.prp2, LEN) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%e, "identify prp refused");
            return (0, Status::DataTransferError.packed());
        }
    };
    match prp::dma_write(bus, &segs, &data) {
        Ok(()) => (0, Status::Success.packed()),
        Err(e) => {
            tracing::warn!(%e, "identify dma failed");
            (0, Status::DataTransferError.packed())
        }
    }
}

fn do_create_cq<D: BlockDevice>(inner: &mut Inner<D>, cmd: &SubmissionEntry) -> (u32, u32) {
    let qid = (cmd.cdw10[0] >> 16) as u16;
    let entries = (cmd.cdw10[0] & 0xFFFF) + 1;
    if qid == 0 || qid >= MAX_IO_QUEUES {
        return (0, Status::InvalidField.packed());
    }
    if inner.queues.contains_key(&qid) {
        return (0, Status::InvalidField.packed());
    }
    inner.queues.insert(
        qid,
        Queue {
            id: qid,
            cq_id: qid,
            sq_base: 0,
            sq_size: 0,
            sq_head: 0,
            sq_tail: 0,
            cq_base: cmd.prp1,
            cq_size: entries,
            cq_next: 0,
            cq_phase: true,
            cid_next: 0,
        },
    );
    (0, Status::Success.packed())
}

fn do_create_sq<D: BlockDevice>(inner: &mut Inner<D>, cmd: &SubmissionEntry) -> (u32, u32) {
    let qid = (cmd.cdw10[0] >> 16) as u16;
    let entries = (cmd.cdw10[0] & 0xFFFF) + 1;
    let cqid = (cmd.cdw10[1] >> 16) as u16;
    if qid == 0 || qid >= MAX_IO_QUEUES {
        return (0, Status::InvalidField.packed());
    }
    let Some(cq) = inner.queues.get(&cqid).filter(|q| q.cq_base != 0) else {
        return (0, Status::InvalidField.packed());
    };
    if cq.cq_size == 0 {
        return (0, Status::InvalidField.packed());
    }
    match inner.queues.get_mut(&qid) {
        Some(q) if q.sq_size == 0 => {
            q.sq_base = cmd.prp1;
            q.sq_size = entries;
            q.sq_head = 0;
            q.sq_tail = 0;
            q.cq_id = cqid;
        }
        _ => return (0, Status::InvalidField.packed()),
    }
    (0, Status::Success.packed())
}

fn do_delete<D: BlockDevice>(inner: &mut Inner<D>, cmd: &SubmissionEntry, sq: bool) -> (u32, u32) {
    let qid = (cmd.cdw10[0] >> 16) as u16;
    if qid == 0 {
        return (0, Status::InvalidField.packed());
    }
    match inner.queues.get_mut(&qid) {
        None => (0, Status::InvalidField.packed()),
        Some(q) => {
            if sq {
                if q.sq_size == 0 {
                    return (0, Status::InvalidField.packed());
                }
                q.sq_base = 0;
                q.sq_size = 0;
                q.sq_head = 0;
                q.sq_tail = 0;
            } else {
                if q.sq_size != 0 {
                    return (0, Status::InvalidField.packed());
                }
                inner.queues.remove(&qid);
            }
            (0, Status::Success.packed())
        }
    }
}

fn do_get_features(cmd: &SubmissionEntry) -> (u32, u32) {
    const FEATURE_NUM_QUEUES: u32 = 0x07;
    let fid = cmd.cdw10[0] & 0xFF;
    if fid != FEATURE_NUM_QUEUES {
        return (0, Status::InvalidField.packed());
    }
    let result = (u32::from(MAX_IO_QUEUES - 1) << 16) | u32::from(MAX_IO_QUEUES - 1);
    (result, Status::Success.packed())
}

fn do_read<D: BlockDevice>(inner: &mut Inner<D>, bus: &Bus, cmd: &SubmissionEntry) -> u32 {
    let blocks = u64::from((cmd.cdw10[2] & 0xFFFF) + 1);
    let slba = u64::from(cmd.cdw10[0]) | (u64::from(cmd.cdw10[1]) << 32);
    if !slba
        .checked_add(blocks)
        .is_some_and(|end| end <= inner.lba_count)
    {
        return Status::LbaOutOfRange.packed();
    }
    let len = blocks * super::LBA_SIZE;
    let Ok(segs) = prp::resolve(bus, cmd.prp1, cmd.prp2, len) else {
        return Status::DataTransferError.packed();
    };
    let mut data = vec![0u8; len as usize];
    if read_exact_at(&mut inner.dev, slba * super::LBA_SIZE, &mut data).is_err() {
        return Status::DataTransferError.packed();
    }
    if prp::dma_write(bus, &segs, &data).is_err() {
        return Status::DataTransferError.packed();
    }
    inner.stats.reads += 1;
    inner.stats.read_bytes += len;
    Status::Success.packed()
}

fn do_write<D: BlockDevice>(inner: &mut Inner<D>, bus: &Bus, cmd: &SubmissionEntry) -> u32 {
    let blocks = u64::from((cmd.cdw10[2] & 0xFFFF) + 1);
    let slba = u64::from(cmd.cdw10[0]) | (u64::from(cmd.cdw10[1]) << 32);
    if !slba
        .checked_add(blocks)
        .is_some_and(|end| end <= inner.lba_count)
    {
        return Status::LbaOutOfRange.packed();
    }
    let len = blocks * super::LBA_SIZE;
    let Ok(segs) = prp::resolve(bus, cmd.prp1, cmd.prp2, len) else {
        return Status::DataTransferError.packed();
    };
    let mut data = vec![0u8; len as usize];
    if prp::dma_read(bus, &segs, &mut data).is_err() {
        return Status::DataTransferError.packed();
    }
    if write_all_at(&mut inner.dev, slba * super::LBA_SIZE, &data).is_err() {
        return Status::DataTransferError.packed();
    }
    inner.stats.writes += 1;
    inner.stats.write_bytes += len;
    Status::Success.packed()
}

/// Positioned read that loops until the buffer is full or EOF.
fn read_exact_at<D: BlockDevice>(dev: &mut D, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = dev.read_at(offset, &mut buf[done..])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short backing read",
            ));
        }
        done += n;
        offset += n as u64;
    }
    Ok(())
}

/// Positioned write that loops until the buffer is drained.
fn write_all_at<D: BlockDevice>(dev: &mut D, mut offset: u64, buf: &[u8]) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = dev.write_at(offset, &buf[done..])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short backing write",
            ));
        }
        done += n;
        offset += n as u64;
    }
    Ok(())
}

impl<D: BlockDevice> MmioDevice for NvmeController<D> {
    fn name(&self) -> &'static str {
        "nvme-ssd"
    }

    fn read(&self, offset: u64, _width: AccessWidth) -> u64 {
        let inner = self.inner.lock().expect("nvme lock poisoned");
        match offset {
            regs::REG_CAP => regs::CAP_VALUE,
            regs::REG_VS => u64::from(regs::VS_VALUE),
            regs::REG_CC => u64::from(inner.cc),
            regs::REG_CSTS => u64::from(inner.csts),
            regs::REG_AQA => u64::from(inner.aqa),
            regs::REG_ASQ => inner.asq,
            regs::REG_ACQ => inner.acq,
            off if off >= regs::NVME_DOORBELL_BASE => {
                let idx = ((off - regs::NVME_DOORBELL_BASE) / 4) as u16;
                let (queue, is_cq) = (idx / 2, idx % 2 == 1);
                inner
                    .queues
                    .get(&queue)
                    .map(|q| if is_cq { 0 } else { u64::from(q.sq_tail) })
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    fn write(&self, offset: u64, _width: AccessWidth, value: u64) {
        let mut inner = self.inner.lock().expect("nvme lock poisoned");
        match offset {
            regs::REG_CC => {
                let new = value as u32;
                let old_enable = inner.cc & regs::CC_EN != 0;
                let new_enable = new & regs::CC_EN != 0;
                inner.cc = new;
                if !old_enable && new_enable {
                    // Enable: the admin queue must be configured.
                    let acqes = (inner.aqa >> 24) & 0xF;
                    let asqes = (inner.aqa >> 16) & 0xF;
                    if inner.asq == 0 || inner.acq == 0 || asqes > 0xF || acqes > 0xF {
                        inner.csts |= regs::CSTS_CFS;
                        tracing::warn!("nvme enable refused: admin queue not configured");
                        return;
                    }
                    let cc = CtrlConfig::from_raw(new);
                    if cc.mps != 0 || cc.css != 0 {
                        // Only 4 KiB pages and the NVM command set.
                        inner.csts |= regs::CSTS_CFS;
                        tracing::warn!(
                            mps = cc.mps,
                            css = cc.css,
                            "nvme enable refused: cc fields"
                        );
                        return;
                    }
                    inner.queues.clear();
                    let (asq_base, acq_base) = (inner.asq, inner.acq);
                    let admin = Queue {
                        id: 0,
                        cq_id: 0,
                        sq_base: asq_base,
                        sq_size: asqes + 1,
                        sq_head: 0,
                        sq_tail: 0,
                        cq_base: acq_base,
                        cq_size: acqes + 1,
                        cq_next: 0,
                        cq_phase: true,
                        cid_next: 0,
                    };
                    let admin_entries = admin.sq_size;
                    inner.queues.insert(0, admin);
                    inner.enabled = true;
                    inner.stats = NvmeStats::default();
                    inner.csts = regs::CSTS_RDY;
                    tracing::info!(
                        asq = asq_base,
                        acq = acq_base,
                        entries = admin_entries,
                        "nvme controller enabled"
                    );
                } else if old_enable && !new_enable {
                    // Reset: queues drop, readiness clears.
                    inner.queues.clear();
                    inner.enabled = false;
                    inner.csts = 0;
                    tracing::info!("nvme controller reset");
                }
                // Shutdown notification: acknowledge completion.
                let shn = (new >> regs::CC_SHN_SHIFT) & 0x3;
                if shn == 0b11 {
                    inner.csts |= 0b10 << regs::CSTS_SHST_SHIFT;
                } else if shn == 0 {
                    inner.csts &= !(0b11 << regs::CSTS_SHST_SHIFT);
                }
            }
            regs::REG_AQA => {
                if !inner.enabled {
                    inner.aqa = value as u32;
                } else {
                    tracing::warn!("ignored AQA write while enabled");
                }
            }
            regs::REG_ASQ => {
                if !inner.enabled {
                    inner.asq = value & !0xFFFu64;
                } else {
                    tracing::warn!("ignored ASQ write while enabled");
                }
            }
            regs::REG_ACQ => {
                if !inner.enabled {
                    inner.acq = value & !0xFFFu64;
                } else {
                    tracing::warn!("ignored ACQ write while enabled");
                }
            }
            off if off >= regs::NVME_DOORBELL_BASE => {
                let idx = ((off - regs::NVME_DOORBELL_BASE) / 4) as u16;
                let (queue, is_cq) = (idx / 2, idx % 2 == 1);
                if !inner.enabled {
                    return;
                }
                if is_cq {
                    // CQ head doorbell: the driver finished consuming CQEs
                    // through this point. Recorded; the phase tag already
                    // handles entry reuse.
                    let _ = inner
                        .queues
                        .get_mut(&queue)
                        .map(|q| q.cq_next = value as u32 % q.cq_size.max(1));
                } else if let Some(q) = inner.queues.get_mut(&queue) {
                    if q.sq_size != 0 {
                        q.sq_tail = (value as u32) % q.sq_size;
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::HostFileBlockDevice;

    fn ssd() -> NvmeController<HostFileBlockDevice> {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ssd.img");
        std::fs::write(&path, vec![0u8; 64 * 1024]).unwrap();
        // Leak the tempdir on purpose: the device keeps the file open for
        // the test's lifetime and tempdirs clean up at process exit anyway.
        std::mem::forget(tmp);
        NvmeController::new(HostFileBlockDevice::open(&path).unwrap())
    }

    fn bus_with_ssd() -> (Bus, Arc<NvmeController<HostFileBlockDevice>>) {
        let mut bus = Bus::new(8 << 20).expect("bus");
        let ssd = ssd().shared();
        bus.attach_mmio(0xE000_0000, regs::NVME_FOOTPRINT, ssd.clone());
        (bus, ssd)
    }

    const BASE: u64 = 0xE000_0000;
    const ASQ: u64 = 0x200_000;
    const ACQ: u64 = 0x201_000;
    const IO_CQ: u64 = 0x300_000;
    const IO_SQ: u64 = 0x310_000;

    fn w32(bus: &Bus, off: u64, v: u32) {
        bus.write(AccessWidth::W32, BASE + off, u64::from(v))
            .unwrap();
    }

    fn r32(bus: &Bus, off: u64) -> u32 {
        bus.read(AccessWidth::W32, BASE + off).unwrap() as u32
    }

    /// Submit one command into the admin SQ at `slot`, ringing the tail
    /// doorbell with the next-free index.
    fn submit_admin(bus: &Bus, cmd: &SubmissionEntry, slot: u32) {
        bus.write_buf(ASQ + u64::from(slot) * 64, &cmd.to_bytes())
            .unwrap();
        bus.write(
            AccessWidth::W32,
            BASE + regs::NVME_DOORBELL_BASE,
            u64::from(slot + 1),
        )
        .unwrap();
    }

    /// Submit into I/O SQ 1 at `slot`.
    fn submit_io(bus: &Bus, cmd: &SubmissionEntry, slot: u32) {
        bus.write_buf(IO_SQ + u64::from(slot) * 64, &cmd.to_bytes())
            .unwrap();
        // Queue 1 SQ tail doorbell: 0x1000 + 2*1*4.
        bus.write(
            AccessWidth::W32,
            BASE + regs::NVME_DOORBELL_BASE + 8,
            u64::from(slot + 1),
        )
        .unwrap();
    }

    fn enable_controller(bus: &Bus) {
        // AQA: 8-entry admin queues (fields hold 7).
        w32(bus, regs::REG_AQA, (7 << 24) | (7 << 16));
        bus.write(AccessWidth::W64, BASE + regs::REG_ASQ, ASQ)
            .unwrap();
        bus.write(AccessWidth::W64, BASE + regs::REG_ACQ, ACQ)
            .unwrap();
        w32(bus, regs::REG_CC, 1);
        let mut spins = 0;
        while r32(bus, regs::REG_CSTS) & regs::CSTS_RDY == 0 {
            spins += 1;
            assert!(spins < 10, "controller never became ready");
        }
    }

    fn cqe_at(bus: &Bus, cq: u64, slot: u32) -> CompletionEntry {
        let mut raw = [0u8; CQE_SIZE];
        bus.read_buf(cq + u64::from(slot) * CQE_SIZE as u64, &mut raw)
            .unwrap();
        CompletionEntry::from_bytes(&raw)
    }

    fn create_io_pair(bus: &Bus, ssd: &NvmeController<HostFileBlockDevice>) {
        submit_admin(bus, &SubmissionEntry::create_io_cq(1, IO_CQ, 4, 0), 0);
        ssd.pump(bus);
        assert_eq!(cqe_at(bus, ACQ, 0).status(), Some(Status::Success));
        submit_admin(bus, &SubmissionEntry::create_io_sq(1, IO_SQ, 4, 1), 1);
        ssd.pump(bus);
        assert_eq!(cqe_at(bus, ACQ, 1).status(), Some(Status::Success));
    }

    #[test]
    fn capability_and_version_are_readable() {
        let (bus, _ssd) = bus_with_ssd();
        let cap = bus.read(AccessWidth::W64, BASE + regs::REG_CAP).unwrap();
        assert_eq!(cap, regs::CAP_VALUE);
        assert_eq!(r32(&bus, regs::REG_VS), regs::VS_VALUE);
    }

    #[test]
    fn enable_handshake_sets_ready() {
        let (bus, _ssd) = bus_with_ssd();
        enable_controller(&bus);
        assert_eq!(r32(&bus, regs::REG_CSTS) & regs::CSTS_RDY, regs::CSTS_RDY);
    }

    #[test]
    fn identify_controller_lands_in_guest_ram() {
        let (bus, ssd) = bus_with_ssd();
        enable_controller(&bus);
        submit_admin(&bus, &SubmissionEntry::identify(0, 0x400_000), 0);
        assert_eq!(ssd.pump(&bus), 1);
        let cqe = cqe_at(&bus, ACQ, 0);
        assert_eq!(cqe.status(), Some(Status::Success));
        let mut sn = [0u8; 4];
        bus.read_buf(0x400_000 + 4, &mut sn).unwrap();
        assert_eq!(&sn, b"PDG-");
    }

    #[test]
    fn queue_lifecycle_and_io_roundtrip() {
        let (bus, ssd) = bus_with_ssd();
        enable_controller(&bus);
        create_io_pair(&bus, &ssd);

        // WRITE a marker from guest RAM at LBA 4.
        bus.write_buf(0x500_000, b"PDG-IO!").unwrap();
        bus.write_buf(0x500_007, b"!").unwrap();
        submit_io(
            &bus,
            &SubmissionEntry::read_write(true, NAMESPACE_ID, 4, 1, 0x500_000, 0),
            0,
        );
        ssd.pump(&bus);
        assert_eq!(cqe_at(&bus, IO_CQ, 0).status(), Some(Status::Success));

        // READ it back into a different page and compare.
        submit_io(
            &bus,
            &SubmissionEntry::read_write(false, NAMESPACE_ID, 4, 1, 0x501_000, 0),
            1,
        );
        ssd.pump(&bus);
        assert_eq!(cqe_at(&bus, IO_CQ, 1).status(), Some(Status::Success));

        let mut out = [0u8; 8];
        bus.read_buf(0x501_000, &mut out).unwrap();
        assert_eq!(&out, b"PDG-IO!!");
        assert_eq!(ssd.stats().writes, 1);
        assert_eq!(ssd.stats().reads, 1);
    }

    #[test]
    fn out_of_range_lba_is_refused_not_wrapped() {
        let (bus, ssd) = bus_with_ssd();
        enable_controller(&bus);
        create_io_pair(&bus, &ssd);
        // The namespace is 64 KiB = 128 LBAs; LBA 200 is out of range.
        submit_io(
            &bus,
            &SubmissionEntry::read_write(false, NAMESPACE_ID, 200, 1, 0x500_000, 0),
            0,
        );
        ssd.pump(&bus);
        assert_eq!(cqe_at(&bus, IO_CQ, 0).status(), Some(Status::LbaOutOfRange));
    }

    #[test]
    fn wrong_namespace_is_refused() {
        let (bus, ssd) = bus_with_ssd();
        enable_controller(&bus);
        create_io_pair(&bus, &ssd);
        submit_io(
            &bus,
            &SubmissionEntry::read_write(false, 42, 0, 1, 0x500_000, 0),
            0,
        );
        ssd.pump(&bus);
        assert_eq!(
            cqe_at(&bus, IO_CQ, 0).status(),
            Some(Status::InvalidNamespaceOrFormat)
        );
    }

    #[test]
    fn phase_tag_inverts_when_the_completion_queue_wraps() {
        let (bus, ssd) = bus_with_ssd();
        enable_controller(&bus);
        // 8-entry admin queue: submit into all slots, one doorbell at a
        // time, so the 9th completion lands in slot 0 *after* one wrap.
        for slot in 0..8u32 {
            let cmd = SubmissionEntry::identify(0, 0x400_000);
            bus.write_buf(ASQ + u64::from(slot) * 64, &cmd.to_bytes())
                .unwrap();
            bus.write(
                AccessWidth::W32,
                BASE + regs::NVME_DOORBELL_BASE,
                u64::from(slot + 1),
            )
            .unwrap();
            ssd.pump(&bus);
            assert!(cqe_at(&bus, ACQ, slot).phase, "slot {slot}");
        }
        // 9th command reuses slot 0; the completion must carry phase 0.
        let cmd = SubmissionEntry::identify(0, 0x400_000);
        bus.write_buf(ASQ, &cmd.to_bytes()).unwrap();
        bus.write(AccessWidth::W32, BASE + regs::NVME_DOORBELL_BASE, 1)
            .unwrap();
        ssd.pump(&bus);
        let cqe = cqe_at(&bus, ACQ, 0);
        assert!(!cqe.phase, "phase tag must invert after wrap");
        assert_eq!(cqe.status(), Some(Status::Success));
    }

    #[test]
    fn reset_drops_queues_and_clears_ready() {
        let (bus, _ssd) = bus_with_ssd();
        enable_controller(&bus);
        w32(&bus, regs::REG_CC, 0);
        assert_eq!(r32(&bus, regs::REG_CSTS) & regs::CSTS_RDY, 0);
    }
}
