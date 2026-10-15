//! # pdg-jit
//!
//! The CPU JIT tier: cranelift-based recompilation of guest x86-64 blocks
//! with loop formation, an inline software TLB over host memory windows,
//! self-modifying-code invalidation, and precise fallback to the interpreter
//! for everything tier-1 does not lift.
//!
//! The execution model is a dispatcher loop:
//!
//! 1. Look up (or compile) the block for `env.rip`.
//! 2. Call it; it returns an exit reason (see [`env::exit`]).
//! 3. Continue, halt, service the syscall, single-step a fallback, or
//!    reproduce a fault precisely by single-stepping the interpreter (which
//!    also drives IDT delivery).
//!
//! Budget semantics: blocks decrement [`JitEnv::budget`] by their
//! instruction count at entry, and loops re-check it every iteration, so
//! `run(n)` terminates even on guest infinite loops. The count is a
//! block-entry approximation, not an exact retirement count (documented in
//! the Phase-2 report).
//!
//! Safety: the generated code and helpers share the `JitEnv` through a raw
//! pointer. The dispatcher exclusively owns `&mut MemoryAccess` for the
//! duration of a block call, and no Rust references into the env are held
//! across calls — the same discipline every JIT uses.

use std::collections::HashMap;

use pdg_cpu::{FlatMemory, Interpreter, MemoryAccess, RunOutcome, SyscallHook, state::CpuState};

pub mod env;
mod lift;

use cranelift_module::Module as _;
pub use env::{JitEnv, TLB_ENTRIES};
use lift::{HelperIds, LiftError, decode_block, lift_block, make_module};

/// How many guest instructions the interpreter executes per fallback
/// episode. Unliftable stretches (push/pop/call/ret today) run in the
/// reference tier in windows of this size: big enough to amortize the
/// env <-> state sync, small enough to return to the JIT promptly when
/// liftable code resumes.
const FALLBACK_WINDOW: u64 = 32;

/// Env-var probes are process-lifetime constants; caching them keeps the
/// per-block-call hot path free of `getenv` (8M lookups dominated fib).
fn trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PDG_JIT_TRACE").is_ok())
}

fn debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PDG_JIT_DEBUG").is_ok())
}

/// A compiled block's entry point.
struct BlockEntry {
    code: extern "C" fn(*mut JitEnv) -> i64,
}

/// JIT execution engine over one logical processor.
pub struct Jit {
    /// Guest state owner + interpreter fallback tier (shares `state`).
    interp: pdg_cpu::Interpreter,
    /// Stable-address execution environment.
    env: *mut JitEnv,
    /// cranelift module (process lifetime; tier 1 does not free code —
    /// SMC invalidation drops the cache entries, and the code leaks until
    /// the Jit dies; bounded by how often the guest rewrites code pages).
    module: Option<cranelift_jit::JITModule>,
    ids: Option<Box<HelperIds>>,
    blocks: HashMap<u64, BlockEntry>,
    /// Guest code page → entry RIPs (for SMC invalidation).
    page_map: HashMap<u64, Vec<u64>>,
    /// Entry RIP → code page of *failed* compiles (negative cache).
    /// Without it, fallback-heavy code re-pays the lifter — which builds
    /// IR before discovering an unliftable instruction — on every
    /// re-entry, making the JIT slower than the interpreter (measured on
    /// recursive fib). Cleared with the page on SMC invalidation.
    failures: HashMap<u64, u64>,
    name_seq: u32,
    /// Memory translation epoch the env TLB was last validated against.
    tlb_epoch: u64,
    stats: JitStats,
}

/// Runtime telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct JitStats {
    /// Blocks compiled.
    pub blocks_compiled: u64,
    /// Block executions.
    pub block_calls: u64,
    /// Guest instructions executed by the interpreter tier (unliftable
    /// stretches, fault reproduction, SMC stores).
    pub fallback_steps: u64,
    /// SMC invalidations.
    pub smc_invalidations: u64,
}

impl Jit {
    /// Build a JIT around `state`.
    pub fn new(state: CpuState, mem: *mut dyn MemoryAccess) -> Self {
        let env = Box::into_raw(Box::new(JitEnv::new(mem)));
        Self {
            interp: Interpreter::new(state),
            env,
            module: None,
            ids: None,
            blocks: HashMap::new(),
            page_map: HashMap::new(),
            failures: HashMap::new(),
            name_seq: 0,
            tlb_epoch: 0,
            stats: JitStats::default(),
        }
    }

    /// Convenience constructor over a `FlatMemory` (tests and benches).
    pub fn with_flat(state: CpuState, mem: &mut FlatMemory) -> Self {
        let ptr = mem as *mut FlatMemory as *mut dyn MemoryAccess;
        Self::new(state, ptr)
    }

    /// Attach an HLE syscall handler (serviced on JIT SYSCALL exits).
    pub fn set_syscall_hook(&mut self, hook: Box<dyn SyscallHook>) {
        self.interp.set_syscall_hook(hook);
    }

    /// The guest CPU state (authoritative between JIT calls).
    pub fn state(&self) -> &CpuState {
        &self.interp.state
    }

    /// Telemetry.
    pub fn stats(&self) -> JitStats {
        self.stats
    }

    /// Run for up to `max_instructions` (block-entry approximation).
    pub fn run(&mut self, mem: &mut dyn MemoryAccess, max_instructions: u64) -> RunOutcome {
        // SAFETY: exclusive access for the whole call; the port rides in the
        // env for the helpers. The lifetime is erased because the env (and
        // the generated code using it) only dereferences it inside this
        // call, while `mem` is borrowed and alive — the standard JIT-context
        // discipline.
        unsafe {
            let static_mem: *mut (dyn MemoryAccess + 'static) =
                std::mem::transmute(mem as *mut dyn MemoryAccess);
            (*self.env).mem = static_mem;
        }
        self.state_to_env();
        // Validate the env TLB (and every rip-keyed cache) against the
        // port's translation epoch — a prior run may have ended after a
        // CR3 reload / INVLPG, which changes what these rips mean.
        let epoch = mem.translation_epoch();
        self.note_epoch(epoch);
        let mut budget = max_instructions as i64;
        let mut executed = 0u64;
        loop {
            let entry = unsafe { (*self.env).rip };
            let code = match self.block_for(entry) {
                Ok(c) => c,
                Err(_) => {
                    // Compile failure (decode/verify): window through the
                    // interpreter, precisely.
                    let (stop, n) = self.fallback_run(mem, FALLBACK_WINDOW);
                    executed += n;
                    if let Some(out) = stop {
                        return out;
                    }
                    continue;
                }
            };
            unsafe { (*self.env).budget = budget };
            self.stats.block_calls += 1;
            // SAFETY: the block ABI is `fn(*mut JitEnv) -> i64` by
            // construction, and the env is exclusively owned here.
            let exit_code = code(self.env);
            let before = budget;
            budget = unsafe { (*self.env).budget };
            executed += before.saturating_sub(budget).max(0) as u64;
            match exit_code {
                env::exit::CONTINUE => {
                    // Hot path: `env.rip` already holds the next entry; skip
                    // the env -> state copy (fallback/syscall arms sync
                    // themselves through their own entry points).
                    if trace_enabled() {
                        eprintln!(
                            "[jit-trace] entry={entry:#x} exit={exit_code} rip={:#x} rflags={:#x} rax={:#x}",
                            unsafe { (*self.env).rip },
                            unsafe { (*self.env).rflags },
                            unsafe { (*self.env).gpr[0] }
                        );
                    }
                    continue;
                }
                env::exit::HALT => {
                    self.env_to_state();
                    return RunOutcome::Halted {
                        instructions: executed + 1,
                        rip: self.interp.state.rip,
                    };
                }
                env::exit::SYSCALL => {
                    if let Some(out) = self.service_syscall(mem) {
                        return out;
                    }
                    unsafe { (*self.env).budget = budget };
                }
                env::exit::BUDGET => {
                    return RunOutcome::MaxInstructions {
                        instructions: executed,
                    };
                }
                env::exit::FALLBACK | env::exit::FAULT => {
                    let (stop, n) = self.fallback_run(mem, FALLBACK_WINDOW);
                    executed += n;
                    if let Some(out) = stop {
                        return out;
                    }
                }
                env::exit::RESTART => {
                    let store_addr = unsafe { (*self.env).exit_aux };
                    self.invalidate_page(store_addr >> 12);
                    self.stats.smc_invalidations += 1;
                    let (stop, n) = self.fallback_run(mem, 1);
                    executed += n;
                    if let Some(out) = stop {
                        return out;
                    }
                }
                _ => {
                    let (stop, _) = self.fallback_run(mem, 1);
                    if let Some(out) = stop {
                        return out;
                    }
                }
            }
        }
    }

    /// Interpreter fallback episode: run up to `window` guest instructions
    /// through the reference tier, then return to the JIT. Returns
    /// `(Some(outcome), steps)` when the run must stop.
    ///
    /// The env TLB is validated through the port's translation epoch —
    /// the old unconditional clear zeroed 128 KiB per fallback step and
    /// made fallback-heavy code ~700x slower than the interpreter
    /// (measured on recursive fib).
    fn fallback_run(
        &mut self,
        mem: &mut dyn MemoryAccess,
        window: u64,
    ) -> (Option<RunOutcome>, u64) {
        self.env_to_state();
        let epoch = mem.translation_epoch();
        self.note_epoch(epoch);
        let out = self.interp.run(mem, window);
        self.state_to_env();
        match out {
            RunOutcome::MaxInstructions { instructions } => {
                self.stats.fallback_steps += instructions;
                (None, instructions)
            }
            RunOutcome::Halted { instructions, .. } => {
                self.stats.fallback_steps += instructions;
                (Some(out), instructions)
            }
            _ => (Some(out), window),
        }
    }

    /// Service a syscall exit through the hook (no instruction executes).
    fn service_syscall(&mut self, mem: &mut dyn MemoryAccess) -> Option<RunOutcome> {
        self.env_to_state();
        let hook = match self.interp.syscall_hook.as_mut() {
            Some(h) => h,
            None => {
                return Some(RunOutcome::Fault(pdg_cpu::Fault::UnhandledSyscall {
                    rip: self.interp.state.rip,
                }));
            }
        };
        match hook.on_syscall(&mut self.interp.state, mem) {
            Ok(()) => {
                self.state_to_env();
                None
            }
            Err(f) => Some(RunOutcome::Fault(f)),
        }
    }

    fn state_to_env(&mut self) {
        // SAFETY: exclusive access between block calls.
        let e = unsafe { &mut *self.env };
        e.gpr = self.interp.state.gpr;
        e.rip = self.interp.state.rip;
        e.rflags = self.interp.state.rflags;
        e.exit_code = env::exit::CONTINUE;
    }

    fn env_to_state(&mut self) {
        // SAFETY: exclusive access between block calls.
        let e = unsafe { &*self.env };
        self.interp.state.gpr = e.gpr;
        self.interp.state.rip = e.rip;
        self.interp.state.rflags = e.rflags;
    }

    /// React to a translation-epoch change: the env TLB and every
    /// rip-keyed cache (blocks, negative cache, SMC bloom) describe the
    /// *previous* address space and must go. (Phase 3 will key blocks by
    /// address-space id instead.)
    fn note_epoch(&mut self, epoch: u64) {
        if epoch != self.tlb_epoch {
            // SAFETY: exclusive access between block calls.
            unsafe {
                (*self.env).clear_tlb();
                (*self.env).bloom_clear();
            }
            self.blocks.clear();
            self.page_map.clear();
            self.failures.clear();
            self.tlb_epoch = epoch;
        }
    }

    /// Look up or compile the block for `entry`.
    fn block_for(&mut self, entry: u64) -> Result<extern "C" fn(*mut JitEnv) -> i64, LiftError> {
        if let Some(b) = self.blocks.get(&entry) {
            return Ok(b.code);
        }
        if self.failures.contains_key(&entry) {
            // Negative cache: the lifter already rejected this entry.
            return Err(LiftError::Decode { rip: entry });
        }
        match self.compile_block(entry) {
            Ok(code) => Ok(code),
            Err(e) => {
                if debug_enabled() {
                    eprintln!("[jit-debug] compile failed at {entry:#x}: {e:?}");
                }
                self.failures.insert(entry, entry >> 12);
                Err(e)
            }
        }
    }

    fn compile_block(
        &mut self,
        entry: u64,
    ) -> Result<extern "C" fn(*mut JitEnv) -> i64, LiftError> {
        if self.module.is_none() {
            let mut module = make_module()?;
            let ids = Box::new(lift::make_helper_ids(&mut module));
            self.module = Some(module);
            self.ids = Some(ids);
        }
        // Decode through the memory port (fetch is read-only).
        let (instrs, back_edge) = {
            // SAFETY: exclusive access during compilation; the port is the
            // same one the dispatcher installed for this run.
            let env = self.env;
            let mem: &mut dyn MemoryAccess = unsafe { &mut *(*env).mem };
            decode_block(mem, entry)?
        };
        if instrs.is_empty() {
            return Err(LiftError::Decode { rip: entry });
        }
        let name = self.name_seq;
        self.name_seq += 1;
        let holder_ids = self.ids.as_deref_mut().expect("helper ids");
        let module = self.module.as_mut().expect("module");
        let mut ctx = lift_block(module, holder_ids, entry, &instrs, back_edge, name)?;
        if std::env::var("PDG_JIT_DUMP").is_ok() {
            eprintln!("{}", ctx.func.display());
        }
        let fid = module
            .declare_function(
                &format!("b{name}"),
                cranelift_module::Linkage::Export,
                &ctx.func.signature,
            )
            .map_err(|e| LiftError::Cranelift(e.to_string()))?;
        module
            .define_function(fid, &mut ctx)
            .map_err(|e| LiftError::Cranelift(e.to_string()))?;
        module
            .finalize_definitions()
            .map_err(|e| LiftError::Cranelift(e.to_string()))?;
        let raw = module.get_finalized_function(fid);
        if std::env::var("PDG_JIT_ASM").is_ok() {
            use iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};
            let mut bytes = Vec::new();
            unsafe {
                let mut p = raw;
                for _ in 0..512 {
                    bytes.push(std::ptr::read(p));
                    p = p.add(1);
                }
            }
            let mut d = Decoder::with_ip(64, &bytes, raw as u64, DecoderOptions::NONE);
            let mut f = NasmFormatter::new();
            let mut out = String::new();
            while d.can_decode() {
                let instr = d.decode();
                f.format(&instr, &mut out);
                println!("  {}", out);
                out.clear();
                if instr.mnemonic() == iced_x86::Mnemonic::Ret {
                    break;
                }
            }
        }
        // SAFETY: the block ABI is `fn(*mut JitEnv) -> i64` by construction.
        let code: extern "C" fn(*mut JitEnv) -> i64 = unsafe { std::mem::transmute(raw) };

        // Register in the caches + SMC bloom.
        let page = entry >> 12;
        // SAFETY: exclusive access between block calls.
        unsafe { (*self.env).bloom_set(page) };
        self.page_map.entry(page).or_default().push(entry);
        self.blocks.insert(entry, BlockEntry { code });
        self.stats.blocks_compiled += 1;
        Ok(code)
    }

    /// Drop every block whose entry lives on `page` (SMC).
    fn invalidate_page(&mut self, page: u64) {
        if let Some(entries) = self.page_map.remove(&page) {
            for e in entries {
                self.blocks.remove(&e);
            }
        }
        // Failed compiles on that page may now succeed on the new bytes.
        self.failures.retain(|_, pg| *pg != page);
        // Rebuild the bloom from the remaining code pages.
        // SAFETY: exclusive access between block calls.
        unsafe {
            (*self.env).bloom_clear();
            for &p in self.page_map.keys() {
                (*self.env).bloom_set(p);
            }
        }
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: no block is executing at drop time.
        unsafe {
            drop(Box::from_raw(self.env));
        }
    }
}

// A Jit owns its execution exclusively (one per vCPU thread); the raw
// pointers inside JitEnv bar the auto-traits, so assert by hand.
unsafe impl Send for Jit {}
