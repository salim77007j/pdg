//! The dynlib layer: the process module table, cross-module linking and
//! NID-form symbol bookkeeping — the Phase-6 kernel work.
//!
//! A PS4/PS5 process is not one image: the main executable carries
//! `DT_NEEDED` entries naming shared libraries (`.sprx`), the kernel's
//! dynamic linker loads the whole closure, resolves each module's symbol
//! relocations against the others' exports, and hands the result to
//! `dlsym`. PDG implements that machinery over the Phase-5 loader:
//!
//! * [`Dynlib::load_main`] registers the main executable (handle 1) and
//!   walks its `DT_NEEDED` closure through a [`ModuleSource`] (the mounted
//!   read-only PFS behind the Orbis layer).
//! * [`Dynlib::load_prx`] is the runtime path — the `sys_dynlib_load_prx`
//!   shape the public payload SDKs pin (`(name, 0, &id, 0)`).
//! * Every module is mapped first ([`crate::loader::map`]), and only after
//!   the whole closure exists are relocations applied
//!   ([`crate::loader::relocate`]) with a resolver that searches the
//!   module's own exports, then its needed closure depth-first — the same
//!   two-phase discipline a real dynamic linker follows, so `JUMP_SLOT`
//!   and `GLOB_DAT` relocations bind across modules.
//!
//! ## Numbering and naming (sources)
//!
//! * The syscall numbers and argument shapes (`dlsym(handle, name, &out)`,
//!   `get_list(int*, max, int*)`, `get_info(handle, &info)` with the
//!   size-prefixed info struct, `load_prx(name, 0, &id, 0)`,
//!   `unload_prx(id, 0, 0)`) are carried in [`crate::orbis`], cited to the
//!   public PS4 payload SDK whose module wrappers call them on real
//!   firmware.
//! * `DT_NEEDED` names resolve through the documented PS4 module layout:
//!   an absolute path as-is; otherwise `/app0/sce_module/<name>` — the
//!   standard application module directory convention.
//!
//! ## NIDs (the honest model)
//!
//! PlayStation modules export through 64-bit NIDs; in real system modules
//! the dynamic symbol *names* are the NIDs' public 11-character encoding
//! (the "aerolib" format), and `dlsym` takes the name string either way —
//! NID-encoded for system libraries, plain for PDG-built modules. Sony's
//! exact name-to-NID hash is not published in a source PDG can verify, so
//! [`nid`] computes PDG's own documented scheme — the first eight bytes
//! (little-endian) of SHA-256 over the symbol name, matching the public
//! 64-bit NID width — recorded beside every export. The scheme is stated
//! as PDG's own, exactly like the Phase-5 TLS model, until a real dump
//! verifies Sony's.
//!
//! ## TLS model v2
//!
//! Phase 5 set `FS` to the single image's TLS block. Phase 6 generalizes:
//! the process keeps **one aggregated TLS arena** — main template at
//! offset 0, each PRX's template appended 8-byte-aligned — and every
//! thread (the main thread included) points `FS` at its own copy of the
//! whole template. `TPOFF64` relocations are applied as
//! `module_tls_offset + addend`, so the same instruction sequence
//! addresses each thread's own block. Single-module processes behave
//! byte-identically to Phase 5 (offset 0).

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::loader::{self, ElfImage, LoadedImage, SymbolResolver};
use crate::pfs::Pfs;
use crate::vm::VmManager;

/// The main module's handle (documented PDG policy; PRXes continue at 2).
pub const MAIN_HANDLE: u64 = 1;
/// Maximum `DT_NEEDED` chain depth (a cycle detector backs it).
const MAX_NEEDED_DEPTH: usize = 8;
/// Where a needed name is searched when it carries no path.
const SCE_MODULE_DIR: &str = "/app0/sce_module/";

/// PDG's documented NID scheme: the first eight bytes (little-endian) of
/// SHA-256 over the symbol name — see the module docs for why this is
/// stated as PDG's own.
pub fn nid(name: &str) -> u64 {
    let digest = Sha256::digest(name.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"))
}

/// Where modules come from: the read-only PFS behind the Orbis layer, or
/// any test double.
pub trait ModuleSource {
    /// Read the module named `name` (a `DT_NEEDED` string or the
    /// `load_prx` path).
    fn read_module(&mut self, name: &str) -> Result<Vec<u8>, DynlibError>;
}

/// The PFS-backed source: `DT_NEEDED` name → path resolution (the
/// documented rule) → [`Pfs::read_file`].
pub struct PfsModuleSource<'a> {
    pfs: &'a mut Pfs,
}

impl<'a> PfsModuleSource<'a> {
    /// Wrap the mounted PFS.
    pub fn new(pfs: &'a mut Pfs) -> Self {
        Self { pfs }
    }
}

impl ModuleSource for PfsModuleSource<'_> {
    fn read_module(&mut self, name: &str) -> Result<Vec<u8>, DynlibError> {
        let path = resolve_needed(name);
        match self.pfs.read_file(&path) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(DynlibError::NotFound {
                name: name.to_owned(),
                tried: path,
                reason: e.to_string(),
            }),
        }
    }
}

/// The documented `DT_NEEDED` name → path rule.
pub fn resolve_needed(name: &str) -> String {
    if name.starts_with('/') {
        name.to_owned()
    } else {
        format!("{SCE_MODULE_DIR}{name}")
    }
}

/// Dynlib errors — each names the exact defect.
#[derive(Debug, thiserror::Error)]
pub enum DynlibError {
    /// The module source could not provide the named module.
    #[error("module {name:?} not found (tried {tried}: {reason})")]
    NotFound {
        /// The requested name.
        name: String,
        /// The resolved path.
        tried: String,
        /// Why the read failed.
        reason: String,
    },
    /// The module did not parse or load as ELF64/SELF.
    #[error("module {name:?} failed to load: {source}")]
    BadModule {
        /// The requested name.
        name: String,
        /// The loader error.
        source: crate::loader::ElfError,
    },
    /// `load_prx` on something that is not `ET_DYN`.
    #[error("module {name:?} is not a shared object (e_type {ty})")]
    NotASharedObject {
        /// The requested name.
        name: String,
        /// The image's `e_type`.
        ty: u16,
    },
    /// The handle names no module.
    #[error("no such module (handle {handle})")]
    UnknownHandle {
        /// The unknown handle.
        handle: u64,
    },
    /// The symbol is not exported by the module.
    #[error("module {module:?} does not export {symbol:?}")]
    UnknownSymbol {
        /// The module name.
        module: String,
        /// The symbol name.
        symbol: String,
    },
    /// The `DT_NEEDED` chain is too deep (a cycle is the usual cause).
    #[error("DT_NEEDED chain too deep at {name:?} (depth {depth}, limit {MAX_NEEDED_DEPTH})")]
    DepthExceeded {
        /// The module that tripped the limit.
        name: String,
        /// The depth reached.
        depth: usize,
    },
    /// The module mapping arena is exhausted.
    #[error("module arena exhausted: need {need:#x} more, have {have:#x}")]
    ArenaExhausted {
        /// Bytes needed.
        need: u64,
        /// Bytes left.
        have: u64,
    },
    /// The module is still needed by another.
    #[error("module {handle} still needed by {needed_by:?}")]
    StillNeeded {
        /// The module handle.
        handle: u64,
        /// The depending module's name.
        needed_by: String,
    },
}

/// One loaded module in the process table.
#[derive(Debug, Clone)]
pub struct Module {
    /// Table handle (main is [`MAIN_HANDLE`]; PRXes continue from 2).
    pub handle: u64,
    /// Registered name: `"main"` for the executable, the file stem for
    /// libraries (`libpdgx.prx` → `libpdgx`).
    pub name: String,
    /// The mapped image: entry, bias, segments, TLS block descriptor.
    pub image: LoadedImage,
    /// This module's TLS-block offset inside the aggregated arena.
    pub tls_offset: u64,
    /// This module's TLS `memsz` (0 when it has no `PT_TLS`).
    pub tls_memsz: u64,
    /// `DT_NEEDED` names, in order.
    pub needed: Vec<String>,
    /// Exports: symbol name → (runtime address, size).
    exports: BTreeMap<String, (u64, u64)>,
}

impl Module {
    /// Reconstruct a module from snapshot parts (the binary codec and
    /// tests use this; the loader fills modules through `Dynlib`).
    #[allow(clippy::too_many_arguments)] // a plain data constructor
    pub fn from_parts(
        handle: u64,
        name: String,
        image: LoadedImage,
        tls_offset: u64,
        tls_memsz: u64,
        needed: Vec<String>,
        exports: BTreeMap<String, (u64, u64)>,
    ) -> Self {
        Self {
            handle,
            name,
            image,
            tls_offset,
            tls_memsz,
            needed,
            exports,
        }
    }

    /// Resolve `name` to a runtime address (the `dlsym` lookup).
    pub fn export(&self, name: &str) -> Option<u64> {
        self.exports.get(name).map(|(a, _)| *a)
    }

    /// Iterate the export table (diagnostics, snapshot, tests).
    pub fn exports(&self) -> impl Iterator<Item = (&str, u64, u64)> {
        self.exports.iter().map(|(n, (a, s))| (n.as_str(), *a, *s))
    }

    /// The NID of an exported symbol (PDG's documented scheme).
    pub fn export_nid(&self, name: &str) -> Option<u64> {
        self.exports.get(name).map(|_| nid(name))
    }
}

/// A [`Dynlib`]'s host-side state (session snapshots).
#[derive(Debug, Clone)]
pub struct DynlibSnapshot {
    /// The committed module table.
    pub modules: Vec<Module>,
    /// Next handle.
    pub next_handle: u64,
    /// PRX mapping cursor.
    pub module_cursor: u64,
    /// Arena cap.
    pub module_limit: u64,
    /// TLS arena base.
    pub tls_base: u64,
    /// Aggregated template size.
    pub tls_total: u64,
    /// TLS-arena bytes mapped.
    pub tls_mapped: u64,
}

/// The combined search table entry: (module name, exports, needed list).
type SearchEntry = (String, BTreeMap<String, (u64, u64)>, Vec<String>);

/// A module mid-load: mapped, not yet relocated.
struct PendingModule {
    handle: u64,
    name: String,
    image: ElfImage,
    loaded: LoadedImage,
    tls_offset: u64,
    tls_memsz: u64,
    needed: Vec<String>,
    exports: BTreeMap<String, (u64, u64)>,
}

/// The process module table: the Phase-6 dynamic-linking state.
pub struct Dynlib {
    modules: Vec<Module>,
    next_handle: u64,
    /// PRX mapping cursor (ET_DYN bias source), page-aligned.
    module_cursor: u64,
    module_limit: u64,
    /// Aggregated TLS arena base.
    tls_base: u64,
    /// Total aggregated template size (8-aligned; 0 when no module has
    /// TLS).
    tls_total: u64,
    /// TLS-arena bytes already mapped.
    tls_mapped: u64,
}

impl std::fmt::Debug for Dynlib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dynlib")
            .field(
                "modules",
                &self
                    .modules
                    .iter()
                    .map(|m| m.name.clone())
                    .collect::<Vec<_>>(),
            )
            .field("tls_total", &self.tls_total)
            .finish_non_exhaustive()
    }
}

impl Dynlib {
    /// An empty table. PRX images map at `[module_base, module_limit)`;
    /// the aggregated TLS arena starts at `tls_base`.
    pub fn new(module_base: u64, module_limit: u64, tls_base: u64) -> Self {
        Self {
            modules: Vec::new(),
            // Main is handle 1; PRX handles continue from 2 (documented).
            next_handle: MAIN_HANDLE + 1,
            module_cursor: module_base,
            module_limit,
            tls_base,
            tls_total: 0,
            tls_mapped: 0,
        }
    }

    /// The main thread's `FS` base: the aggregated TLS arena (0 when no
    /// module has TLS — Phase-5 behavior).
    pub fn tls_base(&self) -> u64 {
        if self.tls_total == 0 {
            0
        } else {
            self.tls_base
        }
    }

    /// Total aggregated TLS template size.
    pub fn tls_total(&self) -> u64 {
        self.tls_total
    }

    /// The module list, load order.
    pub fn modules(&self) -> &[Module] {
        &self.modules
    }

    /// The module handles, load order (the `get_list` payload).
    pub fn module_handles(&self) -> Vec<u64> {
        self.modules.iter().map(|m| m.handle).collect()
    }

    /// Look a module up by handle.
    pub fn module(&self, handle: u64) -> Result<&Module, DynlibError> {
        self.modules
            .iter()
            .find(|m| m.handle == handle)
            .ok_or(DynlibError::UnknownHandle { handle })
    }

    /// Look a module up by registered name.
    fn by_name(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// `dlsym`: resolve `name` in the module `handle` names.
    pub fn dlsym(&self, handle: u64, name: &str) -> Result<u64, DynlibError> {
        let module = self.module(handle)?;
        module.export(name).ok_or(DynlibError::UnknownSymbol {
            module: module.name.clone(),
            symbol: name.to_owned(),
        })
    }

    /// Map and register the main executable (handle 1), loading its whole
    /// `DT_NEEDED` closure and relocating everything cross-module. Returns
    /// the runtime entry point.
    pub fn load_main(
        &mut self,
        elf: ElfImage,
        vm: &mut VmManager,
        source: &mut dyn ModuleSource,
    ) -> Result<u64, DynlibError> {
        if self.modules.first().is_some_and(|m| m.name == "main") {
            return Err(DynlibError::BadModule {
                name: "main".to_owned(),
                source: crate::loader::ElfError::BadDynamic {
                    what: "the main module is already loaded",
                },
            });
        }
        let mut pending = Vec::new();
        self.load_one("main", elf, 0, MAIN_HANDLE, vm, source, &mut pending, 1)?;
        let entry = pending[0].loaded.entry;
        self.finish(pending, vm)?;
        Ok(entry)
    }

    /// `sys_dynlib_load_prx`: load the shared object `name` (plus its own
    /// needed closure), relocating everything against the full table.
    /// Returns the new module handle (or the existing one when the
    /// library is already loaded — the deduplication a real dynamic
    /// linker applies to `DT_NEEDED`).
    pub fn load_prx(
        &mut self,
        name: &str,
        vm: &mut VmManager,
        source: &mut dyn ModuleSource,
    ) -> Result<u64, DynlibError> {
        if let Some(m) = self.by_name(&stem_of(name)) {
            return Ok(m.handle);
        }
        let bytes = source.read_module(name)?;
        let elf =
            crate::loader::parse_executable(&bytes).map_err(|source| DynlibError::BadModule {
                name: name.to_owned(),
                source,
            })?;
        if elf.e_type != crate::loader::ET_DYN {
            return Err(DynlibError::NotASharedObject {
                name: name.to_owned(),
                ty: elf.e_type,
            });
        }
        // The root PRX maps at the module-arena cursor (page-aligned,
        // one guard page) — the same placement dependencies get.
        let span = crate::vm::page_ceil(dep_span(&elf)) + 0x1000;
        let have = self.module_limit - self.module_cursor;
        if span > have {
            return Err(DynlibError::ArenaExhausted { need: span, have });
        }
        let bias = self.module_cursor;
        self.module_cursor += span;
        let handle = self.next_handle;
        self.next_handle += 1;
        let mut pending = Vec::new();
        self.load_one(name, elf, bias, handle, vm, source, &mut pending, 1)?;
        self.finish(pending, vm)?;
        Ok(handle)
    }

    /// `sys_dynlib_unload_prx`: remove the module when nothing needs it.
    pub fn unload_prx(&mut self, handle: u64, vm: &mut VmManager) -> Result<(), DynlibError> {
        let target = self.module(handle)?;
        let name = target.name.clone();
        if let Some(by) = self
            .modules
            .iter()
            .find(|m| m.handle != handle && m.needed.iter().any(|n| stem_of(n) == name))
        {
            return Err(DynlibError::StillNeeded {
                handle,
                needed_by: by.name.clone(),
            });
        }
        let idx = self
            .modules
            .iter()
            .position(|m| m.handle == handle)
            .ok_or(DynlibError::UnknownHandle { handle })?;
        let m = self.modules.remove(idx);
        for seg in &m.image.segments {
            let len = crate::vm::page_ceil(seg.span());
            let _ = vm.unmap(seg.p_vaddr + m.image.bias, len);
        }
        Ok(())
    }

    // ----- internals -------------------------------------------------------

    /// Map one module, register its exports and TLS offset, queue its
    /// needed names, then recursively load the closure.
    #[allow(clippy::too_many_arguments)] // recursion payload: the caller's context is the ABI
    fn load_one(
        &mut self,
        name: &str,
        elf: ElfImage,
        bias: u64,
        handle: u64,
        vm: &mut VmManager,
        source: &mut dyn ModuleSource,
        pending: &mut Vec<PendingModule>,
        depth: usize,
    ) -> Result<(), DynlibError> {
        if depth > MAX_NEEDED_DEPTH {
            return Err(DynlibError::DepthExceeded {
                name: name.to_owned(),
                depth,
            });
        }
        let loaded = loader::map(&elf, vm, bias).map_err(|source| DynlibError::BadModule {
            name: name.to_owned(),
            source,
        })?;
        // Exports: every dynsym entry with a name and a defined section
        // index (st_shndx != SHN_UNDEF; st_value may legitimately be 0 —
        // a function at the start of its module).
        let mut exports = BTreeMap::new();
        for (i, sym) in elf.dynsyms.iter().enumerate() {
            if sym.st_shndx == 0 || sym.st_name == 0 {
                continue; // the null symbol and undefined imports
            }
            let sym_name = elf.sym_name(i as u32);
            if sym_name.is_empty() {
                continue;
            }
            exports.insert(sym_name, (sym.st_value + bias, sym.st_size));
        }
        // Needed names out of the string table.
        let needed: Vec<String> = elf.needed.iter().map(|&off| elf.dynstr_name(off)).collect();
        // TLS block offset in the aggregated arena.
        let tls_memsz = elf.tls().map_or(0, |t| t.p_memsz);
        let tls_offset = self.tls_total;
        self.tls_total = (self.tls_total + tls_memsz + 7) & !7;
        pending.push(PendingModule {
            handle,
            name: stem_of(name),
            image: elf,
            loaded,
            tls_offset,
            tls_memsz,
            needed,
            exports,
        });
        // Recurse into the needed closure (deduplicated by name — a
        // shared dependency loads once; a genuine cycle runs into the
        // depth limit above).
        for n in pending.last().expect("just pushed").needed.clone() {
            let stem = stem_of(&n);
            if self.by_name(&stem).is_some() || pending.iter().any(|p| p.name == stem) {
                continue;
            }
            let bytes = source.read_module(&n)?;
            let dep = crate::loader::parse_executable(&bytes).map_err(|source| {
                DynlibError::BadModule {
                    name: n.clone(),
                    source,
                }
            })?;
            if dep.e_type != crate::loader::ET_DYN {
                return Err(DynlibError::NotASharedObject {
                    name: n.clone(),
                    ty: dep.e_type,
                });
            }
            let span = crate::vm::page_ceil(dep_span(&dep)) + 0x1000;
            let have = self.module_limit - self.module_cursor;
            if span > have {
                return Err(DynlibError::ArenaExhausted { need: span, have });
            }
            let bias = self.module_cursor;
            self.module_cursor += span;
            let dep_handle = self.next_handle;
            self.next_handle += 1;
            self.load_one(&n, dep, bias, dep_handle, vm, source, pending, depth + 1)?;
        }
        Ok(())
    }

    /// Relocate every pending module against the full (committed +
    /// pending) export table, lay out the TLS arena, then commit.
    fn finish(
        &mut self,
        pending: Vec<PendingModule>,
        vm: &mut VmManager,
    ) -> Result<(), DynlibError> {
        // The combined search table: name → (exports, needed) for every
        // module, committed and pending.
        let mut combined: Vec<SearchEntry> = self
            .modules
            .iter()
            .map(|m| (m.name.clone(), m.exports.clone(), m.needed.clone()))
            .collect();
        for p in &pending {
            combined.push((p.name.clone(), p.exports.clone(), p.needed.clone()));
        }
        // Relocate each pending module with the full table in scope.
        let mut committed: Vec<Module> = Vec::with_capacity(pending.len());
        for p in &pending {
            let resolver = Search {
                table: &combined,
                for_module: &p.name,
            };
            let applied =
                loader::relocate(&p.image, vm, p.loaded.bias, p.tls_offset, Some(&resolver))
                    .map_err(|source| DynlibError::BadModule {
                        name: p.name.clone(),
                        source,
                    })?;
            let mut loaded = p.loaded.clone();
            loaded.relocations_applied = applied;
            committed.push(Module {
                handle: p.handle,
                name: p.name.clone(),
                image: loaded,
                tls_offset: p.tls_offset,
                tls_memsz: p.tls_memsz,
                needed: p.needed.clone(),
                exports: p.exports.clone(),
            });
        }
        // Grow the aggregated TLS arena and copy every template in.
        let want = crate::vm::page_ceil(self.tls_total);
        if want > self.tls_mapped {
            vm.map_anon(
                self.tls_base + self.tls_mapped,
                want - self.tls_mapped,
                pdg_memory::PageFlags::PRESENT
                    | pdg_memory::PageFlags::RW
                    | pdg_memory::PageFlags::USER
                    | pdg_memory::PageFlags::NX,
                true,
            )
            .map_err(|source| DynlibError::BadModule {
                name: "tls arena".to_owned(),
                source: crate::loader::ElfError::Vm(source),
            })?;
            self.tls_mapped = want;
        }
        for p in &pending {
            if p.tls_memsz == 0 {
                continue;
            }
            let t = p.image.tls().expect("tls_memsz implies PT_TLS");
            let template = &p.image.bytes[t.p_offset as usize..(t.p_offset + t.p_filesz) as usize];
            vm.write_va_loader(self.tls_base + p.tls_offset, template)
                .map_err(|source| DynlibError::BadModule {
                    name: p.name.clone(),
                    source: crate::loader::ElfError::Vm(source),
                })?;
        }
        self.modules.extend(committed);
        Ok(())
    }

    /// Capture the table's host-side state (session snapshots).
    pub fn snapshot(&self) -> DynlibSnapshot {
        DynlibSnapshot {
            modules: self.modules.clone(),
            next_handle: self.next_handle,
            module_cursor: self.module_cursor,
            module_limit: self.module_limit,
            tls_base: self.tls_base,
            tls_total: self.tls_total,
            tls_mapped: self.tls_mapped,
        }
    }

    /// Restore the table over a fresh instance (the mapped pages live in
    /// guest RAM and are restored with it).
    pub fn restore(&mut self, snap: &DynlibSnapshot) {
        self.modules = snap.modules.clone();
        self.next_handle = snap.next_handle;
        self.module_cursor = snap.module_cursor;
        self.module_limit = snap.module_limit;
        self.tls_base = snap.tls_base;
        self.tls_total = snap.tls_total;
        self.tls_mapped = snap.tls_mapped;
    }

    /// Read the aggregated TLS template back out of the arena (the
    /// authoritative copy lives in mapped guest memory) — what each new
    /// thread's block is copied from.
    pub fn read_tls_template(&self, vm: &mut VmManager) -> Result<Vec<u8>, DynlibError> {
        if self.tls_total == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; self.tls_total as usize];
        vm.read_va(self.tls_base, &mut out)
            .map_err(|source| DynlibError::BadModule {
                name: "tls arena".to_owned(),
                source: crate::loader::ElfError::Vm(source),
            })?;
        Ok(out)
    }
}

/// `libpdgx.prx` → `libpdgx` (basename without extension).
fn stem_of(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name);
    base.split_once('.')
        .map(|(stem, _)| stem.to_owned())
        .unwrap_or_else(|| base.to_owned())
}

/// Total PT_LOAD span of an image (the cursor advance adds a guard page).
fn dep_span(elf: &ElfImage) -> u64 {
    elf.phdrs
        .iter()
        .filter(|p| p.p_type == crate::loader::pt::LOAD)
        .map(|p| p.p_vaddr + p.span())
        .max()
        .unwrap_or(0)
}

/// The rtld-shaped search: the module's own exports, then its needed
/// closure depth-first.
struct Search<'a> {
    table: &'a [SearchEntry],
    for_module: &'a str,
}

impl SymbolResolver for Search<'_> {
    fn resolve(&self, name: &str) -> Option<u64> {
        fn search(
            table: &[SearchEntry],
            current: &str,
            name: &str,
            seen: &mut Vec<String>,
        ) -> Option<u64> {
            if seen.iter().any(|s| s == current) {
                return None; // cycle guard
            }
            seen.push(current.to_owned());
            let (exports, needed) = table
                .iter()
                .find(|(n, _, _)| n == current)
                .map(|(_, e, nd)| (e, nd))?;
            if let Some((addr, _)) = exports.get(name) {
                return Some(*addr);
            }
            for dep in needed {
                let stem = stem_of(dep);
                if let Some(addr) = search(table, &stem, name, seen) {
                    return Some(addr);
                }
            }
            None
        }
        search(self.table, self.for_module, name, &mut Vec::new())
    }
}
impl DynlibSnapshot {
    /// Encode (little-endian, length-prefixed).
    pub fn to_bytes(&self) -> Vec<u8> {
        fn put_str(o: &mut Vec<u8>, s: &str) {
            o.extend_from_slice(&(s.len() as u32).to_le_bytes());
            o.extend_from_slice(s.as_bytes());
        }
        fn put_ph(o: &mut Vec<u8>, p: &crate::loader::ProgramHeader) {
            o.extend_from_slice(&p.p_type.to_le_bytes());
            o.extend_from_slice(&p.p_flags.to_le_bytes());
            o.extend_from_slice(&p.p_offset.to_le_bytes());
            o.extend_from_slice(&p.p_vaddr.to_le_bytes());
            o.extend_from_slice(&p.p_filesz.to_le_bytes());
            o.extend_from_slice(&p.p_memsz.to_le_bytes());
            o.extend_from_slice(&p.p_align.to_le_bytes());
        }
        let mut o = Vec::new();
        o.extend_from_slice(&(self.modules.len() as u32).to_le_bytes());
        for m in &self.modules {
            o.extend_from_slice(&m.handle.to_le_bytes());
            put_str(&mut o, &m.name);
            o.extend_from_slice(&m.image.entry.to_le_bytes());
            o.extend_from_slice(&m.image.bias.to_le_bytes());
            o.extend_from_slice(&(m.image.segments.len() as u32).to_le_bytes());
            for seg in &m.image.segments {
                put_ph(&mut o, seg);
            }
            match &m.image.tls {
                Some(t) => {
                    o.extend_from_slice(&[1u8]);
                    o.extend_from_slice(&t.vaddr.to_le_bytes());
                    o.extend_from_slice(&t.filesz.to_le_bytes());
                    o.extend_from_slice(&t.memsz.to_le_bytes());
                }
                None => o.extend_from_slice(&[0u8]),
            }
            o.extend_from_slice(&(m.image.relocations_applied as u64).to_le_bytes());
            o.extend_from_slice(&m.tls_offset.to_le_bytes());
            o.extend_from_slice(&m.tls_memsz.to_le_bytes());
            o.extend_from_slice(&(m.needed.len() as u32).to_le_bytes());
            for n in &m.needed {
                put_str(&mut o, n);
            }
            o.extend_from_slice(&(m.exports().count() as u64).to_le_bytes());
            for (name, addr, size) in m.exports() {
                put_str(&mut o, name);
                o.extend_from_slice(&addr.to_le_bytes());
                o.extend_from_slice(&size.to_le_bytes());
            }
        }
        o.extend_from_slice(&self.next_handle.to_le_bytes());
        o.extend_from_slice(&self.module_cursor.to_le_bytes());
        o.extend_from_slice(&self.module_limit.to_le_bytes());
        o.extend_from_slice(&self.tls_base.to_le_bytes());
        o.extend_from_slice(&self.tls_total.to_le_bytes());
        o.extend_from_slice(&self.tls_mapped.to_le_bytes());
        o
    }

    /// Decode [`DynlibSnapshot::to_bytes`].
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        fn rd_str(b: &[u8], p: &mut usize) -> Option<String> {
            let n = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?) as usize;
            *p += 4;
            let s = String::from_utf8(b.get(*p..*p + n)?.to_vec()).ok()?;
            *p += n;
            Some(s)
        }
        let rd_u64 = |b: &[u8], p: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let rd_u32 = |b: &[u8], p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let mut p = 0usize;
        let nmod = rd_u32(b, &mut p)?;
        let mut modules = Vec::with_capacity(nmod as usize);
        for _ in 0..nmod {
            let handle = rd_u64(b, &mut p)?;
            let name = rd_str(b, &mut p)?;
            let entry = rd_u64(b, &mut p)?;
            let bias = rd_u64(b, &mut p)?;
            let nseg = rd_u32(b, &mut p)?;
            let mut segments = Vec::with_capacity(nseg as usize);
            for _ in 0..nseg {
                segments.push(crate::loader::ProgramHeader {
                    p_type: rd_u32(b, &mut p)?,
                    p_flags: rd_u32(b, &mut p)?,
                    p_offset: rd_u64(b, &mut p)?,
                    p_vaddr: rd_u64(b, &mut p)?,
                    p_filesz: rd_u64(b, &mut p)?,
                    p_memsz: rd_u64(b, &mut p)?,
                    p_align: rd_u64(b, &mut p)?,
                });
            }
            let tls = if *b.get(p)? == 1 {
                p += 1;
                Some(crate::loader::TlsBlock {
                    vaddr: rd_u64(b, &mut p)?,
                    filesz: rd_u64(b, &mut p)?,
                    memsz: rd_u64(b, &mut p)?,
                })
            } else {
                p += 1;
                None
            };
            let relocations_applied = rd_u64(b, &mut p)? as usize;
            let tls_offset = rd_u64(b, &mut p)?;
            let tls_memsz = rd_u64(b, &mut p)?;
            let nneed = rd_u32(b, &mut p)?;
            let mut needed = Vec::with_capacity(nneed as usize);
            for _ in 0..nneed {
                needed.push(rd_str(b, &mut p)?);
            }
            let nexp = rd_u64(b, &mut p)?;
            let mut exports = std::collections::BTreeMap::new();
            for _ in 0..nexp {
                let ename = rd_str(b, &mut p)?;
                let addr = rd_u64(b, &mut p)?;
                let size = rd_u64(b, &mut p)?;
                exports.insert(ename, (addr, size));
            }
            modules.push(Module::from_parts(
                handle,
                name,
                crate::loader::LoadedImage {
                    entry,
                    bias,
                    segments,
                    tls,
                    relocations_applied,
                },
                tls_offset,
                tls_memsz,
                needed,
                exports,
            ));
        }
        Some(Self {
            modules,
            next_handle: rd_u64(b, &mut p)?,
            module_cursor: rd_u64(b, &mut p)?,
            module_limit: rd_u64(b, &mut p)?,
            tls_base: rd_u64(b, &mut p)?,
            tls_total: rd_u64(b, &mut p)?,
            tls_mapped: rd_u64(b, &mut p)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{ExportSym, Rela, SharedBuilder, rt};
    use crate::pfs::{MemoryBlockDevice, PfsBuilder};
    use pdg_memory::Bus;
    use std::sync::Arc;

    fn vm() -> VmManager {
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

    /// The library: exports `pdgx_add` (`rdi + 0x1357`) and a data symbol
    /// `pdgx_version`, carries a 16-byte TLS template, and applies one
    /// RELATIVE relocation into its own data.
    fn prx_bytes() -> Vec<u8> {
        let mut p = SharedBuilder::new(0, 0x1_0000, 0);
        // pdgx_add: lea rax, [rdi + 0x1357]; ret
        p.text = vec![
            0x48, 0x8D, 0x87, 0x57, 0x13, 0x00, 0x00, // lea rax,[rdi+0x1357]
            0xC3, // ret
        ];
        p.data = vec![0xEF, 0xBE, 0xAD, 0xDE, 0, 0, 0, 0];
        p.exports = vec![
            ExportSym::func("pdgx_add", 0, 8),
            ExportSym::object("pdgx_version", 0x1_0000, 8),
        ];
        p.tls = Some(vec![0xAB; 16]);
        p.relas.push(Rela {
            r_offset: 0x1_0008,
            r_type: rt::X86_64_RELATIVE,
            r_sym: 0,
            r_addend: 0x1_0000,
        });
        p.build()
    }

    /// A main executable needing libpdgx.prx, with a JUMP_SLOT for
    /// `pdgx_add` and a GLOB_DAT for `pdgx_version`.
    fn main_elf() -> (Vec<u8>, u64, u64) {
        let got_add = 0x0041_0000u64;
        let got_ver = 0x0041_0008u64;
        let mut m = SharedBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        m.text = vec![0xC3];
        m.data = vec![0u8; 0x20];
        m.imports = vec!["pdgx_add".to_owned(), "pdgx_version".to_owned()];
        m.needed = vec!["libpdgx.prx".to_owned()];
        let add_idx = m.sym_index("pdgx_add").unwrap();
        let ver_idx = m.sym_index("pdgx_version").unwrap();
        m.relas.push(Rela {
            r_offset: got_add,
            r_type: rt::X86_64_JUMP_SLOT,
            r_sym: add_idx,
            r_addend: 0,
        });
        m.relas.push(Rela {
            r_offset: got_ver,
            r_type: rt::X86_64_GLOB_DAT,
            r_sym: ver_idx,
            r_addend: 0,
        });
        (m.build(), got_add, got_ver)
    }

    struct MemSource(BTreeMap<String, Vec<u8>>);
    impl ModuleSource for MemSource {
        fn read_module(&mut self, name: &str) -> Result<Vec<u8>, DynlibError> {
            self.0
                .get(name)
                .cloned()
                .ok_or_else(|| DynlibError::NotFound {
                    name: name.to_owned(),
                    tried: resolve_needed(name),
                    reason: "not in the test source".to_owned(),
                })
        }
    }

    fn loaded_prx_vm() -> (Dynlib, VmManager) {
        let mut vm = vm();
        let mut source = MemSource(BTreeMap::new());
        source.0.insert("libpdgx.prx".to_owned(), prx_bytes());
        let (main, _, _) = main_elf();
        let elf = crate::loader::parse_executable(&main).unwrap();
        let mut dynlib = Dynlib::new(0x1000_0000, 0x1200_0000, 0x0042_0000);
        dynlib.load_main(elf, &mut vm, &mut source).unwrap();
        (dynlib, vm)
    }

    #[test]
    fn nid_is_sha256_prefix() {
        let a = nid("pdgx_add");
        assert_eq!(a, nid("pdgx_add"));
        assert_ne!(a, nid("pdgx_sub"));
        let d = Sha256::digest(b"pdgx_add");
        assert_eq!(a, u64::from_le_bytes(d[..8].try_into().unwrap()));
    }

    #[test]
    fn load_main_pulls_needed_closure_and_links() {
        let (dynlib, mut vm) = loaded_prx_vm();
        assert_eq!(dynlib.modules().len(), 2);
        assert_eq!(dynlib.modules()[0].handle, MAIN_HANDLE);
        assert_eq!(dynlib.modules()[0].name, "main");
        assert_eq!(dynlib.modules()[1].name, "libpdgx");
        assert_eq!(dynlib.modules()[0].image.entry, 0x0040_0000);
        assert_eq!(
            dynlib.modules()[1].image.relocations_applied,
            1 // the one RELATIVE
        );
        assert_eq!(
            dynlib.modules()[0].image.relocations_applied,
            2 // JUMP_SLOT + GLOB_DAT
        );

        // The JUMP_SLOT slot holds the library's pdgx_add (runtime = bias).
        let lib = &dynlib.modules()[1];
        let mut buf = [0u8; 8];
        vm.read_va(0x0041_0000, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), lib.image.bias);
        // The GLOB_DAT slot holds &pdgx_version; the data is there.
        vm.read_va(0x0041_0008, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), lib.image.bias + 0x1_0000);
        vm.read_va(lib.image.bias + 0x1_0000, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), 0xDEAD_BEEF);
        // The library's RELATIVE relocation was applied.
        vm.read_va(lib.image.bias + 0x1_0008, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), lib.image.bias + 0x1_0000);

        // TLS: main has none (offset 0, size 0); the library's 16-byte
        // template sits at arena offset 0 and reads back correctly.
        assert_eq!(dynlib.modules()[1].tls_memsz, 16);
        assert_eq!(dynlib.tls_base(), 0x0042_0000);
        let template = dynlib.read_tls_template(&mut vm).unwrap();
        assert_eq!(&template, &[0xAB; 16]);
    }

    #[test]
    fn dlsym_resolves_and_reports_errors_precisely() {
        let (dynlib, _) = loaded_prx_vm();
        let lib_handle = dynlib.modules()[1].handle;
        let addr = dynlib.dlsym(lib_handle, "pdgx_add").unwrap();
        assert_eq!(addr, dynlib.modules()[1].image.bias);
        assert!(matches!(
            dynlib.dlsym(lib_handle, "nope"),
            Err(DynlibError::UnknownSymbol { .. })
        ));
        assert!(matches!(
            dynlib.dlsym(99, "pdgx_add"),
            Err(DynlibError::UnknownHandle { .. })
        ));
        // The NID bookkeeping is the documented SHA-256 prefix.
        assert_eq!(
            dynlib.modules()[1].export_nid("pdgx_add"),
            Some(nid("pdgx_add"))
        );
    }

    #[test]
    fn load_prx_at_runtime_returns_handle() {
        let mut vm = vm();
        let mut source = MemSource(BTreeMap::new());
        source.0.insert("libpdgx.prx".to_owned(), prx_bytes());
        let mut m = SharedBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        m.text = vec![0xC3];
        let elf = crate::loader::parse_executable(&m.build()).unwrap();
        let mut dynlib = Dynlib::new(0x1000_0000, 0x1200_0000, 0x0042_0000);
        dynlib.load_main(elf, &mut vm, &mut source).unwrap();
        let h = dynlib
            .load_prx("libpdgx.prx", &mut vm, &mut source)
            .unwrap();
        assert_eq!(h, 2);
        // Reloading deduplicates to the same handle.
        let h2 = dynlib
            .load_prx("libpdgx.prx", &mut vm, &mut source)
            .unwrap();
        assert_eq!(h2, 2);
        assert_eq!(dynlib.modules().len(), 2);
    }

    #[test]
    fn unload_refused_while_needed() {
        let (mut dynlib, mut vm) = loaded_prx_vm();
        assert!(matches!(
            dynlib.unload_prx(2, &mut vm),
            Err(DynlibError::StillNeeded { .. })
        ));
    }

    #[test]
    fn missing_module_is_precise() {
        let mut vm = vm();
        let mut source = MemSource(BTreeMap::new());
        let mut m = SharedBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        m.text = vec![0xC3];
        m.needed = vec!["ghost.prx".to_owned()];
        let elf = crate::loader::parse_executable(&m.build()).unwrap();
        let mut dynlib = Dynlib::new(0x1000_0000, 0x1200_0000, 0x0042_0000);
        assert!(matches!(
            dynlib.load_main(elf, &mut vm, &mut source),
            Err(DynlibError::NotFound { .. })
        ));
    }

    #[test]
    fn needed_resolves_through_pfs_sce_module() {
        let mut b = PfsBuilder::new();
        b.add_file("/app0/sce_module/libpdgx.prx", prx_bytes());
        b.add_file("/app0/data.txt", b"hello");
        let img = b.build().unwrap();
        let mut pfs = Pfs::mount(Box::new(MemoryBlockDevice::new(img))).unwrap();
        let mut src = PfsModuleSource::new(&mut pfs);
        assert_eq!(
            resolve_needed("libpdgx.prx"),
            "/app0/sce_module/libpdgx.prx"
        );
        assert!(src.read_module("libpdgx.prx").is_ok());
        assert!(matches!(
            src.read_module("missing.prx"),
            Err(DynlibError::NotFound { .. })
        ));
    }
}
