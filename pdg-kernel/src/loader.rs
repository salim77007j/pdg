//! ELF64 loading and the SELF container.
//!
//! PDG loads genuine ELF64 images (`ET_EXEC`, and `ET_DYN` with an explicit
//! base) — which is also a real PS5 load path: the PS5 kernel carries three
//! sysvecs ("PS4 SELF", "ELF64", "Native SELF" — public kernel strings), so
//! a plain ELF64 payload is one of the documented executable formats, not a
//! compromise.
//!
//! The loader maps every `PT_LOAD` page-by-page with per-page permissions,
//! zero-fills `memsz > filesz` tails, applies `R_X86_64_RELATIVE` and
//! `R_X86_64_TPOFF64` relocations from `PT_DYNAMIC` (`DT_RELA`), and resolves
//! symbol relocations (`R_X86_64_64`, `GLOB_DAT`, `JUMP_SLOT`) through an
//! optional resolver. Everything else — unknown relocation types, symbol
//! relocations with no resolver, malformed tables — is refused by number,
//! never guessed.
//!
//! The SELF container is parsed per its publicly documented layout (two
//! independent public readers agree on it): magic `4F 15 3D 1D`,
//! identification bytes `00 01 01 12` (version/mode/endian/attributes),
//! then key type, header/meta/file sizes, segment count, and 32-byte
//! entries `{flags, offset, file_size, mem_size}`. The public sources
//! *disagree* on what the entry `flags` bits mean (encryption, block
//! sizes, compression live there), so PDG reads **only entries whose
//! flags are zero** — plain data — and refuses anything else with the
//! exact value in the error. Encrypted or compressed segments need a real
//! dump to verify against; PDG does not guess.
//!
//! ## TLS model (documented, Phase 5)
//!
//! `PT_TLS` identifies the initial TLS block. PDG sets the thread pointer
//! (`FS` base) to the **TLS block's base address** — the host writes
//! `fs_base` after loading — and `R_X86_64_TPOFF64` addends are
//! block-relative offsets, so `fs:[tpoff]` addresses the variable. This is
//! PDG's documented model until a verified Orbis TCB layout lands with the
//! dynlib work; it is self-consistent and enforced on both ends (loader
//! and session) rather than claimed to be Sony's.

use pdg_memory::PageFlags;

use crate::vm::{VmError, VmManager, page_ceil};

// ---------------------------------------------------------------------------
// ELF64 constants (System V ABI, public)
// ---------------------------------------------------------------------------

/// ELF magic.
pub const ELFMAG: [u8; 4] = [0x7F, b'E', b'L', b'F'];
/// 64-bit class.
pub const ELFCLASS64: u8 = 2;
/// Little-endian data encoding.
pub const ELFDATA2LSB: u8 = 1;
/// Executable object.
pub const ET_EXEC: u16 = 2;
/// Shared object (position-independent).
pub const ET_DYN: u16 = 3;
/// x86-64 machine.
pub const EM_X86_64: u16 = 62;

/// Segment types PDG understands.
pub mod pt {
    /// Unused entry.
    pub const NULL: u32 = 0;
    /// Loadable segment.
    pub const LOAD: u32 = 1;
    /// Dynamic linking information.
    pub const DYNAMIC: u32 = 2;
    /// Thread-local storage template.
    pub const TLS: u32 = 7;
}

/// Segment flags.
pub mod pf {
    /// Executable.
    pub const X: u32 = 1;
    /// Writable.
    pub const W: u32 = 2;
    /// Readable.
    pub const R: u32 = 4;
}

/// Dynamic tags (the implemented subset).
pub mod dt {
    /// End of the dynamic array.
    pub const NULL: u64 = 0;
    /// Needed library (a string table offset).
    pub const NEEDED: u64 = 1;
    /// String table *address*.
    pub const STRTAB: u64 = 5;
    /// Symbol table *address*.
    pub const SYMTAB: u64 = 6;
    /// Address of the RELA relocation table.
    pub const RELA: u64 = 7;
    /// Size of the RELA table in bytes.
    pub const RELASZ: u64 = 8;
    /// Size of one RELA entry (24).
    pub const RELAENT: u64 = 9;
    /// Size of the string table in bytes.
    pub const STRSZ: u64 = 10;
    /// Size of one dynamic symbol entry (24).
    pub const SYMENT: u64 = 11;
}

/// x86-64 relocation types (the implemented subset).
pub mod rt {
    /// `*(u64*)target = S + A`.
    pub const X86_64_64: u32 = 1;
    /// `*(u64*)target = S + A` (GOT flavor).
    pub const X86_64_GLOB_DAT: u32 = 6;
    /// `*(u64*)target = S + A` (PLT-slot flavor).
    pub const X86_64_JUMP_SLOT: u32 = 7;
    /// `*(u64*)target = base_bias + A`.
    pub const X86_64_RELATIVE: u32 = 8;
    /// `*(u64*)target = S + A` as an offset from the thread pointer.
    pub const X86_64_TPOFF64: u32 = 18;
}

// ---------------------------------------------------------------------------
// SELF container constants (public layout)
// ---------------------------------------------------------------------------

/// SELF magic bytes (`4F 15 3D 1D`).
pub const SELF_MAGIC: [u8; 4] = [0x4F, 0x15, 0x3D, 0x1D];
/// Total SELF header size before the entry table.
const SELF_HEADER_SIZE: usize = 0x20;
/// One SELF entry (segment table row).
const SELF_ENTRY_SIZE: usize = 32;
/// SELF content type: a SELF (not a PUP entry).
const SELF_CONTENT_SELF: u32 = 1;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Loader errors — each names the exact defect.
#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    /// Not an ELF64 LE image.
    #[error("not a 64-bit little-endian ELF (magic {magic:02x?})")]
    BadIdent {
        /// The first four bytes as read (zeros when truncated).
        magic: [u8; 4],
    },
    /// Wrong machine.
    #[error("ELF machine {machine} is not x86-64")]
    WrongMachine {
        /// e_machine.
        machine: u16,
    },
    /// Wrong object type.
    #[error("ELF type {ty} not loadable (ET_EXEC/ET_DYN only)")]
    BadType {
        /// e_type.
        ty: u16,
    },
    /// Structure beyond the file.
    #[error("ELF structure out of bounds: {what} at {offset:#x} len {len}")]
    OutOfBounds {
        /// What was being read.
        what: &'static str,
        /// File offset.
        offset: u64,
        /// Needed length.
        len: u64,
    },
    /// Segment geometry invalid.
    #[error(
        "segment {index} has invalid geometry (vaddr {vaddr:#x}, filesz {filesz:#x}, memsz {memsz:#x})"
    )]
    BadSegment {
        /// Segment index.
        index: usize,
        /// p_vaddr.
        vaddr: u64,
        /// p_filesz.
        filesz: u64,
        /// p_memsz.
        memsz: u64,
    },
    /// No loadable segments.
    #[error("ELF has no PT_LOAD segments")]
    NoSegments,
    /// Segments overlap in the address space.
    #[error("PT_LOAD segments {a:#x} and {b:#x} overlap")]
    OverlappingSegments {
        /// First segment vaddr.
        a: u64,
        /// Second segment vaddr.
        b: u64,
    },
    /// Relocation outside every PT_LOAD.
    #[error("relocation at {offset:#x} (type {ty}, sym {sym}) outside all PT_LOAD segments")]
    RelocOutside {
        /// r_offset.
        offset: u64,
        /// Relocation type.
        ty: u32,
        /// Symbol index.
        sym: u32,
    },
    /// A symbol relocation with no resolver installed.
    #[error(
        "relocation type {ty} at {offset:#x} names symbol {sym} ({name}) but no symbol resolver is installed"
    )]
    SymbolReloc {
        /// Relocation type.
        ty: u32,
        /// r_offset.
        offset: u64,
        /// Symbol index.
        sym: u32,
        /// Symbol name (when the string table could name it).
        name: String,
    },
    /// Unsupported relocation type.
    #[error("unsupported relocation type {ty} at {offset:#x}")]
    UnsupportedReloc {
        /// Relocation type.
        ty: u32,
        /// r_offset.
        offset: u64,
    },
    /// PT_DYNAMIC malformed.
    #[error("malformed PT_DYNAMIC: {what}")]
    BadDynamic {
        /// What was wrong.
        what: &'static str,
    },
    /// SELF container refusal.
    #[error("SELF container: {0}")]
    Container(#[from] SelfError),
    /// Mapping failed.
    #[error("mapping failed: {0}")]
    Vm(#[from] VmError),
    /// Segment contains data beyond the file.
    #[error("segment data out of file bounds")]
    SegmentBeyondFile,
}

/// SELF container refusals.
#[derive(Debug, thiserror::Error)]
pub enum SelfError {
    /// Bad magic.
    #[error("not a SELF container (magic {magic:02x?})")]
    BadMagic {
        /// The first four bytes as read.
        magic: [u8; 4],
    },
    /// Identification bytes not the documented constants.
    #[error(
        "SELF identification bytes {bytes:02x?} (expected 00 01 01 xx: version 0, mode 1, little-endian)"
    )]
    BadIdent {
        /// The four identification bytes.
        bytes: [u8; 4],
    },
    /// Wrong content type.
    #[error("SELF content type {ty} (only content type 1, a SELF, is loadable)")]
    BadContentType {
        /// The content-type field.
        ty: u32,
    },
    /// Entry flags nonzero (encrypted/compressed/signed — cannot decode).
    #[error(
        "SELF entry {index} has flags {flags:#x}: PDG loads only plain (flags == 0) segments; encryption/compression bits are not verified against a real dump"
    )]
    EncryptedEntry {
        /// Entry index.
        index: usize,
        /// The raw flags value.
        flags: u64,
    },
    /// No entry contains an ELF.
    #[error("no SELF entry contains an ELF64 payload")]
    NoElf,
    /// Entry spans beyond the file.
    #[error("SELF entry {index} spans {offset:#x}..{end:#x} beyond the file ({len:#x} bytes)")]
    EntryBeyondFile {
        /// Entry index.
        index: usize,
        /// Entry offset.
        offset: u64,
        /// Entry end.
        end: u64,
        /// File length.
        len: u64,
    },
    /// Truncated container.
    #[error("SELF container truncated ({len} bytes)")]
    Truncated {
        /// File length.
        len: usize,
    },
}

// ---------------------------------------------------------------------------
// Parsed structures
// ---------------------------------------------------------------------------

/// One program header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramHeader {
    /// p_type.
    pub p_type: u32,
    /// p_flags (see [`pf`]).
    pub p_flags: u32,
    /// p_offset.
    pub p_offset: u64,
    /// p_vaddr.
    pub p_vaddr: u64,
    /// p_filesz.
    pub p_filesz: u64,
    /// p_memsz.
    pub p_memsz: u64,
    /// p_align.
    pub p_align: u64,
}

impl ProgramHeader {
    /// Virtual span `[vaddr, vaddr + span)`.
    pub fn span(&self) -> u64 {
        self.p_memsz.max(self.p_filesz)
    }

    /// PF_* → page flags (user pages, always present).
    pub fn page_flags(&self) -> PageFlags {
        let mut f = PageFlags::PRESENT | PageFlags::USER;
        if self.p_flags & pf::W != 0 {
            f |= PageFlags::RW;
        }
        if self.p_flags & pf::X == 0 {
            f |= PageFlags::NX;
        }
        f
    }
}

/// One dynamic table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dyn {
    /// d_tag.
    pub d_tag: u64,
    /// d_val / d_ptr.
    pub d_val: u64,
}

/// One dynamic symbol (the fields PDG consumes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynSym {
    /// st_name (offset into DT_STRTAB).
    pub st_name: u32,
    /// st_value (link-time address).
    pub st_value: u64,
    /// st_size.
    pub st_size: u64,
}

/// One RELA entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rela {
    /// r_offset (target virtual address).
    pub r_offset: u64,
    /// r_info type half.
    pub r_type: u32,
    /// r_info symbol half.
    pub r_sym: u32,
    /// r_addend.
    pub r_addend: i64,
}

/// A parsed (not yet mapped) ELF64 image.
#[derive(Debug)]
pub struct ElfImage {
    /// e_type.
    pub e_type: u16,
    /// e_entry (link-time).
    pub e_entry: u64,
    /// Program headers.
    pub phdrs: Vec<ProgramHeader>,
    /// Dynamic entries (empty when no PT_DYNAMIC).
    pub dynamic: Vec<Dyn>,
    /// Dynamic symbols (empty when no DT_SYMTAB).
    pub dynsyms: Vec<DynSym>,
    /// Dynamic string table (empty when no DT_STRTAB).
    pub dynstr: Vec<u8>,
    /// RELA entries (empty when no DT_RELA).
    pub relas: Vec<Rela>,
    /// DT_NEEDED string-table offsets.
    pub needed: Vec<u32>,
    /// The raw file.
    pub bytes: Vec<u8>,
}

impl ElfImage {
    /// Parse the ELF64 image at `bytes[0]`.
    pub fn parse(bytes: &[u8]) -> Result<Self, ElfError> {
        Self::parse_at(bytes, 0)
    }

    /// Parse the ELF64 image starting at `base` inside `bytes` (the SELF
    /// path: the payload begins at a container entry offset).
    pub fn parse_at(bytes: &[u8], base: usize) -> Result<Self, ElfError> {
        let file = |o: usize, n: usize, what: &'static str| -> Result<&[u8], ElfError> {
            bytes
                .get(base + o..base + o + n)
                .ok_or(ElfError::OutOfBounds {
                    what,
                    offset: (base + o) as u64,
                    len: n as u64,
                })
        };
        let rd_u16 = |o: usize, what: &'static str| -> Result<u16, ElfError> {
            Ok(u16::from_le_bytes(
                file(o, 2, what)?.try_into().expect("2 bytes"),
            ))
        };
        let rd_u32 = |o: usize, what: &'static str| -> Result<u32, ElfError> {
            Ok(u32::from_le_bytes(
                file(o, 4, what)?.try_into().expect("4 bytes"),
            ))
        };
        let rd_u64 = |o: usize, what: &'static str| -> Result<u64, ElfError> {
            Ok(u64::from_le_bytes(
                file(o, 8, what)?.try_into().expect("8 bytes"),
            ))
        };

        let mut magic = [0u8; 4];
        if bytes.len() >= base + 4 {
            magic.copy_from_slice(&bytes[base..base + 4]);
        }
        let ident = file(0, 16, "e_ident")?;
        if ident[..4] != ELFMAG || ident[4] != ELFCLASS64 || ident[5] != ELFDATA2LSB {
            return Err(ElfError::BadIdent { magic });
        }
        let e_type = rd_u16(0x10, "e_type")?;
        let e_machine = rd_u16(0x12, "e_machine")?;
        if e_machine != EM_X86_64 {
            return Err(ElfError::WrongMachine { machine: e_machine });
        }
        if e_type != ET_EXEC && e_type != ET_DYN {
            return Err(ElfError::BadType { ty: e_type });
        }
        let e_entry = rd_u64(0x18, "e_entry")?;
        let e_phoff = rd_u64(0x20, "e_phoff")? as usize;
        let e_phentsize = rd_u16(0x36, "e_phentsize")? as usize;
        let e_phnum = rd_u16(0x38, "e_phnum")? as usize;
        if e_phentsize != 56 {
            return Err(ElfError::BadDynamic {
                what: "e_phentsize != 56",
            });
        }

        let mut phdrs = Vec::with_capacity(e_phnum);
        for i in 0..e_phnum {
            let o = e_phoff + i * e_phentsize;
            phdrs.push(ProgramHeader {
                p_type: rd_u32(o, "p_type")?,
                p_flags: rd_u32(o + 4, "p_flags")?,
                p_offset: rd_u64(o + 8, "p_offset")?,
                p_vaddr: rd_u64(o + 16, "p_vaddr")?,
                p_filesz: rd_u64(o + 32, "p_filesz")?,
                p_memsz: rd_u64(o + 40, "p_memsz")?,
                p_align: rd_u64(o + 48, "p_align")?,
            });
        }
        if !phdrs.iter().any(|p| p.p_type == pt::LOAD) {
            return Err(ElfError::NoSegments);
        }
        for (i, p) in phdrs.iter().enumerate() {
            if p.p_type == pt::LOAD {
                if p.p_filesz > p.p_memsz {
                    return Err(ElfError::BadSegment {
                        index: i,
                        vaddr: p.p_vaddr,
                        filesz: p.p_filesz,
                        memsz: p.p_memsz,
                    });
                }
                if p.p_offset
                    .checked_add(p.p_filesz)
                    .is_none_or(|e| e as usize > bytes.len() - base)
                {
                    return Err(ElfError::OutOfBounds {
                        what: "PT_LOAD data",
                        offset: p.p_offset,
                        len: p.p_filesz,
                    });
                }
            }
        }
        // PT_LOAD overlap check (page-granular).
        let loads: Vec<&ProgramHeader> = phdrs.iter().filter(|p| p.p_type == pt::LOAD).collect();
        for i in 0..loads.len() {
            for j in i + 1..loads.len() {
                let (x, y) = (loads[i], loads[j]);
                let xe = x.p_vaddr + page_ceil(x.span());
                let ye = y.p_vaddr + page_ceil(y.span());
                if x.p_vaddr < ye && y.p_vaddr < xe {
                    return Err(ElfError::OverlappingSegments {
                        a: x.p_vaddr,
                        b: y.p_vaddr,
                    });
                }
            }
        }

        // PT_DYNAMIC.
        let mut dynamic = Vec::new();
        if let Some(d) = phdrs.iter().find(|p| p.p_type == pt::DYNAMIC) {
            if d.p_filesz % 16 != 0 {
                return Err(ElfError::BadDynamic {
                    what: "PT_DYNAMIC size not a multiple of 16",
                });
            }
            for k in 0..d.p_filesz / 16 {
                let o = (d.p_offset + k * 16) as usize;
                let tag = rd_u64(o, "d_tag")?;
                let val = rd_u64(o + 8, "d_val")?;
                if tag == dt::NULL {
                    break;
                }
                dynamic.push(Dyn {
                    d_tag: tag,
                    d_val: val,
                });
            }
        }
        let find_tag = |tag: u64| dynamic.iter().find(|d| d.d_tag == tag).map(|d| d.d_val);
        // DT_SYMTAB/DT_STRTAB/DT_RELA are *virtual addresses*; map them
        // back to file offsets through the PT_LOAD that contains them.
        let va_to_off = |va: u64| -> Option<usize> {
            phdrs.iter().find_map(|p| {
                (p.p_type == pt::LOAD && va >= p.p_vaddr && va < p.p_vaddr + p.p_filesz)
                    .then(|| (p.p_offset + (va - p.p_vaddr)) as usize)
            })
        };

        let mut dynstr = Vec::new();
        let mut dynsyms = Vec::new();
        let mut needed = Vec::new();
        if let (Some(strtab_va), Some(strsz)) = (find_tag(dt::STRTAB), find_tag(dt::STRSZ)) {
            let st = va_to_off(strtab_va).ok_or(ElfError::BadDynamic {
                what: "DT_STRTAB outside PT_LOAD",
            })?;
            if st + strsz as usize > bytes.len() - base {
                return Err(ElfError::OutOfBounds {
                    what: "DT_STRTAB",
                    offset: st as u64,
                    len: strsz,
                });
            }
            dynstr = bytes[base + st..base + st + strsz as usize].to_vec();
            for d in &dynamic {
                if d.d_tag == dt::NEEDED {
                    needed.push(d.d_val as u32);
                }
            }
        }
        let syment = find_tag(dt::SYMENT).unwrap_or(24);
        if syment != 24 {
            return Err(ElfError::BadDynamic {
                what: "DT_SYMENT != 24",
            });
        }
        if let Some(symtab_va) = find_tag(dt::SYMTAB) {
            if dynstr.is_empty() {
                return Err(ElfError::BadDynamic {
                    what: "DT_SYMTAB without DT_STRTAB",
                });
            }
            let so = va_to_off(symtab_va).ok_or(ElfError::BadDynamic {
                what: "DT_SYMTAB outside PT_LOAD",
            })?;
            let mut o = so;
            while base + o + 24 <= bytes.len() {
                let st_name = rd_u32(o, "st_name")?;
                let st_value = rd_u64(o + 8, "st_value")?;
                let st_size = rd_u64(o + 16, "st_size")?;
                // The symbol table's length is not tagged; the first entry
                // whose name does not fit the string table ends the walk.
                if st_name as usize >= dynstr.len() {
                    break;
                }
                dynsyms.push(DynSym {
                    st_name,
                    st_value,
                    st_size,
                });
                o += syment as usize;
            }
        }

        // DT_RELA.
        let mut relas = Vec::new();
        if let (Some(rela_va), Some(relasz)) = (find_tag(dt::RELA), find_tag(dt::RELASZ)) {
            let relaent = find_tag(dt::RELAENT).unwrap_or(24);
            if relaent != 24 {
                return Err(ElfError::BadDynamic {
                    what: "DT_RELAENT != 24",
                });
            }
            let ro = va_to_off(rela_va).ok_or(ElfError::BadDynamic {
                what: "DT_RELA outside PT_LOAD",
            })?;
            if ro + relasz as usize > bytes.len() - base {
                return Err(ElfError::OutOfBounds {
                    what: "DT_RELA",
                    offset: ro as u64,
                    len: relasz,
                });
            }
            for k in 0..relasz / 24 {
                let o = ro + (k * 24) as usize;
                let r_offset = rd_u64(o, "r_offset")?;
                let info = rd_u64(o + 8, "r_info")?;
                let r_addend = rd_u64(o + 16, "r_addend")? as i64;
                relas.push(Rela {
                    r_offset,
                    r_type: (info & 0xFFFF_FFFF) as u32,
                    r_sym: (info >> 32) as u32,
                    r_addend,
                });
            }
        }

        Ok(Self {
            e_type,
            e_entry,
            phdrs,
            dynamic,
            dynsyms,
            dynstr,
            relas,
            needed,
            bytes: bytes[base..].to_vec(),
        })
    }

    /// Name of dynamic symbol `idx` (empty when unnameable).
    pub fn sym_name(&self, idx: u32) -> String {
        self.dynsyms
            .get(idx as usize)
            .and_then(|s| {
                let end = self.dynstr[s.st_name as usize..]
                    .iter()
                    .position(|b| *b == 0)
                    .map(|p| s.st_name as usize + p)?;
                Some(String::from_utf8_lossy(&self.dynstr[s.st_name as usize..end]).into_owned())
            })
            .unwrap_or_default()
    }

    /// Address of the dynamic symbol named `name` (link-time).
    pub fn sym_addr(&self, name: &str) -> Option<u64> {
        self.dynsyms
            .iter()
            .find(|s| {
                let from = s.st_name as usize;
                from < self.dynstr.len()
                    && self.dynstr[from..].starts_with(name.as_bytes())
                    && self.dynstr[from + name.len()] == 0
            })
            .map(|s| s.st_value)
    }

    /// The PT_TLS header, when present.
    pub fn tls(&self) -> Option<&ProgramHeader> {
        self.phdrs.iter().find(|p| p.p_type == pt::TLS)
    }
}

// ---------------------------------------------------------------------------
// SELF container
// ---------------------------------------------------------------------------

/// A parsed SELF container.
#[derive(Debug)]
pub struct SelfContainer {
    /// The contained ELF image.
    pub elf: ElfImage,
    /// Product type byte (diagnostics).
    pub product_type: u8,
}

impl SelfContainer {
    /// Parse a SELF container, unwrapping the inner ELF64.
    ///
    /// Only plain (flags == 0) entries are accepted; see the module docs
    /// for the refusal rationale.
    pub fn parse(bytes: &[u8]) -> Result<Self, SelfError> {
        if bytes.len() < SELF_HEADER_SIZE {
            return Err(SelfError::Truncated { len: bytes.len() });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[..4]);
        if magic != SELF_MAGIC {
            return Err(SelfError::BadMagic { magic });
        }
        let ident = [bytes[4], bytes[5], bytes[6], bytes[7]];
        if ident[0] != 0x00 || ident[1] != 0x01 || ident[2] != 0x01 {
            return Err(SelfError::BadIdent { bytes: ident });
        }
        let content_type = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        // Low byte is the content type (1 = SELF); the next byte is the
        // product type (0x8 EBOOT/ELF, 0x9 SPRX/...).
        if content_type & 0xFF != SELF_CONTENT_SELF {
            return Err(SelfError::BadContentType { ty: content_type });
        }
        let product_type = (content_type >> 8) as u8;
        let entry_count = u16::from_le_bytes(bytes[0x18..0x1A].try_into().unwrap()) as usize;
        let file_len = bytes.len() as u64;
        let mut elf: Option<ElfImage> = None;
        for i in 0..entry_count {
            let e = SELF_HEADER_SIZE + i * SELF_ENTRY_SIZE;
            let Some(e) = bytes.get(e..e + SELF_ENTRY_SIZE) else {
                return Err(SelfError::Truncated { len: bytes.len() });
            };
            let rd = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap());
            let flags = rd(0);
            let offset = rd(8);
            let file_size = rd(16);
            if flags != 0 {
                return Err(SelfError::EncryptedEntry { index: i, flags });
            }
            let end = offset + file_size;
            if end > file_len {
                return Err(SelfError::EntryBeyondFile {
                    index: i,
                    offset,
                    end,
                    len: file_len,
                });
            }
            // The ELF payload begins where an entry's data starts with
            // the ELF magic; the entry's declared size brackets it.
            if bytes
                .get(offset as usize..offset as usize + 4)
                .is_some_and(|m| m == ELFMAG)
            {
                elf =
                    Some(ElfImage::parse_at(bytes, offset as usize).map_err(|_| SelfError::NoElf)?);
            }
        }
        Ok(Self {
            elf: elf.ok_or(SelfError::NoElf)?,
            product_type,
        })
    }
}

/// Parse `bytes` as ELF64 directly, or as a SELF container wrapping one.
pub fn parse_executable(bytes: &[u8]) -> Result<ElfImage, ElfError> {
    if bytes.len() >= 4 && bytes[..4] == ELFMAG {
        Ok(ElfImage::parse(bytes)?)
    } else {
        Ok(SelfContainer::parse(bytes)?.elf)
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Resolves symbol relocations.
pub trait SymbolResolver {
    /// Address of `name`, or `None` to refuse.
    fn resolve(&self, name: &str) -> Option<u64>;
}

impl<F: Fn(&str) -> Option<u64>> SymbolResolver for F {
    fn resolve(&self, name: &str) -> Option<u64> {
        self(name)
    }
}

/// TLS block description returned by the loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsBlock {
    /// Virtual address of the TLS template.
    pub vaddr: u64,
    /// Initialized bytes in the file.
    pub filesz: u64,
    /// Total size in memory.
    pub memsz: u64,
}

/// A successfully loaded module.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Runtime entry point.
    pub entry: u64,
    /// Load bias applied to link-time addresses (0 for ET_EXEC).
    pub bias: u64,
    /// Mapped segments (PT_LOAD only), sorted by vaddr.
    pub segments: Vec<ProgramHeader>,
    /// The TLS block, when PT_TLS exists.
    pub tls: Option<TlsBlock>,
    /// Applied relocations.
    pub relocations_applied: usize,
}

/// Load `image` into `vm` and return the loaded module.
///
/// `bias` relocates every address (ET_DYN passed with a nonzero base; it
/// must be page-aligned). `resolver` resolves symbol relocations; `None`
/// refuses them.
pub fn load(
    image: &ElfImage,
    vm: &mut VmManager,
    bias: u64,
    resolver: Option<&dyn SymbolResolver>,
) -> Result<LoadedImage, ElfError> {
    if image.e_type == ET_EXEC && bias != 0 {
        return Err(ElfError::BadDynamic {
            what: "ET_EXEC refuses a nonzero load bias",
        });
    }
    if bias % 4096 != 0 {
        return Err(ElfError::BadDynamic {
            what: "load bias not page-aligned",
        });
    }

    // Map every nonempty PT_LOAD page.
    let mut segments = Vec::new();
    for p in &image.phdrs {
        if p.p_type != pt::LOAD || p.span() == 0 {
            continue;
        }
        let va = p.p_vaddr + bias;
        let len = page_ceil(p.span());
        if va % 4096 != 0 {
            return Err(ElfError::BadSegment {
                index: 0,
                vaddr: p.p_vaddr,
                filesz: p.p_filesz,
                memsz: p.p_memsz,
            });
        }
        // PROT: PF_W|PF_R → RW NX; PF_R → R NX; PF_X|PF_R → RX.
        vm.map_anon(va, len, p.page_flags(), false)?;
        segments.push(*p);
    }
    segments.sort_by_key(|p| p.p_vaddr);

    // Copy file bytes (the memsz tail stays zero: alloc_frame zeroes).
    // The loader writes through the supervisor path: segment data lands in
    // RX pages too, exactly like a real kernel's copy under CR0.WP = 0.
    for p in &image.phdrs {
        if p.p_type != pt::LOAD || p.p_filesz == 0 {
            continue;
        }
        let va = p.p_vaddr + bias;
        let data = &image.bytes[p.p_offset as usize..(p.p_offset + p.p_filesz) as usize];
        vm.write_va_loader(va, data)
            .map_err(|_| ElfError::SegmentBeyondFile)?;
    }

    // Apply relocations.
    let mut applied = 0usize;
    for r in &image.relas {
        let target = r.r_offset + bias;
        let inside = segments
            .iter()
            .any(|p| p.p_vaddr + bias <= target && target + 8 <= p.p_vaddr + bias + p.p_memsz);
        if !inside {
            return Err(ElfError::RelocOutside {
                offset: r.r_offset,
                ty: r.r_type,
                sym: r.r_sym,
            });
        }
        let value: u64 = match r.r_type {
            rt::X86_64_RELATIVE => r.r_addend.wrapping_add(bias as i64) as u64,
            rt::X86_64_TPOFF64 => {
                // PDG model: FS base = TLS block base; the addend is the
                // block-relative offset (see module docs).
                r.r_addend as u64
            }
            rt::X86_64_64 | rt::X86_64_GLOB_DAT | rt::X86_64_JUMP_SLOT => {
                let name = image.sym_name(r.r_sym);
                let Some(resolver) = resolver else {
                    return Err(ElfError::SymbolReloc {
                        ty: r.r_type,
                        offset: r.r_offset,
                        sym: r.r_sym,
                        name,
                    });
                };
                let Some(sym) = resolver.resolve(&name) else {
                    return Err(ElfError::SymbolReloc {
                        ty: r.r_type,
                        offset: r.r_offset,
                        sym: r.r_sym,
                        name,
                    });
                };
                sym.wrapping_add(r.r_addend as u64)
            }
            other => {
                return Err(ElfError::UnsupportedReloc {
                    ty: other,
                    offset: r.r_offset,
                });
            }
        };
        vm.write_va_loader(target, &value.to_le_bytes())?;
        applied += 1;
    }

    let tls = image.tls().map(|t| TlsBlock {
        vaddr: t.p_vaddr + bias,
        filesz: t.p_filesz,
        memsz: t.p_memsz,
    });

    Ok(LoadedImage {
        entry: image.e_entry + bias,
        bias,
        segments,
        tls,
        relocations_applied: applied,
    })
}

// ---------------------------------------------------------------------------
// Builder (tests, the Phase-5 demo)
// ---------------------------------------------------------------------------

/// Assembles minimal ELF64 executables in memory.
///
/// One RX `PT_LOAD` (`.text`), one RW `PT_LOAD` (`.data`), an optional
/// `PT_TLS` inside the data segment, and a `PT_DYNAMIC` with `DT_RELA`
/// relocations. Every field is a public ELF64 constant; the builder exists
/// so the test suite and `--demo-kernel` round-trip the loader honestly.
pub struct ElfBuilder {
    /// e_entry (link-time).
    pub entry: u64,
    /// Text bytes.
    pub text: Vec<u8>,
    /// Data bytes.
    pub data: Vec<u8>,
    /// Text virtual address.
    pub text_va: u64,
    /// Data virtual address.
    pub data_va: u64,
    /// TLS template (placed at the start of a dedicated third segment).
    pub tls: Option<(u64, Vec<u8>)>,
    /// RELA entries.
    pub relas: Vec<Rela>,
}

impl ElfBuilder {
    /// A builder with empty sections at the given addresses.
    pub fn new(text_va: u64, data_va: u64, entry: u64) -> Self {
        Self {
            entry,
            text: Vec::new(),
            data: Vec::new(),
            text_va,
            data_va,
            tls: None,
            relas: Vec::new(),
        }
    }

    /// Serialize the ELF64 image.
    pub fn build(&self) -> Vec<u8> {
        const PAGE: usize = 0x1000;
        const PHNUM: usize = 5; // text, data, (tls load), (PT_TLS), dynamic
        let ehsize: usize = 64;
        let phoff: usize = 64;
        let phentsize: usize = 56;
        let header_end = phoff + PHNUM * phentsize;
        // File offsets are page-aligned and one page apart so p_offset ≡
        // p_vaddr (mod page), the ABI congruence rule.
        let text_off = (header_end + PAGE - 1) & !(PAGE - 1);
        let data_off = text_off + PAGE + ((self.text.len() + PAGE - 1) & !(PAGE - 1));
        let tls_off = data_off + PAGE + ((self.data.len() + PAGE - 1) & !(PAGE - 1));

        let tls_bytes = self.tls.as_ref().map_or(&[][..], |(_, b)| b.as_slice());
        // Dynamic array: RELA pointer/sizes + NULL. The RELA data rides at
        // the end of the data segment (the last 8-aligned slot before the
        // segment end is NOT used; we append it after data, within the data
        // PT_LOAD's memsz? Simpler: place .rela inside the data segment's
        // tail, growing the segment to cover it.)
        let rela_off_in_data = self.data.len();
        let rela_bytes_len = self.relas.len() * 24;
        // dynamic entries: RELA(7) addr, RELASZ(8), RELAENT(9), NULL(0)
        let dyn_off_in_data = rela_off_in_data + rela_bytes_len;
        let dyn_bytes_len = 4 * 16;

        let mut data_seg = self.data.clone();
        data_seg.resize(rela_off_in_data + rela_bytes_len + dyn_bytes_len, 0);
        for (i, r) in self.relas.iter().enumerate() {
            let o = rela_off_in_data + i * 24;
            data_seg[o..o + 8].copy_from_slice(&r.r_offset.to_le_bytes());
            let info = (u64::from(r.r_sym) << 32) | u64::from(r.r_type);
            data_seg[o + 8..o + 16].copy_from_slice(&info.to_le_bytes());
            data_seg[o + 16..o + 24].copy_from_slice(&(r.r_addend as u64).to_le_bytes());
        }
        let rela_va = self.data_va + rela_off_in_data as u64;
        let dyn_ent = |buf: &mut Vec<u8>, tag: u64, val: u64| {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        };
        let dyn_start = dyn_off_in_data;
        let mut dbuf = Vec::with_capacity(64);
        dyn_ent(&mut dbuf, dt::RELA, rela_va);
        dyn_ent(&mut dbuf, dt::RELASZ, (self.relas.len() * 24) as u64);
        dyn_ent(&mut dbuf, dt::RELAENT, 24);
        dyn_ent(&mut dbuf, dt::NULL, 0);
        data_seg[dyn_start..dyn_start + dbuf.len()].copy_from_slice(&dbuf);

        let data_memsz = (data_seg.len() + 0xFFF) & !0xFFFusize;
        data_seg.resize(data_memsz, 0);

        let total_off = tls_off + tls_bytes.len().max(1);
        let mut img = vec![0u8; total_off];

        // ELF header.
        let h = &mut img[0..ehsize];
        h[0..4].copy_from_slice(&ELFMAG);
        h[4] = ELFCLASS64;
        h[5] = ELFDATA2LSB;
        h[6] = 1; // EV_CURRENT
        h[0x10..0x12].copy_from_slice(&ET_EXEC.to_le_bytes());
        h[0x12..0x14].copy_from_slice(&EM_X86_64.to_le_bytes());
        h[0x14..0x18].copy_from_slice(&1u32.to_le_bytes());
        h[0x18..0x20].copy_from_slice(&self.entry.to_le_bytes());
        h[0x20..0x28].copy_from_slice(&(phoff as u64).to_le_bytes());
        h[0x36..0x38].copy_from_slice(&(phentsize as u16).to_le_bytes());
        h[0x38..0x3A].copy_from_slice(&(PHNUM as u16).to_le_bytes());

        let tls_len = tls_bytes.len();
        let push_ph = |img: &mut Vec<u8>,
                       idx: usize,
                       ty: u32,
                       flags: u32,
                       off: usize,
                       va: u64,
                       filesz: usize,
                       memsz: usize,
                       align: u64| {
            let o = phoff + idx * phentsize;
            img[o..o + 4].copy_from_slice(&ty.to_le_bytes());
            img[o + 4..o + 8].copy_from_slice(&flags.to_le_bytes());
            img[o + 8..o + 16].copy_from_slice(&(off as u64).to_le_bytes());
            img[o + 16..o + 24].copy_from_slice(&va.to_le_bytes());
            img[o + 24..o + 32].copy_from_slice(&(0u64).to_le_bytes()); // p_paddr
            img[o + 32..o + 40].copy_from_slice(&(filesz as u64).to_le_bytes());
            img[o + 40..o + 48].copy_from_slice(&(memsz as u64).to_le_bytes());
            img[o + 48..o + 56].copy_from_slice(&align.to_le_bytes());
        };
        // 0: text RX
        push_ph(
            &mut img,
            0,
            pt::LOAD,
            pf::R | pf::X,
            text_off,
            self.text_va,
            self.text.len(),
            self.text.len(),
            0x1000,
        );
        // 1: data RW
        push_ph(
            &mut img,
            1,
            pt::LOAD,
            pf::R | pf::W,
            data_off,
            self.data_va,
            data_seg.len(),
            data_memsz,
            0x1000,
        );
        // 2: the TLS block's own PT_LOAD (when present)
        if let Some((tls_va, _)) = &self.tls {
            push_ph(
                &mut img,
                2,
                pt::LOAD,
                pf::R | pf::W,
                tls_off,
                *tls_va,
                tls_len,
                tls_len.max(1),
                8,
            );
        } else {
            push_ph(&mut img, 2, pt::NULL, 0, 0, 0, 0, 0, 0);
        }
        // 3: PT_TLS describing the same bytes (when present)
        if let Some((tls_va, _)) = &self.tls {
            push_ph(
                &mut img,
                3,
                pt::TLS,
                pf::R | pf::W,
                tls_off,
                *tls_va,
                tls_len,
                tls_len.max(1),
                8,
            );
        } else {
            push_ph(&mut img, 3, pt::NULL, 0, 0, 0, 0, 0, 0);
        }
        // 4: DYNAMIC
        push_ph(
            &mut img,
            4,
            pt::DYNAMIC,
            pf::R | pf::W,
            data_off + dyn_start,
            self.data_va + dyn_start as u64,
            dyn_bytes_len,
            dyn_bytes_len,
            8,
        );

        // Section bytes.
        img[text_off..text_off + self.text.len()].copy_from_slice(&self.text);
        img[data_off..data_off + data_seg.len()].copy_from_slice(&data_seg);
        if !tls_bytes.is_empty() {
            img[tls_off..tls_off + tls_bytes.len()].copy_from_slice(tls_bytes);
        }
        img
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::VmManager;
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
            0x3000_0001_0000,
        )
    }

    #[test]
    fn parses_minimal_elf() {
        let b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        let img = b.build();
        let elf = ElfImage::parse(&img).expect("parse");
        assert_eq!(elf.e_type, ET_EXEC);
        assert_eq!(elf.e_entry, 0x0040_0000);
        assert_eq!(elf.phdrs.iter().filter(|p| p.p_type == pt::LOAD).count(), 2);
        assert_eq!(elf.relas.len(), 0);
    }

    #[test]
    fn bad_magic_refused() {
        assert!(matches!(
            ElfImage::parse(&[0u8; 64]),
            Err(ElfError::BadIdent { .. })
        ));
    }

    #[test]
    fn load_maps_segments_with_permissions() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        b.text = vec![0xC3]; // ret
        b.data = b"hello".to_vec();
        let elf = ElfImage::parse(&b.build()).unwrap();
        let mut vm = vm();
        let m = load(&elf, &mut vm, 0, None).expect("load");
        assert_eq!(m.entry, 0x0040_0000);
        assert_eq!(m.relocations_applied, 0);
        let mut out = [0u8; 5];
        vm.read_va(0x0041_0000, &mut out).unwrap();
        assert_eq!(&out, b"hello");
        // Writing into RX text must fault (no PF_W).
        assert!(vm.write_va(0x0040_0000, &[0x90]).is_err());
    }

    #[test]
    fn relative_relocation_applied() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        // Slot 0 of .data holds a pointer, fixed up by RELATIVE.
        b.data = vec![0xEE; 8];
        b.relas.push(Rela {
            r_offset: 0x0041_0000,
            r_type: rt::X86_64_RELATIVE,
            r_sym: 0,
            r_addend: 0x0041_0100,
        });
        let elf = ElfImage::parse(&b.build()).unwrap();
        let mut vm = vm();
        load(&elf, &mut vm, 0, None).unwrap();
        let mut out = [0u8; 8];
        vm.read_va(0x0041_0000, &mut out).unwrap();
        assert_eq!(u64::from_le_bytes(out), 0x0041_0100);
    }

    #[test]
    fn tpoff_relocation_applied() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        b.data = vec![0xEE; 8];
        b.tls = Some((0x0042_0000, vec![0xAB]));
        b.relas.push(Rela {
            r_offset: 0x0041_0008,
            r_type: rt::X86_64_TPOFF64,
            r_sym: 0,
            r_addend: 0, // first variable in the block
        });
        let elf = ElfImage::parse(&b.build()).unwrap();
        assert!(elf.tls().is_some());
        let mut vm = vm();
        let m = load(&elf, &mut vm, 0, None).unwrap();
        assert!(m.tls.is_some());
        let mut out = [0u8; 8];
        vm.read_va(0x0041_0008, &mut out).unwrap();
        assert_eq!(u64::from_le_bytes(out), 0);
        // The TLS block really carries the template byte.
        let mut t = [0u8; 1];
        vm.read_va(0x0042_0000, &mut t).unwrap();
        assert_eq!(t[0], 0xAB);
    }

    #[test]
    fn symbol_relocation_needs_resolver() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        b.data = vec![0; 8];
        b.relas.push(Rela {
            r_offset: 0x0041_0000,
            r_type: rt::X86_64_64,
            r_sym: 0,
            r_addend: 0,
        });
        let elf = ElfImage::parse(&b.build()).unwrap();
        let mut vm = vm();
        assert!(matches!(
            load(&elf, &mut vm, 0, None),
            Err(ElfError::SymbolReloc { .. })
        ));
    }

    #[test]
    fn unsupported_reloc_refused() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        b.data = vec![0; 8];
        b.relas.push(Rela {
            r_offset: 0x0041_0000,
            r_type: 0x2A, // R_X86_64_IRELATIVE-ish, unsupported
            r_sym: 0,
            r_addend: 0,
        });
        let elf = ElfImage::parse(&b.build()).unwrap();
        let mut vm = vm();
        assert!(matches!(
            load(&elf, &mut vm, 0, None),
            Err(ElfError::UnsupportedReloc { .. })
        ));
    }

    #[test]
    fn self_container_roundtrip() {
        let mut b = ElfBuilder::new(0x0040_0000, 0x0041_0000, 0x0040_0000);
        b.text = vec![0xF4]; // hlt
        let elf_bytes = b.build();
        // Wrap: header + one plain entry at 0x20+... the ELF begins at
        // offset 0x20 + 32 (header + entry).
        let elf_off: u64 = (SELF_HEADER_SIZE + SELF_ENTRY_SIZE) as u64;
        let mut c = Vec::new();
        c.extend_from_slice(&SELF_MAGIC); // 0x00: magic
        c.extend_from_slice(&[0x00, 0x01, 0x01, 0x12]); // version/mode/endian/attr
        c.extend_from_slice(&1u32.to_le_bytes()); // 0x08: content type SELF, product 0
        c.extend_from_slice(&0u16.to_le_bytes()); // 0x0C: header size
        c.extend_from_slice(&0u16.to_le_bytes()); // 0x0E: meta size
        c.extend_from_slice(&(elf_off + elf_bytes.len() as u64).to_le_bytes()); // 0x10: file size
        c.extend_from_slice(&1u16.to_le_bytes()); // 0x18: entry count
        c.extend_from_slice(&0u16.to_le_bytes()); // 0x1A: flags
        c.extend_from_slice(&0u32.to_le_bytes()); // 0x1C: padding
        assert_eq!(c.len(), SELF_HEADER_SIZE);
        // entry 0: plain data covering the ELF.
        c.extend_from_slice(&0u64.to_le_bytes()); // flags
        c.extend_from_slice(&elf_off.to_le_bytes()); // offset
        c.extend_from_slice(&(elf_bytes.len() as u64).to_le_bytes()); // file size
        c.extend_from_slice(&(elf_bytes.len() as u64).to_le_bytes()); // mem size
        c.extend_from_slice(&elf_bytes);
        let parsed = SelfContainer::parse(&c).expect("self parse");
        assert_eq!(parsed.elf.e_entry, 0x0040_0000);
        // And the entry-point helper handles both shapes.
        assert!(parse_executable(&c).is_ok());
        assert!(parse_executable(&elf_bytes).is_ok());
    }

    #[test]
    fn self_bad_magic_refused() {
        assert!(matches!(
            SelfContainer::parse(&[0u8; 0x40]),
            Err(SelfError::BadMagic { .. })
        ));
    }

    #[test]
    fn self_encrypted_entry_refused() {
        let mut c = vec![0u8; SELF_HEADER_SIZE + SELF_ENTRY_SIZE];
        c[..4].copy_from_slice(&SELF_MAGIC);
        c[4..8].copy_from_slice(&[0x00, 0x01, 0x01, 0x12]);
        c[8..12].copy_from_slice(&1u32.to_le_bytes());
        c[0x18..0x1A].copy_from_slice(&1u16.to_le_bytes());
        // Entry with nonzero flags.
        c[SELF_HEADER_SIZE..SELF_HEADER_SIZE + 8].copy_from_slice(&0x130006u64.to_le_bytes());
        assert!(matches!(
            SelfContainer::parse(&c),
            Err(SelfError::EncryptedEntry {
                flags: 0x130006,
                ..
            })
        ));
    }
}
