//! Read-only PFS: the PlayStation filesystem, over a block device.
//!
//! The on-disk layout implemented here is the publicly documented one
//! (psdevwiki "PFS", cross-checked against the two independent public
//! readers — the `orbis-pfs` crate and shadPS4, which both cite it):
//!
//! ```text
//! block 0                     : superblock (0x380 bytes; key seed at 0x370)
//!   0x00 u64 version          : 1
//!   0x08 u64 format           : 20130315
//!   0x10 u64 id               : image id
//!   0x18 u8  fmode | clean | ronly | rsv
//!   0x1C u16 mode             : bit0 signed, bit1 64-bit, bit2 encrypted
//!   0x1E u16 unknown
//!   0x20 u32 block_size       : bytes, power of two
//!   0x24 u32 nbackup
//!   0x28 u64 nblock           : total blocks in the image
//!   0x30 u64 ndinode          : inode count
//!   0x38 u64 ndblock           : data block count
//!   0x40 u64 ndinodeblock      : inode block count
//!   0x48 u64 superroot_ino     : root directory inode number
//! blocks 1 ..= ndinodeblock   : inode table
//! following blocks            : data (directory dirents, file contents)
//! ```
//!
//! Inodes are 100-byte headers followed by 12 direct + 5 indirect u32
//! block pointers (168 bytes total). Unsigned inodes only: **signed** PFS
//! (36-byte entries carrying per-block signatures) and **encrypted** PFS
//! are refused with precise errors — verification against those layouts
//! needs a user dump, and PDG does not guess. File data is addressed
//! through the contiguous form (`direct[1] == 0xFFFF_FFFF`), the 12 direct
//! pointers, or the first indirect block; deeper indirections are refused.
//!
//! [`PfsBuilder`] produces images in exactly this layout so the test suite
//! (and `pdg-boot --demo-kernel`) can round-trip the format. The builder's
//! directory encoding (entsize alignment) is PDG's documented choice; the
//! reader accepts any entsize that covers header+name.

use pdg_io::block::BlockDevice;

/// Superblock `version` value.
const HEADER_VERSION: u64 = 1;
/// Superblock `format` value.
const HEADER_FORMAT: u64 = 20_130_315;
/// Total superblock size (includes the key-seed area).
const HEADER_SIZE: usize = 0x380;
/// Inode header size (before block pointers).
const INODE_HEADER_SIZE: usize = 100;
/// Direct block pointers per inode.
const INODE_DIRECT: usize = 12;
/// Indirect block pointers per inode.
const INODE_INDIRECT: usize = 5;
/// Total inode size for unsigned images.
const INODE_SIZE: usize = INODE_HEADER_SIZE + (INODE_DIRECT + INODE_INDIRECT) * 4;

/// PFS errors — every refusal names the exact cause.
#[derive(Debug, thiserror::Error)]
pub enum PfsError {
    /// Not a PFS image (bad superblock version/format).
    #[error("not a PFS image (version {version}, format {format})")]
    BadHeader {
        /// Version field as read.
        version: u64,
        /// Format field as read.
        format: u64,
    },
    /// The image is encrypted; PDG cannot read it.
    #[error(
        "PFS image is encrypted (mode {mode:#x}); PDG does not contain decryption keys — use a decrypted dump"
    )]
    Encrypted {
        /// The mode field.
        mode: u16,
    },
    /// The image is signed (36-byte inode entries); PDG cannot read it.
    #[error("PFS image is signed (mode {mode:#x}); unsigned images only in Phase 5")]
    Signed {
        /// The mode field.
        mode: u16,
    },
    /// 64-bit-inode mode flag set; layout beyond the documented subset.
    #[error("PFS image uses unhandled mode bits {mode:#x}")]
    BadMode {
        /// The mode field.
        mode: u16,
    },
    /// Block size is not a power of two / too small.
    #[error("bad PFS block size {size}")]
    BadBlockSize {
        /// The block size field.
        size: u32,
    },
    /// Structure beyond the image bounds.
    #[error("PFS structure out of range: {what} (block {block})")]
    OutOfRange {
        /// What structure was being read.
        what: &'static str,
        /// The offending block number.
        block: u32,
    },
    /// I/O against the backing block device failed.
    #[error("PFS I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Path lookup failed.
    #[error("no such file: {path}")]
    NotFound {
        /// The looked-up path.
        path: String,
    },
    /// Expected a directory.
    #[error("not a directory: {path}")]
    NotADirectory {
        /// The path.
        path: String,
    },
    /// Expected a regular file.
    #[error("is a directory: {path}")]
    IsADirectory {
        /// The path.
        path: String,
    },
    /// The file needs indirect addressing PDG does not implement.
    #[error("inode {ino} needs unsupported indirect addressing (blocks {blocks})")]
    UnsupportedIndirect {
        /// The inode number.
        ino: u32,
        /// The block count.
        blocks: u32,
    },
    /// Malformed dirent stream.
    #[error("malformed dirent at offset {offset:#x}")]
    BadDirent {
        /// Byte offset of the malformed entry.
        offset: u64,
    },
    /// Inode field inconsistent with the header.
    #[error("inode {ino} inconsistent: {what}")]
    BadInode {
        /// Inode number.
        ino: u32,
        /// What was wrong.
        what: String,
    },
    /// Read past end of file.
    #[error("read past end of file (inode {ino}, size {size})")]
    Eof {
        /// Inode number.
        ino: u32,
        /// File size.
        size: u64,
    },
    /// The builder was given a path it cannot encode.
    #[error("builder cannot encode {what}: {path}")]
    Builder {
        /// What could not be encoded.
        what: &'static str,
        /// The offending path.
        path: String,
    },
}

/// Inode `mode` bits (the documented values).
pub mod inode_mode {
    /// Regular file.
    pub const FILE: u16 = 0x8000;
    /// Directory.
    pub const DIRECTORY: u16 = 0x4000;
}

/// Dirent `type` values (the documented values).
pub mod dirent_type {
    /// Regular file entry.
    pub const FILE: u32 = 2;
    /// Directory entry.
    pub const DIRECTORY: u32 = 3;
    /// "." entry.
    pub const SELF: u32 = 4;
    /// ".." entry.
    pub const PARENT: u32 = 5;
}

/// One parsed inode.
#[derive(Debug, Clone)]
pub struct Inode {
    /// Inode number (index in the inode table).
    pub ino: u32,
    /// `mode` field (0x8000 file / 0x4000 dir).
    pub mode: u16,
    /// Size in bytes.
    pub size: u64,
    /// Number of data blocks.
    pub blocks: u32,
    /// 12 direct block pointers.
    pub direct: [u32; INODE_DIRECT],
    /// First indirect block pointer.
    pub indirect: u32,
}

impl Inode {
    /// True when the mode says directory.
    pub fn is_dir(&self) -> bool {
        self.mode & inode_mode::DIRECTORY != 0
    }
}

/// One directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    /// Target inode number.
    pub ino: u32,
    /// Entry type (see [`dirent_type`]).
    pub ty: u32,
    /// Entry name ("" for self/parent).
    pub name: String,
}

/// A parsed, read-only PFS over a block device.
pub struct Pfs {
    dev: Box<dyn BlockDevice>,
    block_size: u32,
    inode_blocks: u32,
    inode_count: u32,
    superroot: u32,
}

impl std::fmt::Debug for Pfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pfs")
            .field("block_size", &self.block_size)
            .field("inodes", &self.inode_count)
            .field("inode_blocks", &self.inode_blocks)
            .field("superroot", &self.superroot)
            .finish()
    }
}

impl Pfs {
    /// Mount the PFS image on `dev` (the superblock is read from offset 0).
    pub fn mount(mut dev: Box<dyn BlockDevice>) -> Result<Self, PfsError> {
        let mut hdr = [0u8; HEADER_SIZE];
        let n = dev.read_at(0, &mut hdr)?;
        if n != HEADER_SIZE {
            return Err(PfsError::OutOfRange {
                what: "superblock",
                block: 0,
            });
        }
        let version = u64::from_le_bytes(hdr[0x00..0x08].try_into().unwrap());
        let format = u64::from_le_bytes(hdr[0x08..0x10].try_into().unwrap());
        if version != HEADER_VERSION || format != HEADER_FORMAT {
            return Err(PfsError::BadHeader { version, format });
        }
        let mode = u16::from_le_bytes(hdr[0x1C..0x1E].try_into().unwrap());
        if mode & 0x4 != 0 {
            return Err(PfsError::Encrypted { mode });
        }
        if mode & 0x1 != 0 {
            return Err(PfsError::Signed { mode });
        }
        if mode & !0x8 != 0 {
            // Bit 3 is documented as "always set"; anything else is unknown.
            return Err(PfsError::BadMode { mode });
        }
        let block_size = u32::from_le_bytes(hdr[0x20..0x24].try_into().unwrap());
        if !block_size.is_power_of_two() || block_size < 512 {
            return Err(PfsError::BadBlockSize { size: block_size });
        }
        let inode_count = u64::from_le_bytes(hdr[0x30..0x38].try_into().unwrap());
        let inode_blocks = u64::from_le_bytes(hdr[0x40..0x48].try_into().unwrap());
        let superroot = u64::from_le_bytes(hdr[0x48..0x50].try_into().unwrap());
        Ok(Self {
            dev,
            block_size,
            inode_count: u32::try_from(inode_count).map_err(|_| PfsError::BadInode {
                ino: 0,
                what: "inode count overflow".into(),
            })?,
            inode_blocks: u32::try_from(inode_blocks).map_err(|_| PfsError::BadInode {
                ino: 0,
                what: "inode block count overflow".into(),
            })?,
            superroot: u32::try_from(superroot).map_err(|_| PfsError::BadInode {
                ino: 0,
                what: "superroot overflow".into(),
            })?,
        })
    }

    /// Filesystem block size in bytes.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    fn read_block(&mut self, block: u32, out: &mut [u8]) -> Result<(), PfsError> {
        let off = block as u64 * self.block_size as u64;
        let n = self.dev.read_at(off, out)?;
        if n != out.len() {
            return Err(PfsError::OutOfRange {
                what: "block",
                block,
            });
        }
        Ok(())
    }

    /// Read inode number `ino` from the inode table.
    pub fn inode(&mut self, ino: u32) -> Result<Inode, PfsError> {
        if ino >= self.inode_count {
            return Err(PfsError::OutOfRange {
                what: "inode",
                block: ino,
            });
        }
        let per_block = self.block_size as usize / INODE_SIZE;
        let block = 1 + (ino as usize / per_block);
        let slot = ino as usize % per_block;
        let mut buf = vec![0u8; self.block_size as usize];
        self.read_block(block as u32, &mut buf)?;
        let base = slot * INODE_SIZE;
        let raw = &buf[base..base + INODE_SIZE];
        let mode = u16::from_le_bytes(raw[0x00..0x02].try_into().unwrap());
        let size = u64::from_le_bytes(raw[0x08..0x10].try_into().unwrap());
        let blocks = u32::from_le_bytes(raw[0x60..0x64].try_into().unwrap());
        let mut direct = [0u32; INODE_DIRECT];
        for (i, d) in direct.iter_mut().enumerate() {
            *d = u32::from_le_bytes(raw[0x64 + i * 4..0x68 + i * 4].try_into().unwrap());
        }
        let indirect = u32::from_le_bytes(raw[0x94..0x98].try_into().unwrap());
        Ok(Inode {
            ino,
            mode,
            size,
            blocks,
            direct,
            indirect,
        })
    }

    /// Resolve the block map of `inode` (logical index → physical block).
    pub fn block_map(&mut self, inode: &Inode) -> Result<Vec<u32>, PfsError> {
        let n = inode.blocks as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Contiguous form.
        if inode.direct[1] == 0xFFFF_FFFF {
            let start = inode.direct[0];
            return Ok((start..start + inode.blocks).collect());
        }
        let mut map = Vec::with_capacity(n);
        for d in inode.direct {
            map.push(d);
            if map.len() == n {
                return Ok(map);
            }
        }
        // First indirect block: u32 pointers.
        let mut buf = vec![0u8; self.block_size as usize];
        self.read_block(inode.indirect, &mut buf)?;
        for chunk in buf.chunks_exact(4) {
            map.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            if map.len() == n {
                return Ok(map);
            }
        }
        Err(PfsError::UnsupportedIndirect {
            ino: inode.ino,
            blocks: inode.blocks,
        })
    }

    /// Resolve a path (absolute, `/`-separated, leading `/` optional) to an
    /// inode. Intermediate components must be directories.
    pub fn lookup(&mut self, path: &str) -> Result<Inode, PfsError> {
        let path = path.trim_start_matches('/');
        let path = if path.is_empty() { "." } else { path };
        let mut cur = self.inode(self.superroot)?;
        if !cur.is_dir() {
            return Err(PfsError::BadInode {
                ino: self.superroot,
                what: "superroot is not a directory".into(),
            });
        }
        for comp in path.split('/').filter(|c| !c.is_empty() && *c != ".") {
            if comp == ".." {
                // Walk to parent via the ".." dirent when present.
                let entries = self.readdir_inode(&cur)?;
                let parent = entries
                    .iter()
                    .find(|e| e.ty == dirent_type::PARENT)
                    .ok_or(PfsError::NotADirectory { path: path.into() })?;
                cur = self.inode(parent.ino)?;
                continue;
            }
            let entries = self.readdir_inode(&cur)?;
            let hit = entries
                .iter()
                .find(|e| e.name == comp)
                .ok_or(PfsError::NotFound { path: path.into() })?;
            cur = self.inode(hit.ino)?;
        }
        Ok(cur)
    }

    /// List the entries of the directory at `path`.
    pub fn readdir(&mut self, path: &str) -> Result<Vec<DirEntry>, PfsError> {
        let ino = self.lookup(path)?;
        if !ino.is_dir() {
            return Err(PfsError::NotADirectory { path: path.into() });
        }
        self.readdir_inode(&ino)
    }

    /// Parse the dirents of one directory inode into named entries
    /// (self/parent are folded into "." / "..").
    fn readdir_inode(&mut self, ino: &Inode) -> Result<Vec<DirEntry>, PfsError> {
        let map = self.block_map(ino)?;
        let mut data = Vec::with_capacity(map.len() * self.block_size as usize);
        for b in map {
            let start = data.len();
            data.resize(start + self.block_size as usize, 0);
            self.read_block(b, &mut data[start..])?;
        }
        let data = &data[..data.len().min(ino.size as usize)];
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 16 <= data.len() {
            let i = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let ty = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
            let namelen = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
            let entsize = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap()) as usize;
            if entsize < 16 || off + entsize > data.len() || 16 + namelen > entsize {
                return Err(PfsError::BadDirent { offset: off as u64 });
            }
            let name_bytes = &data[off + 16..off + 16 + namelen];
            let name = match ty {
                t if t == dirent_type::SELF => ".".to_owned(),
                t if t == dirent_type::PARENT => "..".to_owned(),
                _ => String::from_utf8_lossy(name_bytes).into_owned(),
            };
            out.push(DirEntry { ino: i, ty, name });
            off += entsize;
        }
        Ok(out)
    }

    /// Read `out.len()` bytes at `offset` from the file at `path`.
    pub fn read_at(&mut self, path: &str, offset: u64, out: &mut [u8]) -> Result<(), PfsError> {
        let ino = self.lookup(path)?;
        if ino.is_dir() {
            return Err(PfsError::IsADirectory { path: path.into() });
        }
        self.read_inode_at(&ino, offset, out)
    }

    /// File size at `path` (a directory counts, too).
    pub fn size_of(&mut self, path: &str) -> Result<u64, PfsError> {
        Ok(self.lookup(path)?.size)
    }

    /// Read from a resolved inode.
    fn read_inode_at(&mut self, ino: &Inode, offset: u64, out: &mut [u8]) -> Result<(), PfsError> {
        if offset + out.len() as u64 > ino.size {
            return Err(PfsError::Eof {
                ino: ino.ino,
                size: ino.size,
            });
        }
        let map = self.block_map(ino)?;
        let bs = self.block_size as u64;
        let mut done = 0usize;
        while done < out.len() {
            let pos = offset + done as u64;
            let idx = (pos / bs) as usize;
            let in_off = (pos % bs) as usize;
            let chunk = ((bs as usize) - in_off).min(out.len() - done);
            let block = map.get(idx).ok_or(PfsError::OutOfRange {
                what: "file block",
                block: idx as u32,
            })?;
            let off = *block as u64 * bs + in_off as u64;
            let n = self.dev.read_at(off, &mut out[done..done + chunk])?;
            if n != chunk {
                return Err(PfsError::OutOfRange {
                    what: "file block",
                    block: *block,
                });
            }
            done += chunk;
        }
        Ok(())
    }

    /// Read the whole file at `path`.
    pub fn read_file(&mut self, path: &str) -> Result<Vec<u8>, PfsError> {
        let ino = self.lookup(path)?;
        if ino.is_dir() {
            return Err(PfsError::IsADirectory { path: path.into() });
        }
        let mut out = vec![0u8; ino.size as usize];
        if !out.is_empty() {
            self.read_inode_at(&ino, 0, &mut out)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// mkfs — build images in this exact layout
// ---------------------------------------------------------------------------

/// Builds read-only PFS images (the demo/test path).
///
/// Files are stored contiguously (`direct[1] == 0xFFFF_FFFF`); directories
/// occupy one or more blocks of packed dirents; entsize is aligned to 8.
pub struct PfsBuilder {
    /// Absolute path → content.
    files: Vec<(String, Vec<u8>)>,
}

impl Default for PfsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PfsBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Add a file (path must be absolute, at least two components, all
    /// intermediate directories are created implicitly).
    pub fn add_file(&mut self, path: &str, content: impl Into<Vec<u8>>) -> &mut Self {
        assert!(path.starts_with('/'), "PFS paths are absolute");
        self.files.push((path.to_owned(), content.into()));
        self
    }

    /// Serialize the image.
    pub fn build(&self) -> Result<Vec<u8>, PfsError> {
        const BS: usize = 0x1000;

        // Deterministic images: inode numbers follow sorted path order, not
        // insertion order.
        let mut files = self.files.clone();
        files.sort_by(|a, b| a.0.cmp(&b.0));

        // ---- assemble the tree ------------------------------------------
        #[derive(Debug)]
        struct Node {
            name: String,
            parent: usize,
            children: Vec<usize>,
            file: Option<Vec<u8>>,
        }
        fn find_or_create_dir(nodes: &mut Vec<Node>, parts: &[&str]) -> usize {
            let mut cur = 0usize;
            for part in parts {
                let hit = nodes[cur]
                    .children
                    .iter()
                    .copied()
                    .find(|c| nodes[*c].name == *part && nodes[*c].file.is_none());
                cur = match hit {
                    Some(c) => c,
                    None => {
                        nodes.push(Node {
                            name: (*part).to_owned(),
                            parent: cur,
                            children: Vec::new(),
                            file: None,
                        });
                        let id = nodes.len() - 1;
                        nodes[cur].children.push(id);
                        id
                    }
                };
            }
            cur
        }

        let mut nodes = vec![Node {
            name: String::new(),
            parent: 0,
            children: Vec::new(),
            file: None,
        }]; // node 0 = root (superroot inode)
        for (path, content) in &files {
            let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
            if parts.is_empty() || parts.iter().any(|p| p.is_empty()) {
                return Err(PfsError::Builder {
                    what: "path (need at least one component)",
                    path: path.clone(),
                });
            }
            let (dir, name) = parts.split_at(parts.len() - 1);
            let dir_id = find_or_create_dir(&mut nodes, dir);
            if nodes[dir_id]
                .children
                .iter()
                .any(|c| nodes[*c].name == name[0])
            {
                return Err(PfsError::Builder {
                    what: "duplicate entry",
                    path: path.clone(),
                });
            }
            nodes.push(Node {
                name: name[0].to_owned(),
                parent: dir_id,
                children: Vec::new(),
                file: Some(content.clone()),
            });
            let new_id = nodes.len() - 1;
            nodes[dir_id].children.push(new_id);
        }

        // ---- layout ------------------------------------------------------
        let n_inodes = nodes.len();
        let per_block = BS / INODE_SIZE;
        let inode_blocks = n_inodes.div_ceil(per_block);
        let mut next_data_block: u32 = 1 + inode_blocks as u32;

        #[derive(Default, Clone)]
        struct InodeOut {
            mode: u16,
            size: u64,
            blocks: u32,
            direct: [u32; INODE_DIRECT],
        }
        let mut inodes_out = vec![InodeOut::default(); n_inodes];

        // Dirent streams per directory (self + parent + sorted children).
        let mut dir_blocks: Vec<Vec<u8>> = vec![Vec::new(); n_inodes];
        for id in 0..nodes.len() {
            if nodes[id].file.is_some() {
                continue;
            }
            let mut ents: Vec<u8> = Vec::new();
            fn push_ent(ents: &mut Vec<u8>, ino: u32, ty: u32, name: &str) {
                let nb = name.as_bytes();
                let entsize = (16 + nb.len() + 7) & !7usize;
                ents.extend_from_slice(&ino.to_le_bytes());
                ents.extend_from_slice(&ty.to_le_bytes());
                ents.extend_from_slice(&(nb.len() as u32).to_le_bytes());
                ents.extend_from_slice(&(entsize as u32).to_le_bytes());
                ents.extend_from_slice(nb);
                ents.resize(ents.len() + (entsize - 16 - nb.len()), 0);
            }
            push_ent(&mut ents, id as u32, dirent_type::SELF, "");
            push_ent(&mut ents, nodes[id].parent as u32, dirent_type::PARENT, "");
            let mut kids: Vec<usize> = nodes[id].children.clone();
            kids.sort_by(|a, b| nodes[*a].name.cmp(&nodes[*b].name));
            for c in kids {
                let ty = if nodes[c].file.is_some() {
                    dirent_type::FILE
                } else {
                    dirent_type::DIRECTORY
                };
                push_ent(&mut ents, c as u32, ty, &nodes[c].name);
            }
            if ents.len() > BS * 8 {
                return Err(PfsError::Builder {
                    what: "directory too large (>8 blocks)",
                    path: nodes[id].name.clone(),
                });
            }
            dir_blocks[id] = ents;
        }

        // Contiguous data-block assignment.
        for id in 0..nodes.len() {
            let len = match nodes[id].file.as_ref() {
                Some(content) => content.len(),
                None => dir_blocks[id].len(),
            };
            let nblocks = len.div_ceil(BS).max(1) as u32;
            inodes_out[id].mode = if nodes[id].file.is_some() {
                inode_mode::FILE
            } else {
                inode_mode::DIRECTORY
            };
            inodes_out[id].size = len as u64;
            inodes_out[id].blocks = nblocks;
            inodes_out[id].direct[0] = next_data_block;
            inodes_out[id].direct[1] = 0xFFFF_FFFF; // contiguous marker
            next_data_block += nblocks;
        }
        let n_blocks_total = next_data_block as u64;
        let data_blocks = n_blocks_total - 1 - inode_blocks as u64;

        // ---- emit --------------------------------------------------------
        let mut img = vec![0u8; (n_blocks_total as usize) * BS];
        let hdr = &mut img[0..HEADER_SIZE];
        hdr[0x00..0x08].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        hdr[0x08..0x10].copy_from_slice(&HEADER_FORMAT.to_le_bytes());
        hdr[0x10..0x18].copy_from_slice(&0x5044_4701u64.to_le_bytes()); // id
        // 0x18: fmode/clean/ronly/rsv = 0
        let mode: u16 = 0x8; // documented "unknown flag always set"
        hdr[0x1C..0x1E].copy_from_slice(&mode.to_le_bytes());
        hdr[0x20..0x24].copy_from_slice(&(BS as u32).to_le_bytes());
        hdr[0x28..0x30].copy_from_slice(&n_blocks_total.to_le_bytes());
        hdr[0x30..0x38].copy_from_slice(&(n_inodes as u64).to_le_bytes());
        hdr[0x38..0x40].copy_from_slice(&data_blocks.to_le_bytes());
        hdr[0x40..0x48].copy_from_slice(&(inode_blocks as u64).to_le_bytes());
        hdr[0x48..0x50].copy_from_slice(&0u64.to_le_bytes()); // superroot ino
        // Inode table.
        for (i, out) in inodes_out.iter().enumerate() {
            let idx = i / per_block;
            let slot = i % per_block;
            let base = (1 + idx) * BS + slot * INODE_SIZE;
            let raw = &mut img[base..base + INODE_SIZE];
            raw[0x00..0x02].copy_from_slice(&out.mode.to_le_bytes());
            raw[0x02..0x04].copy_from_slice(&1u16.to_le_bytes()); // nlink
            raw[0x08..0x10].copy_from_slice(&out.size.to_le_bytes());
            raw[0x10..0x18].copy_from_slice(&out.size.to_le_bytes()); // size_compressed
            raw[0x60..0x64].copy_from_slice(&out.blocks.to_le_bytes());
            for (i, d) in out.direct.iter().enumerate() {
                raw[0x64 + i * 4..0x68 + i * 4].copy_from_slice(&d.to_le_bytes());
            }
            // indirect pointers stay zero (contiguous form)
        }
        // Data blocks.
        let mut cursor = 1 + inode_blocks as u32;
        for (id, out) in inodes_out.iter().enumerate() {
            if out.blocks == 0 {
                continue;
            }
            let start = cursor as usize * BS;
            if let Some(content) = &nodes[id].file {
                img[start..start + content.len()].copy_from_slice(content);
            } else {
                let ents = &dir_blocks[id];
                img[start..start + ents.len()].copy_from_slice(ents);
            }
            cursor += out.blocks;
        }
        Ok(img)
    }
}

/// In-memory read-only block device (tests, builder round-trips).
pub struct MemoryBlockDevice {
    data: std::sync::Mutex<Vec<u8>>,
}

impl MemoryBlockDevice {
    /// Wrap `data`.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: std::sync::Mutex::new(data),
        }
    }
}

impl BlockDevice for MemoryBlockDevice {
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> std::io::Result<usize> {
        let data = self.data.lock().expect("pfs mem dev");
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = buffer.len().min(data.len() - off);
        buffer[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }

    fn write_at(&mut self, _offset: u64, _buffer: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("read-only device"))
    }

    fn byte_len(&mut self) -> std::io::Result<u64> {
        Ok(self.data.lock().expect("pfs mem dev").len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(files: &[(&str, &[u8])]) -> Pfs {
        let mut b = PfsBuilder::new();
        for (p, c) in files {
            b.add_file(p, c.to_vec());
        }
        let img = b.build().expect("build");
        Pfs::mount(Box::new(MemoryBlockDevice::new(img))).expect("mount")
    }

    #[test]
    fn header_roundtrip() {
        let mut pfs = roundtrip(&[("/a/b.txt", b"hello")]);
        assert_eq!(pfs.block_size(), 0x1000);
        let root = pfs.readdir("/").expect("root");
        assert!(
            root.iter()
                .any(|e| e.name == "a" && e.ty == dirent_type::DIRECTORY)
        );
        let a = pfs.readdir("/a").expect("a");
        assert!(
            a.iter()
                .any(|e| e.name == "b.txt" && e.ty == dirent_type::FILE)
        );
        assert!(a.iter().any(|e| e.name == "." && e.ty == dirent_type::SELF));
        assert!(
            a.iter()
                .any(|e| e.name == ".." && e.ty == dirent_type::PARENT)
        );
    }

    #[test]
    fn file_read_roundtrip_small_and_block_spanning() {
        let big: Vec<u8> = (0..0x2300u32).map(|i| (i % 251) as u8).collect();
        let mut pfs = roundtrip(&[("/small.txt", b"tiny"), ("/big.bin", big.as_slice())]);
        assert_eq!(pfs.read_file("/small.txt").unwrap(), b"tiny");
        assert_eq!(pfs.read_file("/big.bin").unwrap(), big);
        assert_eq!(pfs.size_of("/big.bin").unwrap(), 0x2300);
    }

    #[test]
    fn nested_directories() {
        let mut pfs = roundtrip(&[
            ("/app/data.txt", b"data"),
            ("/app/lib/prx.bin", b"prx"),
            ("/app/lib/sub/deep.txt", b"deep"),
        ]);
        assert_eq!(pfs.read_file("/app/lib/sub/deep.txt").unwrap(), b"deep");
        let lib = pfs.readdir("/app/lib").unwrap();
        assert!(lib.iter().any(|e| e.name == "prx.bin"));
        assert!(lib.iter().any(|e| e.name == "sub"));
    }

    #[test]
    fn missing_paths_refused() {
        let mut pfs = roundtrip(&[("/app/x.txt", b"x")]);
        assert!(matches!(
            pfs.lookup("/app/missing"),
            Err(PfsError::NotFound { .. })
        ));
        assert!(matches!(
            pfs.lookup("/nope"),
            Err(PfsError::NotFound { .. })
        ));
        assert!(matches!(
            pfs.read_file("/app"),
            Err(PfsError::IsADirectory { .. })
        ));
        assert!(matches!(
            pfs.readdir("/app/x.txt"),
            Err(PfsError::NotADirectory { .. })
        ));
    }

    #[test]
    fn eof_refused() {
        let mut pfs = roundtrip(&[("/f.txt", b"12345")]);
        let mut out = [0u8; 6];
        assert!(matches!(
            pfs.read_at("/f.txt", 0, &mut out),
            Err(PfsError::Eof { .. })
        ));
    }

    #[test]
    fn bad_headers_refused() {
        let mut b = PfsBuilder::new();
        b.add_file("/a/b", b"x");
        let img = b.build().unwrap();
        // Corrupt version.
        let mut bad = img.clone();
        bad[0..8].copy_from_slice(&7u64.to_le_bytes());
        let err = Pfs::mount(Box::new(MemoryBlockDevice::new(bad))).unwrap_err();
        assert!(matches!(err, PfsError::BadHeader { .. }));
        // Corrupt format.
        let mut bad = img.clone();
        bad[8..16].copy_from_slice(&42u64.to_le_bytes());
        assert!(matches!(
            Pfs::mount(Box::new(MemoryBlockDevice::new(bad))),
            Err(PfsError::BadHeader { .. })
        ));
        // Encrypted mode.
        let mut bad = img;
        bad[0x1C..0x1E].copy_from_slice(&0x4u16.to_le_bytes());
        assert!(matches!(
            Pfs::mount(Box::new(MemoryBlockDevice::new(bad))),
            Err(PfsError::Encrypted { .. })
        ));
    }

    #[test]
    fn truncated_image_refused() {
        let mut b = PfsBuilder::new();
        b.add_file("/a/b", b"x");
        let img = b.build().unwrap();
        let cut = &img[..0x300];
        assert!(Pfs::mount(Box::new(MemoryBlockDevice::new(cut.to_vec()))).is_err());
    }

    #[test]
    fn deterministic_images() {
        let mut b1 = PfsBuilder::new();
        b1.add_file("/a/x", b"1").add_file("/a/y", b"2");
        let mut b2 = PfsBuilder::new();
        b2.add_file("/a/y", b"2").add_file("/a/x", b"1");
        assert_eq!(b1.build().unwrap(), b2.build().unwrap());
    }
}
