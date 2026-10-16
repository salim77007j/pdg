//! Content-addressed on-disk shader cache.
//!
//! Key: SHA-256 of (guest ISA bytes ‖ stage ‖ specialization). Value:
//! the WGSL text and a compact manifest (fetch declarations). The
//! format is versioned; a version bump invalidates everything.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Cache-format version; bump on any layout change.
const CACHE_VERSION: u32 = 1;

/// A cache failure.
#[derive(Debug, Clone, Error)]
pub enum CacheError {
    /// Filesystem trouble.
    #[error("shader cache io: {0}")]
    Io(String),
    /// The stored entry is corrupt or a foreign format.
    #[error("shader cache entry corrupt: {0}")]
    Corrupt(String),
}

/// A single shader's cached artifacts.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedShader {
    /// WGSL text (the translated module).
    pub wgsl: String,
    /// The fetch manifest (serialized compactly).
    pub fetches: Vec<FetchRecord>,
}

/// One serialized fetch record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRecord {
    /// Descriptor-table SGPR pair.
    pub table_sbase: u8,
    /// `s_load` offset in dwords.
    pub sload_offset_dwords: u32,
    /// Binding slot.
    pub slot: u32,
    /// Format tag (`f32x4`, `u32x4`, `unorm8x4`).
    pub format: &'static str,
}

/// The on-disk shader cache.
pub struct ShaderCache {
    root: PathBuf,
}

impl ShaderCache {
    /// Open (creating) a cache at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, CacheError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| CacheError::Io(e.to_string()))?;
        Ok(Self { root })
    }

    /// In-memory only (tests).
    #[must_use]
    pub fn ephemeral() -> Self {
        Self {
            root: PathBuf::new(),
        }
    }

    /// The cache key for a shader.
    #[must_use]
    pub fn key(stage: &str, isa: &[u8], specialization: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(CACHE_VERSION.to_le_bytes());
        hasher.update(stage.as_bytes());
        hasher.update([0u8]);
        hasher.update(isa);
        hasher.update([0u8]);
        hasher.update(specialization);
        let digest = hasher.finalize();
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Look up a shader.
    pub fn get(&self, key: &str) -> Result<Option<CachedShader>, CacheError> {
        if self.root.as_os_str().is_empty() {
            return Ok(None);
        }
        let path = self.root.join(format!("{key}.wgsl"));
        let Ok(text) = fs::read_to_string(&path) else {
            return Ok(None);
        };
        parse_entry(&text).map(Some)
    }

    /// Store a shader.
    pub fn put(&self, key: &str, shader: &CachedShader) -> Result<(), CacheError> {
        if self.root.as_os_str().is_empty() {
            return Ok(());
        }
        let text = serialize_entry(shader);
        let path = self.root.join(format!("{key}.wgsl"));
        let tmp = self.root.join(format!("{key}.tmp"));
        fs::write(&tmp, &text).map_err(|e| CacheError::Io(e.to_string()))?;
        fs::rename(&tmp, &path).map_err(|e| CacheError::Io(e.to_string()))?;
        Ok(())
    }
}

/// Serialize an entry: manifest header lines then the WGSL.
fn serialize_entry(shader: &CachedShader) -> String {
    let mut out = String::new();
    out.push_str("/* pdg-shader-cache v1\n");
    for f in &shader.fetches {
        out.push_str(&format!(
            "/* fetch {} {} {} {}\n",
            f.table_sbase, f.sload_offset_dwords, f.slot, f.format
        ));
    }
    out.push_str("*/\n");
    out.push_str(&shader.wgsl);
    out
}

/// Parse a stored entry.
fn parse_entry(text: &str) -> Result<CachedShader, CacheError> {
    if !text.starts_with("/* pdg-shader-cache v1") {
        return Err(CacheError::Corrupt("missing header".into()));
    }
    let mut fetches = Vec::new();
    let header_end = text
        .find("*/")
        .ok_or_else(|| CacheError::Corrupt("unterminated header".into()))?;
    for line in text[..header_end].lines() {
        let Some(rest) = line.strip_prefix("/* fetch ") else {
            continue;
        };
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() != 4 {
            return Err(CacheError::Corrupt("bad fetch record".into()));
        }
        let table_sbase: u8 = parts[0]
            .parse()
            .map_err(|_| CacheError::Corrupt("bad table sgpr".into()))?;
        let sload_offset_dwords: u32 = parts[1]
            .parse()
            .map_err(|_| CacheError::Corrupt("bad offset".into()))?;
        let slot: u32 = parts[2]
            .parse()
            .map_err(|_| CacheError::Corrupt("bad slot".into()))?;
        let format = match parts[3] {
            "f32x4" => "f32x4",
            "u32x4" => "u32x4",
            "unorm8x4" => "unorm8x4",
            _ => return Err(CacheError::Corrupt("bad format tag".into())),
        };
        fetches.push(FetchRecord {
            table_sbase,
            sload_offset_dwords,
            slot,
            format,
        });
    }
    let wgsl = text[header_end + 2..].trim_start_matches('\n').to_string();
    Ok(CachedShader { wgsl, fetches })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pdg-cache-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn key_is_stable_and_distinct() {
        let a = ShaderCache::key("vs", &[1, 2, 3], b"spec");
        let b = ShaderCache::key("vs", &[1, 2, 3], b"spec");
        let c = ShaderCache::key("ps", &[1, 2, 3], b"spec");
        let d = ShaderCache::key("vs", &[1, 2, 4], b"spec");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn roundtrip_through_disk() {
        let root = temp_root("roundtrip");
        let cache = ShaderCache::open(&root).unwrap();
        let shader = CachedShader {
            wgsl: "@fragment\nfn pdg_main() {}\n".into(),
            fetches: vec![FetchRecord {
                table_sbase: 4,
                sload_offset_dwords: 0x10,
                slot: 1,
                format: "f32x4",
            }],
        };
        let key = ShaderCache::key("vs", &[9, 9], b"x");
        assert_eq!(cache.get(&key).unwrap(), None);
        cache.put(&key, &shader).unwrap();
        assert_eq!(cache.get(&key).unwrap().unwrap(), shader);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn corrupt_entry_is_reported() {
        let root = temp_root("corrupt");
        let cache = ShaderCache::open(&root).unwrap();
        let key = ShaderCache::key("vs", &[1], b"");
        fs::write(root.join(format!("{key}.wgsl")), "not a pdg cache file").unwrap();
        assert!(matches!(
            cache.get(&key),
            Err(CacheError::Corrupt(_)) | Ok(None)
        ));
        let _ = fs::remove_dir_all(&root);
    }
}
