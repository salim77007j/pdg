//! The launcher's data model: settings persistence and save-state
//! metadata — the headless-testable half of the Phase-6 frontend.
//!
//! The GUI ([`crate::gui`]) renders this; `pdg-boot --library` can print
//! it. Settings persist as a TOML file under the user's config home
//! (`PDG_CONFIG` overrides the path for tests and portable installs);
//! save states are `PDGS` files whose headers this module reads without
//! loading the whole machine.

use std::path::{Path, PathBuf};

/// Launcher settings (persisted as TOML).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LauncherSettings {
    /// Where the user keeps their legally-dumped images.
    pub library_dir: Option<PathBuf>,
    /// The user's legally-dumped BIOS, when they set one.
    pub bios_path: Option<PathBuf>,
    /// The default SSD backing image for sessions.
    pub ssd_path: Option<PathBuf>,
    /// The directory save states are written to.
    pub save_dir: Option<PathBuf>,
    /// The instruction budget for guest runs.
    pub max_instructions: u64,
}

impl Default for LauncherSettings {
    fn default() -> Self {
        Self {
            library_dir: None,
            bios_path: None,
            ssd_path: None,
            save_dir: None,
            max_instructions: 50_000_000,
        }
    }
}

impl LauncherSettings {
    /// Load from `path`; a missing file yields defaults (first run),
    /// malformed TOML is surfaced honestly.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(toml::from_str(&text)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Persist to `path` (atomically enough for a launcher: write + rename).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// The resolved save directory (default: `<config>/saves`).
    pub fn save_dir(&self) -> PathBuf {
        self.save_dir
            .clone()
            .unwrap_or_else(|| config_home().join("saves"))
    }
}

/// The user's config directory (`PDG_CONFIG` overrides; otherwise the
/// XDG convention with a `pdg` component).
pub fn config_home() -> PathBuf {
    if let Ok(dir) = std::env::var("PDG_CONFIG") {
        return PathBuf::from(dir);
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_owned());
            PathBuf::from(home).join(".config")
        });
    base.join("pdg")
}

/// The default settings file path.
pub fn settings_path() -> PathBuf {
    config_home().join("launcher.toml")
}

/// One save-state row in the launcher's list.
#[derive(Debug, Clone, PartialEq)]
pub struct SaveStateEntry {
    /// File name (stem) shown in the UI.
    pub name: String,
    /// The file's byte length.
    pub byte_len: u64,
    /// The number of captured threads (from the header).
    pub threads: u32,
    /// The number of nonzero RAM pages (from the header).
    pub ram_pages: u32,
}

/// Scan `dir` for `*.pdgs` files and read their headers (the magic and
/// counts; the full machine state is not decoded here — a listing must
/// stay cheap).
pub fn scan_save_states(dir: &Path) -> std::io::Result<Vec<SaveStateEntry>> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("pdgs") {
            continue;
        }
        let meta = entry.metadata()?;
        if let Some(hdr) = read_pdgs_header(&path) {
            out.push(SaveStateEntry {
                name: path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "save".to_owned()),
                byte_len: meta.len(),
                threads: hdr.0,
                ram_pages: hdr.1,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The `PDGS` header counts: (threads, ram_pages). `None` for anything
/// that is not a version-1 snapshot (refused, not guessed).
fn read_pdgs_header(path: &Path) -> Option<(u32, u32)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).ok()?;
    if &head[..4] != b"PDGS" {
        return None;
    }
    let version = u32::from_le_bytes(head[4..8].try_into().ok()?);
    if version != 1 {
        return None;
    }
    let threads = u32::from_le_bytes(head[8..12].try_into().ok()?);
    // The RAM page count sits after the thread contexts; each thread
    // record is 8 + 16*8 + 8 + 8 + 8 + 8 + 2 = 170 bytes.
    let skip = (threads as u64) * 170;
    std::io::copy(&mut f.by_ref().take(skip), &mut std::io::sink()).ok()?;
    let mut count = [0u8; 4];
    f.read_exact(&mut count).ok()?;
    let ram_pages = u32::from_le_bytes(count.try_into().ok()?);
    Some((threads, ram_pages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_roundtrip_and_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nested").join("launcher.toml");
        // Missing → defaults.
        let s = LauncherSettings::load(&path).unwrap();
        assert_eq!(s, LauncherSettings::default());
        // Save + load round-trips.
        let s = LauncherSettings {
            library_dir: Some(tmp.path().to_owned()),
            bios_path: None,
            ssd_path: Some(tmp.path().join("ssd.img")),
            save_dir: Some(tmp.path().join("states")),
            max_instructions: 1234,
        };
        s.save(&path).unwrap();
        let back = LauncherSettings::load(&path).unwrap();
        assert_eq!(s, back);
        // Malformed TOML is an error, not silent defaults.
        std::fs::write(&path, "not = = toml").unwrap();
        assert!(LauncherSettings::load(&path).is_err());
    }

    #[test]
    fn pdgs_scanning_and_header_reading() {
        let tmp = tempfile::tempdir().unwrap();
        let saves = tmp.path().join("saves");
        std::fs::create_dir_all(&saves).unwrap();
        // A real snapshot: capture one from the session.
        let ssd = tmp.path().join("ssd.img");
        let session = crate::kernel_session::RuntimeSession::new(&ssd).unwrap();
        let (_, snap) = session.run(Some(pdg_kernel::demo::layout6::ST_DLSYM));
        let bytes = crate::kernel_session::SessionSnapshot::to_bytes(&snap.unwrap());
        std::fs::write(saves.join("slot1.pdgs"), &bytes).unwrap();
        // Noise: a wrong extension and a corrupt file.
        std::fs::write(saves.join("noise.txt"), b"hello").unwrap();
        std::fs::write(saves.join("bad.pdgs"), b"PDGSjunkjunk").unwrap();

        let entries = scan_save_states(&saves).unwrap();
        assert_eq!(entries.len(), 1, "only the real snapshot lists");
        assert_eq!(entries[0].name, "slot1");
        assert_eq!(entries[0].byte_len, bytes.len() as u64);
        assert_eq!(entries[0].threads, 1);
        assert!(entries[0].ram_pages >= 15);
        // A missing directory is an empty list, not an error.
        assert!(
            scan_save_states(&tmp.path().join("nope"))
                .unwrap()
                .is_empty()
        );
    }
}
