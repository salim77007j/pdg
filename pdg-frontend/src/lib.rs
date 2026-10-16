//! # pdg-frontend
//!
//! Launcher frontend for the PDG PlayStation 5 emulator.
//!
//! Phase 1 shipped the *model*: [`LibraryEntry`] and [`scan_game_library`]
//! enumerate a directory of user-owned dumps. Phase 4 adds
//! [`io_session`]: the end-to-end I/O machine (NVMe SSD + DualSense +
//! Tempest-shaped audio) the demo driver and `pdg-boot --demo-io` run.
//! The GUI (egui-based: game grid, metadata, settings, BIOS picker, save
//! states) is Phase 6 work and replaces [`HeadlessLauncher`]'s text output.

pub mod io_session;

use std::path::{Path, PathBuf};

/// One entry of the user's game library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryEntry {
    /// Display name (file stem for now; metadata DB later).
    pub name: String,
    /// Path to the user's dump.
    pub path: PathBuf,
    /// Size in bytes.
    pub byte_len: u64,
}

/// Scan `dir` for game images, returning entries sorted by name.
///
/// PDG never downloads or provides game content; every entry points at a
/// file the user owns and dumped themselves.
pub fn scan_game_library(dir: &Path) -> std::io::Result<Vec<LibraryEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            entries.push(LibraryEntry {
                name: entry
                    .path()
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".to_owned()),
                path: entry.path(),
                byte_len: meta.len(),
            });
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Text-mode launcher used until the GUI exists (Phase 6).
pub struct HeadlessLauncher;

impl HeadlessLauncher {
    /// Print the library found in `dir` to stdout.
    pub fn list_library(dir: &Path) -> anyhow::Result<()> {
        let entries = scan_game_library(dir)?;
        if entries.is_empty() {
            println!(
                "No game images found in {} — place your own legally-dumped images there.",
                dir.display()
            );
            return Ok(());
        }
        for entry in entries {
            println!("{:>12} bytes  {}", entry.byte_len, entry.name);
            tracing::debug!(path = %entry.path.display(), "library entry");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn scans_directory_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("zeta.bin"), b"zzz").unwrap();
        fs::write(tmp.path().join("alpha.bin"), b"aaaaaa").unwrap();
        fs::create_dir(tmp.path().join("ignored-dir")).unwrap();

        let entries = scan_game_library(tmp.path()).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["alpha", "zeta"]);
        assert_eq!(entries[0].byte_len, 6);
    }

    #[test]
    fn empty_directory_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(scan_game_library(tmp.path()).unwrap().is_empty());
    }
}
