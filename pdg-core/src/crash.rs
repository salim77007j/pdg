//! Crash reporting: a panic hook that captures the panic and a
//! backtrace into both the log and a `pdg-crash-*.log` file.
//!
//! The emulator runs other people's code; when PDG itself dies, the
//! report is what a contributor gets. `std::backtrace` keeps the
//! dependency count at zero — the hook only formats what the stdlib
//! already captures.

use std::io::Write;

/// Where crash files land when the caller does not say
/// (the current directory, then the temp directory as a fallback).
fn default_crash_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("PDG_CRASH_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::temp_dir()
}

/// Install the hook. Every panic is logged (tracing), printed to stderr,
/// and appended to `pdg-crash-<unix-millis>.log` in the crash
/// directory. The previous hook (if any) still runs afterwards.
pub fn install_panic_hook() {
    install_panic_hook_in(None);
}

/// Install the hook writing crash files into `dir` (tests pin it to a
/// tempdir; the default reads `PDG_CRASH_DIR` or falls back to the
/// system temp directory).
pub fn install_panic_hook_in(dir: Option<std::path::PathBuf>) {
    let dir = dir.unwrap_or_else(default_crash_dir);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = payload(info);
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(panic = %message, "PDG crashed");
        eprintln!("PDG crashed: {message}\n{backtrace}");
        let path = dir.join(format!(
            "pdg-crash-{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        let report = format!("PDG crash report\npanic: {message}\n\nbacktrace:\n{backtrace}\n");
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = f.write_all(report.as_bytes());
            eprintln!("crash report: {}", path.display());
        }
        default_hook(info);
    }));
}

/// Extract the panic message the way the default hook prints it.
fn payload(info: &std::panic::PanicHookInfo<'_>) -> String {
    let p = info.payload();
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::install_panic_hook_in;

    #[test]
    fn hook_writes_a_report_and_still_panics() {
        let dir = tempfile::tempdir().unwrap();
        install_panic_hook_in(Some(dir.path().to_owned()));
        let before = std::fs::read_dir(dir.path()).unwrap().count();
        let result = std::panic::catch_unwind(|| {
            panic!("sentinel crash message 0x5EED");
        });
        assert!(result.is_err());
        let after = std::fs::read_dir(dir.path()).unwrap().count();
        assert!(after > before, "a crash report was written");
        let mut found = false;
        for entry in std::fs::read_dir(dir.path()).unwrap().flatten() {
            if entry.path().extension().and_then(|e| e.to_str()) != Some("log") {
                continue;
            }
            let text = std::fs::read_to_string(entry.path()).unwrap();
            if text.contains("sentinel crash message 0x5EED") {
                found = true;
            }
        }
        assert!(found, "the report carries the panic message");
    }
}
