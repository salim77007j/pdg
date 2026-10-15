//! `tracing` subscriber bootstrap.

use std::sync::OnceLock;

static INIT: OnceLock<()> = OnceLock::new();

/// Install the global tracing subscriber at `level`.
///
/// The `RUST_LOG` environment variable (if set) takes precedence over the
/// configured level, matching the ecosystem convention. Subsequent calls are
/// no-ops, which keeps test binaries and library code from fighting over the
/// global subscriber.
pub fn init_logging(level: &str) {
    if INIT.set(()).is_err() {
        return;
    }
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}
