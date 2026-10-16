//! `pdg-launcher` — the Phase-6 egui launcher (the `gui` feature).

fn main() -> anyhow::Result<()> {
    pdg_core::crash::install_panic_hook();
    pdg_frontend::gui::run_launcher()
}
