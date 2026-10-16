//! The Phase-6 launcher GUI: an egui application over the library model,
//! settings, save states and the runtime session.
//!
//! Compiled behind the `gui` feature (eframe with the glow backend and
//! X11 — the lightest windowing path). The *logic* it renders — settings
//! persistence, save-state scanning, session driving — lives in
//! headless-tested modules; this file is presentation plus the run
//! thread. It cannot be pixel-tested in CI (no display); `cargo check
//! --features gui` keeps it compiling.

use std::path::PathBuf;
use std::sync::mpsc;

use eframe::egui;

use crate::LibraryEntry;
use crate::kernel_session::{RuntimeOutcome, RuntimeSession};
use crate::launcher::{LauncherSettings, SaveStateEntry, scan_save_states};

/// What the run thread reports back to the GUI.
enum RunEvent {
    /// A line of guest console output (chunk).
    Console(String),
    /// The run finished: (outcome string, statuses).
    Done(String, Vec<u64>),
    /// The run thread failed before starting.
    Failed(String),
}

/// The launcher application state.
pub struct LauncherApp {
    settings: LauncherSettings,
    library: Vec<LibraryEntry>,
    saves: Vec<SaveStateEntry>,
    log: String,
    running: bool,
    events: Option<mpsc::Receiver<RunEvent>>,
    selected: Option<usize>,
    status_line: String,
}

impl LauncherApp {
    /// Build the initial state (library + saves scanned, settings
    /// loaded from the default path).
    pub fn new() -> Self {
        let settings =
            LauncherSettings::load(&crate::launcher::settings_path()).unwrap_or_default();
        let library = settings
            .library_dir
            .as_ref()
            .and_then(|d| crate::scan_game_library(d).ok())
            .unwrap_or_default();
        let saves = scan_save_states(&settings.save_dir()).unwrap_or_default();
        Self {
            settings,
            library,
            saves,
            log: String::new(),
            running: false,
            events: None,
            selected: None,
            status_line: "PDG launcher — Phase 6".to_owned(),
        }
    }

    fn refresh(&mut self) {
        self.library = self
            .settings
            .library_dir
            .as_ref()
            .and_then(|d| crate::scan_game_library(d).ok())
            .unwrap_or_default();
        self.saves = scan_save_states(&self.settings.save_dir()).unwrap_or_default();
    }

    fn start_run(&mut self) {
        if self.running {
            return;
        }
        let ssd = self
            .settings
            .ssd_path
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("pdg-ssd.img"));
        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        self.running = true;
        self.log.clear();
        self.status_line = format!("running the runtime demo against {}", ssd.display());
        std::thread::spawn(move || {
            let session = match RuntimeSession::new(&ssd) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(RunEvent::Failed(e.to_string()));
                    return;
                }
            };
            // Stream the console: the session runs synchronously, so poll
            // the shared console from the same thread as it completes
            // chunks of the run — for the Phase-6 demo (a few hundred
            // instructions) one post-run send carries the transcript.
            let (report, _) = session.run(None);
            let _ = tx.send(RunEvent::Console(report.console));
            let outcome = match report.outcome {
                RuntimeOutcome::Exited { code, .. } => {
                    format!("exited({code})")
                }
                RuntimeOutcome::LastThreadExited { .. } => "last thread exited".to_owned(),
                RuntimeOutcome::Halted { .. } => "halted".to_owned(),
                RuntimeOutcome::Fault(ref f) => format!("fault: {f}"),
                RuntimeOutcome::BudgetExhausted => "budget exhausted".to_owned(),
            };
            let _ = tx.send(RunEvent::Done(outcome, report.statuses));
        });
    }

    fn drain_events(&mut self) {
        let mut rx = match self.events.take() {
            Some(rx) => rx,
            None => return,
        };
        loop {
            match rx.try_recv() {
                Ok(RunEvent::Console(text)) => {
                    self.log.push_str(&text);
                }
                Ok(RunEvent::Done(outcome, statuses)) => {
                    let passed = statuses.iter().filter(|s| **s == 1).count();
                    self.status_line =
                        format!("{outcome} — {passed}/{} stages passed", statuses.len());
                    self.running = false;
                    self.refresh();
                }
                Ok(RunEvent::Failed(e)) => {
                    self.status_line = format!("run failed: {e}");
                    self.running = false;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    self.events = Some(rx);
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.running = false;
                    break;
                }
            }
        }
    }
}

impl Default for LauncherApp {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for LauncherApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        if self.running {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("PDG");
                ui.label("— a PlayStation 5 emulator in Rust (Phase 6)");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(if self.running {
                            "Running…"
                        } else {
                            "Run demo"
                        })
                        .clicked()
                    {
                        self.start_run();
                    }
                });
            });
        });

        egui::Panel::bottom("log").show(ui, |ui| {
            ui.weak(&self.status_line);
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(160.0)
                .show(ui, |ui| {
                    ui.monospace(if self.log.is_empty() {
                        "(no output yet)"
                    } else {
                        &self.log
                    });
                });
        });

        egui::Panel::left("side").show(ui, |ui| {
            ui.heading("Library");
            if ui.button("Rescan").clicked() {
                self.refresh();
            }
            if self.library.is_empty() {
                ui.weak(format!(
                    "No images found in {}\n(point Settings at your own legally-dumped images)",
                    self.settings
                        .library_dir
                        .as_ref()
                        .map(|d| d.display().to_string())
                        .unwrap_or_else(|| "(unset)".into()),
                ));
            }
            for (i, entry) in self.library.iter().enumerate() {
                let selected = self.selected == Some(i);
                if ui
                    .selectable_label(
                        selected,
                        format!("{}\n{} bytes", entry.name, entry.byte_len),
                    )
                    .clicked()
                {
                    self.selected = Some(i);
                }
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Settings");
            egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
                ui.label("Library directory");
                let mut lib = self
                    .settings
                    .library_dir
                    .clone()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut lib).changed() {
                    self.settings.library_dir = if lib.trim().is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(lib.trim()))
                    };
                }
                ui.end_row();

                ui.label("SSD backing image");
                let mut ssd = self
                    .settings
                    .ssd_path
                    .clone()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut ssd).changed() {
                    self.settings.ssd_path = if ssd.trim().is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(ssd.trim()))
                    };
                }
                ui.end_row();

                ui.label("BIOS (your own dump)");
                let mut bios = self
                    .settings
                    .bios_path
                    .clone()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut bios).changed() {
                    self.settings.bios_path = if bios.trim().is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(bios.trim()))
                    };
                }
                ui.end_row();
            });
            if ui.button("Save settings").clicked() {
                match self.settings.save(&crate::launcher::settings_path()) {
                    Ok(()) => self.status_line = "settings saved".to_owned(),
                    Err(e) => self.status_line = format!("settings save failed: {e}"),
                }
            }

            ui.add_space(12.0);
            ui.heading("Save states");
            if self.saves.is_empty() {
                ui.weak(format!(
                    "No .pdgs files in {}",
                    self.settings.save_dir().display()
                ));
            }
            egui::Grid::new("saves").num_columns(4).show(ui, |ui| {
                ui.strong("name");
                ui.strong("threads");
                ui.strong("RAM pages");
                ui.strong("bytes");
                ui.end_row();
                for s in &self.saves {
                    ui.label(&s.name);
                    ui.label(s.threads.to_string());
                    ui.label(s.ram_pages.to_string());
                    ui.label(s.byte_len.to_string());
                    ui.end_row();
                }
            });

            ui.add_space(12.0);
            ui.weak(
                "PDG is an independent emulator. We do not distribute BIOS, firmware, or \
                 game files; the library points at files you own and dumped yourself.",
            );
        });
    }
}

/// Run the launcher window (blocks until the window closes).
pub fn run_launcher() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_title("PDG — PlayStation 5 emulator (Phase 6)"),
        ..Default::default()
    };
    eframe::run_native(
        "PDG launcher",
        options,
        Box::new(|_cc| Ok(Box::new(LauncherApp::new()))),
    )
    .map_err(|e| anyhow::anyhow!("launcher window failed: {e}"))
}
