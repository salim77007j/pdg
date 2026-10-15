//! `pdg-boot` — the Phase-1 boot harness.
//!
//! Loads a raw boot image (or your own legally-dumped PS5 BIOS) into the
//! PDG machine, builds the bootstrap address space, and executes the first
//! instructions through the full stack (interpreter → MMU → page tables →
//! guest-physical bus → MMIO). Results are reported honestly: a halt, an
//! instruction budget exhaustion, or a fault with its cause.

use std::path::PathBuf;

use clap::Parser;
use pdg_core::{BootReport, config::PdgConfig, system::System};
use pdg_cpu::RunOutcome;

#[derive(Parser)]
#[command(
    name = "pdg-boot",
    version,
    about = "PDG Phase-1 boot harness — run a raw boot image or your legally-dumped PS5 BIOS",
    after_help = "PDG is an independent emulator. It does not distribute BIOS, firmware, or game files. Users must provide their own legally-dumped system files."
)]
struct Args {
    /// Path to a PS5 BIOS/firmware dump (mapped into the 4 GiB ROM window,
    /// CPU reset to 0xFFFF_FFF0).
    #[arg(long, group = "source")]
    bios: Option<PathBuf>,

    /// Path to a raw boot image (mapped at --image-base).
    #[arg(long, group = "source")]
    image: Option<PathBuf>,

    /// Load address for --image (hex, 4 KiB-aligned).
    #[arg(long, default_value = "0x100000", value_parser = parse_hex)]
    image_base: u64,

    /// Guest RAM size in MiB.
    #[arg(long, default_value_t = 256)]
    ram_mib: u32,

    /// Maximum guest instructions to execute.
    #[arg(long, default_value_t = 50_000_000)]
    max_instructions: u64,

    /// Increase log verbosity (-v: debug, -vv: trace).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn parse_hex(s: &str) -> Result<u64, String> {
    let s = s.trim().trim_start_matches("0x").to_ascii_lowercase();
    u64::from_str_radix(&s, 16).map_err(|e| format!("invalid hex value `{s}`: {e}"))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let level = match args.verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    pdg_core::logging::init_logging(level);

    println!("PDG boot harness — Phase 1 (infrastructure)");
    println!("PDG is an independent emulator. We do not distribute BIOS, firmware, or game files.");
    println!();

    let mut config = PdgConfig::default();
    config.memory.ram_mib = args.ram_mib;
    config.memory.image_base = args.image_base;
    config.boot.max_instructions = args.max_instructions;
    config.validate()?;

    let mut system = System::new(config)?;
    let entry = match (&args.bios, &args.image) {
        (Some(bios), _) => system.load_bios(bios)?,
        (None, Some(image)) => {
            let data = std::fs::read(image)?;
            let base = if args.image_base != 0x0010_0000 {
                args.image_base
            } else {
                0x0010_0000
            };
            system.boot_image(&data, base)?
        }
        (None, None) => {
            anyhow::bail!(
                "no boot source given — pass --bios <path> with your legally-dumped PS5 firmware \
                 or --image <path> with a raw boot image"
            );
        }
    };
    println!("Loaded. Entry point: {entry:#018x}");
    println!("Executing through: interpreter → MMU (4-level x86-64 walk) → guest-physical bus");
    println!();

    let report: BootReport = system.run(args.max_instructions);
    print_report(&report);

    match report.outcome {
        RunOutcome::Halted { .. } | RunOutcome::MaxInstructions { .. } => Ok(()),
        RunOutcome::Fault(_) => std::process::exit(2),
    }
}

fn print_report(report: &BootReport) {
    match &report.outcome {
        RunOutcome::Halted { instructions, rip } => {
            println!("Guest halted at RIP {rip:#018x} after {instructions} instructions.");
        }
        RunOutcome::MaxInstructions { instructions } => {
            println!("Instruction budget exhausted after {instructions} instructions (no halt).");
        }
        RunOutcome::Fault(fault) => {
            println!("Guest fault: {fault}");
            println!("  (Faults are expected in Phase 1: only the integer instruction subset is");
            println!("   implemented; the JIT and full ISA land in Phase 2.)");
        }
    }
    if !report.console.is_empty() {
        println!();
        println!("---- guest console ----");
        println!("{}", report.console);
        println!("-----------------------");
    }
}
