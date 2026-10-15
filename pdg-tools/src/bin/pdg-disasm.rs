//! `pdg-disasm` — x86-64 disassembler front-end over iced-x86.

use std::path::PathBuf;

use clap::Parser;
use pdg_cpu::iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};

#[derive(Parser)]
#[command(
    name = "pdg-disasm",
    version,
    about = "Disassemble a binary as x86-64 (or x86-32) code"
)]
struct Args {
    /// File to disassemble.
    file: PathBuf,

    /// File offset to start at (hex).
    #[arg(long, default_value = "0", value_parser = parse_hex)]
    offset: u64,

    /// Number of bytes to disassemble (default: to end of file).
    #[arg(long)]
    length: Option<usize>,

    /// Bitness: 16, 32 or 64.
    #[arg(long, default_value_t = 64)]
    bitness: u32,

    /// Assume the code starts at this RIP (hex).
    #[arg(long, default_value = "0", value_parser = parse_hex)]
    ip: u64,
}

fn parse_hex(s: &str) -> Result<u64, String> {
    let s = s.trim().trim_start_matches("0x").to_ascii_lowercase();
    u64::from_str_radix(&s, 16).map_err(|e| format!("invalid hex value `{s}`: {e}"))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if !matches!(args.bitness, 16 | 32 | 64) {
        anyhow::bail!("--bitness must be 16, 32 or 64");
    }
    let data = std::fs::read(&args.file)?;
    let offset = (args.offset as usize).min(data.len());
    let end = match args.length {
        Some(len) => offset.saturating_add(len).min(data.len()),
        None => data.len(),
    };
    let bytes = &data[offset..end];

    let mut decoder = Decoder::with_ip(args.bitness, bytes, args.ip, DecoderOptions::NONE);
    let mut formatter = NasmFormatter::new();
    let mut text = String::new();

    println!("{:>18}  {:<24}  INSTRUCTION", "RIP", "BYTES");
    for instruction in &mut decoder {
        formatter.format(&instruction, &mut text);
        let raw = &bytes[usize::try_from(instruction.ip() - args.ip)?..][..instruction.len()];
        println!(
            "{:>18}  {:<24}  {}",
            format!("{:#018x}", instruction.ip()),
            hex(raw),
            text
        );
        text.clear();
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{b:02x}"));
    }
    s
}
