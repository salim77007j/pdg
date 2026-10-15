# Contributing to PDG

Thank you for wanting to build PDG — a legal, independent, clean-room
PlayStation 5 emulator in Rust.

## Ground rules (non-negotiable)

1. **No proprietary code.** No Sony source, firmware, keys, or copyrighted
   data — ever. If a contribution would require any of those, it will be
   rejected. Users dump their own BIOS and games.
2. **No fake implementations.** Every instruction, syscall, and shader must
   genuinely function. Unimplemented paths fault loudly (`#UD`, explicit
   errors); silent no-ops are bugs.
3. **Honest reporting.** Docs, commit messages, and phase reports state what
   actually works.

## Getting started

```bash
git clone https://github.com/salim77007j/pdg
cd pdg
cargo test --workspace          # everything must pass
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Requirements: Rust 1.85+ (`rustup default stable`), any desktop OS.

## Development workflow

- **Branch** from `main` using a descriptive name (`cpu-jit-blocks`,
  `gpu-pm4-opcodes`).
- **Commits**: conventional style, imperative mood — `feat(cpu): implement
  MUL/DIV`, `fix(memory): honor NX at all page levels`.
- **PRs**: small, focused, with tests. A PR that adds behavior adds tests
  that verify the *architectural contract* (register effects, flag bits,
  fault codes), not just "it ran".
- **Benchmarks**: performance-sensitive changes include before/after
  criterion numbers.

## Code standards

- `cargo fmt` is the single source of formatting truth; CI rejects diffs.
- Clippy is enforced at `-D warnings`.
- Public items carry doc comments explaining *why*, not just *what*.
- Every crate compiles warning-free on both Linux and Windows.
- `unsafe` requires a `// SAFETY:` comment proving the invariant.

## Testing philosophy

PDG emulates a real machine; tests assert real architecture semantics:

- Instruction tests assert register values, exact flag bits, and fault
  codes (see `pdg-cpu/tests/interpreter.rs` for the pattern).
- Memory tests assert page-fault error-code bits, not just "it faulted".
- Property tests fuzz the ALU against independent reference models.

If you emulate an instruction, you can explain its architectural effects
from the manual.

## Project phases

Work is organized in six phases; each ends with a report
(`PHASE_N_REPORT.md`) and a pause for review. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the roadmap and
[README.md](README.md) for current status.

## Code of conduct

Be professional, be kind, assume competence and good faith. Harassment and
hostility are not welcome. Keep discussions about code and architecture —
piracy requests are off-topic and will be closed without discussion.
