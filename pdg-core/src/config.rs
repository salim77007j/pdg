//! Layered TOML configuration.

use serde::{Deserialize, Serialize};

/// Top-level PDG configuration. Missing fields fall back to defaults, so a
/// user-supplied TOML file may override a single knob without repeating the
/// rest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct PdgConfig {
    /// Core framework settings.
    pub core: CoreConfig,
    /// CPU model settings.
    pub cpu: CpuConfig,
    /// Memory subsystem settings.
    pub memory: MemoryConfig,
    /// Boot flow settings.
    pub boot: BootConfig,
    /// GPU backend selection.
    pub gpu: GpuConfig,
    /// Storage settings (Phase 4).
    pub storage: StorageConfig,
    /// Input settings (Phase 4).
    pub input: InputConfig,
    /// Audio settings.
    pub audio: AudioConfig,
}

/// Core framework settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreConfig {
    /// `tracing` log level (`error`/`warn`/`info`/`debug`/`trace`).
    pub log_level: String,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_owned(),
        }
    }
}

/// CPU model settings (AMD Zen 2: 8 cores / 16 threads @ ~3.5 GHz).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuConfig {
    /// Physical cores.
    pub cores: u8,
    /// Hardware threads.
    pub threads: u8,
    /// Nominal frequency in Hz.
    pub freq_hz: u64,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            cores: 8,
            threads: 16,
            freq_hz: 3_500_000_000,
        }
    }
}

/// Memory subsystem settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Guest RAM size in MiB (PS5 hardware: 16384).
    pub ram_mib: u32,
    /// MMIO base of the boot console device.
    pub console_mmio_base: u64,
    /// MMIO base of the inter-processor interrupt (IPI) device (SMP).
    pub ipi_mmio_base: u64,
    /// Default load address for raw boot images.
    pub image_base: u64,
    /// Guest-physical address of the bootstrap PML4.
    pub pml4_base: u64,
    /// Bump allocator start for bootstrap page-table frames.
    pub page_frame_alloc_base: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            ram_mib: 256,
            console_mmio_base: 0xF000_0000,
            ipi_mmio_base: 0xF000_1000,
            image_base: 0x0010_0000,
            pml4_base: 0x0020_0000,
            page_frame_alloc_base: 0x0021_0000,
        }
    }
}

/// Boot flow settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BootConfig {
    /// Initial RSP (must lie inside mapped RAM).
    pub stack_top: u64,
    /// Default per-run instruction budget.
    pub max_instructions: u64,
}

impl Default for BootConfig {
    fn default() -> Self {
        Self {
            stack_top: 0x0FFF_F000,
            max_instructions: 50_000_000,
        }
    }
}

/// GPU backend selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// Preferred host graphics API.
    pub backend: GpuBackendKind,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackendKind::Auto,
        }
    }
}

/// Storage settings (the Phase-4 NVMe SSD model).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Host file backing the SSD (created if missing). Empty = no SSD.
    pub ssd_path: String,
    /// Backing-file size in MiB when created.
    pub ssd_size_mib: u32,
    /// Guest-physical MMIO base of the NVMe register window.
    pub ssd_mmio_base: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            ssd_path: String::new(),
            ssd_size_mib: 8192,
            ssd_mmio_base: 0xE000_0000,
        }
    }
}

/// Input settings (the Phase-4 DualSense window).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputConfig {
    /// Guest-physical MMIO base of the pad window.
    pub pad_mmio_base: u64,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            pad_mmio_base: 0xE100_0000,
        }
    }
}

/// Host graphics API preference. The wgpu backend (Phase 3) resolves `Auto`
/// per platform: Vulkan on Linux/Windows desktop, Metal on macOS, DX12 as
/// fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackendKind {
    /// Pick the best backend for the host.
    #[serde(rename = "auto")]
    Auto,
    /// Vulkan.
    #[serde(rename = "vulkan")]
    Vulkan,
    /// Direct3D 12.
    #[serde(rename = "dx12")]
    Dx12,
    /// Metal.
    #[serde(rename = "metal")]
    Metal,
    /// OpenGL.
    #[serde(rename = "gl")]
    Gl,
}

/// Audio settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Output channel count.
    pub channels: u8,
    /// Guest-physical MMIO base of the audio window.
    pub mmio_base: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            mmio_base: 0xE200_0000,
        }
    }
}

impl PdgConfig {
    /// Parse from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Load from `path`; if the file does not exist, fall back to defaults.
    pub fn load_or_default(path: &std::path::Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml_str(&text)
                .map_err(|e| anyhow::anyhow!("invalid config file {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display())),
        }
    }

    /// Serialize back to TOML (used to emit the example file).
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Reject impossible combinations early with actionable messages.
    pub fn validate(&self) -> anyhow::Result<()> {
        let m = &self.memory;
        let ram = u64::from(m.ram_mib) * 1024 * 1024;
        if m.ram_mib < 16 {
            anyhow::bail!("memory.ram_mib must be at least 16 MiB (got {})", m.ram_mib);
        }
        if self.boot.stack_top >= ram {
            anyhow::bail!(
                "boot.stack_top ({:#x}) must lie inside RAM ({:#x} bytes)",
                self.boot.stack_top,
                ram
            );
        }
        if m.image_base + 0x10_0000 > ram {
            anyhow::bail!(
                "memory.image_base ({:#x}) leaves no room inside RAM ({:#x})",
                m.image_base,
                ram
            );
        }
        if m.pml4_base + 0x10_0000 > ram {
            anyhow::bail!(
                "memory.pml4_base ({:#x}) leaves no room inside RAM ({:#x})",
                m.pml4_base,
                ram
            );
        }
        if m.console_mmio_base <= ram {
            anyhow::bail!(
                "memory.console_mmio_base ({:#x}) must sit above RAM ({:#x}) to avoid aliasing",
                m.console_mmio_base,
                ram
            );
        }
        if m.ipi_mmio_base & 0xFFF != 0 {
            anyhow::bail!(
                "memory.ipi_mmio_base ({:#x}) must be 4 KiB-aligned",
                m.ipi_mmio_base
            );
        }
        if m.ipi_mmio_base <= ram {
            anyhow::bail!(
                "memory.ipi_mmio_base ({:#x}) must sit above RAM ({:#x}) to avoid aliasing",
                m.ipi_mmio_base,
                ram
            );
        }
        if m.ipi_mmio_base >> 12 == m.console_mmio_base >> 12 {
            anyhow::bail!(
                "memory.ipi_mmio_base ({:#x}) must not share the boot console's 4 KiB page",
                m.ipi_mmio_base
            );
        }
        // Phase-4 I/O windows: aligned, above RAM, on distinct pages.
        for (name, base) in [
            ("storage.ssd_mmio_base", self.storage.ssd_mmio_base),
            ("input.pad_mmio_base", self.input.pad_mmio_base),
            ("audio.mmio_base", self.audio.mmio_base),
        ] {
            if base & 0xFFF != 0 {
                anyhow::bail!("{name} ({base:#x}) must be 4 KiB-aligned");
            }
            if base <= ram {
                anyhow::bail!("{name} ({base:#x}) must sit above RAM ({ram:#x}) to avoid aliasing");
            }
        }
        let io_bases = [
            ("storage.ssd_mmio_base", self.storage.ssd_mmio_base),
            ("input.pad_mmio_base", self.input.pad_mmio_base),
            ("audio.mmio_base", self.audio.mmio_base),
        ];
        for (i, (name_a, a)) in io_bases.iter().enumerate() {
            for (name_b, b) in io_bases.iter().skip(i + 1) {
                if a >> 12 == b >> 12 {
                    anyhow::bail!("{name_a} and {name_b} must not share a 4 KiB page");
                }
            }
        }
        if self.cpu.threads < self.cpu.cores {
            anyhow::bail!(
                "cpu.threads ({}) must be >= cpu.cores ({})",
                self.cpu.threads,
                self.cpu.cores
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        PdgConfig::default().validate().unwrap();
    }

    #[test]
    fn parse_minimal_toml_overrides() {
        let cfg = PdgConfig::from_toml_str("[memory]\nram_mib = 512\n").unwrap();
        assert_eq!(cfg.memory.ram_mib, 512);
        // Everything else stayed at defaults.
        assert_eq!(cfg.cpu.cores, 8);
        assert_eq!(cfg.boot.stack_top, 0x0FFF_F000);
    }

    #[test]
    fn roundtrip_through_toml() {
        let cfg = PdgConfig::default();
        let text = cfg.to_toml().unwrap();
        let back = PdgConfig::from_toml_str(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn validation_rejects_bad_layout() {
        let mut cfg = PdgConfig::default();
        cfg.memory.ram_mib = 8;
        assert!(cfg.validate().is_err());

        let mut cfg = PdgConfig::default();
        cfg.boot.stack_top = u64::MAX;
        assert!(cfg.validate().is_err());

        let mut cfg = PdgConfig::default();
        cfg.memory.ipi_mmio_base = 0xF000_0800; // shares the console's page
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn missing_config_file_falls_back_to_default() {
        let cfg =
            PdgConfig::load_or_default(std::path::Path::new("/nonexistent/pdg.toml")).unwrap();
        assert_eq!(cfg, PdgConfig::default());
    }

    #[test]
    fn phase4_io_windows_validate() {
        let mut cfg = PdgConfig::default();
        // Default I/O windows are valid.
        cfg.validate().unwrap();
        // Below RAM is refused.
        cfg.storage.ssd_mmio_base = 0x1000;
        assert!(cfg.validate().is_err());
        // Unaligned is refused.
        let mut cfg = PdgConfig::default();
        cfg.audio.mmio_base = 0xE200_0800;
        assert!(cfg.validate().is_err());
        // Two devices sharing a page is refused.
        let mut cfg = PdgConfig::default();
        cfg.input.pad_mmio_base = cfg.storage.ssd_mmio_base;
        assert!(cfg.validate().is_err());
        // Overriding from TOML round-trips.
        let cfg = PdgConfig::from_toml_str("[storage]\nssd_path = '/tmp/disk.img'\n").unwrap();
        assert_eq!(cfg.storage.ssd_path, "/tmp/disk.img");
        assert_eq!(cfg.storage.ssd_mmio_base, 0xE000_0000);
    }
}
