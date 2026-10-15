//! Block devices and host-file backing.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Random-access block device (the shape of a disk the guest would see).
pub trait BlockDevice: Send {
    /// Read up to `buffer.len()` bytes at `offset`; returns bytes read.
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize>;

    /// Write `buffer` at `offset`; returns bytes written.
    fn write_at(&mut self, offset: u64, buffer: &[u8]) -> io::Result<usize>;

    /// Device size in bytes.
    fn byte_len(&mut self) -> io::Result<u64>;
}

/// Host-file-backed block device using synchronous positioned I/O.
pub struct HostFileBlockDevice {
    file: File,
    len: u64,
}

impl HostFileBlockDevice {
    /// Open (creating if necessary) `path` as a read/write block device.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}

impl BlockDevice for HostFileBlockDevice {
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read(buffer)
    }

    fn write_at(&mut self, offset: u64, buffer: &[u8]) -> io::Result<usize> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write(buffer)
    }

    fn byte_len(&mut self) -> io::Result<u64> {
        Ok(self.len)
    }
}

/// Async host-file block device built on tokio positioned I/O — the
/// foundation the Phase-4 SSD pipeline uses to feed the guest without
/// stalling the emulation loop.
pub struct AsyncFileBlockDevice {
    file: tokio::fs::File,
}

impl AsyncFileBlockDevice {
    /// Open `path` for async reading and writing.
    pub async fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            file: tokio::fs::File::open(path).await?,
        })
    }

    /// Open `path` (creating it) for async reading and writing.
    pub async fn open_rw(path: &Path) -> io::Result<Self> {
        Ok(Self {
            file: tokio::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .await?,
        })
    }

    /// Read exactly `buffer.len()` bytes at `offset`.
    ///
    /// Sequential positioned I/O (all methods take `&mut self`); the
    /// concurrent DirectStorage-style path with per-request handles is
    /// Phase-4 work.
    pub async fn read_exact_at(&mut self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        use tokio::io::SeekFrom;
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        self.file.seek(SeekFrom::Start(offset)).await?;
        self.file.read_exact(buffer).await.map(|_| ())
    }

    /// Write all of `buffer` at `offset`.
    pub async fn write_all_at(&mut self, buffer: &[u8], offset: u64) -> io::Result<()> {
        use tokio::io::SeekFrom;
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.seek(SeekFrom::Start(offset)).await?;
        self.file.write_all(buffer).await
    }

    /// Flush buffered writes to disk.
    pub async fn sync(&mut self) -> io::Result<()> {
        use tokio::io::AsyncWriteExt;
        self.file.flush().await?;
        self.file.sync_all().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_file_block_device_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("disk.img");
        let mut dev = HostFileBlockDevice::open(&path).unwrap();
        dev.write_at(0x200, b"PDG-BLOCK").unwrap();
        let mut out = [0u8; 9];
        assert_eq!(dev.read_at(0x200, &mut out).unwrap(), 9);
        assert_eq!(&out, b"PDG-BLOCK");
    }

    #[tokio::test]
    async fn async_device_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("async.img");
        let mut dev = AsyncFileBlockDevice::open_rw(&path).await.unwrap();
        dev.write_all_at(b"ASYNC-PDG", 0x1000).await.unwrap();
        let mut out = [0u8; 9];
        dev.read_exact_at(&mut out, 0x1000).await.unwrap();
        assert_eq!(&out, b"ASYNC-PDG");
    }
}
