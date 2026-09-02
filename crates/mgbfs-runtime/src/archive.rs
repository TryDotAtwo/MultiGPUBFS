use mgbfs_core::Result;
use sha2::{Digest, Sha256};
#[cfg(target_os = "linux")]
pub struct FileExtent(std::fs::File);
#[cfg(target_os = "linux")]
impl FileExtent {
    pub fn create_new(path: &std::path::Path) -> std::io::Result<Self> {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map(Self)
    }
}
#[cfg(target_os = "linux")]
impl Extent for FileExtent {
    fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        extern "C" {
            fn posix_fallocate(fd: i32, offset: i64, len: i64) -> i32;
        }
        let length = i64::try_from(bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "disk extent exceeds off_t",
            )
        })?;
        // Unlike set_len, this reserves physical storage. Unsupported filesystem
        // or ENOSPC is fatal; no sparse-file fallback.
        let status = unsafe { posix_fallocate(self.0.as_raw_fd(), 0, length) };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status));
        }
        Ok(())
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
        std::os::unix::fs::FileExt::write_at(&self.0, bytes, offset)
    }
    fn sync(&mut self) -> std::io::Result<()> {
        self.0.sync_all()
    }
}
pub trait Extent {
    fn reserve(&mut self, bytes: u64) -> std::io::Result<()>;
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize>;
    fn sync(&mut self) -> std::io::Result<()>;
}
/// Synchronous archive codec. The native scheduler must call this from its
/// dedicated disk worker, never from a GPU progress thread.
pub struct Archive<E: Extent> {
    pub extent: E,
    capacity: u64,
    cursor: u64,
    width: usize,
    chain: [u8; 32],
    sequence: u64,
    depth: u64,
    layer_records: u64,
    total: u64,
    poisoned: bool,
    complete: bool,
}
impl<E: Extent> Archive<E> {
    pub fn new(mut extent: E, capacity: u64, state_bytes: usize, config: [u8; 32]) -> Result<Self> {
        if state_bytes == 0 || state_bytes > 33025 || capacity < 48 {
            return Err("ARCHIVE_SHAPE_OR_CAPACITY".into());
        }
        extent
            .reserve(capacity)
            .map_err(|e| format!("ARCHIVE_RESERVE: {e}"))?;
        let mut header = Vec::with_capacity(48);
        header.extend_from_slice(b"MGBFSAR1");
        header.extend_from_slice(&(state_bytes as u64).to_le_bytes());
        header.extend_from_slice(&config);
        let chain = Sha256::digest(&header).into();
        let mut a = Self {
            extent,
            capacity,
            cursor: 0,
            width: state_bytes,
            chain,
            sequence: 0,
            depth: 0,
            layer_records: 0,
            total: 0,
            poisoned: false,
            complete: false,
        };
        a.write(&header)?;
        Ok(a)
    }
    fn live(&self) -> Result<()> {
        if self.poisoned || self.complete {
            Err("ARCHIVE_CLOSED_OR_POISONED".into())
        } else {
            Ok(())
        }
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.live()?;
        let end = self
            .cursor
            .checked_add(bytes.len() as u64)
            .ok_or("ARCHIVE_OFFSET_OVERFLOW")?;
        if end > self.capacity {
            self.poisoned = true;
            return Err("ARCHIVE_CAPACITY".into());
        }
        match self.extent.write_at(self.cursor, bytes) {
            Ok(n) if n == bytes.len() => {
                self.cursor = end;
                Ok(())
            }
            other => {
                self.poisoned = true;
                Err(format!("ARCHIVE_WRITE: {other:?}"))
            }
        }
    }
    fn frame(&mut self, kind: u64, depth: u64, count: u64, payload: &[u8]) -> Result<()> {
        self.live()?;
        let mut header = Vec::with_capacity(80);
        header.extend_from_slice(b"MGBFSFR1");
        for x in [kind, depth, count, payload.len() as u64, self.sequence] {
            header.extend_from_slice(&x.to_le_bytes());
        }
        header.extend_from_slice(&self.chain);
        let next = self
            .sequence
            .checked_add(1)
            .ok_or("ARCHIVE_SEQUENCE_OVERFLOW")?;
        if self
            .cursor
            .checked_add(112)
            .and_then(|v| v.checked_add(payload.len() as u64))
            .filter(|&v| v <= self.capacity)
            .is_none()
        {
            self.poisoned = true;
            return Err("ARCHIVE_CAPACITY".into());
        }
        let mut digest = Sha256::new();
        digest.update(&header);
        digest.update(payload);
        let digest: [u8; 32] = digest.finalize().into();
        self.write(&header)?;
        if !payload.is_empty() {
            self.write(payload)?;
        }
        self.write(&digest)?;
        self.chain = digest;
        self.sequence = next;
        Ok(())
    }
    fn sync(&mut self) -> Result<()> {
        if let Err(e) = self.extent.sync() {
            self.poisoned = true;
            return Err(format!("ARCHIVE_SYNC: {e}"));
        }
        Ok(())
    }
    pub fn records(&mut self, depth: u64, states: &[u8], hashes: &[[u32; 4]]) -> Result<()> {
        self.live()?;
        if depth != self.depth
            || hashes.is_empty()
            || hashes.len().checked_mul(self.width) != Some(states.len())
        {
            return Err("ARCHIVE_RECORD_SHAPE_OR_DEPTH".into());
        }
        let count = hashes.len() as u64;
        let next = self
            .layer_records
            .checked_add(count)
            .ok_or("ARCHIVE_COUNT_OVERFLOW")?;
        // Codec reference implementation: bounded payload allocation. Production
        // pinned slots will supply this wire layout directly, avoiding this copy.
        let length = states
            .len()
            .checked_add(
                hashes
                    .len()
                    .checked_mul(16)
                    .ok_or("ARCHIVE_BYTE_OVERFLOW")?,
            )
            .ok_or("ARCHIVE_BYTE_OVERFLOW")?;
        if self
            .cursor
            .checked_add(112)
            .and_then(|v| v.checked_add(length as u64))
            .filter(|&v| v <= self.capacity)
            .is_none()
        {
            self.poisoned = true;
            return Err("ARCHIVE_CAPACITY".into());
        }
        let mut payload = Vec::with_capacity(length);
        payload.extend_from_slice(states);
        for hash in hashes {
            for word in hash {
                payload.extend_from_slice(&word.to_le_bytes());
            }
        }
        self.frame(1, depth, count, &payload)?;
        self.layer_records = next;
        Ok(())
    }
    pub fn layer_commit(&mut self, depth: u64, expected_records: u64) -> Result<()> {
        self.live()?;
        if depth != self.depth || expected_records != self.layer_records {
            return Err("ARCHIVE_LAYER_COUNT_OR_DEPTH".into());
        }
        let total = self
            .total
            .checked_add(expected_records)
            .ok_or("ARCHIVE_COUNT_OVERFLOW")?;
        let next = self.depth.checked_add(1).ok_or("ARCHIVE_DEPTH_OVERFLOW")?;
        self.frame(2, depth, expected_records, &[])?;
        self.sync()?;
        self.total = total;
        self.depth = next;
        self.layer_records = 0;
        Ok(())
    }
    /// Caller may invoke only after global FinalizeDepth proves exhaustion.
    pub fn run_commit(&mut self) -> Result<()> {
        self.live()?;
        if self.depth == 0 || self.layer_records != 0 {
            return Err("ARCHIVE_UNCOMMITTED_LAYER".into());
        }
        self.frame(3, self.depth, self.total, &[])?;
        self.sync()?;
        self.complete = true;
        Ok(())
    }
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}
pub fn verify(bytes: &[u8]) -> Result<()> {
    fn word(b: &[u8], i: usize) -> u64 {
        u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
    }
    if bytes.len() < 48 || &bytes[..8] != b"MGBFSAR1" {
        return Err("ARCHIVE_HEADER".into());
    }
    let width = word(bytes, 8);
    if width == 0 || width > 33025 {
        return Err("ARCHIVE_WIDTH".into());
    }
    let mut chain: [u8; 32] = Sha256::digest(&bytes[..48]).into();
    let (mut at, mut sequence, mut depth, mut count, mut total) = (48usize, 0u64, 0u64, 0u64, 0u64);
    loop {
        let h = bytes
            .get(at..at.checked_add(80).ok_or("ARCHIVE_OFFSET")?)
            .ok_or("ARCHIVE_TRUNCATED")?;
        if &h[..8] != b"MGBFSFR1"
            || h[48..80] != chain
            || word(h, 40) != sequence
            || word(h, 16) != depth
        {
            return Err("ARCHIVE_CHAIN".into());
        }
        let kind = word(h, 8);
        let records = word(h, 24);
        let size = usize::try_from(word(h, 32)).map_err(|_| "ARCHIVE_SIZE")?;
        let end = at
            .checked_add(80)
            .and_then(|v| v.checked_add(size))
            .ok_or("ARCHIVE_SIZE")?;
        let payload = bytes.get(at + 80..end).ok_or("ARCHIVE_TRUNCATED")?;
        let digest_end = end.checked_add(32).ok_or("ARCHIVE_SIZE")?;
        let stored = bytes.get(end..digest_end).ok_or("ARCHIVE_TRUNCATED")?;
        let mut sha = Sha256::new();
        sha.update(h);
        sha.update(payload);
        chain = sha.finalize().into();
        if stored != chain {
            return Err("ARCHIVE_CHECKSUM".into());
        }
        match kind {
            1 => {
                if records == 0 || records.checked_mul(width + 16) != Some(size as u64) {
                    return Err("ARCHIVE_RECORD_SHAPE".into());
                }
                count = count.checked_add(records).ok_or("ARCHIVE_COUNT")?;
            }
            2 => {
                if size != 0 || records != count {
                    return Err("ARCHIVE_LAYER_COUNT".into());
                }
                total = total.checked_add(count).ok_or("ARCHIVE_COUNT")?;
                count = 0;
                depth = depth.checked_add(1).ok_or("ARCHIVE_DEPTH")?;
            }
            3 => {
                if size != 0 || depth == 0 || count != 0 || records != total {
                    return Err("ARCHIVE_RUN_COUNT".into());
                }
                return Ok(());
            }
            _ => return Err("ARCHIVE_FRAME_KIND".into()),
        }
        at = digest_end;
        sequence = sequence.checked_add(1).ok_or("ARCHIVE_SEQUENCE")?;
    }
}
