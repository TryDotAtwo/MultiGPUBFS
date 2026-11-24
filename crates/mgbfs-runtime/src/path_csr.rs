use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use mgbfs_core::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Transition {
    pub child: u64,
    pub parent_rank: u32,
    pub parent: u64,
    pub generator: u32,
}

const RUN_ROWS: usize = 8192;
const FAN_IN: u64 = 16;
// Run files are private to one expansion. Child and parent are logical rows.
const RECORD_BYTES: usize = 24;
const CSR_EDGE_BYTES: u64 = 16;
const CHECK_ROWS: usize = 4096;
const SHARD_ROW_BITS: u32 = 52;
const SHARD_ROW_LIMIT: u64 = 1 << SHARD_ROW_BITS;
const MAX_SHARDS: usize = 1 << (u64::BITS - SHARD_ROW_BITS);

impl Transition {
    fn write(self, writer: &mut impl Write) -> Result<()> {
        let mut record = [0; RECORD_BYTES];
        record[..8].copy_from_slice(&self.child.to_le_bytes());
        record[8..16].copy_from_slice(&self.parent.to_le_bytes());
        record[16..20].copy_from_slice(&self.parent_rank.to_le_bytes());
        record[20..].copy_from_slice(&self.generator.to_le_bytes());
        writer.write_all(&record).map_err(io_error)
    }

    fn from_record(record: &[u8]) -> Self {
        Self {
            child: u64::from_le_bytes(record[..8].try_into().unwrap()),
            parent: u64::from_le_bytes(record[8..16].try_into().unwrap()),
            parent_rank: u32::from_le_bytes(record[16..20].try_into().unwrap()),
            generator: u32::from_le_bytes(record[20..].try_into().unwrap()),
        }
    }

    fn read(reader: &mut BufReader<File>) -> Result<Option<Self>> {
        let buffered = reader.fill_buf().map_err(io_error)?;
        if let Some(record) = buffered.get(..RECORD_BYTES) {
            let edge = Self::from_record(record);
            reader.consume(RECORD_BYTES);
            return Ok(Some(edge));
        }
        if buffered.is_empty() {
            return Ok(None);
        }
        let mut record = [0; RECORD_BYTES];
        reader.read_exact(&mut record).map_err(io_error)?;
        Ok(Some(Self::from_record(&record)))
    }
}

struct Merge {
    readers: Vec<BufReader<File>>,
    heads: BinaryHeap<Reverse<(Transition, usize)>>,
    tail: std::vec::IntoIter<Transition>,
}

impl Merge {
    fn new(paths: &[PathBuf]) -> Result<Self> {
        let mut merge = Self {
            readers: Vec::new(),
            heads: BinaryHeap::new(),
            tail: Vec::new().into_iter(),
        };
        for path in paths {
            let mut reader =
                BufReader::with_capacity(64 << 10, File::open(path).map_err(io_error)?);
            if let Some(item) = Transition::read(&mut reader)? {
                merge.heads.push(Reverse((item, merge.readers.len())));
            }
            merge.readers.push(reader);
        }
        Ok(merge)
    }

    fn with_tail(mut self, mut tail: Vec<Transition>) -> Self {
        tail.sort_unstable();
        self.tail = tail.into_iter();
        if let Some(edge) = self.tail.next() {
            self.heads.push(Reverse((edge, self.readers.len())));
        }
        self
    }
}

impl Iterator for Merge {
    type Item = Result<Transition>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut head = self.heads.peek_mut()?;
        let Reverse((edge, source)) = *head;
        let next = if source == self.readers.len() {
            Ok(self.tail.next())
        } else {
            Transition::read(&mut self.readers[source])
        };
        match next {
            Ok(Some(next)) => *head = Reverse((next, source)),
            Ok(None) => {
                std::collections::binary_heap::PeekMut::pop(head);
            }
            Err(error) => {
                drop(head);
                self.heads.clear();
                return Some(Err(error));
            }
        }
        Some(Ok(edge))
    }
}

/// One owner layer's transition spool. Run names are computed on demand so the
/// number of on-disk chunks does not grow the in-memory merge metadata.
pub(crate) struct TransitionSpool {
    directory: PathBuf,
    budget: u64,
    live: u64,
    peak: u64,
    runs: u64,
    pending: Vec<Transition>,
}

impl TransitionSpool {
    pub(crate) fn new(directory: PathBuf, budget: u64) -> Result<Self> {
        fs::create_dir(&directory).map_err(io_error)?;
        Ok(Self {
            directory,
            budget,
            live: 0,
            peak: 0,
            runs: 0,
            pending: Vec::with_capacity(RUN_ROWS),
        })
    }

    fn path(&self, generation: u32, run: u64) -> PathBuf {
        self.directory.join(format!("{generation}-{run}.run"))
    }

    fn reserve(&mut self, bytes: u64) -> Result<()> {
        let peak = self.live.checked_add(bytes).ok_or("PATH_LINEAGE_SIZE")?;
        if peak > self.budget {
            return Err("LINEAGE_LIMIT".into());
        }
        self.peak = self.peak.max(peak);
        Ok(())
    }

    pub(crate) fn add(&mut self, records: &mut [Transition]) -> Result<()> {
        self.add_shard(0, records)
    }

    // A provisional child ID sorts by shard and local row. The final merge
    // converts it to a dense rank-local ID once all shard counts are known.
    pub(crate) fn add_shard(&mut self, shard: u32, mut records: &mut [Transition]) -> Result<()> {
        if shard as usize >= MAX_SHARDS || records.iter().any(|edge| edge.child >= SHARD_ROW_LIMIT) {
            return Err("PATH_LINEAGE_SHARD_ROW".into());
        }
        if shard != 0 {
            let prefix = u64::from(shard) << SHARD_ROW_BITS;
            for edge in records.iter_mut() {
                edge.child |= prefix;
            }
        }
        while !records.is_empty() {
            if self.pending.len() == RUN_ROWS {
                self.flush_run()?;
            }
            let take = records.len().min(RUN_ROWS - self.pending.len());
            let (chunk, rest) = records.split_at_mut(take);
            // Keep the first full buffer resident until a further edge
            // actually requires external sorting.
            if self.pending.is_empty() && take == RUN_ROWS && (self.runs != 0 || !rest.is_empty()) {
                let bytes = (take * RECORD_BYTES) as u64;
                self.reserve(bytes)?;
                Self::write_run(self.path(0, self.runs), chunk)?;
                self.live += bytes;
                self.runs += 1;
            } else {
                self.pending.extend_from_slice(chunk);
            }
            records = rest;
        }
        Ok(())
    }

    fn write_run(path: PathBuf, records: &mut [Transition]) -> Result<()> {
        records.sort_unstable();
        let mut writer = BufWriter::new(File::create(path).map_err(io_error)?);
        for &edge in records.iter() {
            edge.write(&mut writer)?;
        }
        writer.flush().map_err(io_error)
    }

    fn flush_run(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let bytes = (self.pending.len() * RECORD_BYTES) as u64;
        self.reserve(bytes)?;
        Self::write_run(self.path(0, self.runs), &mut self.pending)?;
        self.live += bytes;
        self.runs += 1;
        self.pending.clear();
        Ok(())
    }

    pub(crate) fn peak_reserved_bytes(&self) -> u64 {
        self.peak
    }

    pub(crate) fn finish(
        &mut self,
        staging: &Path,
        committed: &Path,
        rows_by_shard: &[u64],
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<(u64, u64)> {
        let (prefixes, rows) = shard_prefixes(rows_by_shard)?;
        let dense = |edge: Result<Transition>| {
            let mut edge = edge?;
            if prefixes.len() == 1 {
                return Ok(edge);
            }
            let shard = (edge.child >> SHARD_ROW_BITS) as usize;
            let row = edge.child & (SHARD_ROW_LIMIT - 1);
            if row >= rows_by_shard.get(shard).copied().unwrap_or(0) {
                return Err("PATH_LINEAGE_SHARD_ROW".into());
            }
            edge.child = prefixes[shard] + row;
            Ok(edge)
        };
        if self.runs == 0 {
            // All edges fit in the bounded pending buffer. Sort once and
            // build the CSR directly, without writing and rereading a run.
            let index_bytes = csr_index_bytes(rows)?;
            let csr_bytes = (self.pending.len() as u64)
                .checked_mul(CSR_EDGE_BYTES)
                .and_then(|size| size.checked_add(index_bytes))
                .ok_or("PATH_LINEAGE_SIZE")?;
            self.reserve(csr_bytes)?;
            self.pending.sort_unstable();
            let bytes = write_owner_csr(
                staging, committed, rows, self.budget,
                self.pending.iter().copied().map(Ok).map(dense), checkpoint,
            )?;
            self.pending.clear();
            return Ok((bytes, self.peak));
        }
        let mut generation = 0;
        let mut count = self.runs;
        while count > FAN_IN {
            for (next, first) in (0..count).step_by(FAN_IN as usize).enumerate() {
                checkpoint()?;
                let inputs = (first..(first + FAN_IN).min(count))
                    .map(|index| self.path(generation, index))
                    .collect::<Vec<_>>();
                let destination = self.path(generation + 1, next as u64);
                if inputs.len() == 1 {
                    fs::rename(&inputs[0], destination).map_err(io_error)?;
                    continue;
                }
                let output_size = inputs.iter().try_fold(0u64, |total, path| {
                    total
                        .checked_add(path.metadata().map_err(io_error)?.len())
                        .ok_or_else(|| "PATH_LINEAGE_SIZE".to_owned())
                })?;
                self.reserve(output_size)?;
                let mut writer = BufWriter::new(File::create(destination).map_err(io_error)?);
                for (index, edge) in Merge::new(&inputs)?.enumerate() {
                    if index % CHECK_ROWS == 0 {
                        checkpoint()?;
                    }
                    edge?.write(&mut writer)?;
                }
                writer.flush().map_err(io_error)?;
                drop(writer);
                for input in inputs {
                    fs::remove_file(input).map_err(io_error)?;
                }
            }
            count = count.div_ceil(FAN_IN);
            generation += 1;
        }
        let inputs = (0..count)
            .map(|i| self.path(generation, i))
            .collect::<Vec<_>>();
        // The merge preserves every transition. Admit the full CSR while
        // the sorted runs still exist, even if writing is later interrupted.
        let edges = (self.live / RECORD_BYTES as u64)
            .checked_add(self.pending.len() as u64)
            .ok_or("PATH_LINEAGE_SIZE")?;
        let index_bytes = csr_index_bytes(rows)?;
        let csr_bytes = edges
            .checked_mul(CSR_EDGE_BYTES)
            .and_then(|size| size.checked_add(index_bytes))
            .ok_or("PATH_LINEAGE_SIZE")?;
        self.reserve(csr_bytes)?;
        let bytes = write_owner_csr(
            staging,
            committed,
            rows,
            self.budget - self.live,
            Merge::new(&inputs)?.with_tail(std::mem::take(&mut self.pending)).map(dense),
            checkpoint,
        )?;
        Ok((bytes, self.peak))
    }
}

fn shard_prefixes(counts: &[u64]) -> Result<(Vec<u64>, u64)> {
    if counts.is_empty() || counts.len() > MAX_SHARDS {
        return Err("PATH_LINEAGE_SHARD_COUNT".into());
    }
    let mut prefixes = Vec::with_capacity(counts.len());
    let mut total = 0u64;
    for &rows in counts {
        if rows > SHARD_ROW_LIMIT {
            return Err("PATH_LINEAGE_SHARD_ROW".into());
        }
        prefixes.push(total);
        total = total.checked_add(rows).ok_or("PATH_LINEAGE_SIZE")?;
    }
    Ok((prefixes, total))
}

impl Drop for TransitionSpool {
    fn drop(&mut self) {
        // This is the exclusive flat directory created by this spool.
        if let Ok(files) = fs::read_dir(&self.directory) {
            for file in files.flatten() {
                let _ = fs::remove_file(file.path());
            }
        }
        let _ = fs::remove_dir(&self.directory);
    }
}

fn csr_index_bytes(rows: u64) -> Result<u64> {
    rows.checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| "PATH_LINEAGE_SIZE".into())
}

/// Consume one owner layer's externally sorted transitions. The directory is
/// published atomically only after both files are closed. The budget includes
/// their full on-disk size and is supplied by the owner of the disk allocation.
pub(crate) fn write_owner_csr(
    staging: &Path,
    committed: &Path,
    rows: u64,
    budget: u64,
    transitions: impl IntoIterator<Item = Result<Transition>>,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<u64> {
    checkpoint()?;
    if committed.exists() {
        return Err("PATH_LINEAGE_ALREADY_COMMITTED".into());
    }
    let index_bytes = csr_index_bytes(rows)?;
    if index_bytes > budget {
        return Err("LINEAGE_LIMIT".into());
    }
    fs::create_dir(staging).map_err(io_error)?;
    let result = (|| {
        let mut offsets = BufWriter::with_capacity(
            64 << 10,
            File::create(staging.join("offsets")).map_err(io_error)?,
        );
        let mut edges = BufWriter::with_capacity(
            64 << 10,
            File::create(staging.join("edges")).map_err(io_error)?,
        );
        let mut next_child = 0;
        let mut count = 0u64;
        let mut bytes = index_bytes;
        for (index, item) in transitions.into_iter().enumerate() {
            if index % CHECK_ROWS == 0 {
                checkpoint()?;
            }
            let edge = item?;
            let child = edge.child;
            if child >= rows || child > next_child || child + 1 < next_child
            {
                return Err("PATH_LINEAGE_CHILD_ORDER".into());
            }
            if child == next_child {
                offsets.write_all(&count.to_le_bytes()).map_err(io_error)?;
                next_child += 1;
            }
            bytes = bytes.checked_add(CSR_EDGE_BYTES).ok_or("PATH_LINEAGE_SIZE")?;
            if bytes > budget {
                return Err("LINEAGE_LIMIT".into());
            }
            let mut record = [0u8; CSR_EDGE_BYTES as usize];
            record[..8].copy_from_slice(&edge.parent.to_le_bytes());
            record[8..12].copy_from_slice(&edge.parent_rank.to_le_bytes());
            record[12..].copy_from_slice(&edge.generator.to_le_bytes());
            edges.write_all(&record).map_err(io_error)?;
            count += 1;
        }
        if next_child != rows {
            return Err("PATH_LINEAGE_MISSING_CHILD".into());
        }
        offsets.write_all(&count.to_le_bytes()).map_err(io_error)?;
        offsets.flush().map_err(io_error)?;
        edges.flush().map_err(io_error)?;
        drop((offsets, edges));
        checkpoint()?;
        fs::rename(staging, committed).map_err(io_error)?;
        Ok(bytes)
    })();
    if result.is_err() {
        let _ = fs::remove_file(staging.join("offsets"));
        let _ = fs::remove_file(staging.join("edges"));
        let _ = fs::remove_dir(staging);
    }
    result
}

pub(crate) fn io_error(error: std::io::Error) -> String {
    #[cfg(target_os = "linux")]
    if matches!(error.raw_os_error(), Some(libc::ENOSPC | libc::EDQUOT)) {
        return "LINEAGE_DISK_FULL".into();
    }
    // Win32 ERROR_HANDLE_DISK_FULL and ERROR_DISK_FULL.
    #[cfg(windows)]
    if matches!(error.raw_os_error(), Some(39 | 112)) {
        return "LINEAGE_DISK_FULL".into();
    }
    format!("PATH_LINEAGE_IO: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_exhaustion_is_distinct_from_the_configured_limit() {
        use std::io::{Error, ErrorKind};
        #[cfg(target_os = "linux")]
        for code in [libc::ENOSPC, libc::EDQUOT] {
            assert_eq!(io_error(Error::from_raw_os_error(code)), "LINEAGE_DISK_FULL");
        }
        #[cfg(windows)]
        for code in [39, 112] {
            assert_eq!(io_error(Error::from_raw_os_error(code)), "LINEAGE_DISK_FULL");
        }
        assert!(io_error(Error::from(ErrorKind::PermissionDenied)).starts_with("PATH_LINEAGE_IO"));
    }

    #[test]
    fn interruption_during_merge_removes_runs_without_publishing_a_layer() {
        let root = std::env::temp_dir().join(format!("mgbfs-path-cancel-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let work = root.join("spool");
        let stage = root.join("stage");
        let committed = root.join("complete");
        let mut spool = TransitionSpool::new(work.clone(), 1 << 24).unwrap();
        let edge = Transition {
            child: 0,
            parent_rank: 0,
            parent: 0,
            generator: 9,
        };
        for _ in 0..17 {
            spool.add(&mut vec![edge; RUN_ROWS]).unwrap();
        }
        let mut calls = 0;
        let result = spool.finish(&stage, &committed, &[1], || {
            calls += 1;
            if calls == 3 {
                Err("CANCELLED".into())
            } else {
                Ok(())
            }
        });
        assert_eq!(result.unwrap_err(), "CANCELLED");
        assert!(spool.peak_reserved_bytes() > spool.live);
        drop(spool);
        assert!(!work.exists() && !stage.exists() && !committed.exists());
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn interrupted_final_write_keeps_the_admitted_peak_without_publishing() {
        let root = std::env::temp_dir().join(format!("mgbfs-csr-finish-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let work = root.join("spool");
        let staging = root.join("stage");
        let committed = root.join("complete");
        let mut spool = TransitionSpool::new(work.clone(), 32).unwrap();
        spool.add(&mut [Transition {
            child: 0,
            parent_rank: 0,
            parent: 0,
            generator: 7,
        }]).unwrap();
        let mut calls = 0;
        let result = spool.finish(&staging, &committed, &[1], || {
            calls += 1;
            if calls == 3 { Err("CANCELLED".into()) } else { Ok(()) }
        });
        assert_eq!(result.unwrap_err(), "CANCELLED");
        assert_eq!(spool.peak_reserved_bytes(), 32);
        assert!(!staging.exists() && !committed.exists());
        drop(spool);
        assert!(!work.exists());
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn full_buffer_needs_only_the_final_csr_budget() {
        let root = std::env::temp_dir().join(format!("mgbfs-csr-full-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let budget = 16 + RUN_ROWS as u64 * CSR_EDGE_BYTES;
        let mut spool = TransitionSpool::new(root.join("spool"), budget).unwrap();
        let edge = Transition { child: 0, parent_rank: 1, parent: 7, generator: 3 };
        spool.add(&mut vec![edge; RUN_ROWS]).unwrap();
        assert_eq!(spool.runs, 0);
        assert_eq!(
            spool.finish(&root.join("stage"), &root.join("csr"), &[1], || Ok(())).unwrap(),
            (budget, budget),
        );
        drop(spool);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn final_merge_includes_the_tail_and_compacts_shard_rows() {
        let root = std::env::temp_dir().join(format!("mgbfs-csr-tail-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let final_bytes = 24 + (RUN_ROWS as u64 + 1) * CSR_EDGE_BYTES;
        let budget = final_bytes + (RUN_ROWS * RECORD_BYTES) as u64;
        let mut spool = TransitionSpool::new(root.join("spool"), budget).unwrap();
        let edge = Transition { child: 0, parent_rank: 1, parent: 7, generator: 3 };
        spool.add_shard(2, &mut vec![edge; RUN_ROWS]).unwrap();
        spool.add(&mut [Transition { child: 0, generator: 11, ..edge }]).unwrap();
        let committed = root.join("csr");
        assert_eq!(
            spool.finish(&root.join("stage"), &committed, &[1, 0, 1], || Ok(())).unwrap(),
            (final_bytes, budget),
        );
        let offsets = fs::read(committed.join("offsets")).unwrap();
        assert_eq!(offsets, [0u64, 1, RUN_ROWS as u64 + 1].map(u64::to_le_bytes).concat());
        let edges = fs::read(committed.join("edges")).unwrap();
        assert_eq!(&edges[12..16], &11u32.to_le_bytes());
        drop(spool);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn spilled_runs_keep_distinct_children_with_equal_low_words() {
        let root = std::env::temp_dir().join(format!("mgbfs-csr-wide-child-{}", std::process::id()));
        let mut spool = TransitionSpool::new(root.clone(), (RUN_ROWS * RECORD_BYTES) as u64).unwrap();
        let low = Transition {
            child: 3,
            parent_rank: 2,
            parent: (1 << 41) + 11,
            generator: 7,
        };
        let high = Transition { child: (1 << 32) + 3, ..low };
        let parallel = Transition { generator: 9, ..high };
        let mut run = vec![high; RUN_ROWS];
        run[0] = parallel;
        spool.add(&mut run).unwrap();
        spool.add(&mut [low]).unwrap();
        let path = spool.path(0, 0);
        assert_eq!(path.metadata().unwrap().len(), (RUN_ROWS * 24) as u64);
        let mut merge = Merge::new(&[path])
            .unwrap()
            .with_tail(std::mem::take(&mut spool.pending));
        assert_eq!(merge.next().unwrap().unwrap(), low);
        for _ in 1..RUN_ROWS {
            assert_eq!(merge.next().unwrap().unwrap(), high);
        }
        assert_eq!(merge.next().unwrap().unwrap(), parallel);
        assert!(merge.next().is_none());
        drop(merge);
        drop(spool);
        assert!(!root.exists());
    }

    #[test]
    fn short_input_batches_write_direct_csr_and_preserve_every_edge() {
        let root = std::env::temp_dir().join(format!("mgbfs-path-sort-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let work = root.join("spool");
        let staging = root.join("stage");
        let committed = root.join("complete");
        let build = |budget| {
            let mut spool = TransitionSpool::new(work.clone(), budget).unwrap();
            for generator in (0..17).rev() {
                spool
                    .add(&mut [Transition {
                        child: u64::from(generator % 2),
                        parent_rank: 0,
                        parent: 9,
                        generator,
                    }])
                    .unwrap();
            }
            assert_eq!(spool.runs, 0);
            spool.finish(&staging, &committed, &[2], || Ok(()))
        };

        assert_eq!(build(295).unwrap_err(), "LINEAGE_LIMIT");
        assert!(!work.exists() && !staging.exists() && !committed.exists());
        assert_eq!(build(296).unwrap(), (296, 296));
        let offsets = fs::read(committed.join("offsets")).unwrap();
        assert_eq!(offsets, [0u64, 9, 17].map(u64::to_le_bytes).concat());
        let edges = fs::read(committed.join("edges")).unwrap();
        let labels = edges
            .chunks_exact(16)
            .map(|edge| edge[12])
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            (0..17)
                .step_by(2)
                .chain((1..17).step_by(2))
                .collect::<Vec<_>>()
        );
        assert!(!work.exists());
        fs::remove_file(committed.join("offsets")).unwrap();
        fs::remove_file(committed.join("edges")).unwrap();
        fs::remove_dir(committed).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn csr_keeps_parallel_labels_and_only_publishes_a_complete_layer() {
        let root = std::env::temp_dir().join(format!("mgbfs-path-csr-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let staging = root.join("stage");
        let committed = root.join("complete");
        let links = [
            Transition {
                child: 0,
                parent_rank: 2,
                parent: 7,
                generator: 4,
            },
            Transition {
                child: 0,
                parent_rank: 2,
                parent: 7,
                generator: 8,
            },
            Transition {
                child: 1,
                parent_rank: 0,
                parent: 0,
                generator: 3,
            },
        ];
        let records = || links.into_iter().map(Ok);
        assert_eq!(
            write_owner_csr(&staging, &committed, 2, 71, records(), || Ok(())).unwrap_err(),
            "LINEAGE_LIMIT"
        );
        assert!(!staging.exists() && !committed.exists());
        assert!(write_owner_csr(&staging, &committed, 2, 72, [Ok(links[2])], || Ok(())).is_err());
        assert!(!staging.exists() && !committed.exists());
        assert_eq!(
            write_owner_csr(&staging, &committed, 2, 72, records(), || Ok(())).unwrap(),
            72
        );
        let offsets = fs::read(committed.join("offsets")).unwrap();
        assert_eq!(offsets, [0u64, 2, 3].map(u64::to_le_bytes).concat());
        let edges = fs::read(committed.join("edges")).unwrap();
        assert_eq!([edges[12], edges[28], edges[44]], [4, 8, 3]);
        fs::remove_file(committed.join("offsets")).unwrap();
        fs::remove_file(committed.join("edges")).unwrap();
        fs::remove_dir(committed).unwrap();
        fs::remove_dir(root).unwrap();
    }
}
