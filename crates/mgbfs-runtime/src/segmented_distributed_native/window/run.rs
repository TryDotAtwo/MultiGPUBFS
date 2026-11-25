use super::super::nearest::Targets;
use super::{
    join::JoinTile,
    lineage::{start_writer, FinishedSpool},
    write_pair, SegmentedDistributedBfs,
};
use crate::generic_distributed_native::DistributedAdvance;
use crate::path_csr::{io_error, TransitionSpool};
use crate::segmented_packet_transport::{
    PACKET_CANCELLED, PACKET_DEADLINE, PACKET_LINEAGE_DISK_FULL, PACKET_LINEAGE_LIMIT,
};
use mgbfs_core::Result;
use serde_json::{json, Value};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const CUDA_MEMORY: u32 = 10;
const ARTIFACT_ERROR: u32 = u32::MAX;

fn query_stop(seconds: u64) -> impl Fn() -> u32 {
    let started = Instant::now();
    move || {
        if crate::generic_run::CANCELLED.load(Ordering::Relaxed) || crate::job_phase::cancelled() {
            1
        } else if seconds != 0 && started.elapsed() >= Duration::from_secs(seconds) {
            2
        } else {
            0
        }
    }
}

pub(super) fn checkpoint(stop: &impl Fn() -> u32) -> Result<()> {
    stop_reason(stop()).map_or(Ok(()), |reason| Err(reason.into()))
}

fn stop_reason(code: u32) -> Option<&'static str> {
    match code {
        1 | PACKET_CANCELLED => Some("CANCELLED"),
        2 | PACKET_DEADLINE => Some("TIME_LIMIT"),
        CUDA_MEMORY => Some("CUDA_MEMORY"),
        PACKET_LINEAGE_LIMIT => Some("LINEAGE_LIMIT"),
        PACKET_LINEAGE_DISK_FULL => Some("LINEAGE_DISK_FULL"),
        _ => None,
    }
}

fn artifact_code(error: &str) -> u32 {
    match error {
        "CANCELLED" => PACKET_CANCELLED,
        "TIME_LIMIT" => PACKET_DEADLINE,
        "CUDA_MEMORY" | "SEGMENTED_WINDOW_LOOKUP_RESOURCE" => CUDA_MEMORY,
        "LINEAGE_LIMIT" => PACKET_LINEAGE_LIMIT,
        "LINEAGE_DISK_FULL" => PACKET_LINEAGE_DISK_FULL,
        _ => ARTIFACT_ERROR,
    }
}

fn unused(path: &Path) -> Result<()> {
    if path.try_exists().map_err(io_error)? {
        return Err("SEGMENTED_WINDOW_ARTIFACT_EXISTS".into());
    }
    Ok(())
}

#[derive(Default)]
struct PendingFiles {
    csr: Option<PathBuf>,
    records: Option<PathBuf>,
}

impl PendingFiles {
    fn claim_records(&mut self, records: PathBuf) -> Result<&Path> {
        unused(&records)?;
        Ok(self.records.insert(records))
    }

    fn commit(mut self) {
        self.csr = None;
        self.records = None;
    }
}

impl Drop for PendingFiles {
    fn drop(&mut self) {
        if let Some(path) = &self.csr {
            let _ = fs::remove_file(path.join("offsets"));
            let _ = fs::remove_file(path.join("edges"));
            let _ = fs::remove_dir(path);
        }
        if let Some(path) = &self.records {
            let _ = fs::remove_file(path);
        }
    }
}

struct Lineage {
    budget: u64,
    bytes: u64,
    csr_bytes: u64,
    peak: u64,
}

impl Lineage {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            bytes: 0,
            csr_bytes: 0,
            peak: 0,
        }
    }

    fn observe(&mut self, additional: u64) -> Result<u64> {
        let occupied = self
            .bytes
            .checked_add(additional)
            .ok_or("PATH_LINEAGE_SIZE")?;
        if occupied > self.budget {
            return Err("LINEAGE_LIMIT".into());
        }
        self.peak = self.peak.max(occupied);
        Ok(occupied)
    }

    fn observe_records(&mut self, csr: u64, file: &Path, result: &Result<u64>) -> Result<u64> {
        let bytes = match result {
            Ok(bytes) => *bytes,
            Err(_) => fs::metadata(file).map_or(0, |meta| meta.len()),
        };
        self.observe(csr.checked_add(bytes).ok_or("PATH_LINEAGE_SIZE")?)
    }

    fn commit(&mut self, bytes: u64, csr_bytes: u64) {
        self.bytes = bytes;
        self.csr_bytes += csr_bytes;
    }
}

enum DiskLayer {
    Ready(FinishedSpool),
    Empty,
    Stopped(&'static str),
}

impl SegmentedDistributedBfs {
    pub(crate) fn solve_window_forward(
        &mut self,
        goals: &[Vec<i64>],
        max_length: u32,
        budget: u64,
        seconds: u64,
        output: &Path,
    ) -> Result<Value> {
        let stop = query_stop(seconds);
        let checkpoint = || checkpoint(&stop);
        self.connect_packet_control()?;
        let mut lineage = Lineage::new(budget);
        let mut rows = Vec::new();
        let targets = if goals.is_empty() {
            Err("NATIVE_PATH_TARGETS".into())
        } else {
            self.path_targets(goals)
        };
        let _ = self.agree_window_artifacts(&targets)?;
        let targets = targets?;
        let mut spool = None;
        let stopped = loop {
            if let Some(reason) =
                self.write_window_depth(&targets, output, spool, &checkpoint, &mut lineage)?
            {
                break Some(reason);
            }
            rows.push(self.counts[self.rank as usize]);
            crate::native_fault::completed(self.depth);
            if self.depth == u64::from(max_length) {
                break None;
            }
            match self.advance_window_disk(output, self.depth + 1, &stop, &mut lineage)? {
                DiskLayer::Stopped(reason) => break Some(reason),
                DiskLayer::Empty => break None,
                DiskLayer::Ready(next) => spool = Some(next),
            }
        };

        let completed_depth = if stopped.is_none() {
            Some(u64::from(max_length))
        } else {
            rows.len().checked_sub(1).map(|depth| depth as u64)
        };
        Ok(json!({
            "status": if stopped.is_none() { "COMPLETE" } else { "INCOMPLETE" },
            "stop_reason": stopped,
            "layer_rows": rows,
            "graph_completed_through_depth": completed_depth,
            "native_state_bytes": self.plan.state_bytes,
            "native_exchange": "COMPACT_PACKET",
            "native_storage": "SEGMENTED_LAYER_BATCH_HASH",
            "lineage_bytes": lineage.bytes,
            "lineage_csr_bytes": lineage.csr_bytes,
            "lineage_peak_bytes": lineage.peak,
            "lineage_peak_scope": "admitted disk bytes including interrupted transition spools",
            "lineage_io": self.window.as_ref().map(|window| window.tile.stats()),
            "native_process_peak_rss_bytes": crate::generic_path_native::process_peak_rss_bytes(),
        }))
    }

    pub(crate) fn solve_window_bidirectional(
        forward: &mut Self,
        reverse: &mut Self,
        goals: &[Vec<i64>],
        min_length: u32,
        max_length: u32,
        budget: u64,
        seconds: u64,
        output: &Path,
    ) -> Result<Value> {
        let stop = query_stop(seconds);
        let checkpoint = || checkpoint(&stop);
        forward.connect_packet_control()?;
        let roots = [output.join("forward"), output.join("reverse")];
        let mut lineage = Lineage::new(budget);
        let mut rows = [Vec::new(), Vec::new()];
        let mut completed = None;
        let mut last_cut = None;
        let mut pending = PendingFiles::default();
        let setup = (|| {
            checkpoint()?;
            if min_length > max_length
                || goals.is_empty()
                || forward.depth != 0
                || reverse.depth != 0
            {
                return Err("SEGMENTED_WINDOW_START_LAYOUT".into());
            }
            for root in &roots {
                fs::create_dir(root).map_err(io_error)?;
            }
            let file = pending.claim_records(roots[1].join("0.targets"))?;
            let written = reverse.write_window_root_targets(goals.len(), file, budget, &checkpoint);
            let bytes = lineage.observe_records(0, file, &written)?;
            written?;
            Ok((bytes, JoinTile::new(forward.base.device)?))
        })();

        let stopped = 'window: {
            if let Some(reason) = forward.agree_window_artifacts(&setup)? {
                break 'window Some(reason);
            }
            let (bytes, mut join) = setup?;
            lineage.commit(bytes, 0);
            pending.commit();
            rows[0].push(forward.counts[forward.rank as usize]);
            rows[1].push(reverse.counts[reverse.rank as usize]);

            for length in 0..=max_length {
                let side = (length % 2) as usize;
                let cut = length >= min_length;
                let spool = if length == 0 {
                    None
                } else {
                    // Odd lengths grow reverse, then even lengths grow forward.
                    let active = if side == 0 {
                        reverse.transfer_window_packet(forward)?;
                        &mut *forward
                    } else {
                        forward.transfer_window_packet(reverse)?;
                        &mut *reverse
                    };
                    match active.advance_window_disk(
                        &roots[side],
                        u64::from(length),
                        &stop,
                        &mut lineage,
                    )? {
                        DiskLayer::Stopped(reason) => break 'window Some(reason),
                        DiskLayer::Empty => {
                            completed = Some(max_length);
                            break 'window None;
                        }
                        DiskLayer::Ready(spool) => Some(spool),
                    }
                };

                // A length is published as one local CSR plus the cut (when in
                // the requested window), and only after all ranks agree.
                let mut pending = PendingFiles::default();
                let written = (|| {
                    let half = length / 2;
                    if forward.depth != u64::from(half)
                        || reverse.depth != u64::from(half + length % 2)
                    {
                        return Err("SEGMENTED_WINDOW_CUT_LAYOUT".into());
                    }
                    let csr_bytes = if let Some(spool) = spool {
                        let active = if side == 0 { &*forward } else { &*reverse };
                        active.finish_window_csr(
                            &roots[side],
                            spool,
                            &mut pending,
                            &checkpoint,
                            &mut lineage,
                        )?
                    } else {
                        checkpoint()?;
                        0
                    };
                    let occupied = lineage.observe(csr_bytes)?;
                    if !cut {
                        return Ok((occupied, csr_bytes));
                    }
                    let file = pending.claim_records(output.join(format!("{length}.matches")))?;
                    let result = join.write(forward, reverse, file, budget - occupied, &checkpoint);
                    let bytes = lineage.observe_records(csr_bytes, file, &result)?;
                    result?;
                    Ok((bytes, csr_bytes))
                })();
                let active = if side == 0 {
                    &mut *forward
                } else {
                    &mut *reverse
                };
                if let Some(reason) = active.agree_window_artifacts(&written)? {
                    break 'window Some(reason);
                }
                let (bytes, csr_bytes) = written?;
                lineage.commit(bytes, csr_bytes);
                pending.commit();
                if length != 0 {
                    rows[side].push(active.counts[active.rank as usize]);
                }
                if cut {
                    last_cut = Some(length);
                }
                completed = Some(length);
                crate::native_fault::completed(u64::from(length));
            }
            None
        };

        Ok(json!({
            "status": if stopped.is_none() { "COMPLETE" } else { "INCOMPLETE" },
            "stop_reason": stopped,
            "side_layer_rows": rows,
            "first_cut_length": min_length,
            "last_cut_length": last_cut,
            "graph_completed_through_depth": completed,
            "native_state_bytes": forward.plan.state_bytes,
            "native_exchange": "COMPACT_PACKET",
            "native_storage": "SEGMENTED_LAYER_BATCH_HASH",
            "lineage_bytes": lineage.bytes,
            "lineage_csr_bytes": lineage.csr_bytes,
            "lineage_peak_bytes": lineage.peak,
            "lineage_peak_scope": "admitted disk bytes including interrupted transition spools",
            "lineage_io": [forward.window.as_ref().map(|window| window.tile.stats()),
                           reverse.window.as_ref().map(|window| window.tile.stats())],
            "native_process_peak_rss_bytes": crate::generic_path_native::process_peak_rss_bytes(),
        }))
    }

    fn write_window_root_targets(
        &self,
        goals: usize,
        output: &Path,
        budget: u64,
        checkpoint: &impl Fn() -> Result<()>,
    ) -> Result<u64> {
        let window = self.window.as_ref().ok_or("SEGMENTED_WINDOW_LAYOUT")?;
        let ids = &window.root_targets;
        if ids.len() as u64 != self.counts[self.rank as usize] {
            return Err("SEGMENTED_WINDOW_ROOT_ROWS".into());
        }
        let size = (ids.len() as u64)
            .checked_mul(16)
            .ok_or("PATH_LINEAGE_SIZE")?;
        if size > budget {
            return Err("LINEAGE_LIMIT".into());
        }
        let mut writer = BufWriter::new(fs::File::create(output).map_err(io_error)?);
        let mut bytes = 0;
        for (row, &target) in ids.iter().enumerate() {
            if row % 4096 == 0 {
                checkpoint()?;
            }
            if target as usize >= goals {
                return Err("SEGMENTED_WINDOW_ROOT_TARGET".into());
            }
            write_pair(
                &mut writer,
                &mut bytes,
                budget,
                u64::from(target),
                row as u64,
            )?;
        }
        writer.flush().map_err(io_error)?;
        Ok(bytes)
    }

    fn advance_window_disk(
        &mut self,
        output: &Path,
        next_length: u64,
        stop: &impl Fn() -> u32,
        lineage: &mut Lineage,
    ) -> Result<DiskLayer> {
        let next_depth = self.depth + 1;
        let writer = checkpoint(stop).and_then(|()| {
            start_writer(TransitionSpool::new(
                output.join(format!("{next_depth}.runs")),
                lineage.budget - lineage.bytes,
            )?)
        });
        if let Some(reason) = self.agree_window_artifacts(&writer)? {
            return Ok(DiskLayer::Stopped(reason));
        }
        crate::native_fault::layer(next_length, next_length - 1);
        let (advanced, boundary_stop, spool) = self.advance_window(writer?, stop)?;
        lineage.observe(spool.spool.peak_reserved_bytes())?;
        match advanced {
            // A drained, empty frontier proves that every later cut is empty.
            DistributedAdvance::Complete => Ok(DiskLayer::Empty),
            // The packet protocol already agreed this terminal code. Do not
            // enter another collective on the aborted communicator.
            DistributedAdvance::Resource { fatal } => stop_reason(fatal)
                .map(DiskLayer::Stopped)
                .ok_or_else(|| format!("SEGMENTED_WINDOW_OWNER_RESOURCE_{fatal}")),
            DistributedAdvance::Layer { .. } if boundary_stop != 0 => stop_reason(boundary_stop)
                .map(DiskLayer::Stopped)
                .ok_or_else(|| format!("SEGMENTED_WINDOW_BOUNDARY_STOP_{boundary_stop}")),
            DistributedAdvance::Layer { .. } => Ok(DiskLayer::Ready(spool)),
        }
    }

    fn write_window_depth(
        &mut self,
        targets: &Targets<'_>,
        output: &Path,
        spool: Option<FinishedSpool>,
        checkpoint: &impl Fn() -> Result<()>,
        lineage: &mut Lineage,
    ) -> Result<Option<&'static str>> {
        let mut pending = PendingFiles::default();
        let local = (|| {
            let csr_bytes = match spool {
                Some(spool) => {
                    self.finish_window_csr(output, spool, &mut pending, checkpoint, lineage)?
                }
                None => {
                    checkpoint()?;
                    0
                }
            };
            let target_file =
                pending.claim_records(output.join(format!("{}.targets", self.depth)))?;
            let before_targets = lineage.observe(csr_bytes)?;
            let written = self.window_write_targets(
                targets,
                target_file,
                lineage.budget - before_targets,
                checkpoint,
            );
            let bytes = lineage.observe_records(csr_bytes, target_file, &written)?;
            written?;
            Ok((bytes, csr_bytes))
        })();
        if let Some(reason) = self.agree_window_artifacts(&local)? {
            return Ok(Some(reason));
        }
        let (bytes, csr_bytes) = local?;
        lineage.commit(bytes, csr_bytes);
        pending.commit();
        Ok(None)
    }

    fn finish_window_csr(
        &self,
        output: &Path,
        finished: FinishedSpool,
        pending: &mut PendingFiles,
        checkpoint: &impl Fn() -> Result<()>,
        lineage: &mut Lineage,
    ) -> Result<u64> {
        let FinishedSpool { mut spool, failure } = finished;
        if let Some(error) = failure {
            return Err(error);
        }
        checkpoint()?;
        let by_shard = self.window_rows_by_shard()?;
        if by_shard
            .iter()
            .try_fold(0u64, |sum, rows| sum.checked_add(*rows))
            != Some(self.counts[self.rank as usize])
        {
            return Err("SEGMENTED_WINDOW_LAYER_ROWS".into());
        }
        let depth = self.depth;
        let csr = output.join(format!("{depth}.csr"));
        unused(&csr)?;
        let csr = pending.csr.insert(csr);
        let written = spool.finish(
            &output.join(format!("{depth}.staging")),
            csr,
            &by_shard,
            checkpoint,
        );
        lineage.observe(spool.peak_reserved_bytes())?;
        Ok(written?.0)
    }

    pub(super) fn agree_window_artifacts<T>(
        &mut self,
        local: &Result<T>,
    ) -> Result<Option<&'static str>> {
        let code = local.as_ref().err().map_or(0, |error| artifact_code(error));
        match self.vote(code)? {
            0 => Ok(None),
            ARTIFACT_ERROR => Err(local
                .as_ref()
                .err()
                .filter(|error| artifact_code(error) == ARTIFACT_ERROR)
                .cloned()
                .unwrap_or_else(|| "SEGMENTED_WINDOW_REMOTE_LINEAGE".into())),
            code => stop_reason(code)
                .map(Some)
                .ok_or_else(|| format!("SEGMENTED_WINDOW_ARTIFACT_{code}")),
        }
    }
}
