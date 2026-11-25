//! Layer-local exact indexes and bounded lineage for the segmented Window.

mod join;
mod lineage;
mod run;

use self::lineage::{FinishedSpool, LineageCatalog, LineageSource, LineageTile, Writer};
use super::nearest::{target_lookup_kernel, FindTargets, TargetLookup, Targets};
use super::{
    ArenaPurpose, BatchHashShard, PacketOwnerShard, Plan, Range, RootSpec, Segment,
    SegmentedDistributedBfs,
};
use crate::generic_distributed_native::DistributedAdvance;
use crate::path_csr::io_error;
use crate::segmented_host_ready::HostAgreement;
use crate::segmented_packet_transport::{PacketJobToken, PACKET_CANCELLED, PACKET_DEADLINE};
use mgbfs_core::{graph_definition::GraphDefinitionV2, Result};
use mgbfs_cuda::ffi::cudaMemcpy;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

pub(super) struct WindowState {
    tile: LineageTile,
    expansion: Option<Expansion>,
    segment_bases: Vec<Arc<Vec<u64>>>,
    root_targets: Vec<u32>,
    target_lookup: Option<TargetLookup>,
    find_targets: FindTargets,
}

// Source allocations stay live until routing and every owner lease have
// completed. They are absent from the destination's exact lookup directory.
struct Expansion {
    source: Vec<Vec<Segment>>,
    source_index: Vec<Option<BatchHashShard>>,
    source_totals: [u64; 3],
    writer: Writer,
}

impl WindowState {
    pub(super) fn abi_ready() -> bool {
        LineageTile::abi_ready() && target_lookup_kernel().is_some()
    }

    pub(super) fn new(shards: usize, elements: u32, device: i32) -> Result<Self> {
        Ok(Self {
            tile: LineageTile::new(device)?,
            expansion: None,
            segment_bases: (0..shards).map(|_| Arc::new(Vec::new())).collect(),
            root_targets: Vec::new(),
            target_lookup: Some(TargetLookup::new(elements, device)?),
            find_targets: target_lookup_kernel().ok_or("SEGMENTED_PATH_TARGET_ABI_UNAVAILABLE")?,
        })
    }
}

fn number_ranges(ranges: &mut [Range]) -> Result<()> {
    let mut next = 0u64;
    for range in ranges {
        range.logical_start = next;
        next = next
            .checked_add(range.end - range.begin)
            .ok_or("SEGMENTED_WINDOW_ROW_OVERFLOW")?;
    }
    Ok(())
}

impl SegmentedDistributedBfs {
    pub(crate) fn segmented_window_available(bidirectional: bool) -> bool {
        WindowState::abi_ready() && (!bidirectional || join::JoinTile::abi_ready())
    }

    pub(crate) fn segmented_window_workspace(elements: usize, bidirectional: bool) -> u64 {
        let side_bytes =
            lineage::DEVICE_BYTES as u64 + (256u64 << 10).max(elements as u64 * 8 + 28);
        if bidirectional {
            2 * side_bytes + join::JoinTile::DEVICE_BYTES as u64
        } else {
            side_bytes
        }
    }

    pub(super) fn transfer_window_packet(&mut self, to: &mut Self) -> Result<()> {
        if self.window.is_none()
            || to.window.is_none()
            || !std::rc::Rc::ptr_eq(&self.comm, &to.comm)
            || self.packet.is_some() == to.packet.is_some()
        {
            return Err("SEGMENTED_WINDOW_PACKET_OWNERSHIP".into());
        }
        if self.packet.is_some() {
            std::mem::swap(&mut self.packet, &mut to.packet);
        }
        Ok(())
    }

    pub(crate) fn new_window(
        graph: &GraphDefinitionV2,
        inverse: Option<(&GraphDefinitionV2, &[Vec<i64>])>,
        device: u32,
        rank: u32,
        plan: Plan,
        id: &[u8; 128],
        packet_job_token: Option<PacketJobToken>,
        seed: u64,
        bits: u32,
        cuts: Option<&[u64]>,
        ready: &HostAgreement,
    ) -> Result<(Self, Option<Self>)> {
        lineage::prepare_pinned_pool(if inverse.is_some() { 2 } else { 1 });
        if let Some((inverse, targets)) = inverse {
            let (forward, reverse) = Self::new_path_pair(
                graph,
                inverse,
                targets,
                device,
                rank,
                plan,
                id,
                packet_job_token,
                seed,
                bits,
                cuts,
                ready,
                ArenaPurpose::Window,
            )?;
            Ok((forward, Some(reverse)))
        } else {
            let bfs = Self::new_arena(
                graph,
                device,
                rank,
                plan,
                id,
                packet_job_token,
                seed,
                bits,
                cuts,
                ready,
                ArenaPurpose::Window,
                RootSpec::Single,
                false,
                None,
            )?;
            Ok((bfs, None))
        }
    }

    pub(super) fn set_window_root_targets(&mut self, targets: Vec<u32>) -> Result<()> {
        self.window
            .as_mut()
            .ok_or("SEGMENTED_WINDOW_LAYOUT")?
            .root_targets = targets;
        Ok(())
    }

    pub(super) fn number_window_roots(&mut self) -> Result<()> {
        if self.window.is_some() {
            number_ranges(&mut self.current)?;
        }
        Ok(())
    }

    pub(super) fn packet_parent_segment(&self, range: Range) -> &Segment {
        let source = self
            .window
            .as_ref()
            .and_then(|window| window.expansion.as_ref())
            .map_or(&self.segments, |active| &active.source);
        &source[range.shard][range.segment]
    }

    pub(super) fn window_record_edges(
        &mut self,
        span: PacketOwnerShard,
        source: u32,
        stop: &impl Fn() -> u32,
    ) -> Result<()> {
        let shard = span.shard;
        let (view, directory, max_series) = self.window_catalog(shard)?;
        let window = self.window.as_mut().ok_or("SEGMENTED_WINDOW_LAYOUT")?;
        let active = window.expansion.as_mut().ok_or("SEGMENTED_WINDOW_LAYER")?;
        let segments = &self.segments[shard];
        let bases = &mut window.segment_bases[shard];
        // Row counts of an extent freeze when its successor becomes active.
        // Keep the base stable across owner waves and spill/replay.
        if bases.len() < segments.len() {
            let bases = Arc::make_mut(bases);
            while bases.len() < segments.len() {
                if segments[bases.len()].desc.banks != 1 {
                    return Err("SEGMENTED_WINDOW_EDGE_ROW".into());
                }
                let base = match bases.last() {
                    None => 0,
                    Some(&previous) => previous
                        .checked_add(segments[bases.len() - 1].banks[0])
                        .ok_or("SEGMENTED_WINDOW_ROW_OVERFLOW")?,
                };
                bases.push(base);
            }
        }
        window.tile.resolve(
            LineageSource {
                plan: &self.plan,
                span,
                rank: source,
                parent_rows: *self
                    .counts
                    .get(source as usize)
                    .ok_or("SEGMENTED_WINDOW_PARENT_RANK")?,
                generators: self.base.generators,
            },
            LineageCatalog {
                prepared: &view,
                directory: &directory,
                max_series,
                bases: Arc::clone(bases),
                last_rows: segments.last().map_or(0, |segment| segment.banks[0]),
            },
            &self.base.stream,
            &mut active.writer,
            || match stop() {
                1 => Err("CANCELLED".into()),
                2 => Err("TIME_LIMIT".into()),
                _ => Ok(()),
            },
        )
    }

    fn release_window_arena(
        &mut self,
        segments: Vec<Vec<Segment>>,
        indexes: Vec<Option<BatchHashShard>>,
    ) -> Result<()> {
        let mut freed = 0u64;
        for item in indexes.into_iter().flatten() {
            freed += item.dir.device_bytes();
            freed += item.view.map_or(0, |view| view.data.device_bytes());
            freed += item
                .snapshots
                .iter()
                .map(|snapshot| snapshot.storage.device_bytes())
                .sum::<u64>();
        }
        for mut segment in segments.into_iter().flatten() {
            if self.reuse_hash_segments
                && self.spare_hash_segment.is_none()
                && segment.desc.capacity == self.layout.rows_per_bank
            {
                // The shared spare is reset only if another shard takes it.
                segment.banks = [0; 3];
                self.spare_hash_segment = Some(segment);
            } else {
                freed += segment.device_bytes;
            }
        }
        self.remaining_bytes = self
            .remaining_bytes
            .checked_add(freed)
            .filter(|&value| value <= self.storage_budget)
            .ok_or("SEGMENTED_STORAGE_ACCOUNTING")?;
        Ok(())
    }

    pub(super) fn window_rows_by_shard(&self) -> Result<Vec<u64>> {
        self.segments
            .iter()
            .map(|segments| {
                segments.iter().try_fold(0u64, |rows, segment| {
                    rows.checked_add(segment.banks[0])
                        .ok_or_else(|| "SEGMENTED_WINDOW_ROW_OVERFLOW".into())
                })
            })
            .collect()
    }

    fn window_logical_row(&self, shard: usize, segment: u32, row: u32) -> Result<u64> {
        let key = (shard, segment as usize);
        let index = self
            .current
            .binary_search_by_key(&key, |range| (range.shard, range.segment))
            .map_err(|_| "SEGMENTED_WINDOW_TARGET_EXTENT")?;
        let range = self.current[index];
        let row = u64::from(row);
        if row < range.begin || row >= range.end {
            return Err("SEGMENTED_WINDOW_TARGET_ROW".into());
        }
        range
            .logical_start
            .checked_add(row - range.begin)
            .ok_or_else(|| "SEGMENTED_WINDOW_ROW_OVERFLOW".into())
    }

    fn window_catalog(
        &mut self,
        shard: usize,
    ) -> Result<(super::OwnerPrepared, super::HashSeriesDirectory, u32)> {
        let view = self
            .hash_view(shard, false)?
            .ok_or("SEGMENTED_WINDOW_LOOKUP_RESOURCE")?;
        let limit = self
            .hash_lookup_limit
            .ok_or("SEGMENTED_WINDOW_LOOKUP_BOUND")?;
        if self.hash_shards[shard]
            .as_ref()
            .is_some_and(|item| item.searched_series() > limit)
        {
            self.refresh_hash_series_snapshots(2, Some(shard))?;
        }
        let catalog = self.hash_shards[shard]
            .as_ref()
            .ok_or("SEGMENTED_WINDOW_LOOKUP_MISSING")?;
        if catalog.searched_series() > limit {
            return Err("SEGMENTED_WINDOW_LOOKUP_RESOURCE".into());
        }
        Ok((view, catalog.snapshot_directory(), limit))
    }

    fn window_write_targets(
        &mut self,
        targets: &Targets<'_>,
        output: &Path,
        budget: u64,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<u64> {
        let window = self.window.as_mut().ok_or("SEGMENTED_WINDOW_LAYOUT")?;
        let find = window.find_targets;
        let mut lookup = window
            .target_lookup
            .take()
            .ok_or("SEGMENTED_WINDOW_TARGETS")?;
        let result = (|| {
            let mut writer = BufWriter::new(File::create(output).map_err(io_error)?);
            let mut bytes = 0u64;
            for (shard, local) in targets.local.iter().enumerate() {
                if local.is_empty() || self.segments[shard].iter().all(|seg| seg.banks[0] == 0) {
                    continue;
                }
                let (view, directory, limit) = self.window_catalog(shard)?;
                for tile in local.chunks(lookup.tile_rows()) {
                    checkpoint()?;
                    let hits =
                        lookup.find(self, find, &view, &directory, limit, targets.states, tile)?;
                    for (target, hit) in tile.iter().zip(hits) {
                        if hit.found == 0 {
                            continue;
                        }
                        if hit.found != 1 || hit.bank != 0 {
                            return Err("SEGMENTED_WINDOW_TARGET_BANK".into());
                        }
                        let row = self.window_logical_row(shard, hit.segment, hit.row)?;
                        write_pair(
                            &mut writer,
                            &mut bytes,
                            budget,
                            u64::from(target.index),
                            row,
                        )?;
                    }
                }
            }
            writer.flush().map_err(io_error)?;
            Ok(bytes)
        })();
        self.window.as_mut().unwrap().target_lookup = Some(lookup);
        result
    }

    fn advance_window(
        &mut self,
        writer: Writer,
        stop: &impl Fn() -> u32,
    ) -> Result<(DistributedAdvance, u32, FinishedSpool)> {
        self.connect_packet_control()?;
        let window = self.window.as_mut().ok_or("SEGMENTED_WINDOW_LAYOUT")?;
        if window.expansion.is_some() {
            return Err("SEGMENTED_WINDOW_LAYER".into());
        }
        for bases in &mut window.segment_bases {
            Arc::make_mut(bases).clear();
        }
        let shards = self.segments.len();
        let old_source = std::mem::replace(
            &mut self.segments,
            (0..shards).map(|_| Vec::new()).collect(),
        );
        let old_index =
            std::mem::replace(&mut self.hash_shards, (0..shards).map(|_| None).collect());
        window.expansion = Some(Expansion {
            source: old_source,
            source_index: old_index,
            source_totals: std::mem::take(&mut self.bank_totals),
            writer,
        });
        let mut boundary = 0;
        let result = self.expand_window(stop, &mut boundary);
        let Expansion {
            source,
            source_index,
            source_totals,
            writer,
        } = self.window.as_mut().unwrap().expansion.take().unwrap();
        let complete = matches!(result, Ok(DistributedAdvance::Layer { .. })) && boundary == 0;
        let finished = self.window.as_mut().unwrap().tile.finish(writer, complete);
        if matches!(result, Ok(DistributedAdvance::Layer { .. })) {
            self.release_window_arena(source, source_index)?;
        } else {
            let partial = std::mem::replace(&mut self.segments, source);
            let partial_index = std::mem::replace(&mut self.hash_shards, source_index);
            self.bank_totals = source_totals;
            self.release_window_arena(partial, partial_index)?;
        }
        Ok((result?, boundary, finished?))
    }

    fn expand_window(
        &mut self,
        stop: &impl Fn() -> u32,
        boundary: &mut u32,
    ) -> Result<DistributedAdvance> {
        if self.capacity_limit.is_some() {
            let zero = 0u64;
            crate::generic_native::checked!(cudaMemcpy, unsafe {
                cudaMemcpy(self.control.at(24), (&zero as *const u64).cast(), 8, 1)
            })?;
        }
        let work = std::mem::take(&mut self.current);
        let result = (|| {
            let mut batch = self.active_batch.max(1).min(self.queue_batch_ceiling);
            if let Err(code) = self.advance_packets(&work, 0, &mut batch, false, stop)? {
                *boundary = match code {
                    PACKET_CANCELLED => 1,
                    PACKET_DEADLINE => 2,
                    _ => 0,
                };
                return Ok(DistributedAdvance::Resource { fatal: code });
            }
            let layer = self.prepare_layer_commit(0, &[], stop, boundary)?;
            if layer.global == 0 {
                return Ok(DistributedAdvance::Complete);
            }
            self.publish_layer(layer)
        })();
        if !matches!(result, Ok(DistributedAdvance::Layer { .. })) {
            self.current = work;
        }
        result
    }
}

fn write_pair(
    writer: &mut impl Write,
    bytes: &mut u64,
    budget: u64,
    left: u64,
    right: u64,
) -> Result<()> {
    *bytes = bytes.checked_add(16).ok_or("PATH_LINEAGE_SIZE")?;
    if *bytes > budget {
        return Err("LINEAGE_LIMIT".into());
    }
    writer.write_all(&left.to_le_bytes()).map_err(io_error)?;
    writer.write_all(&right.to_le_bytes()).map_err(io_error)
}
