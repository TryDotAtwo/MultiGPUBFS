//! The completed-layer boundary after either candidate transport has drained.

use super::{Range, SegmentedDistributedBfs};
use crate::generic_distributed_native::DistributedAdvance;
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpyAsync, mgbfs_nccl_all_gather_pair_u64, mgbfs_nccl_all_gather_u64};
use std::ffi::c_void;

pub(super) struct CompletedLayer {
    current: Vec<Range>,
    counts: Vec<u64>,
    pub(super) local: u64,
    pub(super) global: u64,
}

fn total(counts: &[u64]) -> Result<u64> {
    counts.iter().try_fold(0u64, |sum, &count| {
        sum.checked_add(count)
            .ok_or_else(|| "SEGMENTED_GLOBAL_COUNT_OVERFLOW".into())
    })
}

impl SegmentedDistributedBfs {
    pub(super) fn prepare_layer_commit(
        &mut self,
        bank: u32,
        before: &[Vec<u64>],
        boundary_stop: &impl Fn() -> u32,
        next_stop: &mut u32,
    ) -> Result<CompletedLayer> {
        let mut current = Vec::new();
        let mut local = 0u64;
        for (shard, bucket) in self.segments.iter().enumerate() {
            for (segment, descriptor) in bucket.iter().enumerate() {
                let end = descriptor.banks[bank as usize];
                let begin = before
                    .get(shard)
                    .and_then(|rows| rows.get(segment))
                    .copied()
                    .unwrap_or(0);
                if end < begin {
                    return Err("SEGMENTED_COUNT_DECREASED".into());
                }
                if end > begin {
                    current.push(Range {
                        shard,
                        segment,
                        bank,
                        begin,
                        end,
                        logical_start: 0,
                    });
                    local = local
                        .checked_add(end - begin)
                        .ok_or("SEGMENTED_LOCAL_COUNT_OVERFLOW")?;
                }
            }
        }
        if let Some(nearest) = &self.nearest {
            nearest.check_layer_size(local)?;
        }

        let counts = if self.plan.world == 1 {
            vec![local]
        } else {
            self.gather_layer_boundary(local, boundary_stop, next_stop)?
        };
        let global = total(&counts)?;
        Ok(CompletedLayer {
            current,
            counts,
            local,
            global,
        })
    }

    fn gather_layer_boundary(
        &mut self,
        local: u64,
        boundary_stop: &impl Fn() -> u32,
        next_stop: &mut u32,
    ) -> Result<Vec<u64>> {
        let world = self.plan.world as usize;
        let stream = self.base.stream.ptr;
        let packet = self.uses_packet_transport();
        let boundary = [
            local,
            if packet {
                u64::from(boundary_stop())
            } else {
                0
            },
        ];
        let words = if packet { 2 } else { 1 };
        crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                self.control.at::<c_void>(32),
                boundary.as_ptr().cast(),
                words * 8,
                1,
                stream,
            )
        })?;
        self.note_sync(|counts| counts.nccl_layer_count_gathers += 1);
        if packet {
            crate::generic_native::checked!(mgbfs_nccl_all_gather_pair_u64, unsafe {
                mgbfs_nccl_all_gather_pair_u64(
                    self.comm.ptr,
                    self.control.at(32),
                    self.control.at(64),
                    stream,
                )
            })?;
        } else {
            crate::generic_native::checked!(mgbfs_nccl_all_gather_u64, unsafe {
                mgbfs_nccl_all_gather_u64(
                    self.comm.ptr,
                    self.control.at(32),
                    self.control.at(64),
                    stream,
                )
            })?;
        }
        let mut counts = vec![0u64; world * words];
        crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                counts.as_mut_ptr().cast(),
                self.control.at::<c_void>(64),
                counts.len() * 8,
                2,
                stream,
            )
        })?;
        self.wait_boundary_stream()?;
        if packet {
            for rank in 0..world {
                *next_stop = (*next_stop).max(counts[2 * rank + 1] as u32);
                counts[rank] = counts[2 * rank];
            }
            counts.truncate(world);
        }
        Ok(counts)
    }

    pub(super) fn publish_layer(&mut self, mut layer: CompletedLayer) -> Result<DistributedAdvance> {
        self.previous_global = total(&self.counts)?;
        if let Some(nearest) = &mut self.nearest {
            nearest.publish(&mut layer.current, &self.segments, self.depth + 1);
        }
        self.current = layer.current;
        self.previous_snapshot_valid = false;
        self.counts = layer.counts;
        self.depth += 1;
        Ok(DistributedAdvance::Layer {
            local: layer.local,
            global: layer.global,
        })
    }
}
