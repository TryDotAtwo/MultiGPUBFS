//! Host control-plane job splitting from GPU-produced bucket directories.
//! Never receives state/hash payloads. Output storage is allocated before run.
use mgbfs_core::{
    owner_job::{BucketJob, Range},
    Result,
};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobSpan {
    pub first: usize,
    pub buckets: u32,
    pub source_begin: u64,
    pub rows: u32,
}
pub fn split(
    incoming: &[Range],
    prev: &[Range],
    curr: &[Range],
    i: u32,
    j: u32,
    per_shard: u32,
    generation: u32,
    descriptors: &mut [BucketJob],
    spans: &mut [JobSpan],
) -> Result<(usize, usize)> {
    if i == 0
        || i > i32::MAX as u32
        || j == 0
        || !per_shard.is_power_of_two()
        || incoming.len() != prev.len()
        || incoming.len() != curr.len()
        || incoming.len() > u32::MAX as usize
    {
        return Err("JOB_SHAPE".into());
    }
    let mut end = 0;
    for r in incoming {
        if r.begin != end {
            return Err("JOB_DIRECTORY".into());
        }
        end = end.checked_add(r.count).ok_or("JOB_OVERFLOW")?;
    }
    let (mut bucket, mut offset, mut nd, mut ns) = (0usize, 0u64, 0usize, 0usize);
    while bucket < incoming.len() {
        while bucket < incoming.len() && incoming[bucket].count == offset {
            bucket += 1;
            offset = 0;
        }
        if bucket == incoming.len() {
            break;
        }
        let shard = bucket / per_shard as usize;
        let mut span = JobSpan {
            first: nd,
            buckets: 0,
            source_begin: incoming[bucket].begin + offset,
            rows: 0,
        };
        while bucket < incoming.len()
            && bucket / per_shard as usize == shard
            && span.buckets < j
            && span.rows < i
        {
            let remaining = incoming[bucket].count - offset;
            if remaining == 0 {
                bucket += 1;
                offset = 0;
                continue;
            }
            let free = u64::from(i - span.rows);
            // Keep a small bucket whole when it can fit in the next job.
            // Oversized buckets must split; no truncation or hidden growth.
            if span.rows > 0 && remaining <= u64::from(i) && remaining > free {
                break;
            }
            let take = remaining.min(free);
            let d = descriptors.get_mut(nd).ok_or("JOB_DESCRIPTOR_CAPACITY")?;
            *d = BucketJob {
                bucket: bucket as u32,
                lane: shard as u32,
                incoming: Range {
                    begin: u64::from(span.rows),
                    count: take,
                },
                prev: prev[bucket],
                curr: curr[bucket],
                accepted_count: 0,
                generation,
            };
            nd += 1;
            span.buckets += 1;
            span.rows += take as u32;
            offset += take;
            if offset == incoming[bucket].count {
                bucket += 1;
                offset = 0;
            }
        }
        *spans.get_mut(ns).ok_or("JOB_SPAN_CAPACITY")? = span;
        ns += 1;
    }
    Ok((nd, ns))
}
