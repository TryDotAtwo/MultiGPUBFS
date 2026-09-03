//! Bounded owner job ABI. Admission does not commit hashes or reserve states.
use crate::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Range {
    pub begin: u64,
    pub count: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct BucketJob {
    pub bucket: u32,
    pub lane: u32,
    pub incoming: Range,
    pub prev: Range,
    pub curr: Range,
    pub accepted_count: u32,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct JobLimits {
    pub incoming: u32,
    pub touched_buckets: u32,
    pub bucket_capacity: u32,
    pub buckets_per_shard: u32,
    pub shard_count: u32,
    pub lane: u32,
    pub generation: u32,
    pub prev_arena_count: u64,
    pub curr_arena_count: u64,
}

/// Read-only, allocation-free validation of one lane's job descriptor slice.
/// Input must be packed from row zero in ascending bucket order, with one
/// descriptor per touched bucket. Returns its live row count, not capacity.
/// The scheduler must hold the shard's exclusive writer lease and refresh
/// accepted_count after every commit. This check neither inspects hash data
/// nor proves sortedness, directory authenticity or available commit credits.
pub fn admit(jobs: &[BucketJob], limits: JobLimits) -> Result<u32> {
    if limits.incoming == 0
        || limits.incoming > i32::MAX as u32
        || limits.touched_buckets == 0
        || limits.bucket_capacity == 0
        || !limits.buckets_per_shard.is_power_of_two()
        || !limits.shard_count.is_power_of_two()
        || u64::from(limits.buckets_per_shard) * u64::from(limits.shard_count)
            > u64::from(u32::MAX) + 1
    {
        return Err("OWNER_JOB_LIMITS".into());
    }
    if jobs.is_empty() {
        return Err("OWNER_EMPTY_JOB".into());
    }
    if jobs.len() > limits.touched_buckets as usize {
        return Err("OWNER_JOB_CAPACITY".into());
    }
    let shard = jobs[0].bucket / limits.buckets_per_shard;
    if shard >= limits.shard_count {
        return Err("OWNER_SHARD".into());
    }
    let mut incoming_end = 0u64;
    let mut last_bucket = None;
    for job in jobs {
        if job.lane != limits.lane {
            return Err("OWNER_LANE".into());
        }
        if job.generation != limits.generation {
            return Err("OWNER_GENERATION".into());
        }
        if job.bucket / limits.buckets_per_shard != shard {
            return Err("OWNER_SHARD".into());
        }
        if last_bucket.is_some_and(|last| job.bucket <= last) {
            return Err("OWNER_BUCKET_ORDER".into());
        }
        last_bucket = Some(job.bucket);
        if job.incoming.begin != incoming_end || job.incoming.count == 0 {
            return Err("OWNER_INCOMING_RANGE".into());
        }
        incoming_end = incoming_end
            .checked_add(job.incoming.count)
            .filter(|&end| end <= u64::from(limits.incoming))
            .ok_or("OWNER_INCOMING_CAPACITY")?;
        if job.accepted_count > limits.bucket_capacity
            || job.prev.count > u64::from(limits.bucket_capacity)
            || job.curr.count > u64::from(limits.bucket_capacity)
        {
            return Err("OWNER_BUCKET_CAPACITY".into());
        }
        for (range, capacity, error) in [
            (job.prev, limits.prev_arena_count, "OWNER_PREV_RANGE"),
            (job.curr, limits.curr_arena_count, "OWNER_CURR_RANGE"),
        ] {
            if range
                .begin
                .checked_add(range.count)
                .map_or(true, |end| end > capacity)
            {
                return Err(error.into());
            }
        }
    }
    Ok(incoming_end as u32)
}
