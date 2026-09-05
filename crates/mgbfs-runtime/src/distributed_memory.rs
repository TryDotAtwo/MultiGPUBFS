//! Actual shared Buffer allocations of the two-rank reference runtime.
//! Excludes library plans, HASH_FIRST-only storage, NCCL and pinned archive.
use mgbfs_core::{memory::AllocationLedger, Result};
use mgbfs_cuda::native_owner::{BucketJob, Control, Counts, Extent, Range, Ring};

#[derive(Clone, Copy)]
pub struct SharedBufferShape {
    pub state_stride: u64,
    pub packet_stride: u64,
    pub batch: u64,
    pub candidates: u64,
    pub layer_capacity: u64,
    pub state_ring_capacity: u64,
    pub buckets: u64,
    pub bucket_capacity: u64,
    pub job_buckets: u64,
    pub archive_width: u64,
}

pub fn shared_buffers(s: SharedBufferShape) -> Result<AllocationLedger> {
    if [
        s.state_stride,
        s.packet_stride,
        s.batch,
        s.candidates,
        s.layer_capacity,
        s.state_ring_capacity,
        s.buckets,
        s.bucket_capacity,
        s.job_buckets,
        s.archive_width,
    ]
    .contains(&0)
        || s.state_stride % 16 != 0
        || s.packet_stride % 16 != 0
        || !s.buckets.is_power_of_two()
        || s.job_buckets > s.buckets
    {
        return Err("DISTRIBUTED_MEMORY_SHAPE".into());
    }
    let mut l = AllocationLedger::new(u64::MAX, 0)?;
    let slots = s.buckets.checked_add(1).ok_or("BYTE_OVERFLOW")?;
    for (name, count, stride) in [
        ("states", s.state_ring_capacity, s.state_stride),
        ("prev", s.layer_capacity, 16),
        ("curr", s.layer_capacity, 16),
        (
            "accepted",
            s.buckets
                .checked_mul(s.bucket_capacity)
                .ok_or("BYTE_OVERFLOW")?,
            16,
        ),
        ("lengths", s.buckets, 4),
        ("children", s.candidates, s.packet_stride),
        ("child_hashes", s.candidates, 16),
        ("archive_hashes", s.batch, 16),
        ("archive_states", s.batch, s.archive_width),
        ("sorted_hashes", s.candidates, 16),
        ("sorted_refs", s.candidates, 8),
        ("route_count", 1, 4),
        ("packed_states", s.candidates, s.packet_stride),
        ("owner_counts", 1, 8),
        ("recv_states", s.candidates, s.packet_stride),
        ("recv_hashes", s.candidates, 16),
        ("recv_count", 1, 4),
        ("identity_refs", s.candidates, 8),
        ("directory", s.buckets, std::mem::size_of::<Range>() as u64),
        ("fatal", 1, 4),
        ("jobs_gpu", slots, std::mem::size_of::<BucketJob>() as u64),
        (
            "counts",
            s.job_buckets,
            std::mem::size_of::<Counts>() as u64,
        ),
        ("control", 1, std::mem::size_of::<Control>() as u64),
        ("selected", s.candidates, 4),
        ("ring", 1, std::mem::size_of::<Ring>() as u64),
        ("extent", 1, std::mem::size_of::<Extent>() as u64),
        ("layer_count", 1, 4),
        ("collective_send", 1, 4),
        ("collective_recv", 1, 8),
    ] {
        l.add(name, count, stride, 256)?;
    }
    Ok(l)
}
