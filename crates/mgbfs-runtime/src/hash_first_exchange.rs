//! Native two-rank HASH_FIRST request/response epoch. No allocation or host sync.
//! The scheduler owns count agreement, source leases, slots and final publication.
use mgbfs_core::Result;
use mgbfs_cuda::ffi::*;
use std::ffi::c_void;

pub struct MatrixSource {
    pub n: u32,
    pub moves: u32,
    pub modulus: u32,
    pub stride: u32,
    pub rank: u32,
    pub parent_begin: u64,
    pub parent_count: u32,
    pub parents: *const u8,
    pub generators: *const u8,
}

pub struct ExchangeBuffers {
    pub capacity: u32,
    pub outgoing_count: u32,
    pub incoming_count: u32,
    pub incoming_count_device: *const u32,
    pub outgoing_requests: *const RegenerateOrigin,
    pub incoming_requests: *mut RegenerateOrigin,
    pub outgoing_responses: *mut u8,
    pub incoming_responses: *mut u8,
    pub local_fatal: *mut u32,
    pub group_fatal: *mut u32,
}

fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("HASH_FIRST_EXCHANGE_STATUS_{status}"))
    }
}

/// Enqueue requests -> selected regeneration -> responses -> group fatal vote.
/// Zero-count peers issue exactly the same operations, in the same order.
/// Success means enqueue only: no StateReady publication occurs here.
///
/// # Safety
/// `comm` is a live two-rank communicator; both ranks enter matching epochs.
/// Counts must have been agreed before entry; incoming_count_device must equal
/// incoming_count. Inputs are canonical and buffers disjoint, device resident,
/// preallocated for capacity rows (16-byte origins, stride-byte responses).
/// Response buffers must be initialized: a poisoned regeneration still drains
/// the agreed transfers before the fatal vote. Do not consume any response when
/// group_fatal is nonzero. All buffers, including source parents, remain alive
/// until stream completion; only then may the scheduler close the source lease.
/// A host error requires rank-group abort, never retry or partial continuation.
pub unsafe fn enqueue_round_trip(
    comm: *mut c_void,
    peer: u32,
    source: &MatrixSource,
    buffers: &ExchangeBuffers,
    stream: *mut c_void,
) -> Result<()> {
    let b = buffers;
    let s = source;
    if comm.is_null()
        || s.rank >= 2
        || peer != (s.rank ^ 1)
        || s.n == 0
        || u64::from(s.n) * u64::from(s.n) > 33025
        || u64::from(s.n) * u64::from(s.n) > u64::from(s.stride)
        || s.stride % 16 != 0
        || !(1..=65536).contains(&s.moves)
        || !(2..=256).contains(&s.modulus)
        || b.capacity == 0
        || b.incoming_count > b.capacity
        || b.outgoing_count > b.capacity
        || s.parents.is_null()
        || s.generators.is_null()
        || b.incoming_count_device.is_null()
        || b.outgoing_requests.is_null()
        || b.incoming_requests.is_null()
        || b.outgoing_responses.is_null()
        || b.incoming_responses.is_null()
        || b.local_fatal.is_null()
        || b.group_fatal.is_null()
    {
        return Err("HASH_FIRST_EXCHANGE_CONTRACT".into());
    }
    check(mgbfs_nccl_send_recv(
        comm,
        b.outgoing_requests.cast(),
        u64::from(b.outgoing_count) * 16,
        peer,
        b.incoming_requests.cast(),
        u64::from(b.incoming_count) * 16,
        stream,
    ))?;
    check(mgbfs_regenerate_selected(
        s.n,
        s.moves,
        s.modulus,
        s.stride,
        b.capacity,
        s.rank,
        s.parent_begin,
        s.parent_count,
        s.parents,
        s.generators,
        b.incoming_requests,
        b.incoming_count_device,
        b.outgoing_responses,
        b.local_fatal,
        stream,
    ))?;
    check(mgbfs_nccl_send_recv(
        comm,
        b.outgoing_responses.cast(),
        u64::from(b.incoming_count) * u64::from(s.stride),
        peer,
        b.incoming_responses.cast(),
        u64::from(b.outgoing_count) * u64::from(s.stride),
        stream,
    ))?;
    check(mgbfs_nccl_all_reduce_max_u32(
        comm,
        b.local_fatal,
        b.group_fatal,
        stream,
    ))
}
