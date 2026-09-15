//! Experimental raw ABI matching experiments/library_owner/owner_abi.h.
//! No safe wrapper, CPU fallback, or production scheduler integration is implied.
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KeysV1 {
    pub words: [*const u32; 4],
    pub rows: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CandidatesV1 {
    pub keys: KeysV1,
    pub source_indices: *const u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SurvivorsV1 {
    pub source_indices: *const u32,
    pub epoch: u64,
    pub rows: u32,
    pub reserved: u32,
}

const _: [(); 40] = [(); std::mem::size_of::<KeysV1>()];
const _: [(); 48] = [(); std::mem::size_of::<CandidatesV1>()];
const _: [(); 24] = [(); std::mem::size_of::<SurvivorsV1>()];
const _: [(); 8] = [(); std::mem::align_of::<KeysV1>()];
const _: [(); 8] = [(); std::mem::align_of::<CandidatesV1>()];
const _: [(); 8] = [(); std::mem::align_of::<SurvivorsV1>()];

/// Opaque handle: only the C++ adapter creates/destroys the object.
pub type OwnerHandle = *mut c_void;
pub type PoolHandle = *mut c_void;

#[cfg(feature = "library-owner")]
extern "C" {
    /// Convert native Hash128 AoS into five aligned u32 planes in preallocated
    /// scratch. No input/output aliasing. Data readiness is stream-ordered.
    pub fn mgbfs_library_candidates_from_aos_v1(
        hashes: *const c_void,
        rows: u32,
        capacity: u32,
        scratch: *mut c_void,
        scratch_bytes: u64,
        cuda_stream: *mut c_void,
        result: *mut CandidatesV1,
    ) -> i32;
    /// Convert committed SoA hashes back to native AoS in preallocated storage.
    pub fn mgbfs_library_keys_to_aos_v1(
        keys: KeysV1,
        output: *mut c_void,
        capacity: u32,
        cuda_stream: *mut c_void,
    ) -> i32;
    /// Install one fixed-size RMM pool on the current device before allocating
    /// owner inputs. Calls are serialized; no nested pool or runtime growth.
    pub fn mgbfs_library_pool_create_v1(
        bytes: u64,
        reserve_bytes: u64,
        pool: *mut PoolHandle,
    ) -> i32;
    /// Drain all users first. Live suballocations reject destruction and leave
    /// the handle valid. Success restores the previous resource, invalidating it.
    pub fn mgbfs_library_pool_destroy_v1(pool: PoolHandle) -> i32;
    /// Caller keeps history and installed fixed RMM pool alive on the same GPU.
    /// Zero status is success. A factory failure must leave `owner` null.
    pub fn mgbfs_library_owner_create_v1(
        history: KeysV1,
        capacity: u32,
        cuda_stream: *mut c_void,
        owner: *mut OwnerHandle,
    ) -> i32;
    /// Result indices are device pointers borrowed until the next compare/destroy.
    /// Count synchronization performed by the library belongs in measured time.
    pub fn mgbfs_library_owner_compare_v1(
        owner: OwnerHandle,
        epoch: u64,
        input: CandidatesV1,
        result: *mut SurvivorsV1,
    ) -> i32;
    /// `granted_rows` is actual reserved state/archive credit. Success enqueues
    /// writes; it does not replace a completed CUDA event or host publication gate.
    pub fn mgbfs_library_owner_commit_v1(owner: OwnerHandle, epoch: u64, granted_rows: u32) -> i32;
    /// Borrow append-order committed keys. Consume on the owner stream or after
    /// its completion event; pending/poisoned owners are rejected.
    pub fn mgbfs_library_owner_export_v1(owner: OwnerHandle, keys: *mut KeysV1) -> i32;
    /// Drain the stream before destroying the handle or external leases.
    pub fn mgbfs_library_owner_destroy_v1(owner: OwnerHandle);
}
