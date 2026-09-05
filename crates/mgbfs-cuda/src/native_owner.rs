//! Stable owner/state/directory C ABI; ownership is enforced by runtime lanes.
pub use mgbfs_core::owner_job::{BucketJob, Range};
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundedOwnerBytes {
    pub flags: u64,
    pub indices: u64,
    pub merged: u64,
    pub refinement_errors: u64,
}
const _: [(); 32] = [(); std::mem::size_of::<BoundedOwnerBytes>()];
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub duplicates: u32,
    pub prev: u32,
    pub curr: u32,
    pub accepted: u32,
    pub survivors: u32,
    pub new_count: u32,
    pub output_offset: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Control {
    pub error: u32,
    pub stage: u32,
    pub survivors: u32,
    pub reserved: u32,
    pub padding: [u64; 6],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Ring {
    pub head: u64,
    pub tail: u64,
    pub descriptor_head: u64,
    pub descriptor_tail: u64,
    pub capacity: u64,
    pub descriptor_capacity: u64,
    pub fatal: u32,
    pub reserved: u32,
    pub padding: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Extent {
    pub sequence: u64,
    pub begin: u64,
    pub count: u64,
    pub descriptor: u64,
    pub granted_rows: u32,
    pub ready: u32,
    pub padding: [u64; 3],
}
const _: [(); 64] = [(); std::mem::size_of::<Control>()];
const _: [(); 64] = [(); std::mem::size_of::<Ring>()];
const _: [(); 64] = [(); std::mem::size_of::<Extent>()];
const _: [(); 32] = [(); std::mem::size_of::<Counts>()];
#[cfg(feature = "cuda")]
mod calls {
    use super::*;
    use std::ffi::c_void;
    extern "C" {
        pub fn mgbfs_state_reserve_layer(
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            count: *mut u32,
            capacity: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_state_retire_dense_prefix(
            ring: *mut Ring,
            extent: *mut Extent,
            records: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_archive_pack_permutation_u8(
            n: u32,
            stride: u32,
            states: *const u8,
            count: u32,
            permutations: *mut u8,
            ring: *mut Ring,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_bounded_owner_query(
            i: u32,
            j: u32,
            k: u32,
            backend: u32,
            refinement_capacity: u32,
            tile_limit: u32,
            out: *mut BoundedOwnerBytes,
        ) -> i32;
        pub fn mgbfs_bounded_owner_create(i: u32, j: u32, k: u32, out: *mut *mut c_void) -> i32;
        pub fn mgbfs_bounded_owner_create_backend(
            i: u32,
            j: u32,
            k: u32,
            backend: u32,
            refinement_capacity: u32,
            tile_limit: u32,
            out: *mut *mut c_void,
        ) -> i32;
        pub fn mgbfs_bounded_owner_destroy(plan: *mut c_void);
        pub fn mgbfs_bounded_owner_compare(
            plan: *mut c_void,
            jobs: *const BucketJob,
            j: u32,
            rows: u32,
            input: *const c_void,
            prev: *const c_void,
            pn: u64,
            curr: *const c_void,
            cn: u64,
            accepted: *const c_void,
            lengths: *const u32,
            buckets: u32,
            per_shard: u32,
            lane: u32,
            generation: u32,
            counts: *mut Counts,
            control: *mut Control,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_bounded_owner_commit(
            plan: *mut c_void,
            jobs: *const BucketJob,
            j: u32,
            input: *const c_void,
            accepted: *mut c_void,
            lengths: *mut u32,
            counts: *const Counts,
            control: *mut Control,
            grant: *const u32,
            selected: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_state_reserve(
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_state_materialize(
            input: *const u8,
            candidates: u32,
            refs: *const u64,
            sorted: u32,
            selected: *const u32,
            capacity: u32,
            stride: u32,
            output: *mut u8,
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            stream: *mut c_void,
        ) -> i32;
        /// Build HASH_FIRST requests after owner commit, without publishing ready.
        /// All pointers are device resident; target refs are absolute sequences.
        pub fn mgbfs_state_build_requests(
            origins: *const crate::ffi::RegenerateOrigin,
            candidates: u32,
            refs: *const u64,
            sorted: u32,
            selected: *const u32,
            capacity: u32,
            requests: *mut crate::ffi::RegenerateOrigin,
            targets: *mut u64,
            count: *mut u32,
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            stream: *mut c_void,
        ) -> i32;
        /// Sort and validate complete target coverage before dense publication.
        pub fn mgbfs_state_apply_responses(
            plan: *mut c_void,
            responses: *const u8,
            targets: *const u64,
            count: *const u32,
            group_fatal: *const u32,
            states: *mut u8,
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_state_apply_response_span(
            plan: *mut c_void,
            responses: *const u8,
            targets: *const u64,
            count: *const u32,
            sorted_offset: u32,
            group_fatal: *const u32,
            states: *mut u8,
            ring: *mut Ring,
            control: *mut Control,
            extent: *mut Extent,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_bucket_directory(
            sorted: *const c_void,
            count: *const u32,
            capacity: u32,
            buckets: u32,
            dir: *mut Range,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_bucket_directory(
            sorted: *const c_void,
            count: *const u32,
            capacity: u32,
            buckets: u32,
            owner: u32,
            dir: *mut Range,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_bind_owner_jobs(
            jobs: *mut BucketJob,
            n: u32,
            counts: *const u32,
            buckets: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_compact_hash_layer(
            accepted: *const c_void,
            counts: *const u32,
            buckets: u32,
            k: u32,
            output: *mut c_void,
            capacity: u32,
            dir: *mut Range,
            total: *mut u32,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaHostAlloc(out: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
        pub fn cudaMemcpy2DAsync(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
        pub fn cudaEventQuery(event: *mut c_void) -> i32;
        pub fn cudaGetDevice(device: *mut i32) -> i32;
        pub fn cudaSetDevice(device: i32) -> i32;
        pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
}
#[cfg(feature = "cuda")]
pub use calls::*;
