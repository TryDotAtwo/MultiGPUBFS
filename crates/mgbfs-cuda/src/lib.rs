//! Raw C ABI. CUDA is opt-in; there is no CPU implementation of these calls.
pub mod allocation;
pub mod native_owner;
#[cfg(feature = "cuda")]
pub mod ffi {
    pub use crate::allocation::{
        FutureMergeBytes, GenerateBytes, HashBytes, MaterializeBytes, RouteBytes,
    };
    use std::ffi::{c_char, c_void};
    /// Device ABI matching cuda/regenerate.h and little-endian wire OriginRef.
    /// Do not transmute the Rust core OriginRef (it has no C layout).
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug)]
    pub struct RegenerateOrigin {
        pub source: u32,
        pub movement: u16,
        pub reserved: u16,
        pub parent: u64,
    }
    const _: [(); 16] = [(); std::mem::size_of::<RegenerateOrigin>()];
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug)]
    pub struct FrontierState {
        pub count: u32,
        pub fatal: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug)]
    pub struct OwnerState {
        pub last_epoch: u64,
        pub count: u32,
        pub initialized: u32,
        pub fatal: u32,
        pub reserved: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
    pub struct MacroSettleBytes {
        pub indices: u64,
        pub selected: u64,
        pub flags: u64,
        pub count: u64,
        pub scratch: u64,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
    pub struct MacroSettleState {
        pub last_epoch: u64,
        pub count: u32,
        pub fatal: u32,
    }
    extern "C" {
        /// Scalar CUDA reference: emit only hashes and origins (no child state).
        /// Device coefficients are row-major [n*n,4] canonical F_p residues.
        /// Caller validates canonical inputs and retains all buffers until stream
        /// completion. Device fatal is sticky; candidate_count is zero on fatal.
        pub fn mgbfs_generate_hash_only(
            n: u32,
            moves: u32,
            modulus: u32,
            stride: u32,
            parent_capacity: u32,
            candidate_capacity: u32,
            source: u32,
            parent_begin: u64,
            parents: *const u8,
            generators: *const u8,
            coefficients: *const u32,
            offsets: *const u32,
            parent_count: *const u32,
            hashes: *mut u32,
            origins: *mut RegenerateOrigin,
            candidate_count: *mut u32,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        /// Enqueues selected-parent matrix regeneration; no allocation or host sync.
        /// Parents and requests must remain live through stream completion.
        /// All pointers except stream are device pointers. Fatal is sticky;
        /// output is dense request order, with zero padding. Status 0 is enqueue
        /// success only; inspect fatal after the consuming stream completes.
        pub fn mgbfs_regenerate_selected(
            n: u32,
            moves: u32,
            modulus: u32,
            stride: u32,
            capacity: u32,
            source_rank: u32,
            parent_begin: u64,
            parent_count: u32,
            parents: *const u8,
            generators: *const u8,
            requests: *const RegenerateOrigin,
            count: *const u32,
            output: *mut u8,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_generate_query(
            n: u32,
            moves: u32,
            modulus: u32,
            capacity: u32,
            variant: u32,
            out: *mut GenerateBytes,
        ) -> i32;
        pub fn mgbfs_hash_query(bytes: u32, capacity: u32, out: *mut HashBytes) -> i32;
        pub fn mgbfs_materialize_query(
            stride: u32,
            capacity: u32,
            frontier: u32,
            out: *mut MaterializeBytes,
        ) -> i32;
        pub fn mgbfs_future_merge_query(
            stride: u32,
            future: u32,
            incoming: u32,
            out: *mut FutureMergeBytes,
        ) -> i32;
        pub fn mgbfs_materialize_create(
            stride: u32,
            candidate_capacity: u32,
            frontier_capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_materialize_run(
            plan: *mut c_void,
            source: *const u8,
            source_count: u32,
            hashes: *const c_void,
            refs: *const u64,
            count: *const u32,
            states: *mut u8,
            out_hashes: *mut c_void,
            state: *mut FrontierState,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_materialize_destroy(plan: *mut c_void);
        /// Stable per-source parent sorting, preserving the target StateRef pair.
        /// Reuses the materialize plan's scratch; device fatal is sticky.
        pub fn mgbfs_materialize_sort_origins(
            plan: *mut c_void,
            source_rank: u32,
            origins: *const RegenerateOrigin,
            targets: *const u64,
            count: *const u32,
            sorted_origins: *mut RegenerateOrigin,
            sorted_targets: *mut u64,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_create(
            candidate_capacity: u32,
            bucket_capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_owner_run(
            plan: *mut c_void,
            prev: *const c_void,
            prev_count: u32,
            curr: *const c_void,
            curr_count: u32,
            accepted: *mut c_void,
            state: *mut OwnerState,
            candidates: *const c_void,
            refs: *const u64,
            candidate_count: *const u32,
            survivors: *mut c_void,
            survivor_refs: *mut u64,
            survivor_count: *mut u32,
            epoch: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_destroy(plan: *mut c_void);
        pub fn mgbfs_macro_settle_query(
            candidate_capacity: u32,
            history_layers: u32,
            history_capacity: u32,
            out: *mut MacroSettleBytes,
        ) -> i32;
        pub fn mgbfs_macro_settle_create(
            candidate_capacity: u32,
            history_layers: u32,
            history_capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_macro_settle_run(
            plan: *mut c_void,
            future: *const c_void,
            refs: *const u64,
            count: *const u32,
            history: *const c_void,
            history_counts: *const u32,
            survivors: *mut c_void,
            survivor_refs: *mut u64,
            survivor_count: *mut u32,
            state: *mut MacroSettleState,
            epoch: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_macro_settle_destroy(plan: *mut c_void);
        pub fn mgbfs_future_merge_create(
            stride: u32,
            future_capacity: u32,
            incoming_capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_future_merge_run(
            plan: *mut c_void,
            future_states: *mut u8,
            future_hashes: *mut c_void,
            future_state: *mut FrontierState,
            source_states: *const u8,
            source_count: u32,
            incoming_hashes: *const c_void,
            incoming_refs: *const u64,
            incoming_count: *const u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_future_merge_run_bounded(
            plan: *mut c_void,
            future_states: *mut u8,
            future_hashes: *mut c_void,
            future_state: *mut FrontierState,
            old_count_bound: u32,
            source_states: *const u8,
            source_count: u32,
            incoming_hashes: *const c_void,
            incoming_refs: *const u64,
            incoming_count: *const u32,
            incoming_count_bound: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_future_merge_destroy(plan: *mut c_void);
        pub fn mgbfs_exchange_pack(
            stride: u32,
            capacity: u32,
            source_states: *const u8,
            source_count: u32,
            sorted_hashes: *const c_void,
            sorted_refs: *const u64,
            count: u32,
            packed_states: *mut u8,
            owner_counts: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_unique_id(id128: *mut c_void) -> i32;
        pub fn mgbfs_nccl_create(
            rank: u32,
            world: u32,
            device: u32,
            id128: *const c_void,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_nccl_send_recv(
            comm: *mut c_void,
            send: *const c_void,
            send_bytes: u64,
            peer: u32,
            receive: *mut c_void,
            receive_bytes: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_all_gather_u32(
            comm: *mut c_void,
            send: *const u32,
            receive: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_all_reduce_max_u32(
            comm: *mut c_void,
            send: *const u32,
            receive: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_destroy(comm: *mut c_void);
        pub fn mgbfs_route_query(capacity: u32, out: *mut RouteBytes) -> i32;
        pub fn mgbfs_route_create(
            capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_route_run(
            plan: *mut c_void,
            hashes: *const c_void,
            refs: *const u64,
            sorted_hashes: *mut c_void,
            sorted_refs: *mut u64,
            output_count: *mut u32,
            count: u32,
            pre_dedup: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_route_destroy(plan: *mut c_void);
        pub fn mgbfs_generate_create_variant(
            n: u32,
            moves: u32,
            modulus: u32,
            capacity: u32,
            generators: *const u8,
            variant: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_generate_create_macro_variant(
            n: u32,
            moves: u32,
            modulus: u32,
            capacity: u32,
            generators: *const u8,
            weights: *const u32,
            variant: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_generate_profile_run(
            plan: *mut c_void,
            parents: *const u8,
            children: *mut u8,
            count: u32,
            stream: *mut c_void,
            marks: *const *mut c_void,
        ) -> i32;
        pub fn mgbfs_generate_create(
            n: u32,
            moves: u32,
            modulus: u32,
            capacity: u32,
            generators: *const u8,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_generate_run(
            plan: *mut c_void,
            parents: *const u8,
            children: *mut u8,
            count: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_generate_destroy(plan: *mut c_void);
        pub fn mgbfs_hash_create(
            bytes: u32,
            capacity: u32,
            limbs: *const u8,
            offsets: *const u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_hash_run(
            plan: *mut c_void,
            input: *const u8,
            output: *mut u32,
            count: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_hash_destroy(plan: *mut c_void);
        pub fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
        pub fn cudaMemsetAsync(
            ptr: *mut c_void,
            value: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}
