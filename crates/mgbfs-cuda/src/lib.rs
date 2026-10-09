//! Raw C ABI. CUDA is opt-in; there is no CPU implementation of these calls.
pub mod allocation;
pub mod shard_ab;
pub mod library_owner;
pub mod native_owner;
#[cfg(any(feature = "cuda", feature = "library-owner"))]
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
        pub fn mgbfs_trace_ranges_available() -> i32;
        pub fn mgbfs_trace_range_push(label: *const c_char);
        pub fn mgbfs_trace_range_pop();
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
        /// Experimental SM75 integer-MMA generation, register-only hash reduction.
        pub fn mgbfs_hash_first_tc_validate_device() -> i32;
        /// Requires successful SM75 admission on the same current device.
        /// Same shapes/lifetimes as legacy tc; no per-batch hardware query.
        pub fn mgbfs_generate_hash_only_tc_admitted(
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
        /// Same pointer/lifetime contract as mgbfs_generate_hash_only.
        pub fn mgbfs_generate_hash_only_tc(
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
        pub fn mgbfs_macro_settle_run_frontier(
            plan: *mut c_void,
            future: *const c_void,
            refs: *const u64,
            future_state: *const FrontierState,
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
        pub fn mgbfs_future_merge_run_bounded_checked(
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
            input_fatal: *const u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_future_merge_destroy(plan: *mut c_void);
        pub fn mgbfs_exchange_pack_frame(
            stride: u32,
            source_states: *const u8,
            source_count: u32,
            sorted_hashes: *const c_void,
            sorted_refs: *const u64,
            sorted_count: u32,
            begin: u32,
            count: u32,
            output: *mut u8,
            output_capacity: u64,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_macro_exchange_pack_frame(
            stride: u32,
            source_depth: u32,
            weight: u32,
            source_states: *const u8,
            source_count: u32,
            sorted_hashes: *const c_void,
            sorted_refs: *const u64,
            sorted_count: u32,
            begin: u32,
            count: u32,
            output: *mut u8,
            output_capacity: u64,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_macro_validate_refs(
            refs: *const c_void,
            count: u32,
            source_depth: u32,
            target_depth: u32,
            max_weight: u32,
            max_state_ref: u64,
            fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_frame_write_header(
            host_header: *const u8,
            device_prefix: *mut u8,
            stream: *mut c_void,
        ) -> i32;
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
        pub fn mgbfs_exchange_pack_n(
            world: u32,
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
        pub fn mgbfs_exchange_pack_device_n(
            world: u32,
            stride: u32,
            capacity: u32,
            source_states: *const u8,
            source_count: u32,
            sorted_hashes: *const c_void,
            sorted_refs: *const u64,
            count: *const u32,
            packed_states: *mut u8,
            owner_counts: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_device_store_u32(
            destination: *mut u32,
            value: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_window_from_counts(
            world: u32,
            packed_capacity: u32,
            logical_owner: u32,
            owner_counts: *const u32,
            routed_count: *const u32,
            begin: *mut u32,
            rows: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_import_transport_fatal(
            transport_fatal: *const u32,
            ring: *mut crate::native_owner::Ring,
            owner: *mut crate::native_owner::Control,
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
        pub fn mgbfs_nccl_bind_cancel(
            comm: *mut c_void,
            probe: Option<extern "C" fn(*mut c_void) -> i32>,
            context: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_session_park(comm: *mut c_void) -> i32;
        pub fn mgbfs_nccl_create_with_cancel(
            rank: u32,
            world: u32,
            device: u32,
            id128: *const c_void,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
            probe: Option<extern "C" fn(*mut c_void) -> i32>,
            context: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_bind_retirement(
            comm: *mut c_void,
            probe: Option<extern "C" fn(*mut c_void, i32) -> i32>,
            context: *mut c_void,
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
        pub fn mgbfs_nccl_send_recv_pair(
            comm: *mut c_void,
            hashes: *const c_void,
            hash_bytes: u64,
            states: *const c_void,
            state_bytes: u64,
            peer: u32,
            receive_hashes: *mut c_void,
            receive_hash_bytes: u64,
            receive_states: *mut c_void,
            receive_state_bytes: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_prepare(
            comm: *mut c_void,
            candidate_capacity: u32,
            state_stride: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_activate(
            comm: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_exchange(
            comm: *mut c_void,
            sorted_hashes: *const c_void,
            packed_states: *const c_void,
            owner_counts: *const u32,
            group_fatal: *const u32,
            logical_owner: u32,
            peer: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_exchange_rows(
            comm: *mut c_void,
            sorted_hashes: *const c_void,
            packed_rows: *const c_void,
            owner_counts: *const u32,
            group_fatal: *const u32,
            logical_owner: u32,
            peer: u32,
            row_stride: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_view(
            comm: *mut c_void,
            count: *mut *const u32,
            fatal: *mut *const u32,
            hashes: *mut *const c_void,
            states: *mut *const c_void,
        ) -> i32;
        pub fn mgbfs_nccl_cancel_words(
            comm: *mut c_void,
            host: *mut *mut u32,
            device: *mut *mut u32,
        ) -> i32;
        pub fn mgbfs_owner_local_fatal_gate(
            comm: *mut c_void,
            ring: *mut crate::native_owner::Ring,
            owner: *mut crate::native_owner::Control,
            local_fatal: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_owner_global_fatal_gate(
            comm: *mut c_void,
            ring: *mut crate::native_owner::Ring,
            owner: *mut crate::native_owner::Control,
            send: *mut u32,
            receive: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_nccl_lsa_cancel_word(comm: *mut c_void, word: *mut *mut u32) -> i32;
        pub fn mgbfs_owner_lsa_fatal_gate(
            comm: *mut c_void,
            ring: *mut crate::native_owner::Ring,
            owner: *mut crate::native_owner::Control,
            send: *mut u32,
            receive: *mut u32,
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
        /// Terminal abort; serialize with communicator calls, then destroy wrapper.
        pub fn mgbfs_nccl_abort(comm: *mut c_void) -> i32;
        /// Nonblocking health query, not transfer completion; zero means healthy.
        pub fn mgbfs_nccl_poll(comm: *mut c_void) -> i32;
        /// Dense rank-ordered single-source scatter; local source range stays a view.
        /// Caller agrees counts and retains all device buffers until completion.
        pub fn mgbfs_nccl_scatter(
            comm: *mut c_void,
            source: u32,
            send: *const c_void,
            send_capacity: u64,
            sizes: *const u64,
            recv: *mut c_void,
            recv_bytes: u64,
            recv_capacity: u64,
            stream: *mut c_void,
        ) -> i32;
        pub fn mgbfs_route_query(capacity: u32, out: *mut RouteBytes) -> i32;
        pub fn mgbfs_route_create(
            capacity: u32,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        pub fn mgbfs_nccl_all_reduce_sum_u64(comm:*mut c_void,send:*const u64,recv:*mut u64,stream:*mut c_void)->i32;
        pub fn mgbfs_nccl_send_recv_triplet(comm:*mut c_void,count:*const c_void,count_bytes:u64,metadata:*const c_void,metadata_bytes:u64,states:*const c_void,state_bytes:u64,peer:u32,recv_count:*mut c_void,recv_metadata:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
        pub fn mgbfs_nccl_all_reduce_sum_u32(comm: *mut c_void, send: *const u32, recv: *mut u32, stream: *mut c_void) -> i32;
        pub fn mgbfs_route_run_sharded(plan: *mut c_void, hashes: *const c_void, refs: *const u64, output: *mut c_void, outrefs: *mut u64, output_count: *mut u32, count: u32, partitions: u32, stream: *mut c_void) -> i32;
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
        pub fn mgbfs_compact_map_query(n:u32,moves:u32,capacity:u32,bytes:*mut u64)->i32;
        pub fn mgbfs_compact_map_create(n:u32,moves:u32,capacity:u32,permutation:*const u8,out:*mut *mut c_void,error:*mut c_char,error_capacity:usize)->i32;
        pub fn mgbfs_compact_hash_query(n:u32,moves:u32,capacity:u32,bytes:*mut u64)->i32;
        pub fn mgbfs_compact_hash_create(n:u32,moves:u32,capacity:u32,permutation:*const u8,limbs:*const u8,offsets:*const u32,move_major:u32,out:*mut *mut c_void,error:*mut c_char,error_capacity:usize)->i32;
        pub fn mgbfs_compact_hash_run(plan:*mut c_void,parents:*const u8,output:*mut u32,count:u32,stream:*mut c_void)->i32;
        pub fn mgbfs_compact_hash_destroy(plan:*mut c_void);
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
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaProfilerStart() -> i32;
        pub fn cudaProfilerStop() -> i32;
        pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaStreamQuery(stream: *mut c_void) -> i32;
        pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaEventQuery(event: *mut c_void) -> i32;
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

pub mod generic_graph;
