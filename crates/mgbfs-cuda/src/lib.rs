//! Raw C ABI. CUDA is opt-in; there is no CPU implementation of these calls.
pub mod allocation;
pub mod native_owner;
#[cfg(feature = "cuda")]
pub mod ffi {
    pub use crate::allocation::{GenerateBytes, HashBytes, RouteBytes};
    use std::ffi::{c_char, c_void};
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
        pub fn mgbfs_generate_query(
            n: u32,
            moves: u32,
            modulus: u32,
            capacity: u32,
            variant: u32,
            out: *mut GenerateBytes,
        ) -> i32;
        pub fn mgbfs_hash_query(bytes: u32, capacity: u32, out: *mut HashBytes) -> i32;
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
