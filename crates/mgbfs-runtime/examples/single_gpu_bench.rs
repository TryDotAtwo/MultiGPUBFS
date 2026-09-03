//! Development executor benchmark. No archive and no production RunCommit.
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::dense_device::DenseDeviceStepper;
use sha2::{Digest, Sha256};
use std::time::Instant;

extern "C" {
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}
fn used_bytes() -> usize {
    let (mut free, mut total) = (0, 0);
    unsafe {
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(cudaMemGetInfo(&mut free, &mut total), 0);
    }
    total - free
}
fn make(g: &MatrixGroup, batch: u32, pre: bool) -> DenseDeviceStepper {
    let capacity = std::env::var("MGBFS_BENCH_CAPACITY")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(u32::try_from(g.expected_max_unique_states).unwrap());
    let generation_variant = std::env::var("MGBFS_BENCH_GENERATION")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(0);
    let result = DenseDeviceStepper::new_pipelined_with_generation(
        g,
        20260828u128.to_le_bytes(),
        batch,
        capacity,
        pre,
        generation_variant,
    )
    .unwrap();
    if std::env::var_os("MGBFS_BENCH_RESERVE_GIB").is_some() {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe {
            assert_eq!(cudaMemGetInfo(&mut free, &mut total), 0);
        }
        assert!(free >= 1usize << 30, "BENCH_UNTOUCHED_RESERVE");
    }
    result
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: u16 = args[1].parse().unwrap();
    let batch: u32 = args[2].parse().unwrap();
    let pre = args[3] == "1";
    let validate = args[4] == "verify";
    let g = MatrixGroup::unitriangular(4, m).unwrap();
    // Same-workload warmup, including all depth-dependent library paths.
    let mut warm = make(&g, batch, pre);
    while warm.advance().unwrap() {}
    drop(warm);
    let context_used = used_bytes();
    let setup_start = Instant::now();
    let mut bfs = make(&g, batch, pre);
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let allocated_used = used_bytes();
    let mut peak_used = allocated_used;
    let mut counts = vec![1u32];
    let mut digests = Vec::new();
    let mut depth_seconds = Vec::new();
    if validate {
        digest(&bfs, &mut digests);
    }
    let start = Instant::now();
    loop {
        let depth_start = Instant::now();
        let alive = bfs.advance().unwrap();
        depth_seconds.push(depth_start.elapsed().as_secs_f64());
        if !alive {
            break;
        }
        counts.push(bfs.frontier_len());
        if validate {
            digest(&bfs, &mut digests);
        }
    }
    let search_seconds = start.elapsed().as_secs_f64();
    peak_used = peak_used.max(used_bytes());
    assert_eq!(
        counts.iter().map(|&v| u64::from(v)).sum::<u64>(),
        g.expected_max_unique_states
    );
    println!("{{\"status\":\"COMPLETE\",\"backend\":\"native_ping_pong\",\"modulus\":{m},\"batch\":{batch},\"prededup\":{pre},\"verification_only\":{validate},\"setup_seconds\":{setup_seconds},\"search_seconds\":{search_seconds},\"layer_sizes\":{counts:?},\"layer_sha256\":{digests:?},\"depth_seconds\":{depth_seconds:?},\"cuda_context_used_bytes\":{context_used},\"cuda_fixed_allocation_delta_bytes\":{},\"cuda_observed_used_bytes\":{peak_used}}}", allocated_used.saturating_sub(context_used));
}
fn digest(bfs: &DenseDeviceStepper, out: &mut Vec<String>) {
    let mut states = bfs.snapshot().unwrap();
    states.sort_unstable();
    let mut h = Sha256::new();
    for state in states {
        h.update(state);
    }
    out.push(format!("{:x}", h.finalize()));
}
