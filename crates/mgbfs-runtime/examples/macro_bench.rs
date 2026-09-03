//! Archived single-rank weighted macro BFS benchmark for matrix Cayley graphs.
use mgbfs_core::{matrix::MatrixGroup, Result};
use mgbfs_cuda::native_owner::cudaMemGetInfo;
use mgbfs_runtime::{
    archive::FileExtent,
    macro_native::{MacroNativeBfs, MacroNativeConfig},
    pinned_archive::PinnedArchive,
};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).map(|v| v.parse().expect(key)).unwrap_or(default)
}
fn used() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0, 0);
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if status != 0 { return Err(format!("CUDA_MEMORY_{status}")); }
    Ok((total - free, free))
}
fn graph(spec: &str) -> Result<MatrixGroup> {
    if let Some(n) = spec.strip_prefix('s') {
        MatrixGroup::symmetric_permutation_matrices(n.parse().map_err(|_| "GROUP")?)
    } else if let Some(m) = spec.strip_prefix("u4-") {
        MatrixGroup::unitriangular(4, m.parse().map_err(|_| "GROUP")?)
    } else {
        Err("GROUP_EXPECTED_sN_OR_u4-M".into())
    }
}
fn execute() -> Result<()> {
    if env_u32("WORLD_SIZE", 1) != 1 || env_u32("LOCAL_WORLD_SIZE", 1) != 1 {
        return Err("MACRO_BENCH_SINGLE_RANK_ONLY".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 { return Err("ARGS_group_batch_macro_depth_prededup_verify_archive".into()); }
    let group = &args[1];
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let macro_depth: u32 = args[3].parse().map_err(|_| "MACRO_DEPTH")?;
    let prededup = args[4] == "1";
    let verify = args[5] == "verify";
    let g = graph(group)?;
    let layer_capacity = env_u32(
        "MGBFS_BENCH_CAPACITY",
        u32::try_from(g.expected_max_unique_states).map_err(|_| "CAPACITY")?,
    );
    let future_capacity_per_depth = env_u32("MGBFS_FUTURE_CAPACITY", layer_capacity);
    let config = MacroNativeConfig {
        macro_depth,
        batch,
        layer_capacity,
        future_capacity_per_depth,
        prededup,
        generation_variant: env_u32("MGBFS_BENCH_GENERATION", 1),
    };
    let seed = 20260828u128.to_le_bytes();
    // Initialize context and every native primitive without repeating the measured graph.
    let warm_graph = MatrixGroup::unitriangular(3, 2)?;
    let mut warm = MacroNativeBfs::new(&warm_graph, seed, MacroNativeConfig {
        macro_depth: macro_depth.min(3), batch: 8, layer_capacity: 8,
        future_capacity_per_depth: 64, prededup, generation_variant: config.generation_variant,
    })?;
    while warm.advance()? {}
    drop(warm);
    let context = used()?.0;
    let description = format!("macro-native-v1;group={group};batch={batch};k={macro_depth};layer={layer_capacity};future={future_capacity_per_depth};pre={prededup};seed=20260828");
    let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
    let disk_bytes = g.expected_max_unique_states
        .checked_mul((g.start.len() + 16) as u64)
        .and_then(|v| v.checked_add(64 << 20)).ok_or("DISK_OVERFLOW")?;
    let extent = FileExtent::create_new(Path::new(&args[6])).map_err(|e| e.to_string())?;
    let archive_rows = env_u32("MGBFS_ARCHIVE_ROWS", batch);
    let mut archive = PinnedArchive::new(
        extent, disk_bytes, g.start.len(), digest, archive_rows,
        env_u32("MGBFS_ARCHIVE_SLOTS", 64) as usize,
    )?;
    let pinned_bytes = archive.pinned_bytes();
    let setup = Instant::now();
    let mut bfs = MacroNativeBfs::new(&g, seed, config)?;
    let (allocated, free) = used()?;
    if free < (1usize << 30) { return Err("UNTOUCHED_VRAM_RESERVE".into()); }
    let setup_seconds = setup.elapsed().as_secs_f64();
    let mut layers = Vec::new();
    let mut times = Vec::new();
    let mut layer_sha256 = Vec::new();
    let start = Instant::now();
    loop {
        let layer_start = Instant::now();
        layers.push(bfs.frontier_len());
        if verify {
            let mut states = bfs.snapshot()?; states.sort();
            let mut hash = Sha256::new(); for state in states { hash.update(state); }
            layer_sha256.push(format!("{:x}", hash.finalize()));
        }
        bfs.archive_current(&mut archive)?;
        let alive = bfs.advance()?;
        times.push(layer_start.elapsed().as_secs_f64());
        if !alive { break; }
    }
    let search = start.elapsed().as_secs_f64();
    let total: u64 = layers.iter().map(|&v| u64::from(v)).sum();
    if total != g.expected_max_unique_states { return Err(format!("CARDINALITY_{total}")); }
    archive.finish()?;
    let durable = start.elapsed().as_secs_f64();
    println!("{{\"status\":\"COMPLETE\",\"backend\":\"native_macro_archived_reference\",\"group\":\"{group}\",\"macro_depth\":{macro_depth},\"batch\":{batch},\"prededup\":{prededup},\"verification_only\":{verify},\"search_complete_seconds\":{search},\"durable_run_commit_seconds\":{durable},\"setup_seconds\":{setup_seconds},\"unique_states\":{total},\"layer_sizes\":{layers:?},\"layer_sha256\":{layer_sha256:?},\"per_depth_seconds\":{times:?},\"cuda_context_used_bytes\":{context},\"cuda_allocated_used_bytes\":{allocated},\"cuda_peak_observed_bytes\":{},\"pinned_bytes\":{pinned_bytes},\"archive_rows\":{archive_rows},\"disk_reserved_bytes\":{disk_bytes}}}", used()?.0.max(allocated));
    Ok(())
}
fn main() {
    if let Err(error) = execute() {
        eprintln!("MACRO_NATIVE_INCOMPLETE: {error}");
        std::process::exit(1);
    }
}
