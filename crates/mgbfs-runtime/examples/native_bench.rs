//! Assembled single-rank DENSE reference with mandatory asynchronous V1 archive.
//! Not yet the overlapped multi-rank production runtime.
use mgbfs_core::{matrix::MatrixGroup, Result};
use mgbfs_cuda::native_owner::cudaMemGetInfo;
use mgbfs_runtime::{
    archive::Extent,
    native::{NativeBfs, NativeConfig},
    pinned_archive::PinnedArchive,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn used() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0, 0);
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if status != 0 {
        return Err(format!("CUDA_MEMORY_{status}"));
    }
    Ok((total - free, free))
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .map(|v| v.parse().expect(key))
        .unwrap_or(default)
}
fn execute<E: Extent + Send + 'static>(disk: E) -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let m: u16 = args[1].parse().map_err(|_| "MODULUS")?;
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let prededup = args[3] == "1";
    let verify = args[4] == "verify";
    let g = MatrixGroup::unitriangular(4, m)?;
    let f = env_u32(
        "MGBFS_BENCH_CAPACITY",
        u32::try_from(g.expected_max_unique_states).map_err(|_| "CAPACITY")?,
    );
    let buckets = env_u32("MGBFS_BUCKETS", 256);
    let cfg = NativeConfig {
        batch,
        layer_capacity: f,
        buckets,
        shards: env_u32("MGBFS_SHARDS", 16),
        job_buckets: env_u32("MGBFS_JOB_BUCKETS", 16),
        bucket_capacity: env_u32(
            "MGBFS_BUCKET_CAPACITY",
            ((u64::from(f) + u64::from(buckets) - 1) / u64::from(buckets) * 2 + 256) as u32,
        ),
        prededup,
    };
    let seed = 20260828u128.to_le_bytes();
    // Full same-workload CUDA warmup is excluded from both setup and BFS time.
    let mut warm = NativeBfs::new(&g, seed, cfg)?;
    while warm.advance()? {}
    drop(warm);
    let context = used()?.0;
    let setup = Instant::now();
    let description = format!("native-dense-v1;m={m};batch={batch};f={f};b={buckets};k={};j={};h={};pre={prededup};seed=20260828", cfg.bucket_capacity, cfg.job_buckets, cfg.shards);
    let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
    // Fixed extent: all state/hash payloads plus 64 MiB of frame/commit metadata.
    // Exceeding this explicit metadata allowance is fatal, never a resize.
    let disk_bytes = g
        .expected_max_unique_states
        .checked_mul(32)
        .and_then(|v| v.checked_add(64 << 20))
        .ok_or("DISK_OVERFLOW")?;
    let mut archive = PinnedArchive::new(
        disk,
        disk_bytes,
        16,
        digest,
        batch,
        env_u32("MGBFS_ARCHIVE_SLOTS", 64) as usize,
    )?;
    let pinned_bytes = archive.pinned_bytes();
    let mut bfs = NativeBfs::new(&g, seed, cfg)?;
    let device_requested = bfs.requested_device_bytes();
    let (allocated, free) = used()?;
    if free < (1usize << 30) {
        return Err("UNTOUCHED_VRAM_RESERVE".into());
    }
    let setup_seconds = setup.elapsed().as_secs_f64();
    let mut layers = Vec::new();
    let mut times = Vec::new();
    let mut digests = Vec::new();
    let start = Instant::now();
    loop {
        let depth = Instant::now();
        layers.push(bfs.frontier_len());
        if verify {
            let mut states = bfs.snapshot()?;
            states.sort();
            let mut sha = Sha256::new();
            for state in states {
                sha.update(state);
            }
            digests.push(format!("{:x}", sha.finalize()));
        }
        bfs.archive_current(&mut archive)?;
        let alive = bfs.advance()?;
        times.push(depth.elapsed().as_secs_f64());
        if !alive {
            break;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let total: u64 = layers.iter().map(|&n| u64::from(n)).sum();
    if total != g.expected_max_unique_states {
        return Err(format!("CARDINALITY_{total}"));
    }
    archive.finish()?;
    let durable = start.elapsed().as_secs_f64();
    let peak = allocated.max(used()?.0);
    println!("{{\"status\":\"COMPLETE\",\"backend\":\"native_dense_archived_reference\",\"archive_format\":\"MGBFSAR1\",\"modulus\":{m},\"batch\":{batch},\"verification_only\":{verify},\"search_seconds\":{seconds},\"search_complete_seconds\":{seconds},\"durable_run_commit_seconds\":{durable},\"setup_seconds\":{setup_seconds},\"layer_sizes\":{layers:?},\"layer_sha256\":{digests:?},\"per_depth_seconds\":{times:?},\"requested_device_bytes\":{device_requested},\"cuda_context_used_bytes\":{context},\"cuda_peak_used_bytes\":{peak},\"pinned_bytes\":{pinned_bytes},\"disk_reserved_bytes\":{disk_bytes},\"bucket_capacity\":{},\"buckets\":{buckets},\"prededup\":{prededup}}}", cfg.bucket_capacity);
    Ok(())
}
#[cfg(target_os = "linux")]
fn main() {
    let path = std::env::args().nth(5).expect("archive path required");
    let result = mgbfs_runtime::archive::FileExtent::create_new(std::path::Path::new(&path))
        .map_err(|e| e.to_string())
        .and_then(execute);
    if let Err(e) = result {
        eprintln!("NATIVE_INCOMPLETE: {e}");
        std::process::exit(1);
    }
}
#[cfg(not(target_os = "linux"))]
fn main() {
    let _ = used;
    eprintln!("native_bench requires Linux physical file extents");
    std::process::exit(1);
}
