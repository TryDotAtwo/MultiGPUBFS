use mgbfs_core::{
    config::ReferenceSelection,
    matrix::MatrixGroup,
    rank_plan::{cluster_capacity_plan, CapacityMode},
    Result,
};
use mgbfs_cuda::{
    ffi::mgbfs_nccl_unique_id,
    native_owner::{cudaMemGetInfo, cudaSetDevice},
};
use mgbfs_runtime::{
    archive::{create_archive_extent, Extent, StreamExtent},
    distributed_native::{DistributedConfig, DistributedNativeBfs},
    pinned_archive::PinnedArchive,
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .map(|x| x.parse().expect(key))
        .unwrap_or(default)
}
fn required(key: &str) -> Result<u32> {
    std::env::var(key)
        .map_err(|_| format!("ENV_{key}"))?
        .parse()
        .map_err(|_| format!("ENV_{key}"))
}
fn capacity_mode() -> Result<CapacityMode> {
    match std::env::var("MGBFS_CAPACITY_MODE").as_deref() {
        Ok("equal_global") => Ok(CapacityMode::EqualGlobal),
        Ok("max_per_rank") | Err(_) => Ok(CapacityMode::MaxPerRank),
        _ => Err("ENV_MGBFS_CAPACITY_MODE".into()),
    }
}
fn bootstrap(path: &Path, rank: u32, world: u32) -> Result<[u8; 128]> {
    const MAGIC: &[u8; 8] = b"MGBNCCL1";
    if rank == 0 {
        let mut id = [0; 128];
        if unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) } != 0 {
            return Err("NCCL_ID".into());
        }
        let mut x = Vec::new();
        x.extend_from_slice(MAGIC);
        x.extend_from_slice(&world.to_le_bytes());
        x.extend_from_slice(&id);
        let tmp = path.with_extension("rank0.tmp");
        if tmp.exists() || path.exists() {
            return Err("BOOTSTRAP_EXISTS".into());
        }
        std::fs::write(&tmp, x).map_err(|e| e.to_string())?;
        std::fs::rename(tmp, path).map_err(|e| e.to_string())?;
        return Ok(id);
    }
    let start = Instant::now();
    loop {
        if let Ok(x) = std::fs::read(path) {
            if x.len() != 140
                || &x[..8] != MAGIC
                || u32::from_le_bytes(x[8..12].try_into().unwrap()) != world
            {
                return Err("BOOTSTRAP_FORMAT".into());
            }
            return Ok(x[12..].try_into().unwrap());
        }
        if start.elapsed() > Duration::from_secs(60) {
            return Err("BOOTSTRAP_TIMEOUT".into());
        }
        std::thread::sleep(Duration::from_millis(10))
    }
}
fn used() -> Result<usize> {
    let (mut free, mut total) = (0, 0);
    let x = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if x != 0 {
        return Err(format!("CUDA_MEMORY_{x}"));
    }
    Ok(total - free)
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("ARGS_group_batch_bootstrap_archive_prefix_output_dir".into());
    }
    let n: usize = args[1]
        .strip_prefix('s')
        .ok_or("GROUP")?
        .parse()
        .map_err(|_| "GROUP")?;
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let rank = required("RANK")?;
    let local = required("LOCAL_RANK")?;
    let world = required("WORLD_SIZE")?;
    if world != 2 || rank != local {
        return Err("TOPOLOGY".into());
    }
    if unsafe { cudaSetDevice(local as i32) } != 0 {
        return Err("CUDA_SET_DEVICE".into());
    }
    let graph = MatrixGroup::symmetric_permutation_matrices(n)?;
    let id = bootstrap(Path::new(&args[3]), rank, world)?;
    let declared_capacity = match std::env::var("MGBFS_BENCH_CAPACITY") {
        Ok(value) => value.parse::<u32>().map_err(|_| "CAPACITY")?,
        Err(std::env::VarError::NotPresent) => u32::try_from(graph.expected_max_unique_states)
            .map_err(|_| "CAPACITY_EXPLICIT_REQUIRED")?,
        Err(_) => return Err("CAPACITY".into()),
    };
    let declared_future = env_u32("MGBFS_FUTURE_CAPACITY", declared_capacity);
    let mode = capacity_mode()?;
    let capacity_plan = cluster_capacity_plan(mode, u64::from(declared_capacity), world)?;
    let future_plan = cluster_capacity_plan(mode, u64::from(declared_future), world)?;
    let capacity = u32::try_from(capacity_plan.rank_records(rank)?).map_err(|_| "CAPACITY")?;
    let future = u32::try_from(future_plan.rank_records(rank)?).map_err(|_| "CAPACITY")?;
    let rank_map = match std::env::var("MGBFS_RANK_MAP").as_deref() {
        Ok("1,0") => [1, 0],
        Ok("0,1") | Err(_) => [0, 1],
        _ => return Err("RANK_MAP".into()),
    };
    // Archive config identity is cluster-wide.  Rank is already carried by the
    // stream frames; including it here prevents otherwise compatible rank
    // archives from being atomically combined.
    let compact_states = match std::env::var("MGBFS_STATE_CODEC").as_deref() {
        Ok("permutation_u8") => true,
        Ok("matrix_u8") | Err(_) => false,
        _ => return Err("STATE_CODEC".into()),
    };
    let archive_width = match std::env::var("MGBFS_ARCHIVE_CODEC").as_deref() {
        Ok("permutation_u8") => n,
        Err(_) if compact_states => n,
        Ok("matrix_u8") | Err(_) => graph.start.len(),
        _ => return Err("ARCHIVE_CODEC".into()),
    };
    if compact_states && archive_width != n {
        return Err("COMPACT_STATE_REQUIRES_COMPACT_ARCHIVE".into());
    }
    let profile = std::env::var("MGBFS_PROFILE").unwrap_or_else(|_| "DENSE".into());
    let owner = std::env::var("MGBFS_OWNER_BACKEND").unwrap_or_else(|_| "CUB_SORT_MERGE".into());
    let pre = std::env::var("MGBFS_PRE_DEDUP").unwrap_or_else(|_| "ON".into());
    let hash_first_generation =
        std::env::var("MGBFS_HASH_FIRST_GENERATION").unwrap_or_else(|_| "SCALAR".into());
    let selection = ReferenceSelection::parse(
        &profile,
        &owner,
        &pre,
        compact_states,
        env_u32(
            "MGBFS_MATERIALIZATION_CAPACITY",
            batch
                .checked_mul(graph.generators.len() as u32)
                .ok_or("CANDIDATE_OVERFLOW")?,
        ),
        env_u32("MGBFS_BMMA_TILE_LIMIT", 256),
    )?
    .with_hash_first_generation(&hash_first_generation)?;
    let description=format!("distributed-native-ring-v2;s{n};batch={batch};capacity_mode={mode:?};declared_capacity={declared_capacity};declared_ring={declared_future};global_capacity={};global_ring={};map={rank_map:?};seed=20260828;archive_width={archive_width}", capacity_plan.global_records, future_plan.global_records);
    let description = format!("{description};compact_states={compact_states}");
    let description = format!("{description};reference_selection={selection:?}");
    let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
    let archive_path = format!("{}-rank-{rank}.mgbfsar1", args[4]);
    let disk_bytes = graph
        .expected_max_unique_states
        .checked_mul((archive_width + 16) as u64)
        .and_then(|x| x.checked_add(64 << 20))
        .ok_or("DISK")?;
    let archive_rows = env_u32("MGBFS_ARCHIVE_ROWS", batch);
    let stream_archive = std::env::var("MGBFS_ARCHIVE_STREAM").as_deref() == Ok("1");
    // Test-only A/B switch. Production runs retain the mandatory archive.
    let archive_enabled = std::env::var("MGBFS_BENCH_SKIP_ARCHIVE").as_deref() != Ok("1");
    let extent: Box<dyn Extent + Send> = if archive_enabled {
        create_archive_extent(Path::new(&archive_path), stream_archive)
            .map_err(|e| format!("ARCHIVE_EXTENT: {e}"))?
    } else {
        Box::new(StreamExtent::new(std::io::sink()))
    };
    let mut archive = PinnedArchive::new(
        extent,
        disk_bytes,
        archive_width,
        digest,
        archive_rows,
        env_u32("MGBFS_ARCHIVE_SLOTS", 64) as usize,
    )?;
    let pinned = archive.pinned_bytes();
    let setup = Instant::now();
    let cfg = DistributedConfig {
        rank,
        world,
        logical_owner_to_rank: rank_map,
        batch,
        layer_capacity: capacity,
        state_ring_capacity: future,
        buckets: env_u32("MGBFS_BUCKETS", 256),
        shards: env_u32("MGBFS_SHARDS", 64),
        job_buckets: env_u32("MGBFS_JOB_BUCKETS", 4),
        bucket_capacity: env_u32(
            "MGBFS_BUCKET_CAPACITY",
            capacity.div_ceil(128).saturating_add(4096),
        ),
        prededup: selection.prededup,
        generation_variant: if compact_states { 5 } else { 1 },
    };
    let mut bfs = if selection.tensor_generation {
        DistributedNativeBfs::new_hash_first_tc_with_owner(
            &graph,
            20260828u128.to_le_bytes(),
            id,
            cfg,
            selection
                .materialization_capacity
                .ok_or("REFERENCE_HASH_FIRST_CAPACITY")?,
            selection.owner,
            selection.tile_limit,
        )?
    } else {
        DistributedNativeBfs::new_reference_with_owner(
            &graph,
            20260828u128.to_le_bytes(),
            id,
            cfg,
            selection.materialization_capacity,
            selection.owner,
            selection.tile_limit,
        )?
    };
    let allocated = used()?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let trace = std::env::var_os("MGBFS_TRACE_DEPTHS").is_some();
    let start = Instant::now();
    let mut layers = Vec::new();
    let mut times = Vec::new();
    loop {
        let tick = Instant::now();
        let depth = bfs.depth();
        let count = bfs.frontier_len();
        layers.push(count);
        if trace {
            eprintln!("MGBFS_DEPTH_BEGIN rank={rank} depth={depth} count={count}");
        }
        let alive = if archive_enabled {
            bfs.advance_archived(&mut archive)?
        } else {
            bfs.advance()?
        };
        if trace {
            eprintln!("MGBFS_ARCHIVE_SUBMITTED rank={rank} depth={depth} count={count}");
        }
        let elapsed = tick.elapsed().as_secs_f64();
        times.push(elapsed);
        if trace {
            eprintln!("MGBFS_DEPTH_END rank={rank} depth={depth} seconds={elapsed:.6} next={} alive={alive}",bfs.frontier_len());
        }
        if !alive {
            break;
        }
    }
    let search = start.elapsed().as_secs_f64();
    if archive_enabled {
        archive.finish()?;
    }
    let durable = start.elapsed().as_secs_f64();
    std::fs::create_dir_all(&args[5]).map_err(|e| e.to_string())?;
    let record=format!("{{\"status\":\"COMPLETE\",\"backend\":\"native_nccl_dense_ring_v2\",\"rank\":{rank},\"group\":\"s{n}\",\"batch\":{batch},\"capacity_mode\":\"{mode:?}\",\"archive_enabled\":{archive_enabled},\"archive_state_bytes\":{archive_width},\"declared_capacity_records\":{declared_capacity},\"global_capacity_records\":{},\"rank_capacity_records\":{capacity},\"declared_state_ring_records\":{declared_future},\"global_state_ring_records\":{},\"rank_state_ring_records\":{future},\"search_complete_seconds\":{search},\"durable_run_commit_seconds\":{durable},\"setup_seconds\":{setup_seconds},\"local_layer_sizes\":{layers:?},\"per_depth_seconds\":{times:?},\"cuda_allocated_used_bytes\":{allocated},\"cuda_peak_observed_bytes\":{},\"pinned_bytes\":{pinned},\"disk_reserved_bytes\":{disk_bytes}}}",capacity_plan.global_records,future_plan.global_records,used()?.max(allocated));
    // Keep the existing timing schema, but never label HASH_FIRST as DENSE.
    let record = if selection.materialization_capacity.is_some() {
        record.replace(
            "native_nccl_dense_ring_v2",
            "native_nccl_hash_first_reference_v1",
        )
    } else {
        record
    };
    let record = format!("{},\"frontier_profile\":\"{profile}\",\"owner_backend\":\"{owner}\",\"pre_dedup\":\"{pre}\",\"generation_variant\":{},\"materialization_capacity\":{},\"bmma_tile_limit\":{}}}",
        record.strip_suffix('}').ok_or("RECORD_FORMAT")?,
        if compact_states { 5 } else { 1 },
        selection.materialization_capacity.unwrap_or(0), selection.tile_limit);
    let record = format!(
        "{},\"hash_first_generation\":\"{hash_first_generation}\"}}",
        record.strip_suffix('}').ok_or("RECORD_FORMAT")?
    );
    std::fs::write(
        Path::new(&args[5]).join(format!("rank-{rank}.json")),
        record,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: {e}");
        std::process::exit(1)
    }
}
