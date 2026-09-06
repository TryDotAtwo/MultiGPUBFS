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
fn bootstrap(
    path: &Path,
    rank: u32,
    world: u32,
    digest: [u8; 32],
) -> Result<mgbfs_runtime::bootstrap::BootstrapGroup> {
    let launch =
        std::env::var("TORCHELASTIC_RUN_ID").map_err(|_| "BOOTSTRAP_LAUNCH_ID_REQUIRED")?;
    if launch.is_empty() || launch == "none" {
        return Err("BOOTSTRAP_LAUNCH_ID_REQUIRED".into());
    }
    let run_digest =
        Sha256::digest(serde_json::to_vec(&(launch, path)).map_err(|e| e.to_string())?);
    let identity = mgbfs_runtime::control_handshake::RunIdentity {
        config_digest: digest,
        run_id: run_digest[..16].try_into().unwrap(),
    };
    mgbfs_runtime::bootstrap::rendezvous(
        path,
        rank,
        world,
        identity,
        Duration::from_secs(60),
        || {
            let mut id = [0; 128];
            if unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) } != 0 {
                return Err("NCCL_ID".into());
            }
            Ok(id)
        },
    )
}
fn used() -> Result<usize> {
    let (mut free, mut total) = (0, 0);
    let x = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if x != 0 {
        return Err(format!("CUDA_MEMORY_{x}"));
    }
    Ok(total - free)
}
fn run_pass(args: &[String], warmup_completed: bool) -> Result<()> {
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
    if !(1..=2).contains(&world) || rank != local {
        return Err("TOPOLOGY".into());
    }
    if unsafe { cudaSetDevice(local as i32) } != 0 {
        return Err("CUDA_SET_DEVICE".into());
    }
    let graph = MatrixGroup::symmetric_permutation_matrices(n)?;
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
        Ok("0") if world == 1 => [0, 0],
        Err(_) if world == 1 => [0, 0],
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
        untouched_vram_reserve: 1 << 30,
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
    // Reference launch agreement includes geometry and archive settings omitted
    // by the older archive digest. Rank-local capacities are derived from the
    // shared declared capacity and rank map, not compared as equal across ranks.
    let bootstrap_description = serde_json::json!({
        "schema": "reference-bootstrap-v1", "archive_digest": digest,
        "world": world, "buckets": cfg.buckets, "shards": cfg.shards,
        "job_buckets": cfg.job_buckets,
        "bucket_capacity_override": std::env::var("MGBFS_BUCKET_CAPACITY").ok(),
        "reserve": cfg.untouched_vram_reserve, "archive_rows": archive_rows,
        "archive_slots": std::env::var("MGBFS_ARCHIVE_SLOTS").ok(),
        "stream_archive": stream_archive, "archive_enabled": archive_enabled,
    });
    let bootstrap_digest: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&bootstrap_description).map_err(|e| e.to_string())?)
            .into();
    // Keep control sockets alive throughout this reference run. Dispatching GPU
    // epochs on them is a separate integration step, not claimed here.
    let _control_group = bootstrap(Path::new(&args[3]), rank, world, bootstrap_digest)?;
    let id = _control_group.nccl_id;
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
        "{},\"world_size\":{world},\"hash_first_generation\":\"{hash_first_generation}\",\"warmup_completed\":{warmup_completed}}}",
        record.strip_suffix('}').ok_or("RECORD_FORMAT")?
    );
    let owned_payload: u64 = bfs
        .owned_memory()
        .allocations
        .iter()
        .map(|a| a.payload_bytes)
        .sum();
    let record=format!("{},\"explicit_device_payload_bytes\":{owned_payload},\"explicit_device_aligned_bytes\":{},\"untouched_vram_reserve_bytes\":{},\"allocation_scope\":\"explicit_runtime_and_library_device_buffers_excludes_nccl_driver_and_pinned_archive\"}}",
        record.strip_suffix('}').ok_or("RECORD_FORMAT")?,bfs.owned_memory().total(),cfg.untouched_vram_reserve);
    std::fs::write(Path::new(&args[5]).join(format!("rank-{rank}.json")), {
        let mut value: serde_json::Value =
            serde_json::from_str(&record).map_err(|e| format!("RECORD_JSON: {e}"))?;
        value["device_allocation_plan"] =
            mgbfs_runtime::distributed_memory::allocation_report(bfs.owned_memory());
        serde_json::to_vec(&value).map_err(|e| format!("RECORD_JSON: {e}"))?
    })
    .map_err(|e| e.to_string())?;
    Ok(())
}
fn run() -> Result<()> {
    use mgbfs_runtime::benchmark::{run_phases, Phase};
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("ARGS_group_batch_bootstrap_archive_prefix_output_dir".into());
    }
    let warmup = match std::env::var("MGBFS_BENCH_WARMUP").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        _ => return Err("BENCH_WARMUP_CONFIG".into()),
    };
    if warmup && std::env::var("MGBFS_ARCHIVE_STREAM").as_deref() == Ok("1") {
        return Err("BENCH_WARMUP_REQUIRES_FILE_ARCHIVE".into());
    }
    run_phases(warmup, |phase| {
        if phase == Phase::Measure {
            return run_pass(&args, warmup);
        }
        let mut warm_args = args.clone();
        for index in [3, 4, 5] {
            warm_args[index].push_str(".warmup");
        }
        run_pass(&warm_args, false)?;
        // FileExtent uses create_new: this exact rank-local warmup archive
        // belongs to this completed pass. Keep its small timing JSON/logs.
        if std::env::var("MGBFS_BENCH_SKIP_ARCHIVE").as_deref() != Ok("1") {
            let rank = required("RANK")?;
            std::fs::remove_file(format!("{}-rank-{rank}.mgbfsar1", warm_args[4]))
                .map_err(|e| format!("WARMUP_ARCHIVE_RELEASE: {e}"))?;
        }
        Ok(())
    })
}
fn main() {
    if let Err(e) = run() {
        eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: {e}");
        std::process::exit(1)
    }
}
