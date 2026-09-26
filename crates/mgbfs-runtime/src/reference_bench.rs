use crate::{
    archive::{create_archive_extent, ArchiveRingPlan},
    distributed_native::{DistributedConfig, DistributedNativeBfs},
    macro_native::{MacroNativeBfs, MacroNativeConfig},
    pinned_archive::PinnedArchive,
};
use mgbfs_core::{
    config::{ReferenceOwner, ReferenceSelection},
    macro_memory::MacroStateLayout,
    matrix::MatrixGroup,
    rank_plan::{cluster_capacity_plan, CapacityMode},
    Result,
};
use mgbfs_cuda::{
    ffi::{cudaProfilerStart, cudaProfilerStop, mgbfs_nccl_unique_id},
    native_owner::{cudaMemGetInfo, cudaSetDevice},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn parse_u32_config(key: &str, value: Option<&str>, default: u32) -> Result<u32> {
    match value {
        Some(value) => value.parse().map_err(|_| format!("ENV_{key}")),
        None => Ok(default),
    }
}
fn env_u32(key: &str, default: u32) -> Result<u32> {
    match std::env::var(key) {
        Ok(value) => parse_u32_config(key, Some(&value), default),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(_) => Err(format!("ENV_{key}")),
    }
}
#[cfg(test)]
mod env_tests {
    use super::parse_u32_config;

    #[test]
    fn optional_u32_rejects_invalid_value_without_panicking() {
        assert_eq!(parse_u32_config("MGBFS_SHARDS", None, 64).unwrap(), 64);
        assert_eq!(parse_u32_config("MGBFS_SHARDS", Some("8"), 64).unwrap(), 8);
        assert_eq!(parse_u32_config("MGBFS_SHARDS", Some("bad"), 64).unwrap_err(), "ENV_MGBFS_SHARDS");
        assert_eq!(parse_u32_config("MGBFS_SHARDS", Some("4294967296"), 64).unwrap_err(), "ENV_MGBFS_SHARDS");
    }
}
fn profiler_window_start() -> Result<bool> {
    match std::env::var("MGBFS_PROFILE_SEARCH").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(false),
        Ok("1") => {
            let code = unsafe { cudaProfilerStart() };
            if code != 0 {
                return Err(format!("CUDA_PROFILER_START_{code}"));
            }
            Ok(true)
        }
        _ => Err("MGBFS_PROFILE_SEARCH_EXPECTED_0_OR_1".into()),
    }
}
fn profiler_window_stop(active: bool) -> Result<()> {
    if active {
        let code = unsafe { cudaProfilerStop() };
        if code != 0 {
            return Err(format!("CUDA_PROFILER_STOP_{code}"));
        }
    }
    Ok(())
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
) -> Result<crate::bootstrap::BootstrapGroup> {
    let launch =
        std::env::var("TORCHELASTIC_RUN_ID").map_err(|_| "BOOTSTRAP_LAUNCH_ID_REQUIRED")?;
    if launch.is_empty() || launch == "none" {
        return Err("BOOTSTRAP_LAUNCH_ID_REQUIRED".into());
    }
    let run_digest =
        Sha256::digest(serde_json::to_vec(&(launch, path)).map_err(|e| e.to_string())?);
    let identity = crate::control_handshake::RunIdentity {
        config_digest: digest,
        run_id: run_digest[..16].try_into().unwrap(),
    };
    crate::bootstrap::rendezvous(path, rank, world, identity, Duration::from_secs(60), || {
        let mut id = [0; 128];
        if unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) } != 0 {
            return Err("NCCL_ID".into());
        }
        Ok(id)
    })
}
fn used() -> Result<usize> {
    let (mut free, mut total) = (0, 0);
    let x = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if x != 0 {
        return Err(format!("CUDA_MEMORY_{x}"));
    }
    Ok(total - free)
}
struct PreparedPass {
    multiset: Option<mgbfs_core::lrx_multiset::LrxMultiset>,
    group: String,
    graph: MatrixGroup,
    n: usize,
    batch: u32,
    declared_capacity: u32,
    declared_future: u32,
    mode: CapacityMode,
    global_capacity: u64,
    global_future: u64,
    capacity: u32,
    future: u32,
    compact_states: bool,
    archive_width: usize,
    profile: String,
    owner: String,
    pre: String,
    seed: [u8; 16],
    seed_hex: String,
    hash_first_generation: String,
    selection: ReferenceSelection,
    digest: [u8; 32],
    archive_path: String,
    archive_enabled: bool,
    disk_bytes: u64,
    archive_rows: u32,
    archive_slots: usize,
    stream_archive: bool,
    cfg: DistributedConfig,
    bootstrap_digest: [u8; 32],
}
fn run_pass(args: &[String], warmup_completed: bool, is_measure: bool) -> Result<()> {
    if args.len() != 6 {
        return Err("ARGS_group_batch_bootstrap_archive_prefix_output_dir".into());
    }
    if env_u32("MGBFS_MACRO_DEPTH", 1)? > 1 {
        return run_macro_pass(args, warmup_completed, is_measure);
    }
    let rank = required("RANK")?;
    let local = required("LOCAL_RANK")?;
    let world = required("WORLD_SIZE")?;
    if !world.is_power_of_two() || world > 8 || rank != local {
        return Err("TOPOLOGY".into());
    }
    // Rendezvous by launch identity first. Config digest is agreed over the
    // connected control channel, so an invalid local config can report failure
    // instead of leaving its peer in a stale-config bootstrap timeout.
    let mut control_group = bootstrap(Path::new(&args[3]), rank, world, [0; 32])?;
    let prepared = (|| -> Result<PreparedPass> {
    let multiset = if args[1].starts_with("lrx") {
        Some(mgbfs_core::lrx_multiset::LrxMultiset::from_label(&args[1])?)
    } else { None };
    let (group, graph) = if let Some(word) = &multiset {
        (word.label(), word.position_group()?)
    } else { MatrixGroup::from_reference_label(&args[1])? };
    let expected_states = multiset.as_ref().map_or(graph.expected_max_unique_states, |x| x.order());
    let n = graph.rows;
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let declared_capacity = match std::env::var("MGBFS_BENCH_CAPACITY") {
        Ok(value) => value.parse::<u32>().map_err(|_| "CAPACITY")?,
        Err(std::env::VarError::NotPresent) => u32::try_from(expected_states)
            .map_err(|_| "CAPACITY_EXPLICIT_REQUIRED")?,
        Err(_) => return Err("CAPACITY".into()),
    };
    let declared_future = env_u32("MGBFS_FUTURE_CAPACITY", declared_capacity)?;
    let mode = capacity_mode()?;
    let capacity_plan = cluster_capacity_plan(mode, u64::from(declared_capacity), world)?;
    let future_plan = cluster_capacity_plan(mode, u64::from(declared_future), world)?;
    let capacity = u32::try_from(capacity_plan.rank_records(rank)?).map_err(|_| "CAPACITY")?;
    let future = u32::try_from(future_plan.rank_records(rank)?).map_err(|_| "CAPACITY")?;
    let rank_map = crate::topology::reference_rank_map(
        world,
        std::env::var("MGBFS_RANK_MAP").ok().as_deref(),
    )?;
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
    if group.starts_with('u') && (compact_states || archive_width != graph.start.len()) {
        return Err("UNITRIANGULAR_REQUIRES_MATRIX_CODEC".into());
    }
    let profile = std::env::var("MGBFS_PROFILE").unwrap_or_else(|_| "DENSE".into());
    let owner = std::env::var("MGBFS_OWNER_BACKEND").unwrap_or_else(|_| "CUB_SORT_MERGE".into());
    let pre = std::env::var("MGBFS_PRE_DEDUP").unwrap_or_else(|_| "ON".into());
    let seed = match std::env::var("MGBFS_HASH_SEED_HEX") {
        Ok(value) => mgbfs_core::hash::parse_seed_hex(&value)?,
        Err(std::env::VarError::NotPresent) => 20260828u128.to_le_bytes(),
        Err(_) => return Err("HASH_SEED_HEX_32".into()),
    };
    let seed_hex = format!("{:032x}", u128::from_le_bytes(seed));
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
        )?,
        env_u32("MGBFS_BMMA_TILE_LIMIT", 256)?,
    )?
    .with_hash_first_generation(&hash_first_generation)?
    .with_transport(&std::env::var("MGBFS_TRANSPORT_BACKEND")
        .unwrap_or_else(|_| "HOST_SIZED_NCCL".into()))?
    .with_library_pool(
        std::env::var("MGBFS_LIBRARY_POOL_BYTES").ok().as_deref(),
        cfg!(feature = "library-owner"),
    )?;
    if multiset.is_some() && (!compact_states || profile != "DENSE" || selection.tensor_generation) {
        return Err("LRX_MULTISET_REQUIRES_COMPACT_DENSE".into());
    }
    let description=format!("distributed-native-ring-v2;{group};batch={batch};capacity_mode={mode:?};declared_capacity={declared_capacity};declared_ring={declared_future};global_capacity={};global_ring={};map={rank_map:?};seed=0x{seed_hex};archive_width={archive_width}", capacity_plan.global_records, future_plan.global_records);
    let description = format!("{description};compact_states={compact_states}");
    let description = format!("{description};reference_selection={selection:?}");
    let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
    let archive_path = format!("{}-rank-{rank}.mgbfsar1", args[4]);
    let archive_enabled = std::env::var("MGBFS_BENCH_SKIP_ARCHIVE").as_deref() != Ok("1");
    let disk_bytes = if archive_enabled {
        ArchiveRingPlan::reference_extent_bytes(archive_width, expected_states, capacity)?
    } else { 0 };
    let archive_rows = env_u32("MGBFS_ARCHIVE_ROWS", batch)?;
    let archive_slots = env_u32("MGBFS_ARCHIVE_SLOTS", 64)? as usize;
    let stream_archive = std::env::var("MGBFS_ARCHIVE_STREAM").as_deref() == Ok("1");
    selection.validate_archive_contract(archive_enabled,
        std::env::var("MGBFS_SEARCH_ONLY").as_deref() == Ok("1"))?;
    let buckets = env_u32("MGBFS_BUCKETS", 256)?;
    let shards = env_u32("MGBFS_SHARDS", 64)?;
    let (local_buckets, _) =
        crate::topology::reference_owner_geometry(world, rank, &rank_map, buckets, shards)?;
    let default_bucket_capacity =
        crate::topology::reference_bucket_capacity(capacity, local_buckets, 4096)?;
    let cfg = DistributedConfig {
        untouched_vram_reserve: 1 << 30,
        rank,
        world,
        logical_owner_to_rank: rank_map,
        transport: selection.transport,
        batch,
        layer_capacity: capacity,
        state_ring_capacity: future,
        buckets,
        shards,
        job_buckets: env_u32("MGBFS_JOB_BUCKETS", 4)?,
        bucket_capacity: env_u32("MGBFS_BUCKET_CAPACITY", default_bucket_capacity)?,
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
        "transport": format!("{:?}", cfg.transport),
    });
    let bootstrap_digest: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&bootstrap_description).map_err(|e| e.to_string())?)
            .into();
    Ok(PreparedPass {
        multiset, group, graph, n, batch, declared_capacity, declared_future,
        mode, global_capacity: capacity_plan.global_records,
        global_future: future_plan.global_records, capacity, future,
        compact_states, archive_width, profile, owner, pre, seed, seed_hex,
        hash_first_generation, selection, digest, archive_path, archive_enabled,
        disk_bytes, archive_rows, archive_slots, stream_archive, cfg,
        bootstrap_digest,
    })
    })();
    let device_admission = if prepared.is_ok() {
        let code = unsafe { cudaSetDevice(local as i32) };
        if code == 0 { Ok(()) } else { Err("CUDA_SET_DEVICE".to_string()) }
    } else {
        Ok(())
    };
    let config_digest = prepared.as_ref().map_or([0; 32], |p| p.bootstrap_digest);
    if control_group.agree_configuration(
        config_digest, prepared.is_err() || device_admission.is_err(),
        Duration::from_secs(60),
    )? {
        return Err(prepared.err().or_else(|| device_admission.err())
            .unwrap_or_else(|| "REMOTE_CONFIGURATION_FATAL".into()));
    }
    let PreparedPass {
        multiset, group, graph, n, batch, declared_capacity, declared_future,
        mode, global_capacity, global_future, capacity, future, compact_states,
        archive_width, profile, owner, pre, seed, seed_hex,
        hash_first_generation, selection, digest, archive_path, archive_enabled,
        disk_bytes, archive_rows, archive_slots, stream_archive, cfg,
        bootstrap_digest,
    } = prepared?;
    // Keep control sockets alive throughout this reference run. Dispatching GPU
    // epochs on them is a separate integration step, not claimed here.
    let id = control_group.nccl_id;
    let archive_setup = if archive_enabled {
        create_archive_extent(Path::new(&archive_path), stream_archive)
            .map_err(|e| format!("ARCHIVE_EXTENT: {e}"))
            .and_then(|extent| PinnedArchive::new(
                extent, disk_bytes, archive_width, digest, archive_rows,
                archive_slots,
            ))
            .map(Some)
    } else { Ok(None) };
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::ArchiveAdmission,
        archive_setup.is_err(), Duration::from_secs(600),
    )? {
        return Err(archive_setup.err().unwrap_or_else(|| "REMOTE_ARCHIVE_ADMISSION_FATAL".into()));
    }
    let mut archive = archive_setup?;
    let pinned = archive.as_ref().map_or(0, |a| a.pinned_bytes());
    let setup = Instant::now();
    let mut bfs = if let Some(word) = &multiset {
        DistributedNativeBfs::new_lrx_multiset_reference(word, seed,
            id, cfg.clone(), selection.owner, selection.library_pool_bytes)?
    } else { match selection.owner {
        ReferenceOwner::CudfRelational | ReferenceOwner::CucoIndexed | ReferenceOwner::CucoRank => {
            #[cfg(feature = "library-owner")]
            {
                DistributedNativeBfs::new_library_reference_with_owner(
                    &graph,
                    seed,
                    id,
                    cfg.clone(),
                    selection.materialization_capacity,
                    selection
                        .library_pool_bytes
                        .ok_or("REFERENCE_LIBRARY_POOL_REQUIRED")?,
                    selection.tensor_generation,
                    selection.owner,
                )?
            }
            #[cfg(not(feature = "library-owner"))]
            {
                return Err("REFERENCE_LIBRARY_NOT_COMPILED".into());
            }
        }
        ReferenceOwner::Native(owner) => {
            if selection.tensor_generation {
                DistributedNativeBfs::new_hash_first_tc_with_owner(
                    &graph,
                    seed,
                    id,
                    cfg.clone(),
                    selection
                        .materialization_capacity
                        .ok_or("REFERENCE_HASH_FIRST_CAPACITY")?,
                    owner,
                    selection.tile_limit,
                )?
            } else {
                DistributedNativeBfs::new_reference_with_owner(
                    &graph,
                    seed,
                    id,
                    cfg.clone(),
                    selection.materialization_capacity,
                    owner,
                    selection.tile_limit,
                )?
            }
        }
    }};
    let allocated = used()?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let trace = std::env::var_os("MGBFS_TRACE_DEPTHS").is_some();
    let profile_window = if is_measure { profiler_window_start()? } else { false };
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
        let alive = if let Some(archive) = archive.as_mut() {
            bfs.advance_archived(archive)?
        } else {
            bfs.advance()?
        };
        if trace && archive_enabled {
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
    profiler_window_stop(profile_window)?;
    let archive_commit = archive.take().map_or(Ok(()), PinnedArchive::finish);
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::ArchiveCommitted,
        archive_commit.is_err(), Duration::from_secs(7200),
    )? {
        return Err(archive_commit.err().unwrap_or_else(|| "REMOTE_ARCHIVE_COMMIT_FATAL".into()));
    }
    archive_commit?;
    let durable = start.elapsed().as_secs_f64();
    let output = (|| -> Result<()> {
    std::fs::create_dir_all(&args[5]).map_err(|e| e.to_string())?;
    let record=format!("{{\"status\":\"COMPLETE\",\"backend\":\"native_nccl_dense_ring_v2\",\"rank\":{rank},\"group\":\"s{n}\",\"batch\":{batch},\"capacity_mode\":\"{mode:?}\",\"archive_enabled\":{archive_enabled},\"archive_state_bytes\":{archive_width},\"declared_capacity_records\":{declared_capacity},\"global_capacity_records\":{global_capacity},\"rank_capacity_records\":{capacity},\"declared_state_ring_records\":{declared_future},\"global_state_ring_records\":{global_future},\"rank_state_ring_records\":{future},\"search_complete_seconds\":{search},\"durable_run_commit_seconds\":{durable},\"setup_seconds\":{setup_seconds},\"local_layer_sizes\":{layers:?},\"per_depth_seconds\":{times:?},\"cuda_allocated_used_bytes\":{allocated},\"cuda_peak_observed_bytes\":{},\"pinned_bytes\":{pinned},\"disk_reserved_bytes\":{disk_bytes}}}",used()?.max(allocated));
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
    crate::group_commit::write_rank_result(Path::new(&args[5]), rank, &{
        let mut value: serde_json::Value =
            serde_json::from_str(&record).map_err(|e| format!("RECORD_JSON: {e}"))?;
        value["output_contract"] = serde_json::json!(if archive_enabled {
            "archive_and_layer_counts"
        } else {
            "search_only_layer_counts"
        });
        if !archive_enabled || stream_archive {
            value["durable_run_commit_seconds"] = serde_json::Value::Null;
        }
        // Legacy consumers use durable_run_commit_seconds, but this timestamp
        // precedes rank-result fsync and group marker publication. State the
        // actual boundary explicitly without changing the old field's shape.
        value["archive_file_commit_seconds"] = if archive_enabled && !stream_archive {
            serde_json::json!(durable)
        } else {
            serde_json::Value::Null
        };
        if stream_archive && archive_enabled {
            value["stream_handoff_seconds"] = serde_json::json!(durable);
        }
        value["archive_commit_scope"] = serde_json::json!(if !archive_enabled {
            "search_only"
        } else if stream_archive {
            "fifo_flush"
        } else {
            "file_fsync"
        });
        value["device_allocation_plan"] =
            crate::distributed_memory::allocation_report(bfs.owned_memory());
        value["hash_seed_hex"] = serde_json::json!(seed_hex);
        value["bootstrap_digest"] = serde_json::json!(bootstrap_digest);
        value["transport_backend"] = serde_json::json!(format!("{:?}", cfg.transport));
        value["group"] = serde_json::json!(group);
        if let Some(word) = &multiset {
            value["graph_kind"] = serde_json::json!("lrx_multiset_schreier");
            value["start_state"] = serde_json::json!(word.start());
            value["expected_unique_states"] = serde_json::json!(word.order());
            value["generators"] = serde_json::json!(["L", "R", "X"]);
        }
        value["cuda_memory_sampling"] = serde_json::json!("setup_and_final_only_not_full_peak");
        value["dense_lookahead_batches"] = serde_json::json!(bfs.dense_lookahead_batches());
        value["library_pool_reserved_bytes"] = serde_json::json!(selection.library_pool_bytes);
        #[cfg(feature = "library-owner")]
        if let Some(usage) = bfs.library_pool_usage()? {
            value["library_control_pinned_bytes"] =
                serde_json::json!(mgbfs_cuda::library_owner::CONTROL_TRANSFER_PINNED_BYTES);
            value["library_pool_usage"] = serde_json::json!({
                "reserved_bytes": usage.reserved_bytes,
                "live_requested_bytes": usage.live_bytes,
                "peak_requested_bytes": usage.peak_bytes,
                "scope": "since_pool_creation_suballocations_not_full_vram_not_fragmentation_bound"
            });
        }
        if let Some(label) = crate::benchmark::library_backend_label(
            selection.owner, selection.profile,
        ) {
            value["backend"] = serde_json::json!(label);
        }
        serde_json::to_vec(&value).map_err(|e| format!("RECORD_JSON: {e}"))?
    })?;
    Ok(())
    })();
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::OutputWritten,
        output.is_err(), Duration::from_secs(60),
    )? {
        return Err(output.err().unwrap_or_else(|| "REMOTE_OUTPUT_WRITE_FATAL".into()));
    }
    output?;
    if rank == 0 {
        crate::group_commit::write_group_commit(Path::new(&args[5]), world, bootstrap_digest)?;
    }
    Ok(())
}

/// The existing weighted CUDA backend is single-rank. It remains separate
/// from the unit-cost NCCL runtime until distributed weighted settlement exists.
fn run_macro_pass(args: &[String], warmup_completed: bool, is_measure: bool) -> Result<()> {
    let rank = required("RANK")?;
    let world = required("WORLD_SIZE")?;
    let local = required("LOCAL_RANK")?;
    if rank != 0 || local != 0 || world != 1 {
        return Err("MACRO_REFERENCE_SINGLE_RANK_ONLY".into());
    }
    let macro_depth: u32 = std::env::var("MGBFS_MACRO_DEPTH")
        .map_err(|_| "MACRO_DEPTH")?
        .parse()
        .map_err(|_| "MACRO_DEPTH")?;
    if macro_depth <= 1 {
        return Err("MACRO_DEPTH".into());
    }
    if std::env::var("MGBFS_PROFILE").as_deref().is_ok_and(|x| x != "DENSE")
        || std::env::var("MGBFS_OWNER_BACKEND").as_deref().is_ok_and(|x| x != "CUB_SORT_MERGE")
    {
        return Err("MACRO_REFERENCE_DENSE_CUB_ONLY".into());
    }
    if unsafe { cudaSetDevice(local as i32) } != 0 {
        return Err("CUDA_SET_DEVICE".into());
    }
    let (group, graph) = MatrixGroup::from_reference_label(&args[1])?;
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let capacity = match std::env::var("MGBFS_BENCH_CAPACITY") {
        Ok(value) => value.parse().map_err(|_| "CAPACITY")?,
        Err(std::env::VarError::NotPresent) => graph.expected_max_unique_states
            .try_into().map_err(|_| "CAPACITY_EXPLICIT_REQUIRED")?,
        Err(_) => return Err("CAPACITY".into()),
    };
    let future = env_u32("MGBFS_FUTURE_CAPACITY", capacity)?;
    let compact = match std::env::var("MGBFS_STATE_CODEC").as_deref() {
        Ok("permutation_u8") => true,
        Ok("matrix_u8") | Err(_) => false,
        _ => return Err("STATE_CODEC".into()),
    };
    let generation_variant = if compact { 5 } else { 1 };
    let layout = MacroStateLayout::derive(&graph, generation_variant)?;
    match std::env::var("MGBFS_ARCHIVE_CODEC").as_deref() {
        Ok("permutation_u8") if compact => (),
        Ok("matrix_u8") if !compact => (),
        Err(_) => (),
        _ => return Err("MACRO_ARCHIVE_CODEC_MISMATCH".into()),
    }
    let prededup = match std::env::var("MGBFS_PRE_DEDUP").as_deref() {
        Ok("OFF") => false,
        Ok("ON") | Err(_) => true,
        _ => return Err("PRE_DEDUP".into()),
    };
    let seed = match std::env::var("MGBFS_HASH_SEED_HEX") {
        Ok(value) => mgbfs_core::hash::parse_seed_hex(&value)?,
        Err(std::env::VarError::NotPresent) => 20260828u128.to_le_bytes(),
        Err(_) => return Err("HASH_SEED_HEX_32".into()),
    };
    let seed_hex = format!("{:032x}", u128::from_le_bytes(seed));
    let archive_enabled = std::env::var("MGBFS_BENCH_SKIP_ARCHIVE").as_deref() != Ok("1");
    let stream_archive = std::env::var("MGBFS_ARCHIVE_STREAM").as_deref() == Ok("1");
    let archive_rows = env_u32("MGBFS_ARCHIVE_ROWS", batch)?;
    let cfg = MacroNativeConfig {
        macro_depth,
        batch,
        layer_capacity: capacity,
        future_capacity_per_depth: future,
        prededup,
        generation_variant,
        untouched_vram_reserve_bytes: 1 << 30,
    };
    let description = format!("macro-reference-v1;group={group};batch={batch};capacity={capacity};future={future};K={macro_depth};pre={prededup};generation={generation_variant};seed=0x{seed_hex};archive_width={};archive_enabled={archive_enabled}", layout.width);
    let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
    let disk_bytes = if archive_enabled {
        ArchiveRingPlan::reference_extent_bytes(
            layout.width, graph.expected_max_unique_states, capacity)?
    } else {
        0
    };
    let archive_path = format!("{}-rank-0.mgbfsar1", args[4]);
    let mut archive = if archive_enabled {
        let extent = create_archive_extent(Path::new(&archive_path), stream_archive)
            .map_err(|e| format!("ARCHIVE_EXTENT: {e}"))?;
        Some(PinnedArchive::new(
            extent, disk_bytes, layout.width, digest, archive_rows,
            env_u32("MGBFS_ARCHIVE_SLOTS", 64)? as usize,
        )?)
    } else {
        None
    };
    let pinned = archive.as_ref().map_or(0, |a| a.pinned_bytes());
    let setup_start = Instant::now();
    let mut bfs = MacroNativeBfs::new(&graph, seed, cfg)?;
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let allocated = used()?;
    let profile_window = if is_measure { profiler_window_start()? } else { false };
    let start = Instant::now();
    let mut layers = Vec::new();
    let mut times = Vec::new();
    loop {
        let tick = Instant::now();
        layers.push(bfs.frontier_len());
        if let Some(archive) = archive.as_mut() {
            bfs.archive_current(archive)?;
        }
        let alive = bfs.advance()?;
        times.push(tick.elapsed().as_secs_f64());
        if !alive {
            break;
        }
    }
    let search = start.elapsed().as_secs_f64();
    profiler_window_stop(profile_window)?;
    if let Some(archive) = archive.take() {
        archive.finish()?;
    }
    let durable = start.elapsed().as_secs_f64();
    std::fs::create_dir_all(&args[5]).map_err(|e| e.to_string())?;
    let record = serde_json::json!({
        "status": "COMPLETE", "backend": "macro_native_single_rank_v1",
        "rank": 0, "world_size": 1, "group": group, "batch": batch,
        "macro_depth": macro_depth, "frontier_profile": "DENSE",
        "owner_backend": "CUB_SORT_MERGE", "pre_dedup": if prededup { "ON" } else { "OFF" },
        "hash_seed_hex": seed_hex, "generation_variant": generation_variant,
        "archive_enabled": archive_enabled, "archive_state_bytes": layout.width,
        "output_contract": if archive_enabled { "archive_and_layer_counts" } else { "search_only_layer_counts" },
        "search_complete_seconds": search,
        "durable_run_commit_seconds": if archive_enabled { Some(durable) } else { None },
        "setup_seconds": setup_seconds, "local_layer_sizes": layers,
        "per_depth_seconds": times, "declared_capacity_records": capacity,
        "future_capacity_per_depth": future,
        "explicit_device_aligned_bytes": bfs.requested_device_bytes(),
        "cuda_allocated_used_bytes": allocated,
        "cuda_peak_observed_bytes": used()?.max(allocated),
        "cuda_memory_sampling": "setup_and_final_only_not_full_peak",
        "pinned_bytes": pinned, "disk_reserved_bytes": disk_bytes,
        "warmup_completed": warmup_completed,
    });
    std::fs::write(
        Path::new(&args[5]).join("rank-0.json"),
        serde_json::to_vec(&record).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    Ok(())
}
/// Shared reference benchmark entry point. Argument zero is the launcher name;
/// the remaining arguments are group, batch, bootstrap, archive and output.
/// This does not implement the production RunConfigV1 dispatcher.
pub fn run(args: Vec<String>) -> Result<()> {
    use crate::benchmark::{run_phases, Phase};
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
            return run_pass(&args, warmup, true);
        }
        let mut warm_args = args.clone();
        for index in [3, 4, 5] {
            warm_args[index].push_str(".warmup");
        }
        run_pass(&warm_args, false, false)?;
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
