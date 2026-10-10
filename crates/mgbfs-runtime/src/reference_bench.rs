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
#[cfg(debug_assertions)]
fn test_fault_rank(name: &str, rank: u32, world: u32) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => {
            let selected: u32 = value.parse().map_err(|_| format!("{name}_INVALID"))?;
            if selected >= world {
                return Err(format!("{name}_OUT_OF_RANGE"));
            }
            Ok(selected == rank)
        }
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(_) => Err(format!("{name}_INVALID")),
    }
}
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
        assert_eq!(
            parse_u32_config("MGBFS_SHARDS", Some("bad"), 64).unwrap_err(),
            "ENV_MGBFS_SHARDS"
        );
        assert_eq!(
            parse_u32_config("MGBFS_SHARDS", Some("4294967296"), 64).unwrap_err(),
            "ENV_MGBFS_SHARDS"
        );
    }
}
#[cfg(debug_assertions)]
fn archive_fault_extent(
    extent: Box<dyn crate::archive::Extent + Send>,
    write_fault: bool,
    sync_fault: bool,
) -> Box<dyn crate::archive::Extent + Send> {
    // Test-only disk boundary: real reservation/header writes still happen.
    // Fail records in the worker, not an artificial producer-side check.
    struct FaultExtent {
        inner: Box<dyn crate::archive::Extent + Send>,
        write_fault: bool,
        sync_fault: bool,
    }
    impl crate::archive::Extent for FaultExtent {
        fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
            self.inner.reserve(bytes)
        }
        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
            if self.write_fault && offset >= 48 {
                return Err(std::io::Error::other(
                    "TEST_INJECTED_ARCHIVE_WORKER_WRITE_ERROR",
                ));
            }
            self.inner.write_at(offset, bytes)
        }
        fn sync(&mut self) -> std::io::Result<()> {
            if self.sync_fault {
                return Err(std::io::Error::other(
                    "TEST_INJECTED_ARCHIVE_WORKER_SYNC_ERROR",
                ));
            }
            self.inner.sync()
        }
    }
    if write_fault || sync_fault {
        Box::new(FaultExtent {
            inner: extent,
            write_fault,
            sync_fault,
        })
    } else {
        extent
    }
}
#[cfg(all(test, debug_assertions, target_os = "linux"))]
mod archive_fault_tests {
    use super::archive_fault_extent;
    use crate::archive::{Archive, FileExtent};

    fn exercise(write_fault: bool, sync_fault: bool) -> (bool, bool) {
        let path = std::env::temp_dir().join(format!(
            "mgbfs-archive-fault-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let extent = FileExtent::create_new(&path).unwrap();
        let mut archive = Archive::new_run_durable(
            archive_fault_extent(Box::new(extent), write_fault, sync_fault),
            4096,
            4,
            [0; 32],
        )
        .unwrap();
        let write_ok = archive.records(0, &[0; 4], &[[0; 4]]).is_ok();
        if write_ok {
            archive.layer_commit(0, 1).unwrap();
        }
        let finish_ok = archive.run_commit().is_ok();
        drop(archive);
        std::fs::remove_file(path).unwrap();
        (write_ok, finish_ok)
    }
    #[test]
    fn disk_write_fault_preserves_admission_but_rejects_records_and_commit() {
        assert_eq!(exercise(true, false), (false, false));
    }
    #[test]
    fn disk_sync_fault_preserves_records_but_rejects_durable_commit() {
        assert_eq!(exercise(false, true), (true, false));
    }
    #[test]
    fn healthy_disk_extent_commits() {
        assert_eq!(exercise(false, false), (true, true));
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
    macro_operators: Option<mgbfs_core::macro_generators::MacroGeneratorSet>,
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
    calibration_layers: Option<u32>,
    cfg: DistributedConfig,
    bootstrap_digest: [u8; 32],
}
impl PreparedPass {
    fn ensure_native_dispatch(&self) -> Result<()> {
        // Composed operators stay weighted through the existing constructor
        // and original-depth driver; never send them to the unit-cost path.
        if self.macro_operators.is_some()
            && (self.profile != "DENSE"
                || self.selection.materialization_capacity.is_some()
                || !matches!(
                    self.selection.owner,
                    ReferenceOwner::Native(_) | ReferenceOwner::CucoRank
                ))
        {
            return Err("RUN_MACRO_DISPATCH_UNAVAILABLE".into());
        }
        Ok(())
    }
}
fn prepare_production(args: &[String], rank: u32, world: u32) -> Result<PreparedPass> {
    use mgbfs_core::config::{FrontierProfile, RunConfigV1};
    let file = std::fs::File::open(&args[1]).map_err(|e| format!("CONFIG_OPEN: {e}"))?;
    let config: RunConfigV1 = serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| format!("CONFIG_PARSE: {e}"))?;
    let digest = config.digest()?;
    if config.topology.world_size != world {
        return Err("RUN_TOPOLOGY_MISMATCH".into());
    }
    if !(2..=4).contains(&config.capacities.route_slot_count) {
        return Err("RUN_ROUTE_BANK_COUNT_UNAVAILABLE".into());
    }
    let narrow = |value: u64| -> Result<u32> {
        let value = u32::try_from(value).map_err(|_| "RUN_CAPACITY_ABI")?;
        if value == 0 || value > i32::MAX as u32 {
            return Err("RUN_CAPACITY_ABI".into());
        }
        Ok(value)
    };
    let batch = narrow(config.parent_batch)?;
    let capacity = narrow(config.capacities.layer_hash_records_per_arena)?;
    let future = narrow(config.capacities.state_ring_records)?;
    let slots = narrow(config.capacities.route_slot_records)?;
    let macro_operators = if config.macro_depth > 1 {
        Some(
            mgbfs_core::macro_generators::MacroGeneratorSet::compile_bounded(
                &config.graph,
                config.macro_depth,
                (slots / batch) as usize,
            )?,
        )
    } else {
        None
    };
    let moves = u32::try_from(
        macro_operators
            .as_ref()
            .map_or(config.graph.generators.len(), |schedule| {
                schedule.transitions.len()
            }),
    )
    .map_err(|_| "RUN_CANDIDATE_OVERFLOW")?;
    let candidates = batch.checked_mul(moves).ok_or("RUN_CANDIDATE_OVERFLOW")?;
    // Capacity is an admission bound, not the payload size. Unused records
    // are not added to candidates or sent to peers.
    if slots < candidates {
        return Err("RUN_ROUTE_RECORD_COUNT_UNAVAILABLE".into());
    }
    let buckets = config
        .topology
        .shards_per_rank
        .checked_mul(config.topology.buckets_per_shard)
        .and_then(|x| x.checked_mul(world))
        .ok_or("RUN_TOPOLOGY_OVERFLOW")?;
    let shards = config
        .topology
        .shards_per_rank
        .checked_mul(world)
        .ok_or("RUN_TOPOLOGY_OVERFLOW")?;
    let width = config.graph.start.len();
    let slot_stride = (width as u64).checked_add(16).ok_or("RUN_ARCHIVE_BYTES")?;
    if config.capacities.pinned_archive_slot_bytes % slot_stride != 0 {
        return Err("RUN_PINNED_SLOT_ALIGNMENT".into());
    }
    let archive_rows = narrow(config.capacities.pinned_archive_slot_bytes / slot_stride)?;
    if archive_rows < batch {
        return Err("RUN_PINNED_SLOT_BATCH_CAPACITY".into());
    }
    let profile = match config.frontier_profile {
        FrontierProfile::Dense => "DENSE",
        FrontierProfile::HashFirst => "HASH_FIRST",
    }
    .to_owned();
    let pre = if config.local_pre_dedup { "ON" } else { "OFF" }.to_owned();
    let owner = match config.owner_backend {
        mgbfs_core::config::RunOwnerBackend::CubSortMerge => "CUB_SORT_MERGE",
        mgbfs_core::config::RunOwnerBackend::BmmaBucket => "BMMA_BUCKET",
        mgbfs_core::config::RunOwnerBackend::CucoRank => "CUCO_RANK",
    }
    .to_owned();
    let hash_first_generation = if config.frontier_profile == FrontierProfile::HashFirst {
        "INT_MMA_SM75"
    } else {
        "SCALAR"
    }
    .to_owned();
    let selection = ReferenceSelection::parse(&profile, &owner, &pre, false, candidates, 256)?
        .with_hash_first_generation(&hash_first_generation)?
        .with_transport(match config.transport_backend {
            mgbfs_core::config::ReferenceTransport::HostSizedNccl => "HOST_SIZED_NCCL",
            mgbfs_core::config::ReferenceTransport::Lsa => "NCCL_LSA",
        })?;
    let pool = config.library_pool_bytes.map(|bytes| bytes.to_string());
    let selection =
        selection.with_library_pool(pool.as_deref(), cfg!(feature = "library-owner"))?;
    let cfg = DistributedConfig {
        route_banks: config.capacities.route_slot_count as usize,
        epoch_window: usize::try_from(config.completion_epoch_window)
            .map_err(|_| "RUN_EPOCH_WINDOW_ABI")?,
        rank,
        world,
        logical_owner_to_rank: if world == 1 {
            vec![0, 0]
        } else {
            config.topology.logical_owner_to_rank.clone()
        },
        transport: config.transport_backend,
        batch,
        layer_capacity: capacity,
        state_ring_capacity: future,
        state_descriptor_capacity: narrow(config.capacities.state_extent_descriptors)?,
        buckets,
        shards,
        job_buckets: config.topology.buckets_per_shard.min(candidates),
        bucket_capacity: narrow(config.capacities.next_bucket_capacity_records)?,
        prededup: config.local_pre_dedup,
        generation_variant: 1,
        untouched_vram_reserve: config.capacities.untouched_vram_reserve_bytes,
    };
    let group = format!(
        "matrix-{}",
        digest
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect::<String>()
    );
    Ok(PreparedPass {
        macro_operators,
        multiset: None,
        group,
        n: config.graph.rows,
        graph: config.graph,
        batch,
        declared_capacity: capacity,
        declared_future: future,
        mode: CapacityMode::MaxPerRank,
        global_capacity: u64::from(capacity) * u64::from(world),
        global_future: u64::from(future) * u64::from(world),
        capacity,
        future,
        compact_states: false,
        archive_width: width,
        profile,
        owner,
        pre,
        seed: config.seed,
        seed_hex: format!("{:032x}", u128::from_le_bytes(config.seed)),
        hash_first_generation,
        selection,
        digest,
        archive_path: format!("{}-rank-{rank}.mgbfsar1", args[4]),
        archive_enabled: true,
        disk_bytes: config.capacities.disk_extent_bytes_per_rank,
        archive_rows,
        archive_slots: config.capacities.pinned_archive_slots as usize,
        stream_archive: false,
        calibration_layers: None,
        cfg,
        bootstrap_digest: digest,
    })
}
#[cfg(test)]
mod production_tests {
    use super::*;
    #[test]
    fn production_prepare_counts_weighted_operators_not_base_moves() {
        for (depth, operators) in [(2, 8), (10, 23)] {
            let mut config: mgbfs_core::config::RunConfigV1 =
                serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
            config.macro_depth = depth;
            config.capacities.route_slot_records = 32;
            // Keep the job geometry above both independent operator counts;
            // job_buckets is bounded by geometry as well as candidate count.
            config.topology.buckets_per_shard = 32;
            let original = config.graph.clone();
            let mut prepared = prepare(config).unwrap();
            // Independent S4 ball counts: layers 3+5, or all 24 minus identity.
            assert_eq!(prepared.cfg.job_buckets, operators);
            let schedule = prepared.macro_operators.as_ref().unwrap();
            assert_eq!(schedule.transitions.len(), operators as usize);
            assert_eq!(schedule.requested_depth, depth);
            assert_eq!(
                schedule
                    .transitions
                    .iter()
                    .filter(|x| x.weight == 1)
                    .count(),
                3
            );
            assert_eq!(
                schedule
                    .transitions
                    .iter()
                    .filter(|x| x.weight == 2)
                    .count(),
                5
            );
            prepared.ensure_native_dispatch().unwrap();
            assert_eq!(prepared.graph.generators, original.generators);
            assert_eq!(prepared.cfg.state_ring_capacity, 128);
            prepared.cfg.world = 1;
            prepared.cfg.rank = 0;
            prepared.cfg.logical_owner_to_rank = vec![0];
            assert!(prepared.ensure_native_dispatch().is_ok());
            prepared.selection.materialization_capacity = Some(32);
            assert_eq!(
                prepared.ensure_native_dispatch().unwrap_err(),
                "RUN_MACRO_DISPATCH_UNAVAILABLE"
            );
            prepared.selection.materialization_capacity = None;
            prepared.selection.owner = ReferenceOwner::CucoRank;
            prepared.ensure_native_dispatch().unwrap();
            // Other library owners still lack weighted target membership.
            prepared.selection.owner = ReferenceOwner::CucoIndexed;
            assert_eq!(
                prepared.ensure_native_dispatch().unwrap_err(),
                "RUN_MACRO_DISPATCH_UNAVAILABLE"
            );
        }
    }
    #[test]
    fn production_prepare_admits_spare_route_capacity_without_changing_job_geometry() {
        let mut config: mgbfs_core::config::RunConfigV1 =
            serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
        config.capacities.route_slot_records = 32;
        let prepared = prepare(config).unwrap();
        // Fixture geometry has two buckets/shard, independently of three moves.
        assert_eq!(prepared.cfg.job_buckets, 2);
        assert_eq!(prepared.cfg.batch, 1);
        assert!(prepared.macro_operators.is_none());
        prepared.ensure_native_dispatch().unwrap();
    }
    #[test]
    fn typed_route_banks_are_independent_of_completion_credits() {
        for banks in [2, 3, 4] {
            let mut config: mgbfs_core::config::RunConfigV1 =
                serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
            config.capacities.route_slot_count = banks;
            config.completion_epoch_window = 3;
            let prepared = prepare(config).unwrap();
            assert_eq!(prepared.cfg.route_banks, banks as usize);
            assert_eq!(prepared.cfg.epoch_window, 3);
            assert_eq!(prepared.cfg.state_ring_capacity, 128);
        }
    }
    #[test]
    #[cfg(feature = "library-owner")]
    fn typed_cuco_uses_the_declared_pool_and_existing_rank_owner() {
        let mut config: mgbfs_core::config::RunConfigV1 =
            serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
        config.owner_backend = mgbfs_core::config::RunOwnerBackend::CucoRank;
        config.library_pool_bytes = Some(96 << 20);
        config.completion_epoch_window = 3;
        let prepared = prepare(config).unwrap();
        assert_eq!(prepared.owner, "CUCO_RANK");
        assert_eq!(
            prepared.selection.owner,
            mgbfs_core::config::ReferenceOwner::CucoRank
        );
        assert_eq!(prepared.selection.library_pool_bytes, Some(96 << 20));
        assert_eq!(prepared.cfg.epoch_window, 3);
        assert_eq!(prepared.cfg.state_ring_capacity, 128);
        assert!(prepared.archive_enabled);
    }
    #[test]
    fn physical_gate_fixture_has_the_independent_s4_layers() {
        let config: mgbfs_core::config::RunConfigV1 =
            serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
        config.validate().unwrap();
        assert_eq!(u128::from_le_bytes(config.seed), 20260828);
        assert_eq!(
            config
                .graph
                .exact_layers(24)
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![1, 3, 5, 6, 5, 3, 1]
        );
        assert!(prepare(config).unwrap().archive_enabled);
    }
    fn prepare(config: mgbfs_core::config::RunConfigV1) -> Result<PreparedPass> {
        let path = std::env::temp_dir().join(format!(
            "mgbfs-production-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        let result = prepare_production(
            &[
                "run".into(),
                path.to_str().unwrap().into(),
                "unused".into(),
                "bootstrap".into(),
                "archive".into(),
                "output".into(),
            ],
            1,
            2,
        );
        std::fs::remove_file(path).unwrap();
        result
    }
    #[test]
    fn admitted_dense_buckets_form_one_bounded_job_per_shard() {
        use crate::jobs::{split, JobSpan};
        use mgbfs_core::owner_job::{BucketJob, Range};
        let mut config = mgbfs_core::config::RunConfigV1::fixture(3).unwrap();
        config.topology.shards_per_rank = 4;
        config.topology.buckets_per_shard = 64;
        config.parent_batch = 128;
        config.capacities.route_slot_records = 128 * config.graph.generators.len() as u64;
        config.capacities.pinned_archive_slot_bytes = 128 * (config.graph.start.len() as u64 + 16);
        let prepared = prepare(config).unwrap();
        let incoming: Vec<Range> = (0..256).map(|i| Range { begin: i, count: 1 }).collect();
        let history = vec![Range::default(); 256];
        let mut jobs = vec![BucketJob::default(); 256];
        let mut spans = vec![JobSpan::default(); 256];
        let (descriptors, count) = split(
            &incoming,
            &history,
            &history,
            512,
            prepared.cfg.job_buckets,
            64,
            0,
            &mut jobs,
            &mut spans,
        )
        .unwrap();
        assert_eq!(descriptors, 256);
        assert_eq!(
            count, 4,
            "small dense buckets must not become 64 serial owner jobs"
        );
        assert_eq!(
            spans[..count].iter().map(|s| s.rows).collect::<Vec<_>>(),
            vec![64; 4]
        );
        let (_, bounded) = split(
            &incoming,
            &history,
            &history,
            32,
            prepared.cfg.job_buckets,
            64,
            0,
            &mut jobs,
            &mut spans,
        )
        .unwrap();
        assert_eq!(bounded, 8);
        assert!(spans[..bounded].iter().all(|s| s.rows <= 32));
    }
    #[test]
    fn small_batch_owner_jobs_satisfy_native_memory_query() {
        let mut config = mgbfs_core::config::RunConfigV1::fixture(3).unwrap();
        config.topology.buckets_per_shard = 64;
        config.parent_batch = 1;
        config.capacities.route_slot_records = config.graph.generators.len() as u64;
        let prepared = prepare(config.clone()).unwrap();
        let candidates = prepared.cfg.batch * config.graph.generators.len() as u32;
        let mut bytes = mgbfs_cuda::native_owner::BoundedOwnerBytes::default();
        let status = unsafe {
            mgbfs_cuda::native_owner::mgbfs_bounded_owner_query(
                candidates,
                prepared.cfg.job_buckets,
                prepared.cfg.bucket_capacity,
                0,
                0,
                0,
                &mut bytes,
            )
        };
        assert_eq!(
            status, 0,
            "production jobs must fit the native candidate capacity"
        );
    }
    #[test]
    fn typed_run_preserves_admitted_config_and_independent_capacities() {
        let mut config = mgbfs_core::config::RunConfigV1::fixture(3).unwrap();
        config.capacities.route_slot_count = 2;
        config.capacities.state_extent_descriptors = 321;
        config.seed = [42; 16];
        config.topology.logical_owner_to_rank = vec![1, 0];
        let expected = config.digest().unwrap();
        let prepared = prepare(config.clone()).unwrap();
        assert_eq!(prepared.digest, expected);
        assert_eq!(prepared.bootstrap_digest, expected);
        assert_eq!(prepared.seed, config.seed);
        assert_eq!(prepared.cfg.state_descriptor_capacity, 321);
        assert_eq!(
            prepared.cfg.state_ring_capacity as u64,
            config.capacities.state_ring_records
        );
        assert_eq!(prepared.cfg.logical_owner_to_rank, vec![1, 0]);
        assert_eq!(
            prepared.disk_bytes,
            config.capacities.disk_extent_bytes_per_rank
        );
        assert_eq!(
            u64::from(prepared.archive_rows) * (prepared.archive_width as u64 + 16),
            config.capacities.pinned_archive_slot_bytes
        );
        assert!(prepared.archive_enabled);
    }
    #[test]
    fn typed_run_does_not_coerce_unimplemented_contracts() {
        let mut config = mgbfs_core::config::RunConfigV1::fixture(3).unwrap();
        config.capacities.route_slot_count = 5;
        assert_eq!(
            prepare(config.clone()).err().unwrap(),
            "RUN_ROUTE_BANK_COUNT_UNAVAILABLE"
        );
        let mut config = config;
        config.capacities.route_slot_count = 2;
        config.capacities.pinned_archive_slot_bytes -= 1;
        assert_eq!(prepare(config).err().unwrap(), "RUN_PINNED_SLOT_ALIGNMENT");
    }
}
fn run_pass(
    args: &[String],
    warmup_completed: bool,
    is_measure: bool,
    manifest: bool,
    production: bool,
) -> Result<()> {
    if args.len() != 6 {
        return Err("ARGS_group_batch_bootstrap_archive_prefix_output_dir".into());
    }
    let rank = required("RANK")?;
    let local = required("LOCAL_RANK")?;
    let world = required("WORLD_SIZE")?;
    if !world.is_power_of_two() || world > 8 || rank >= world || local > i32::MAX as u32 {
        return Err("TOPOLOGY".into());
    }
    if !production && world == 1 && crate::reference_launch::macro_depth_from_env(world)? {
        return run_macro_pass(args, warmup_completed, is_measure, manifest);
    }
    // Rendezvous by launch identity first. Config digest is agreed over the
    // connected control channel, so an invalid local config can report failure
    // instead of leaving its peer in a stale-config bootstrap timeout.
    let mut control_group = bootstrap(Path::new(&args[3]), rank, world, [0; 32])?;
    let prepared = (|| -> Result<PreparedPass> {
        crate::cuda_loading::validate_requested()?;
        if production {
            crate::reference_launch::bench_warmup_for_launch(
                std::env::var("MGBFS_BENCH_WARMUP").ok().as_deref(),
                std::env::var("MGBFS_ARCHIVE_STREAM").ok().as_deref(),
            )?;
            let prepared = prepare_production(args, rank, world)?;
            prepared.ensure_native_dispatch()?;
            return Ok(prepared);
        }
        let warmup_requested = crate::reference_launch::bench_warmup_for_launch(
            std::env::var("MGBFS_BENCH_WARMUP").ok().as_deref(),
            std::env::var("MGBFS_ARCHIVE_STREAM").ok().as_deref(),
        )?;
        crate::reference_launch::macro_depth_from_env(world)?;
        let multiset = if !manifest && args[1].starts_with("lrx") {
            Some(mgbfs_core::lrx_multiset::LrxMultiset::from_label(&args[1])?)
        } else {
            None
        };
        let (group, graph) = if let Some(word) = &multiset {
            (word.label(), word.position_action()?)
        } else if manifest {
            crate::reference_launch::load_matrix_manifest(Path::new(&args[1]))?
        } else {
            MatrixGroup::from_reference_label(&args[1])?
        };
        let expected_states = multiset
            .as_ref()
            .map_or(graph.expected_max_unique_states, |x| x.order());
        let n = graph.rows;
        let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
        let declared_capacity = match std::env::var("MGBFS_BENCH_CAPACITY") {
            Ok(value) => value.parse::<u32>().map_err(|_| "CAPACITY")?,
            Err(std::env::VarError::NotPresent) => {
                u32::try_from(expected_states).map_err(|_| "CAPACITY_EXPLICIT_REQUIRED")?
            }
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
        if manifest && compact_states {
            return Err("MATRIX_MANIFEST_REQUIRES_MATRIX_CODEC".into());
        }
        let archive_width = match std::env::var("MGBFS_ARCHIVE_CODEC").as_deref() {
            Ok("permutation_u8") => n,
            Err(_) if compact_states => n,
            Ok("matrix_u8") | Err(_) => graph.start.len(),
            _ => return Err("ARCHIVE_CODEC".into()),
        };
        if compact_states && archive_width != n {
            return Err("COMPACT_STATE_REQUIRES_COMPACT_ARCHIVE".into());
        }
        if manifest && archive_width != graph.start.len() {
            return Err("MATRIX_MANIFEST_REQUIRES_MATRIX_CODEC".into());
        }
        if group.starts_with('u') && (compact_states || archive_width != graph.start.len()) {
            return Err("UNITRIANGULAR_REQUIRES_MATRIX_CODEC".into());
        }
        let profile = std::env::var("MGBFS_PROFILE").unwrap_or_else(|_| "DENSE".into());
        let owner =
            std::env::var("MGBFS_OWNER_BACKEND").unwrap_or_else(|_| "CUB_SORT_MERGE".into());
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
        .with_transport(
            &std::env::var("MGBFS_TRANSPORT_BACKEND").unwrap_or_else(|_| "HOST_SIZED_NCCL".into()),
        )?
        .with_library_pool(
            std::env::var("MGBFS_LIBRARY_POOL_BYTES").ok().as_deref(),
            cfg!(feature = "library-owner"),
        )?;
        if selection.tensor_generation {
            // This preparation result is agreed by the existing control channel
            // before archive/pinned admission or communicator creation.
            crate::failure::check_native_status(unsafe { cudaSetDevice(local as i32) })?;
            match unsafe { mgbfs_cuda::ffi::mgbfs_hash_first_tc_validate_device() } {
                0 => (),
                3 => return Err("HASH_FIRST_TC_DEVICE_UNSUPPORTED".into()),
                status => return Err(format!("HASH_FIRST_TC_DEVICE_QUERY_{status}")),
            }
        }
        if multiset.is_some()
            && (!compact_states || profile != "DENSE" || selection.tensor_generation)
        {
            return Err("LRX_MULTISET_REQUIRES_COMPACT_DENSE".into());
        }
        let description=format!("distributed-native-ring-v2;{group};batch={batch};capacity_mode={mode:?};declared_capacity={declared_capacity};declared_ring={declared_future};global_capacity={};global_ring={};map={rank_map:?};seed=0x{seed_hex};archive_width={archive_width}", capacity_plan.global_records, future_plan.global_records);
        let description = format!("{description};compact_states={compact_states}");
    let description = format!("{description};reference_selection={selection:?}");
    let archive_selection = std::env::var("MGBFS_ARCHIVE_SELECTION").ok();
    if archive_selection.as_deref().is_some_and(|mode| mode != "last_complete_small_1000" && mode != "last_complete_prefix_1000" && mode != "all_states") {
        return Err("ENV_MGBFS_ARCHIVE_SELECTION".into());
    }
    if archive_selection.is_some() && !compact_states {
        return Err("SELECTED_ARCHIVE_REQUIRES_COMPACT_STATES".into());
    }
    let description = format!("{description};archive_selection={archive_selection:?}");
    let description = format!("{description};exact_packed_keys={:?}",std::env::var("MGBFS_EXACT_PACKED_KEYS").ok());

        let digest: [u8; 32] = Sha256::digest(description.as_bytes()).into();
        let archive_path = format!("{}-rank-{rank}.mgbfsar1", args[4]);
        let archive_enabled = crate::reference_launch::bench_archive_for_launch(
            std::env::var("MGBFS_BENCH_SKIP_ARCHIVE").ok().as_deref(),
            std::env::var("MGBFS_SEARCH_ONLY").as_deref() == Ok("1"),
        )?;
        let stream_archive = std::env::var("MGBFS_ARCHIVE_STREAM").as_deref() == Ok("1");
        let disk_bytes = if archive_enabled {
            ArchiveRingPlan::reference_output_limit(
                archive_width,
                expected_states,
                capacity,
                stream_archive,
            )?
        } else {
            0
        };
        let archive_rows = env_u32("MGBFS_ARCHIVE_ROWS", batch)?;
        let archive_slots = env_u32("MGBFS_ARCHIVE_SLOTS", 64)? as usize;
        selection.validate_archive_contract(
            archive_enabled,
            std::env::var("MGBFS_SEARCH_ONLY").as_deref() == Ok("1"),
        )?;
        let buckets = env_u32("MGBFS_BUCKETS", 256)?;
        let shards = env_u32("MGBFS_SHARDS", 64)?;
        let (local_buckets, _) =
            crate::topology::reference_owner_geometry(world, rank, &rank_map, buckets, shards)?;
        let default_bucket_capacity =
            crate::topology::reference_bucket_capacity(capacity, local_buckets, 4096)?;
        let cfg = DistributedConfig {
            route_banks: env_u32("MGBFS_ROUTE_BANKS", 2)? as usize,
            epoch_window: match std::env::var("MGBFS_EPOCH_WINDOW") {
                Ok(value) => crate::reference_launch::epoch_window_for_launch(Some(&value))?,
                Err(std::env::VarError::NotPresent) => {
                    crate::reference_launch::epoch_window_for_launch(None)?
                }
                Err(_) => return Err("ENV_MGBFS_EPOCH_WINDOW".into()),
            },
            untouched_vram_reserve: crate::reference_launch::vram_reserve_bytes(std::env::var("MGBFS_VRAM_RESERVE_BYTES").ok().as_deref())?,
            rank,
            world,
            logical_owner_to_rank: rank_map,
            transport: selection.transport,
            batch,
            layer_capacity: capacity,
            state_ring_capacity: future,
            state_descriptor_capacity: env_u32("MGBFS_STATE_DESCRIPTOR_CAPACITY", future)?,
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
            "archive_selection": archive_selection,
            "shard_ab_capacity": std::env::var("MGBFS_SHARD_AB_CAPACITY").ok(),
            "shard_ab_sort_slots": std::env::var("MGBFS_SHARD_AB_SORT_SLOTS").ok(),
            "shard_ab_shards": std::env::var("MGBFS_SHARD_AB_SHARDS").ok(),
            "inflight_batches": crate::reference_launch::inflight_batches(std::env::var("MGBFS_INFLIGHT_BATCHES").ok().as_deref())?,
            "graph_batches": crate::reference_launch::graph_batches(std::env::var("MGBFS_CUDA_GRAPH_BATCHES").ok().as_deref())?,
            "calibration_layers": std::env::var("MGBFS_CALIBRATION_LAYERS").ok(),
            "shard_key_first": std::env::var("MGBFS_SHARD_AB_KEY_FIRST").ok(),
            "compact_direct": std::env::var("MGBFS_COMPACT_DIRECT_HASH").ok(),
            "exact_packed_keys": std::env::var("MGBFS_EXACT_PACKED_KEYS").ok(),
            "peer_metadata": std::env::var("MGBFS_SHARD_AB_PEER_METADATA").ok(),
            "reuse_preowner_status": std::env::var("MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS").ok(),
            "combined_status": std::env::var("MGBFS_SHARD_AB_COMBINED_STATUS").ok(),
            "world": world, "buckets": cfg.buckets, "shards": cfg.shards,
            "job_buckets": cfg.job_buckets,
            "bucket_capacity_override": std::env::var("MGBFS_BUCKET_CAPACITY").ok(),
            "reserve": cfg.untouched_vram_reserve, "archive_rows": archive_rows,
            "epoch_window": cfg.epoch_window,
            "route_banks": cfg.route_banks,
            "state_descriptor_capacity": cfg.state_descriptor_capacity,
            "archive_slots": std::env::var("MGBFS_ARCHIVE_SLOTS").ok(),
            "stream_archive": stream_archive, "archive_enabled": archive_enabled,
            "warmup_requested": warmup_requested,
            "transport": format!("{:?}", cfg.transport),
        });
        let bootstrap_digest: [u8; 32] =
            Sha256::digest(serde_json::to_vec(&bootstrap_description).map_err(|e| e.to_string())?)
                .into();
        Ok(PreparedPass {
            macro_operators: None,
            multiset,
            group,
            graph,
            n,
            batch,
            declared_capacity,
            declared_future,
            mode,
            global_capacity: capacity_plan.global_records,
            global_future: future_plan.global_records,
            capacity,
            future,
            compact_states,
            archive_width,
            profile,
            owner,
            pre,
            seed,
            seed_hex,
            hash_first_generation,
            selection,
            digest,
            archive_path,
            archive_enabled,
            disk_bytes,
            archive_rows,
            archive_slots,
            stream_archive,
            calibration_layers: crate::reference_launch::calibration_layers(std::env::var("MGBFS_CALIBRATION_LAYERS").ok().as_deref())?,
            cfg,
            bootstrap_digest,
        })
    })();
    let device_admission = if prepared.is_ok() {
        let code = unsafe { cudaSetDevice(local as i32) };
        if code == 0 {
            crate::cuda_loading::verify_driver_before_allocations()
        } else {
            Err("CUDA_SET_DEVICE".to_string())
        }
    } else {
        Ok(())
    };
    let config_digest = prepared.as_ref().map_or([0; 32], |p| {
        crate::benchmark::phase_digest(
            p.bootstrap_digest,
            warmup_completed || !is_measure,
            is_measure,
        )
    });
    if control_group.agree_configuration(
        config_digest,
        prepared.is_err() || device_admission.is_err(),
        Duration::from_secs(60),
    )? {
        return Err(prepared
            .err()
            .or_else(|| device_admission.err())
            .unwrap_or_else(|| "REMOTE_CONFIGURATION_FATAL".into()));
    }
    let archive_selection = std::env::var("MGBFS_ARCHIVE_SELECTION").ok();
    let PreparedPass {
        macro_operators,
        multiset,
        group,
        graph,
        n,
        batch,
        declared_capacity,
        declared_future,
        mode,
        global_capacity,
        global_future,
        capacity,
        future,
        compact_states,
        archive_width,
        profile,
        owner,
        pre,
        seed,
        seed_hex,
        hash_first_generation,
        selection,
        digest,
        archive_path,
        archive_enabled,
        disk_bytes,
        archive_rows,
        archive_slots,
        stream_archive,
        calibration_layers,
        cfg,
        bootstrap_digest,
    } = prepared?;
    // Keep control sockets alive throughout this reference run. Dispatching GPU
    // epochs on them is a separate integration step, not claimed here.
    let id = control_group.nccl_id;
    // Allocated before archive admission; both the disk worker and existing
    // search sideband observe this one sticky failure signal.
    let search_failure = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
    let archive_setup = (|| -> Result<Option<PinnedArchive>> {
        #[cfg(debug_assertions)]
        if test_fault_rank("MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK", rank, world)? {
            return Err("TEST_INJECTED_ARCHIVE_ADMISSION_ERROR".into());
        }
        if archive_enabled {
            #[cfg(debug_assertions)]
            let write_fault =
                test_fault_rank("MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK", rank, world)?;
            #[cfg(debug_assertions)]
            let sync_fault =
                test_fault_rank("MGBFS_TEST_ARCHIVE_WORKER_SYNC_FAULT_RANK", rank, world)?;
            create_archive_extent(Path::new(&archive_path), stream_archive)
                .map_err(|e| format!("ARCHIVE_EXTENT: {e}"))
                .and_then(|extent| {
                    #[cfg(debug_assertions)]
                    let extent = archive_fault_extent(extent, write_fault, sync_fault);
                    PinnedArchive::new_with_failure_report(
                        extent,
                        disk_bytes,
                        archive_width,
                        digest,
                        archive_rows,
                        archive_slots,
                        Some(search_failure.clone()),
                    )
                })
                .map(Some)
        } else {
            Ok(None)
        }
    })();
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::ArchiveAdmission,
        archive_setup.is_err() || search_failure.load(std::sync::atomic::Ordering::Acquire) == 2,
        Duration::from_secs(600),
    )? {
        return Err(archive_setup
            .err()
            .unwrap_or_else(|| "REMOTE_ARCHIVE_ADMISSION_FATAL".into()));
    }
    let mut archive = archive_setup?;
    let pinned = archive.as_ref().map_or(0, |a| a.pinned_bytes());
    #[cfg(target_os="linux")]
    let _search_signals = crate::bootstrap::SearchSignals::install()?;
    let sideband = control_group
        .start_search_sideband_with_report(Duration::from_secs(7200), search_failure)?;
    let mut search_result = (|| -> Result<_> {
        #[cfg(debug_assertions)]
        if test_fault_rank("MGBFS_TEST_NCCL_STARTUP_FAULT_RANK", rank, world)? {
            return Err("TEST_INJECTED_NCCL_STARTUP_ERROR".into());
        }
        let setup = Instant::now();
        let mut bfs = if let Some(schedule) = &macro_operators {
            if selection.materialization_capacity.is_some() {
                return Err("WEIGHTED_HASH_FIRST_NOT_READY".into());
            }
            DistributedNativeBfs::new_weighted_with_owner_and_cancel(
                &graph,
                schedule,
                seed,
                id,
                cfg.clone(),
                selection.owner,
                selection.library_pool_bytes,
                selection.tile_limit,
                Some(sideband.cancel_token()),
                Some(sideband.failure_token()),
            )?
        } else if let Some(word) = &multiset {
            DistributedNativeBfs::new_lrx_multiset_reference_and_cancel(
                word,
                seed,
                id,
                cfg.clone(),
                selection.owner,
                selection.library_pool_bytes,
                Some(sideband.cancel_token()),
                Some(sideband.failure_token()),
            )?
        } else {
            match selection.owner {
                ReferenceOwner::CudfRelational
                | ReferenceOwner::CucoIndexed
                | ReferenceOwner::CucoRank => {
                    #[cfg(feature = "library-owner")]
                    {
                        DistributedNativeBfs::new_library_reference_with_owner_and_cancel(
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
                            Some(sideband.cancel_token()),
                            Some(sideband.failure_token()),
                        )?
                    }
                    #[cfg(not(feature = "library-owner"))]
                    {
                        return Err("REFERENCE_LIBRARY_NOT_COMPILED".into());
                    }
                }
                ReferenceOwner::Native(owner) => {
                    DistributedNativeBfs::new_reference_with_owner_and_cancel(
                        &graph,
                        seed,
                        id,
                        cfg.clone(),
                        selection.materialization_capacity,
                        owner,
                        selection.tile_limit,
                        selection.tensor_generation,
                        Some(sideband.cancel_token()),
                        Some(sideband.failure_token()),
                    )?
                }
            }
        };
        bfs.set_cancel_token(sideband.cancel_token())?;
        bfs.set_failure_token(sideband.failure_token());
        bfs.set_retirement_token(sideband.retirement_token())?;
        #[cfg(debug_assertions)]
        if test_fault_rank("MGBFS_TEST_OWNER_HOST_FAULT_RANK", rank, world)? {
            crate::distributed_native::inject_owner_host_error_once_for_test();
        }
        #[cfg(debug_assertions)]
        if test_fault_rank("MGBFS_TEST_MATERIALIZER_SORT_FATAL_RANK", rank, world)? {
            crate::distributed_native::inject_materializer_local_error_once_for_test(false);
        }
        #[cfg(debug_assertions)]
        if test_fault_rank("MGBFS_TEST_MATERIALIZER_REGENERATE_FATAL_RANK", rank, world)? {
            crate::distributed_native::inject_materializer_local_error_once_for_test(true);
        }
    let allocated = used()?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let trace = std::env::var_os("MGBFS_TRACE_DEPTHS").is_some();
    let profile_window = if is_measure { profiler_window_start()? } else { false };
    let start = Instant::now();
    let mut layers = Vec::new();
    let mut times = Vec::new();
    let mut calibration_stopped = false;
    loop {
        let tick = Instant::now();
        let depth = bfs.depth();
        let count = bfs.frontier_len();
        layers.push(count);
        if trace {
            let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?.as_secs_f64();
            std::io::Write::write_all(&mut std::io::stderr(),
                format!("MGBFS_DEPTH_BEGIN rank={rank} depth={depth} count={count} unix={unix:.6}\n").as_bytes())
                .map_err(|e| e.to_string())?;
        }
        let advance = if sideband.cancel_requested() {
            if let Some(archive)=archive.as_mut(){
                if archive.selected_whole || archive.selected_prefix {bfs.archive_selected_terminal_snapshot(archive)?;}
            }
            Err("REMOTE_SEARCH_CANCELLED".into())
        } else if let Some(archive) = archive.as_mut() {
            if archive.selected { bfs.advance_selected(archive) }
            else { bfs.advance_archived(archive) }
        } else {
            bfs.advance()
        };
        // Publish a local search failure before `bfs` is dropped and its
        // communicator is aborted. Peers need the sideband cancellation even
        // when their own GPU path has not yet observed the NCCL error.
        if let Err(error) = &advance {
            if archive.as_ref().is_some_and(|a|a.selected_whole || a.selected_prefix) {
                let unix=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .map_err(|e|e.to_string())?.as_secs_f64();
                // Keep failure-boundary metadata atomic on the shared pipe.
                std::io::Write::write_all(&mut std::io::stderr(),format!(
                    "MGBFS_DEPTH_END rank={rank} depth={depth} seconds={} next={} alive=false unix={unix:.6}\n",tick.elapsed().as_secs_f64(),bfs.frontier_len()).as_bytes())
                    .map_err(|e|e.to_string())?;
                if let Some(a)=archive.take(){if let Err(drain)=a.finish(){eprintln!("MGBFS_ARCHIVE_SNAPSHOT_ERROR {drain}");}}
            }
            sideband.report_failure();
            eprintln!("MGBFS_SELECTED_STOP {error}");
        }
        let alive = advance?;
        if trace && archive_enabled {
            eprintln!("MGBFS_ARCHIVE_SUBMITTED rank={rank} depth={depth} count={count}");
        }
        let elapsed = tick.elapsed().as_secs_f64();
        times.push(elapsed);
        if trace {
            let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?.as_secs_f64();
            std::io::Write::write_all(&mut std::io::stderr(),
                format!("MGBFS_DEPTH_END rank={rank} depth={depth} seconds={elapsed:.6} next={} alive={alive} unix={unix:.6}\n",bfs.frontier_len()).as_bytes())
                .map_err(|e| e.to_string())?;
        }
        if !alive {
            break;
        }
        if calibration_layers.is_some_and(|limit| layers.len() >= limit as usize) {
            calibration_stopped = true;
            break;
        }
    }
    let search = start.elapsed().as_secs_f64();
    profiler_window_stop(profile_window)?;
    Ok((bfs, allocated, setup_seconds, search, layers, times, start, calibration_stopped))
    })();
    // Query-only admission is a successful control outcome, not peer cancellation.
    let query_only = search_result.as_ref().err().is_some_and(|e| e == "MEMORY_QUERY_DONE");
    if search_result.is_err() && !query_only {
        sideband.report_failure();
        // The failed closure no longer owns a BFS: its abort/drop has returned.
        // Constructor errors can also arrive here before a transport reader exists.
        sideband.report_retired();
    } else {
        sideband.report_success();
    }
    let remote_failed = match sideband.finish_with_cleanup(&mut control_group, || {
        if let Ok((bfs, ..)) = &mut search_result {
            bfs.abort_group();
        }
    }) {
        Ok(failed) => failed,
        Err(error) => {
            if let Ok((bfs, ..)) = &mut search_result {
                bfs.abort_group();
            }
            return Err(error);
        }
    };
    if remote_failed {
        if let Ok((bfs, ..)) = &mut search_result {
            bfs.abort_group();
        }
        return Err(search_result
            .err()
            .unwrap_or_else(|| "REMOTE_SEARCH_FATAL".into()));
    }
    let (mut bfs, allocated, setup_seconds, search, layers, times, start, calibration_stopped) = search_result?;
    let archive_ram_ring = archive.as_ref().map(PinnedArchive::ring_stats);
    let archive_commit = archive.take().map_or(Ok(()), PinnedArchive::finish);
    #[cfg(debug_assertions)]
    let archive_commit = archive_commit.and_then(|()| {
        if test_fault_rank("MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK", rank, world)? {
            Err("TEST_INJECTED_ARCHIVE_FINISH_ERROR".into())
        } else {
            Ok(())
        }
    });
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::ArchiveCommitted,
        archive_commit.is_err(),
        Duration::from_secs(7200),
    )? {
        return Err(archive_commit
            .err()
            .unwrap_or_else(|| "REMOTE_ARCHIVE_COMMIT_FATAL".into()));
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
            value["archive_ram_ring"] = serde_json::json!(archive_ram_ring);
            if let Some([maximum,capacity,slots]) = bfs.shard_ab_geometry() {
                value["backend"] = serde_json::json!("native_shard_ab_hash_pipeline_v1");
                value["shard_ab"] = serde_json::json!({"maximum_shards":maximum,"buffer_capacity":capacity,"active_job_slots":slots});
            }
            value["batch_graph"] = serde_json::json!(bfs.batch_graph_stats()?);
            value["calibration_layers"] = serde_json::json!(calibration_layers);
            if calibration_stopped {
                value["status"] = serde_json::json!("INCOMPLETE");
                value["stop_reason"] = serde_json::json!("calibration layer limit");
                value["last_completed_layer"] = serde_json::json!(layers.len()-1);
                value["search_prefix_seconds"] = serde_json::json!(search);
                value.as_object_mut().ok_or("RECORD_JSON_OBJECT")?.remove("search_complete_seconds");
            }
            if matches!(archive_selection.as_deref(),Some("last_complete_small_1000" | "last_complete_prefix_1000")) {
                value["output_contract"] = serde_json::json!("selected_states_and_all_layer_counts");
                value["archive_wire_format"] = serde_json::json!("MGBFSAS2");
                value["archive_prefix_limit_per_rank"] = serde_json::json!(1000);
                value["archive_per_state_hashes"] = serde_json::json!(false);
            }
            if archive_selection.as_deref()==Some("all_states") {
                value["archive_wire_format"] = serde_json::json!("MGBFSAS3");
                value["archive_per_state_hashes"] = serde_json::json!(false);
            }
            value["archive_wire_limit_bytes"] = serde_json::json!(disk_bytes);
            value["disk_reserved_bytes"] =
                serde_json::json!(if stream_archive { 0 } else { disk_bytes });
            value["output_contract"] = serde_json::json!(if archive_enabled {
                if is_measure {
                    "archive_and_layer_counts"
                } else {
                    "warmup_layer_counts"
                }
            } else {
                "search_only_layer_counts"
            });
            value["transport_control_pinned_payload_bytes"] =
                serde_json::json!(bfs.transport_control_pinned_payload_bytes());
            value["pinned_bytes_scope"] = serde_json::json!("archive_only");
            if !archive_enabled || stream_archive || !is_measure {
                value["durable_run_commit_seconds"] = serde_json::Value::Null;
            }
            // Legacy consumers use durable_run_commit_seconds, but this timestamp
            // precedes rank-result fsync and group marker publication. State the
            // actual boundary explicitly without changing the old field's shape.
            value["archive_file_commit_seconds"] =
                if archive_enabled && !stream_archive && is_measure {
                    serde_json::json!(durable)
                } else {
                    serde_json::Value::Null
                };
            if stream_archive && archive_enabled {
                value["stream_handoff_seconds"] = serde_json::json!(durable);
            }
            value["archive_commit_scope"] = serde_json::json!(if !is_measure {
                "warmup_ephemeral"
            } else if !archive_enabled {
                "search_only"
            } else if stream_archive {
                "fifo_flush"
            } else {
                "file_fsync"
            });
            value["device_allocation_plan"] =
                crate::distributed_memory::allocation_report(bfs.owned_memory());
            value["hash_seed_hex"] = serde_json::json!(seed_hex);
            value["state_key_codec"] = serde_json::json!(if std::env::var("MGBFS_EXACT_PACKED_KEYS").ok().as_deref()==Some("1"){"lossless_bitpack128_feistel_v1"}else{"affine_fingerprint128_v1"});
            value["cuda_loading"] = crate::cuda_loading::evidence();
            value["bootstrap_digest"] = serde_json::json!(bootstrap_digest);
            value["logical_owner_to_rank"] = serde_json::json!(cfg.logical_owner_to_rank);
            value["transport_backend"] = serde_json::json!(format!("{:?}", cfg.transport));
            value["group"] = serde_json::json!(group);
            if let Some(word) = &multiset {
                value["graph_kind"] = serde_json::json!("lrx_multiset_schreier");
                value["start_state"] = serde_json::json!(word.start());
                value["expected_unique_states"] = serde_json::json!(word.order());
                value["expected_unique_states_u64_is_bound"] = serde_json::json!(!word.order_fits_u64());
            value["expected_unique_states_words_le"] = serde_json::json!(word.order_words());
            value["generators"] = serde_json::json!(["L", "R", "X"]);
            }
            value["cuda_memory_sampling"] = serde_json::json!("setup_and_final_only_not_full_peak");
            value["dense_lookahead_batches"] = serde_json::json!(bfs.dense_lookahead_batches());
            value["epoch_window"] = serde_json::json!(bfs.epoch_window());
            value["route_banks"] = serde_json::json!(bfs.route_bank_count());
            value["route_bank_reuses"] = serde_json::json!(bfs.route_bank_reuses());
            value["run_contract"] = serde_json::json!(if production {
                "RunConfigV1"
            } else {
                "reference_bench"
            });
            value["state_descriptor_capacity"] = serde_json::json!(bfs.state_descriptor_capacity());
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
            if let Some(label) =
                crate::benchmark::library_backend_label(selection.owner, selection.profile)
            {
                value["backend"] = serde_json::json!(label);
            }
            if matches!(archive_selection.as_deref(),Some("last_complete_small_1000" | "last_complete_prefix_1000")) { value["output_contract"] = serde_json::json!("selected_states_and_all_layer_counts"); }
            serde_json::to_vec(&value).map_err(|e| format!("RECORD_JSON: {e}"))?
        })?;
        if !is_measure && archive_enabled {
            // Warmup has no durable archive contract. Release its file while the
            // control group is alive so any local failure reaches every rank.
            std::fs::remove_file(&archive_path)
                .map_err(|e| format!("WARMUP_ARCHIVE_RELEASE: {e}"))?;
        }
        Ok(())
    })();
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::OutputWritten,
        output.is_err(),
        Duration::from_secs(60),
    )? {
        return Err(output
            .err()
            .unwrap_or_else(|| "REMOTE_OUTPUT_WRITE_FATAL".into()));
    }
    output?;
    let publication = if rank == 0 && is_measure {
        if calibration_stopped {
            crate::group_commit::write_calibration_commit(Path::new(&args[5]), world,
                bootstrap_digest,calibration_layers.ok_or("CALIBRATION_LIMIT_MISSING")?)
        } else {
            crate::group_commit::write_group_commit(Path::new(&args[5]), world, bootstrap_digest)
        }
    } else { Ok(()) };
    if control_group.agree_boundary(
        crate::bootstrap::BoundaryPhase::GroupPublished,
        publication.is_err(),
        Duration::from_secs(60),
    )? {
        return Err(publication
            .err()
            .unwrap_or_else(|| "REMOTE_GROUP_PUBLICATION_FATAL".into()));
    }
    publication?;
    if crate::session_cache::enabled() {
        let parked = bfs.session_parkable();
        let rejected = control_group.agree_boundary(
            crate::bootstrap::BoundaryPhase::SessionReuse, !parked,
            Duration::from_secs(60),
        )?;
        crate::session_cache::permit_comm(!rejected);
    }

    Ok(())
}

/// The existing weighted CUDA backend is single-rank. It remains separate
/// from the unit-cost NCCL runtime until distributed weighted settlement exists.
fn run_macro_pass(
    args: &[String],
    warmup_completed: bool,
    is_measure: bool,
    manifest: bool,
) -> Result<()> {
    crate::reference_launch::bench_warmup_for_launch(
        std::env::var("MGBFS_BENCH_WARMUP").ok().as_deref(),
        std::env::var("MGBFS_ARCHIVE_STREAM").ok().as_deref(),
    )?;
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
    if std::env::var("MGBFS_PROFILE")
        .as_deref()
        .is_ok_and(|x| x != "DENSE")
        || std::env::var("MGBFS_OWNER_BACKEND")
            .as_deref()
            .is_ok_and(|x| x != "CUB_SORT_MERGE")
    {
        return Err("MACRO_REFERENCE_DENSE_CUB_ONLY".into());
    }
    crate::cuda_loading::validate_requested()?;
    if unsafe { cudaSetDevice(local as i32) } != 0 {
        return Err("CUDA_SET_DEVICE".into());
    }
    crate::cuda_loading::verify_driver_before_allocations()?;
    let (group, graph) = if manifest {
        crate::reference_launch::load_matrix_manifest(Path::new(&args[1]))?
    } else {
        MatrixGroup::from_reference_label(&args[1])?
    };
    let batch: u32 = args[2].parse().map_err(|_| "BATCH")?;
    let capacity = match std::env::var("MGBFS_BENCH_CAPACITY") {
        Ok(value) => value.parse().map_err(|_| "CAPACITY")?,
        Err(std::env::VarError::NotPresent) => graph
            .expected_max_unique_states
            .try_into()
            .map_err(|_| "CAPACITY_EXPLICIT_REQUIRED")?,
        Err(_) => return Err("CAPACITY".into()),
    };
    let future = env_u32("MGBFS_FUTURE_CAPACITY", capacity)?;
    let compact = match std::env::var("MGBFS_STATE_CODEC").as_deref() {
        Ok("permutation_u8") => true,
        Ok("matrix_u8") | Err(_) => false,
        _ => return Err("STATE_CODEC".into()),
    };
    let generation_variant = if compact { 5 } else { 1 };
    if manifest && compact {
        return Err("MATRIX_MANIFEST_REQUIRES_MATRIX_CODEC".into());
    }
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
        ArchiveRingPlan::reference_output_limit(
            layout.width,
            graph.expected_max_unique_states,
            capacity,
            stream_archive,
        )?
    } else {
        0
    };
    let archive_path = format!("{}-rank-0.mgbfsar1", args[4]);
    let mut archive = if archive_enabled {
        let extent = create_archive_extent(Path::new(&archive_path), stream_archive)
            .map_err(|e| format!("ARCHIVE_EXTENT: {e}"))?;
        Some(PinnedArchive::new(
            extent,
            disk_bytes,
            layout.width,
            digest,
            archive_rows,
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
    let profile_window = if is_measure {
        profiler_window_start()?
    } else {
        false
    };
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
    let archive_commit = archive.take().map_or(Ok(()), PinnedArchive::finish);
    #[cfg(debug_assertions)]
    let archive_commit = archive_commit.and_then(|()| {
        if test_fault_rank("MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK", rank, world)? {
            Err("TEST_INJECTED_ARCHIVE_FINISH_ERROR".into())
        } else {
            Ok(())
        }
    });
    archive_commit?;
    let durable = start.elapsed().as_secs_f64();
    std::fs::create_dir_all(&args[5]).map_err(|e| e.to_string())?;
    let record = serde_json::json!({
        "status": "COMPLETE", "backend": "macro_native_single_rank_v1",
        "rank": 0, "world_size": 1, "group": group, "batch": batch,
        "macro_depth": macro_depth, "frontier_profile": "DENSE",
        "owner_backend": "CUB_SORT_MERGE", "pre_dedup": if prededup { "ON" } else { "OFF" },
        "hash_seed_hex": seed_hex, "generation_variant": generation_variant,
        "archive_enabled": archive_enabled, "archive_state_bytes": layout.width,
        "output_contract": if !is_measure { "warmup_layer_counts" }
            else if archive_enabled { "archive_and_layer_counts" } else { "search_only_layer_counts" },
        "search_complete_seconds": search,
        "durable_run_commit_seconds": if is_measure && archive_enabled { Some(durable) } else { None },
        "setup_seconds": setup_seconds, "local_layer_sizes": layers,
        "per_depth_seconds": times, "declared_capacity_records": capacity,
        "future_capacity_per_depth": future,
        "explicit_device_aligned_bytes": bfs.requested_device_bytes(),
        "cuda_allocated_used_bytes": allocated,
        "cuda_peak_observed_bytes": used()?.max(allocated),
        "cuda_memory_sampling": "setup_and_final_only_not_full_peak",
        "pinned_bytes": pinned, "disk_reserved_bytes": if stream_archive { 0 } else { disk_bytes },
        "archive_wire_limit_bytes": disk_bytes,
        "warmup_completed": warmup_completed,
        "bootstrap_digest": digest,
        "archive_commit_scope": if !is_measure { "warmup_ephemeral" }
            else if !archive_enabled { "search_only" }
            else if stream_archive { "fifo_flush" } else { "file_fsync" },
    });
    let output = Path::new(&args[5]);
    crate::group_commit::write_rank_result(
        output,
        0,
        &serde_json::to_vec(&record).map_err(|e| e.to_string())?,
    )?;
    if !is_measure && archive_enabled {
        std::fs::remove_file(&archive_path)
            .map_err(|error| format!("WARMUP_ARCHIVE_RELEASE: {error}"))?;
    }
    if is_measure {
        crate::group_commit::write_group_commit(output, 1, digest)?;
    }
    Ok(())
}
/// Shared reference benchmark entry point. Argument zero is the launcher name;
/// the remaining arguments are group, batch, bootstrap, archive and output.
/// This does not implement the production RunConfigV1 dispatcher.
pub fn run(args: Vec<String>) -> Result<()> {
    run_source(args, false, false)
}
/// Manifest input uses the same rank admission, owner, transport and archive.
/// This is still the benchmark contract, not the RunConfigV1 dispatcher.
pub fn run_manifest(args: Vec<String>) -> Result<()> {
    run_source(args, true, false)
}
/// Typed configuration enters the same admission, cancellation, GPU pipeline
/// and group publication as bench. Unsupported config contracts fail in the
/// cross-rank preparation vote, before communicator/archive construction.
pub fn run_config(
    config: String,
    bootstrap: String,
    archive: String,
    output: String,
) -> Result<()> {
    run_source(
        vec![
            "mgbfs-run".into(),
            config,
            "unused".into(),
            bootstrap,
            archive,
            output,
        ],
        true,
        true,
    )
}
fn run_source(args: Vec<String>, manifest: bool, production: bool) -> Result<()> {
    use crate::benchmark::{run_phases, Phase};
    if args.len() != 6 {
        return Err("ARGS_group_batch_bootstrap_archive_prefix_output_dir".into());
    }
    // Invalid local values still enter the same first rendezvous. The
    // prepared configuration vote reports the error to every rank.
    let warmup = crate::reference_launch::bench_warmup_for_launch(
        std::env::var("MGBFS_BENCH_WARMUP").ok().as_deref(),
        std::env::var("MGBFS_ARCHIVE_STREAM").ok().as_deref(),
    )
    .unwrap_or(false);
    run_phases(warmup, |phase| {
        let phase = if phase == Phase::Measure {
            crate::reference_launch::BenchPhase::Measure
        } else {
            crate::reference_launch::BenchPhase::Warmup
        };
        let paths = crate::reference_launch::bench_phase_paths(
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
            warmup,
            phase,
        )?;
        run_pass(
            &paths,
            warmup && phase == crate::reference_launch::BenchPhase::Measure,
            phase == crate::reference_launch::BenchPhase::Measure,
            manifest,
            production,
        )
    })
}
