use mgbfs_runtime::reference_launch::macro_depth_for_launch;

#[test]
fn calibration_limit_is_optional_positive_and_not_graph_completion() {
    use mgbfs_runtime::reference_launch::calibration_layers;
    assert_eq!(calibration_layers(None).unwrap(),None);
    assert_eq!(calibration_layers(Some("34")).unwrap(),Some(34));
    for value in ["0","-1","bad","4294967296"] {
        assert_eq!(calibration_layers(Some(value)).unwrap_err(),"ENV_MGBFS_CALIBRATION_LAYERS");
    }
}

#[test]
fn reserve_is_explicit_with_driver_headroom_and_rejects_invalid_bytes() {
    use mgbfs_runtime::reference_launch::vram_reserve_bytes as reserve;
    assert_eq!(reserve(None).unwrap(), 1 << 30);
    assert_eq!(reserve(Some("268435456")).unwrap(), 256 << 20);
    assert_eq!(reserve(Some("67108864")).unwrap(), 64 << 20);
    for value in ["0", "1", "67108863", "-1", "bad", "18446744073709551616"] {
        assert_eq!(reserve(Some(value)).unwrap_err(), "ENV_MGBFS_VRAM_RESERVE_BYTES");
    }
}

#[test]
fn full_graph_window_policy_is_explicit_and_cluster_agreed() {
    use mgbfs_runtime::reference_launch::graph_batches;
    assert_eq!(graph_batches(None).unwrap(), 0);
    assert_eq!(graph_batches(Some("0")).unwrap(), 0);
    assert_eq!(graph_batches(Some("32")).unwrap(), 32);
    for value in ["1", "2", "31", "33", "-1", "bad", ""] {
        assert_eq!(graph_batches(Some(value)).unwrap_err(), "ENV_MGBFS_CUDA_GRAPH_BATCHES");
    }
}

#[test]
fn graph_window_archive_credits_cover_preparation_and_all_inflight_windows() {
    use mgbfs_runtime::reference_launch::graph_archive_credits as credits;
    assert_eq!(credits(1000, 32768, 32768, 2).unwrap(), 96);
    assert_eq!(credits(1000, 32768, 8192, 2).unwrap(), 384);
    assert_eq!(credits(3, 32768, 32768, 2).unwrap(), 9);
    assert_eq!(credits(1000, 9, 4, 1).unwrap(), 192);
    for args in [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)] {
        assert_eq!(credits(args.0,args.1,args.2,args.3).unwrap_err(), "GRAPH_ARCHIVE_CREDIT_SHAPE");
    }
    assert!(credits(u32::MAX,u32::MAX,1,usize::MAX).is_err());
}

#[test]
fn bounded_device_epoch_queue_preserves_legacy_default_and_accepts_32() {
    use mgbfs_runtime::reference_launch::inflight_batches;
    assert_eq!(inflight_batches(None).unwrap(),2);
    for value in 1..=32 {
        assert_eq!(inflight_batches(Some(&value.to_string())).unwrap(),value);
    }
    for value in ["0","33","-1","bad","999999999999999999999999999"] {
        assert_eq!(inflight_batches(Some(value)).unwrap_err(),"ENV_MGBFS_INFLIGHT_BATCHES");
    }
}
use mgbfs_runtime::reference_launch::{bench_archive_for_launch, bench_phase_paths, bench_warmup_for_launch, BenchPhase};

#[test]
fn multi_rank_macro_depth_is_rejected_before_path_dispatch() {
    assert_eq!(macro_depth_for_launch(Some("0"), 1).unwrap_err(),
               "ENV_MGBFS_MACRO_DEPTH");
    assert_eq!(macro_depth_for_launch(Some("0"), 2).unwrap_err(),
               "ENV_MGBFS_MACRO_DEPTH");
    assert_eq!(macro_depth_for_launch(Some("bad"), 2).unwrap_err(),
               "ENV_MGBFS_MACRO_DEPTH");
    assert_eq!(macro_depth_for_launch(Some("2"), 2).unwrap_err(),
               "MACRO_MULTI_GPU_UNSUPPORTED");
    assert_eq!(macro_depth_for_launch(Some("10"), 2).unwrap_err(),
               "MACRO_MULTI_GPU_UNSUPPORTED");
    assert!(!macro_depth_for_launch(None, 2).unwrap());
    assert!(!macro_depth_for_launch(Some("1"), 2).unwrap());
    assert!(macro_depth_for_launch(Some("2"), 1).unwrap());
}

#[test]
fn warmup_and_non_warmup_share_first_rendezvous_path() {
    let args = ["mgbfs", "s4", "16", "bootstrap", "archive", "results"];
    let warm = bench_phase_paths(&args, true, BenchPhase::Warmup).unwrap();
    let direct = bench_phase_paths(&args, false, BenchPhase::Measure).unwrap();
    assert_eq!(warm[3], "bootstrap");
    assert_eq!(direct[3], "bootstrap");
    assert_eq!(warm[4], "archive.warmup");
    assert_eq!(warm[5], "results.warmup");
    let measured = bench_phase_paths(&args, true, BenchPhase::Measure).unwrap();
    assert_eq!(measured[3], "bootstrap.measure");
    assert_eq!(measured[4], "archive");
    assert_eq!(measured[5], "results");
}

#[test]
fn warmup_configuration_rejects_invalid_and_streamed_warmup() {
    assert!(!bench_warmup_for_launch(None, None).unwrap());
    assert!(!bench_warmup_for_launch(Some("0"), Some("1")).unwrap());
    assert!(bench_warmup_for_launch(Some("1"), Some("0")).unwrap());
    assert_eq!(bench_warmup_for_launch(Some("bad"), None).unwrap_err(),
               "BENCH_WARMUP_CONFIG");
    assert_eq!(bench_warmup_for_launch(Some("1"), Some("1")).unwrap_err(),
               "BENCH_WARMUP_REQUIRES_FILE_ARCHIVE");
}

#[test]
fn archive_is_disabled_only_by_explicit_search_only() {
    assert!(bench_archive_for_launch(None, false).unwrap());
    assert!(bench_archive_for_launch(Some("0"), false).unwrap());
    assert!(!bench_archive_for_launch(Some("1"), true).unwrap());
    assert_eq!(bench_archive_for_launch(Some("1"), false).unwrap_err(),
               "CLI_BENCH_ARCHIVE_REQUIRED");
    assert_eq!(bench_archive_for_launch(Some("bad"), false).unwrap_err(),
               "CLI_BENCH_ARCHIVE_REQUIRED");
}
