use mgbfs_runtime::reference_launch::macro_depth_for_launch;
use mgbfs_runtime::reference_launch::{bench_archive_for_launch, bench_phase_paths, bench_warmup_for_launch, BenchPhase};

#[test]
fn multi_rank_macro_depth_is_rejected_before_path_dispatch() {
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
