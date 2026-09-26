use mgbfs_runtime::reference_launch::macro_depth_for_launch;

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
