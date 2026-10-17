use mgbfs_runtime::native_fault;

#[test]
fn failed_call_carries_rank_device_target_layer_and_last_completed_layer() {
    native_fault::reset(Some(5), Some(5));
    native_fault::layer(7, 6);
    native_fault::begin("cudaMemcpyAsync", "ffi.rs", 42);
    let error = native_fault::finish(999).unwrap_err();
    assert!(error.starts_with("GENERIC_CUDA_999"));
    let details = native_fault::snapshot();
    assert_eq!(details["native_error"]["call"], "cudaMemcpyAsync");
    assert_eq!(details["native_error"]["rank"], 5);
    assert_eq!(details["native_error"]["device"], 5);
    assert_eq!(details["native_error"]["layer"], 7);
    assert_eq!(details["native_error"]["last_completed_layer"], 6);
    assert!(error.contains("cudaMemcpyAsync") && error.contains("layer=7"));
}

#[test]
fn success_and_new_job_do_not_inherit_a_previous_failure() {
    native_fault::reset(Some(0), Some(2));
    native_fault::begin("mgbfs_nccl_all_reduce_max_u32", "transport.rs", 10);
    assert!(native_fault::finish(0).is_ok());
    assert!(native_fault::snapshot()["native_error"].is_null());
    assert!(native_fault::finish(999).is_err());
    native_fault::reset(Some(1), Some(3));
    let snapshot = native_fault::snapshot();
    assert!(snapshot["native_error"].is_null());
    assert!(snapshot["layer"].is_null());
    assert_eq!(snapshot["rank"], 1);
}

#[test]
fn separate_threads_have_separate_call_context() {
    native_fault::reset(Some(1), Some(1));
    native_fault::layer(4, 3);
    std::thread::spawn(|| {
        native_fault::reset(Some(7), Some(7));
        native_fault::layer(9, 8);
        assert_eq!(native_fault::snapshot()["layer"], 9);
    }).join().unwrap();
    assert_eq!(native_fault::snapshot()["layer"], 4);
    assert_eq!(native_fault::snapshot()["rank"], 1);
}

#[test]
fn teardown_does_not_replace_the_primary_native_error() {
    native_fault::reset(Some(1), Some(5));
    native_fault::layer(2, 1);
    native_fault::begin("cudaMemcpyAsync", "advance.rs", 275);
    assert!(native_fault::finish(999).is_err());
    let first = native_fault::snapshot()["native_error"].clone();
    for line in 1..=32 {
        native_fault::begin("cudaDeviceSynchronize", "cleanup.rs", line);
        assert!(native_fault::finish(999).is_err());
    }
    let snapshot = native_fault::snapshot();
    assert_eq!(snapshot["native_error"], first);
    assert_eq!(snapshot["native_secondary_error_count"], 32);
    assert_eq!(snapshot["native_last_error"]["call"], "cudaDeviceSynchronize");
    native_fault::begin("cleanup_success", "cleanup.rs", 40);
    assert!(native_fault::finish(0).is_ok());
    assert_eq!(native_fault::snapshot()["native_error"], first);
    native_fault::reset(Some(0), Some(0));
    assert!(native_fault::snapshot()["native_error"].is_null());
    assert!(native_fault::snapshot()["native_last_error"].is_null());
    assert_eq!(native_fault::snapshot()["native_secondary_error_count"], 0);
}
