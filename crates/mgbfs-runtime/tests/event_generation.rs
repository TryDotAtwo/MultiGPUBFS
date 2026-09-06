use mgbfs_runtime::event_generation::EventGeneration;
#[test]
fn native_query_status_distinguishes_not_ready_from_async_error() {
    use mgbfs_runtime::event_generation::cuda_query_status;
    assert_eq!(cuda_query_status(0).unwrap(), true);
    assert_eq!(cuda_query_status(600).unwrap(), false);
    for code in [1, 400, 700, 719, -1] {
        assert!(cuda_query_status(code).is_err());
    }
}
#[test]
fn unrecorded_event_never_queries_cuda_empty_work_success() {
    let mut event = EventGeneration::default();
    let mut queried = false;
    assert!(event
        .poll(0, || {
            queried = true;
            Ok(true)
        })
        .is_err());
    assert!(!queried);
    assert!(event.record(0, || Ok(())).is_err());
}
#[test]
fn live_event_cannot_be_overwritten_even_after_it_becomes_ready() {
    let mut event = EventGeneration::default();
    event.record(0, || Ok(())).unwrap();
    assert!(event.poll(0, || Ok(true)).unwrap());
    let mut submitted = false;
    assert!(event
        .record(1, || {
            submitted = true;
            Ok(())
        })
        .is_err());
    assert!(!submitted);
}
#[test]
fn generation_reuse_requires_completion_and_explicit_retirement() {
    let mut event = EventGeneration::default();
    event.record(7, || Ok(())).unwrap();
    assert!(!event.poll(7, || Ok(false)).unwrap());
    assert!(event.poll(7, || Ok(true)).unwrap());
    event.retire(7).unwrap();
    event.record(8, || Ok(())).unwrap();
    assert!(!event.poll(8, || Ok(false)).unwrap());
    assert!(event.poll(7, || Ok(true)).is_err());
    assert!(event.poll(8, || Ok(true)).is_err());
}
#[test]
fn record_and_query_errors_are_terminal() {
    let mut event = EventGeneration::default();
    assert!(event
        .record(0, || Err("CUDA_RECORD_FAILURE".into()))
        .is_err());
    assert!(event.poll(0, || Ok(true)).is_err());
    let mut event = EventGeneration::default();
    event.record(0, || Ok(())).unwrap();
    assert!(event.poll(0, || Err("CUDA_QUERY_FAILURE".into())).is_err());
    assert!(event.poll(0, || Ok(true)).is_err());
    let mut event = EventGeneration::default();
    event.record(0, || Ok(())).unwrap();
    assert!(event.retire(0).is_err());
}
