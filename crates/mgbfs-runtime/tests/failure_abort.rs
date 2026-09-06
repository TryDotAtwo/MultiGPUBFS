use mgbfs_runtime::failure::abort_on_error;

#[test]
fn error_poisoning_runs_terminal_cleanup_and_preserves_original_error() {
    let mut failed = false;
    let mut cleanups = 0;
    let result: Result<(), &str> = abort_on_error(Err("original"), &mut failed, || cleanups += 1);
    assert_eq!(result, Err("original"));
    assert!(failed);
    assert_eq!(cleanups, 1);
}

#[test]
fn success_preserves_value_without_poisoning_or_cleanup() {
    let mut failed = false;
    let result = abort_on_error::<_, &str>(Ok(42), &mut failed, || panic!("unexpected abort"));
    assert_eq!(result, Ok(42));
    assert!(!failed);
}
