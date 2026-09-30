use mgbfs_runtime::failure::report_and_abort_on_error;
use std::sync::{atomic::{AtomicU8, Ordering}, mpsc};
use std::time::Duration;

#[test]
fn peer_observes_archive_failure_before_cleanup_can_finish() {
    let token = AtomicU8::new(0);
    let mut failed = false;
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let peer_token = &token;
        let peer = scope.spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while peer_token.load(Ordering::Acquire) != 2 {
                assert!(std::time::Instant::now() < deadline, "failure was hidden behind cleanup");
                std::thread::yield_now();
            }
            tx.send(()).unwrap();
        });
        let result: Result<(), &str> = report_and_abort_on_error(
            Err("ARCHIVE_D2H"), &mut failed, Some(&token),
            || rx.recv_timeout(Duration::from_secs(3)).expect("peer cannot cancel before cleanup"));
        assert_eq!(result, Err("ARCHIVE_D2H"));
        peer.join().unwrap();
    });
    assert!(failed);
}

#[test]
fn successful_archive_submission_does_not_signal_or_cleanup() {
    let token = AtomicU8::new(0);
    let mut failed = false;
    assert_eq!(report_and_abort_on_error::<_, &str>(Ok(7), &mut failed, Some(&token),
        || panic!("healthy work must retain its consumers")), Ok(7));
    assert_eq!(token.load(Ordering::Acquire), 0);
    assert!(!failed);
}

#[test]
fn absent_sideband_still_poison_and_preserve_originating_error() {
    let mut failed = false;
    let mut cleaned = false;
    let result: Result<(), &str> = report_and_abort_on_error(
        Err("original"), &mut failed, None, || cleaned = true);
    assert_eq!(result, Err("original"));
    assert!(failed && cleaned);
}

#[test]
fn previously_poisoned_rank_still_reports_and_runs_dispatcher_cleanup() {
    let token = AtomicU8::new(0);
    let mut failed = true;
    let mut cleanup_calls = 0;
    let result: Result<(), &str> = report_and_abort_on_error(
        Err("OWNER_API"), &mut failed, Some(&token), || {
            assert_eq!(token.load(Ordering::Acquire), 2);
            cleanup_calls += 1;
        });
    assert_eq!(result, Err("OWNER_API"));
    assert_eq!(cleanup_calls, 1);
    assert!(failed);
}
