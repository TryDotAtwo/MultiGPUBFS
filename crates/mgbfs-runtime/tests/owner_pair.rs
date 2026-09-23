use mgbfs_runtime::failure::{process_owner_pair, vote_group_error};
use std::cell::RefCell;

#[test]
fn local_work_does_not_wait_for_remote_but_remote_never_reads_before_ready() {
    let events = RefCell::new(Vec::new());
    process_owner_pair(
        "local",
        "remote",
        |value| {
            events.borrow_mut().push(value);
            Ok::<_, &str>(())
        },
        || {
            events.borrow_mut().push("ready");
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(*events.borrow(), ["local", "ready", "remote"]);
}

#[test]
fn local_failure_still_establishes_dependency_and_preserves_first_error() {
    let events = RefCell::new(Vec::new());
    let result = process_owner_pair(
        "local",
        "remote",
        |value| {
            events.borrow_mut().push(value);
            Err(value)
        },
        || {
            events.borrow_mut().push("ready");
            Ok(())
        },
    );
    assert_eq!(result, Err("local"));
    assert_eq!(*events.borrow(), ["local", "ready", "remote"]);
}

#[test]
fn failed_readiness_does_not_expose_remote_payload() {
    for local_error in [false, true] {
        let events = RefCell::new(Vec::new());
        let result = process_owner_pair(
            "local",
            "remote",
            |value| {
                events.borrow_mut().push(value);
                if local_error {
                    Err("local")
                } else {
                    Ok(())
                }
            },
            || Err("ready"),
        );
        assert_eq!(result, Err(if local_error { "local" } else { "ready" }));
        assert_eq!(*events.borrow(), ["local"]);
    }
}

#[test]
fn local_retirement_failure_still_enters_group_vote_before_owner_work() {
    let events = RefCell::new(Vec::new());
    let result = vote_group_error(
        Err("local_retirement"),
        |failed| {
            events.borrow_mut().push(if failed { "vote_failed" } else { "vote_clear" });
            Ok(true)
        },
        "remote_retirement",
    );
    assert_eq!(result, Err("local_retirement"));
    assert_eq!(*events.borrow(), ["vote_failed"]);

    let result = vote_group_error(
        Ok::<(), &str>(()),
        |failed| {
            events.borrow_mut().push(if failed { "vote_failed" } else { "vote_clear" });
            Ok(true)
        },
        "remote_retirement",
    );
    assert_eq!(result, Err("remote_retirement"));
    assert_eq!(*events.borrow(), ["vote_failed", "vote_clear"]);
}
