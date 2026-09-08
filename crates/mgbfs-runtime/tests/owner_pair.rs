use mgbfs_runtime::failure::process_owner_pair;
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
