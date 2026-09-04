use mgbfs_runtime::failure::attempt_all;

#[test]
fn later_rank_safe_work_is_attempted_after_the_first_local_error() {
    let mut attempted = Vec::new();
    let error = attempt_all(0..2, |lane| {
        attempted.push(lane);
        if lane == 0 {
            Err("LOCAL_CAPACITY")
        } else {
            Ok(())
        }
    });

    assert_eq!(attempted, vec![0, 1]);
    assert_eq!(error, Err("LOCAL_CAPACITY"));
}

#[test]
fn the_first_error_is_preserved_after_all_safe_work_is_attempted() {
    let error = attempt_all(0..3, |lane| match lane {
        0 => Err("FIRST"),
        1 => Err("SECOND"),
        _ => Ok(()),
    });

    assert_eq!(error, Err("FIRST"));
}
