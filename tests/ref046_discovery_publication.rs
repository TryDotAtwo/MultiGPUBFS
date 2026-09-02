use multigpubfs_gpu::discovery_publication::{
    enumerate_chain_single_stop_schedules, enumerate_single_stop_schedules, Protocol,
};

#[test]
fn blind_drop_has_schedules_with_visited_but_unpublished_state() {
    let outcomes = enumerate_single_stop_schedules(Protocol::BlindDrop);

    assert_eq!(outcomes.len(), 6);
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| outcome.visited && !outcome.published)
            .count(),
        3
    );
}

#[test]
fn helpable_descriptor_preserves_publication_in_every_single_stop_schedule() {
    let outcomes = enumerate_single_stop_schedules(Protocol::HelpableDescriptor);

    assert_eq!(outcomes.len(), 6);
    assert!(outcomes
        .iter()
        .all(|outcome| outcome.visited && outcome.published));
}

#[test]
fn logged_intent_preserves_publication_in_every_single_stop_schedule() {
    let outcomes = enumerate_single_stop_schedules(Protocol::LoggedIntent);

    assert_eq!(outcomes.len(), 6);
    assert!(outcomes
        .iter()
        .all(|outcome| outcome.visited && outcome.published));
}

#[test]
fn idempotent_set_commit_hides_duplicate_physical_publication_attempts() {
    for protocol in [Protocol::HelpableDescriptor, Protocol::LoggedIntent] {
        let outcomes = enumerate_single_stop_schedules(protocol);

        assert!(outcomes.iter().any(|outcome| {
            outcome.publication_attempts == 2 && outcome.unique_publications == 1
        }));
        assert!(outcomes
            .iter()
            .all(|outcome| outcome.unique_publications <= 1));
    }
}

#[test]
fn one_interrupted_publication_anywhere_on_three_edge_path_has_declared_coverage() {
    let blind = enumerate_chain_single_stop_schedules(Protocol::BlindDrop, 3);
    let helpable = enumerate_chain_single_stop_schedules(Protocol::HelpableDescriptor, 3);
    let logged = enumerate_chain_single_stop_schedules(Protocol::LoggedIntent, 3);

    assert_eq!(blind.len(), 18);
    assert_eq!(
        blind
            .iter()
            .filter(|outcome| outcome.target_reached)
            .count(),
        9
    );
    assert!(helpable.iter().all(|outcome| outcome.target_reached));
    assert!(logged.iter().all(|outcome| outcome.target_reached));
}
