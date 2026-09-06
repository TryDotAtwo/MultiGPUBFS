use mgbfs_runtime::{
    control_wire::{Action, ControlFrame, Plane, NO_SLOT},
    epoch_coordinator::EpochCoordinator,
};
fn ready(rank: u32, plane: Plane, slot: u64) -> ControlFrame {
    ControlFrame {
        action: Action::Ready,
        rank,
        depth: 0,
        epoch: 0,
        slot,
        plane,
        source_rank: 0,
        fatal_code: 0,
        destination_rank: 0,
        payload_bytes: 0,
    }
}
fn ack(rank: u32, plane: Plane, epoch: u64, action: Action) -> ControlFrame {
    ControlFrame {
        action,
        epoch,
        slot: NO_SLOT,
        ..ready(rank, plane, 0)
    }
}
#[test]
fn response_priority_empty_ranks_and_one_global_order() {
    let mut coordinator = EpochCoordinator::new(2, 2).unwrap();
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator.receive(ready(1, Plane::Response, 9)).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(
        (
            frames[0].epoch,
            frames[0].plane,
            frames[0].slot,
            frames[1].slot
        ),
        (0, Plane::Response, NO_SLOT, 9)
    );
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(
        (
            frames[0].epoch,
            frames[0].plane,
            frames[0].slot,
            frames[1].slot
        ),
        (1, Plane::Candidate, 7, NO_SLOT)
    );
    assert!(!coordinator.issue(&mut frames).unwrap());
}
#[test]
fn credit_stays_until_every_rank_consumes_but_other_planes_progress() {
    let mut coordinator = EpochCoordinator::new(2, 1).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator.issue(&mut frames).unwrap();
    for rank in 0..2 {
        coordinator
            .receive(ack(rank, Plane::Candidate, 0, Action::Complete))
            .unwrap();
    }
    coordinator
        .receive(ack(0, Plane::Candidate, 0, Action::Consumed))
        .unwrap();
    coordinator.receive(ready(0, Plane::Candidate, 8)).unwrap();
    assert!(!coordinator.issue(&mut frames).unwrap());
    coordinator.receive(ready(1, Plane::Response, 9)).unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(frames[0].plane, Plane::Response);
    coordinator
        .receive(ack(1, Plane::Candidate, 0, Action::Consumed))
        .unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!((frames[0].epoch, frames[0].slot), (2, 8));
}
#[test]
fn wrong_ack_plane_poisoning_prevents_following_issuance() {
    let mut coordinator = EpochCoordinator::new(2, 1).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator.issue(&mut frames).unwrap();
    assert!(coordinator
        .receive(ack(0, Plane::Response, 0, Action::Complete))
        .is_err());
    assert!(coordinator.issue(&mut frames).is_err());
}

#[test]
fn one_source_per_ticket_bounds_aggregate_input_and_round_robins_sources() {
    let mut coordinator = EpochCoordinator::new(2, 2).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator.receive(ready(0, Plane::Candidate, 8)).unwrap();
    coordinator.receive(ready(1, Plane::Candidate, 9)).unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!((frames[0].slot, frames[1].slot), (7, NO_SLOT));
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!((frames[0].slot, frames[1].slot), (NO_SLOT, 9));
}

#[test]
fn source_close_does_not_discard_pending_candidates_or_block_responses() {
    let mut coordinator = EpochCoordinator::new(2, 2).unwrap();
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator
        .receive(ack(0, Plane::None, 0, Action::SourceClosed))
        .unwrap();
    coordinator.receive(ready(0, Plane::Response, 8)).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(frames[0].plane, Plane::Response);
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!((frames[0].plane, frames[0].slot), (Plane::Candidate, 7));
    assert!(coordinator.receive(ready(0, Plane::Candidate, 9)).is_err());
    assert!(coordinator.issue(&mut frames).is_err());
}

#[test]
fn duplicate_or_wrong_depth_source_close_is_terminal() {
    for wrong_depth in [false, true] {
        let mut coordinator = EpochCoordinator::new(2, 1).unwrap();
        let close = ack(1, Plane::None, 0, Action::SourceClosed);
        if !wrong_depth {
            coordinator.receive(close).unwrap();
        }
        assert!(coordinator
            .receive(ControlFrame {
                depth: u64::from(wrong_depth),
                ..close
            })
            .is_err());
        assert!(coordinator.receive(ready(0, Plane::Response, 0)).is_err());
    }
}

#[test]
fn finalization_waits_for_all_sources_and_consumer_retirement() {
    let mut coordinator = EpochCoordinator::new(2, 1).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    coordinator.receive(ready(0, Plane::Candidate, 7)).unwrap();
    coordinator.issue(&mut frames).unwrap();
    for rank in 0..2 {
        coordinator
            .receive(ack(rank, Plane::None, 0, Action::SourceClosed))
            .unwrap();
        coordinator
            .receive(ack(rank, Plane::Candidate, 0, Action::Complete))
            .unwrap();
    }
    assert!(!coordinator.issue(&mut frames).unwrap());
    coordinator
        .receive(ack(0, Plane::Candidate, 0, Action::Consumed))
        .unwrap();
    assert!(!coordinator.issue(&mut frames).unwrap());
    coordinator
        .receive(ack(1, Plane::Candidate, 0, Action::Consumed))
        .unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!((frames[0].action, frames[0].epoch), (Action::Finalize, 1));
    assert_eq!(frames[0], frames[1]);
    assert!(coordinator.receive(ready(1, Plane::Response, 9)).is_err());
}

#[test]
fn all_finalization_acks_precede_publication_and_next_depth_data() {
    let mut coordinator = EpochCoordinator::new(2, 1).unwrap();
    let mut frames = [ready(0, Plane::Candidate, 0); 2];
    for rank in 0..2 {
        coordinator
            .receive(ack(rank, Plane::None, 0, Action::SourceClosed))
            .unwrap();
    }
    assert!(coordinator.issue(&mut frames).unwrap());
    let finalize = frames[0];
    coordinator
        .receive(ControlFrame {
            action: Action::Finalized,
            rank: 1,
            ..finalize
        })
        .unwrap();
    assert!(!coordinator.issue(&mut frames).unwrap());
    coordinator
        .receive(ControlFrame {
            action: Action::Finalized,
            rank: 0,
            ..finalize
        })
        .unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(
        (frames[0].action, frames[0].depth, frames[0].epoch),
        (Action::Publish, 1, 0)
    );
    coordinator
        .receive(ControlFrame {
            depth: 1,
            ..ready(0, Plane::Candidate, 7)
        })
        .unwrap();
    assert!(coordinator.issue(&mut frames).unwrap());
    assert_eq!(
        (frames[0].action, frames[0].depth, frames[0].epoch),
        (Action::Begin, 1, 1)
    );
}
#[test]
fn three_rank_begin_identifies_nonzero_source_to_every_receiver() {
    let mut coordinator = mgbfs_runtime::epoch_coordinator::EpochCoordinator::new(3, 2).unwrap();
    let offer = mgbfs_runtime::control_wire::ControlFrame {
        action: mgbfs_runtime::control_wire::Action::Ready,
        rank: 2,
        depth: 0,
        epoch: 0,
        slot: 17,
        plane: mgbfs_runtime::control_wire::Plane::Candidate,
        source_rank: 0,
        fatal_code: 0,
        destination_rank: 0,
        payload_bytes: 0,
    };
    coordinator.receive(offer).unwrap();
    let mut commands = [offer; 3];
    assert!(coordinator.issue(&mut commands).unwrap());
    for (rank, command) in commands.iter().enumerate() {
        assert_eq!(command.source_rank, 2);
        assert_eq!(command.slot, if rank == 2 { 17 } else { u64::MAX });
    }
}
