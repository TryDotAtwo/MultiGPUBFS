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
        fatal_code: 0,
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
