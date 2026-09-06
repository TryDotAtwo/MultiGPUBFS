use mgbfs_runtime::{
    control_wire::{Action, ControlFrame, Plane, NO_SLOT},
    rank_epochs::RankEpochs,
};
fn begin(epoch: u64, slot: u64, plane: Plane) -> ControlFrame {
    ControlFrame {
        action: Action::Begin,
        rank: 0,
        depth: 0,
        epoch,
        slot,
        plane,
        source_rank: 0,
        fatal_code: 0,
    }
}
#[test]
fn finalize_requires_drain_and_preserves_sequence_across_depths() {
    let finalize = ControlFrame {
        action: Action::Finalize,
        plane: Plane::None,
        ..begin(1, NO_SLOT, Plane::Candidate)
    };
    let mut rank = RankEpochs::new(2, 1, 1).unwrap();
    rank.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    rank.transfer_complete(0).unwrap();
    rank.consume(0).unwrap();
    rank.finish_depth(finalize, true).unwrap();
    rank.publish(ControlFrame {
        action: Action::Publish,
        depth: 1,
        ..finalize
    })
    .unwrap();
    assert_eq!(rank.offer(Plane::Candidate, 0).unwrap().depth, 1);
    rank.begin(ControlFrame {
        depth: 1,
        ..begin(2, 0, Plane::Candidate)
    })
    .unwrap();
    for drained in [false, true] {
        let mut rank = RankEpochs::new(2, 1, 1).unwrap();
        rank.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
        rank.transfer_complete(0).unwrap();
        if !drained {
            rank.consume(0).unwrap();
        }
        assert!(rank.finish_depth(finalize, drained).is_err());
    }
}
#[test]
fn begin_pins_offered_slot_not_latest_ready_and_holds_it_until_consumed() {
    let mut rank = RankEpochs::new(2, 1, 2).unwrap();
    rank.offer(Plane::Candidate, 7).unwrap();
    rank.offer(Plane::Candidate, 9).unwrap();
    rank.begin(begin(0, 7, Plane::Candidate)).unwrap();
    rank.begin(begin(1, 9, Plane::Candidate)).unwrap();
    rank.transfer_complete(0).unwrap();
    let done = rank.consume(0).unwrap();
    assert_eq!(
        (done.action, done.rank, done.epoch, done.plane),
        (Action::Consumed, 1, 0, Plane::Candidate)
    );
    rank.offer(Plane::Candidate, 7).unwrap();
    rank.begin(begin(2, 7, Plane::Candidate)).unwrap();
}
#[test]
fn all_planes_share_sequence_and_empty_offers_hold_receive_credit() {
    let mut rank = RankEpochs::new(2, 1, 1).unwrap();
    rank.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    rank.offer(Plane::Response, 3).unwrap();
    rank.begin(begin(1, 3, Plane::Response)).unwrap();
    assert!(rank.begin(begin(2, NO_SLOT, Plane::Candidate)).is_err());
    assert!(
        rank.consume(0).is_err(),
        "invalid admission must poison session"
    );
}
#[test]
fn unoffered_stale_wrong_depth_and_reused_slots_are_terminal() {
    for bad in [
        begin(0, 4, Plane::Candidate),
        begin(1, 3, Plane::Candidate),
        ControlFrame {
            depth: 1,
            ..begin(0, 3, Plane::Candidate)
        },
    ] {
        let mut rank = RankEpochs::new(2, 1, 2).unwrap();
        rank.offer(Plane::Candidate, 3).unwrap();
        assert!(rank.begin(bad).is_err());
        assert!(rank.offer(Plane::Response, 8).is_err());
    }
    let mut rank = RankEpochs::new(2, 1, 2).unwrap();
    rank.offer(Plane::Candidate, 3).unwrap();
    rank.begin(begin(0, 3, Plane::Candidate)).unwrap();
    assert!(rank.offer(Plane::Candidate, 3).is_err());
}

#[test]
fn fixed_capacity_invalid_topology_and_duplicate_retirement_fail_closed() {
    for (world, rank, slots) in [(0, 0, 1), (2, 2, 1), (2, 1, 0), (2, 1, usize::MAX)] {
        assert!(RankEpochs::new(world, rank, slots).is_err());
    }
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs.offer(Plane::Candidate, 0).unwrap();
    assert!(epochs.offer(Plane::Candidate, 1).is_err());
    assert!(epochs.begin(begin(0, 0, Plane::Candidate)).is_err());
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    epochs.transfer_complete(0).unwrap();
    epochs.consume(0).unwrap();
    assert!(epochs.consume(0).is_err());
    assert!(epochs.offer(Plane::Response, 0).is_err());
}

#[test]
fn consumers_cannot_retire_before_transfer_completion() {
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    assert!(epochs.consume(0).is_err());
}

#[test]
fn ordered_transfer_completions_allow_independent_consumer_retirement() {
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    epochs.begin(begin(1, NO_SLOT, Plane::Response)).unwrap();
    assert_eq!(
        epochs.transfer_complete(0).unwrap().action,
        Action::Complete
    );
    epochs.transfer_complete(1).unwrap();
    assert_eq!(epochs.consume(1).unwrap().action, Action::Consumed);
    epochs.begin(begin(2, NO_SLOT, Plane::Response)).unwrap();
    epochs.consume(0).unwrap();
    epochs.begin(begin(3, NO_SLOT, Plane::Candidate)).unwrap();
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs.begin(begin(0, NO_SLOT, Plane::Candidate)).unwrap();
    epochs.begin(begin(1, NO_SLOT, Plane::Response)).unwrap();
    assert!(epochs.transfer_complete(1).is_err());
}

#[test]
fn local_finalization_does_not_authorize_next_depth_before_publication() {
    let mut epochs = RankEpochs::new(2, 1, 1).unwrap();
    epochs
        .finish_depth(
            ControlFrame {
                action: Action::Finalize,
                plane: Plane::None,
                ..begin(0, NO_SLOT, Plane::Candidate)
            },
            true,
        )
        .unwrap();
    assert!(epochs.offer(Plane::Candidate, 7).is_err());
}
