use mgbfs_core::{hash::Hash128, wire::{validate_macro_candidate_payload, payload_layout, FrameHeader, FrameKind, MacroCandidateRef}};
use mgbfs_runtime::{macro_owner::{CandidateKey, FutureOffer, MacroOwner}, transport::{Kind, Transport}};

fn deliver(
    transport: &mut Transport,
    owner: &mut MacroOwner,
    source: usize,
    slot: u64,
    target: u32,
    weight: u32,
    state_ref: u64,
    hash: Hash128,
) {
    transport.offer_at(Kind::Candidate, target, source, slot, vec![0, 1]).unwrap();
    let ticket = transport.issue().unwrap().unwrap();
    assert_eq!((ticket.target_depth, ticket.source), (target, source));
    let header = FrameHeader {
        kind: FrameKind::MacroDense,
        run_tag: 1,
        sequence: ticket.seq,
        batch: slot,
        depth: target,
        source: source as u32,
        destination: 1,
        count: 1,
    };
    let layout = payload_layout(header.kind, 1, 16).unwrap();
    let mut payload = vec![0; layout.bytes as usize];
    payload[..16].copy_from_slice(&hash.to_le_bytes());
    let reference = MacroCandidateRef { source_depth: 0, weight, state_ref };
    let begin = layout.planes[1].offset as usize;
    payload[begin..begin + 16].copy_from_slice(&reference.encode());
    for rank in 0..2 {
        transport.complete(rank, ticket.seq).unwrap();
    }
    validate_macro_candidate_payload(&payload, &header, 16, 3).unwrap();
    owner.offer(FutureOffer::new(target, hash, state_ref,
        CandidateKey::new(0, weight, source as u32, slot, 0))).unwrap();
    transport.consume(ticket.seq).unwrap();
}

fn finalize(transport: &mut Transport) {
    transport.close_source(0).unwrap();
    transport.close_source(1).unwrap();
    let ticket = transport.issue().unwrap().unwrap();
    assert_eq!(ticket.kind, Kind::Finalize);
    transport.complete(0, ticket.seq).unwrap();
    transport.complete(1, ticket.seq).unwrap();
    transport.advance_depth().unwrap();
}

#[test]
fn weighted_frames_keep_distant_offers_provisional_across_rank_depth_finalization() {
    let hash = Hash128([7, 8, 9, 10]);
    let mut transport = Transport::new_macro(2, 2, 8, 3).unwrap();
    let mut owner = MacroOwner::new(3, 8, 8).unwrap();
    owner.seed(0, []).unwrap();
    // The earlier delivery has a *longer* route from source depth zero.
    deliver(&mut transport, &mut owner, 0, 0, 3, 3, 30, hash);
    deliver(&mut transport, &mut owner, 1, 1, 1, 1, 10, hash);
    finalize(&mut transport);
    assert_eq!(owner.settle(1).unwrap(), vec![(hash, 10)]);
    finalize(&mut transport);
    assert!(owner.settle(2).unwrap().is_empty());
    finalize(&mut transport);
    assert!(owner.settle(3).unwrap().is_empty());
}
