use mgbfs_runtime::transport::{Kind::*, Transport};

#[test]
fn candidate_metadata_cannot_exhaust_response_ticket_capacity() {
    let mut t = Transport::new(2, 1, 8).unwrap();
    for slot in 0..2 {
        t.offer(Candidate, 0, slot, vec![0, 0]).unwrap();
        let x = t.issue().unwrap().unwrap();
        ack(&mut t, x.seq);
    }
    t.offer(Candidate, 0, 2, vec![0, 0]).unwrap();
    assert!(t.issue().unwrap().is_none());
    t.offer(Response, 1, 0, vec![1, 0]).unwrap();
    assert_eq!(t.issue().unwrap().unwrap().kind, Response);
}

#[test]
fn depth_rotation_preserves_total_sequence_and_requires_all_final_acks() {
    let mut t = Transport::new(2, 1, 8).unwrap();
    assert!(t.advance_depth().is_err());
    t.close_source(0).unwrap();
    t.close_source(1).unwrap();
    let f = t.issue().unwrap().unwrap();
    t.complete(0, f.seq).unwrap();
    assert!(t.advance_depth().is_err());
    t.complete(1, f.seq).unwrap();
    t.advance_depth().unwrap();
    assert!(!t.finished());
    t.offer(Candidate, 0, 0, vec![1, 0]).unwrap();
    let c = t.issue().unwrap().unwrap();
    assert_eq!(c.seq, 1);
    ack(&mut t, c.seq);
    t.consume(c.seq).unwrap();
    t.close_source(0).unwrap();
    t.close_source(1).unwrap();
    let f = t.issue().unwrap().unwrap();
    assert_eq!(f.seq, 2);
    ack(&mut t, f.seq);
    assert!(t.finished());
}
fn ack(t: &mut Transport, seq: u64) {
    t.complete(1, seq).unwrap();
    t.complete(0, seq).unwrap();
}

#[test]
fn all_kinds_share_one_order_with_multiple_inflight_tickets() {
    let mut t = Transport::new(2, 4, 16).unwrap();
    for (kind, slot) in [
        (Candidate, 10),
        (Request, 11),
        (Response, 12),
        (Receipt, 13),
    ] {
        t.offer(kind, 0, slot, vec![0, 4]).unwrap();
    }
    let mut tickets = vec![];
    for expected in [Response, Request, Receipt, Candidate] {
        let x = t.issue().unwrap().unwrap();
        assert_eq!(x.kind, expected);
        assert_eq!(x.seq, tickets.len() as u64);
        tickets.push(x);
    }
    assert!(t.complete(0, 1).is_err()); // per-rank comm stream order
    for x in tickets {
        ack(&mut t, x.seq);
        t.consume(x.seq).unwrap();
    }
    t.close_source(0).unwrap();
    t.close_source(1).unwrap();
    let f = t.issue().unwrap().unwrap();
    assert_eq!((f.seq, f.kind), (4, Finalize));
    t.complete(0, f.seq).unwrap();
    assert!(!t.finished());
    t.complete(1, f.seq).unwrap();
    assert!(t.finished());
}

#[test]
fn receive_credit_is_held_until_consumed_not_just_transfer_complete() {
    let mut t = Transport::new(2, 1, 8).unwrap();
    t.offer(Candidate, 0, 1, vec![0, 8]).unwrap();
    let a = t.issue().unwrap().unwrap();
    ack(&mut t, a.seq);
    t.offer(Candidate, 0, 2, vec![0, 8]).unwrap();
    assert!(t.issue().unwrap().is_none());
    // Separate response credits permit progress while candidate receive is full.
    t.offer(Response, 1, 1, vec![8, 0]).unwrap();
    let b = t.issue().unwrap().unwrap();
    assert_eq!(b.kind, Response);
    ack(&mut t, b.seq);
    t.consume(b.seq).unwrap();
    t.consume(a.seq).unwrap();
    assert_eq!(t.issue().unwrap().unwrap().slot, 2);
}

#[test]
fn closed_source_may_serve_materialization_but_cannot_generate_more_candidates() {
    let mut t = Transport::new(2, 2, 8).unwrap();
    t.work(0, true).unwrap();
    t.close_source(0).unwrap();
    t.close_source(1).unwrap();
    assert!(t.offer(Candidate, 0, 0, vec![0, 1]).is_err());
    assert!(t.issue().unwrap().is_none());
    t.offer(Response, 0, 0, vec![0, 1]).unwrap();
    let x = t.issue().unwrap().unwrap();
    ack(&mut t, x.seq);
    assert!(t.issue().unwrap().is_none());
    t.consume(x.seq).unwrap();
    assert!(t.issue().unwrap().is_none());
    t.work(0, false).unwrap();
    assert_eq!(t.issue().unwrap().unwrap().kind, Finalize);
    assert!(t.offer(Receipt, 0, 1, vec![0, 1]).is_err());
}

#[test]
fn malformed_offers_do_not_consume_slots_or_sequence_numbers() {
    let mut t = Transport::new(2, 1, 8).unwrap();
    assert!(t.offer(Candidate, 2, 0, vec![1, 0]).is_err());
    assert!(t.offer(Candidate, 0, 0, vec![1]).is_err());
    assert!(t.offer(Candidate, 0, 0, vec![0, 9]).is_err());
    assert!(t.offer(Candidate, 0, 0, vec![5, 5]).is_err());
    assert!(t.offer(Finalize, 0, 0, vec![0, 0]).is_err());
    t.offer(Candidate, 0, 7, vec![0, 1]).unwrap();
    assert!(t.offer(Candidate, 0, 7, vec![0, 1]).is_err());
    assert!(t.offer(Candidate, 0, 8, vec![0, 1]).is_err());
    let a = t.issue().unwrap().unwrap();
    assert_eq!(a.seq, 0);
    assert!(t.consume(a.seq).is_err());
    assert!(t.complete(2, a.seq).is_err());
    t.complete(0, a.seq).unwrap();
    assert!(t.complete(0, a.seq).is_err());
    t.complete(1, a.seq).unwrap();
    t.consume(a.seq).unwrap();
    assert!(t.consume(a.seq).is_err());
    assert!(t.work(0, false).is_err());
}

#[test]
fn round_robin_sources_and_empty_rank_participation() {
    let mut t = Transport::new(2, 3, 8).unwrap();
    t.offer(Candidate, 0, 1, vec![0, 1]).unwrap();
    t.offer(Candidate, 0, 2, vec![0, 1]).unwrap();
    t.offer(Candidate, 1, 1, vec![0, 0]).unwrap();
    let a = t.issue().unwrap().unwrap();
    let b = t.issue().unwrap().unwrap();
    assert_eq!((a.source, b.source), (0, 1));
    ack(&mut t, a.seq);
    ack(&mut t, b.seq);
    t.consume(a.seq).unwrap();
    t.consume(b.seq).unwrap();
    assert_eq!(t.issue().unwrap().unwrap().source, 0);
}

#[test]
fn macro_candidate_tickets_preserve_target_depth_across_ready_reordering() {
    let mut t = Transport::new(2, 3, 8).unwrap();
    t.offer_at(Candidate, 3, 0, 10, vec![0, 2]).unwrap();
    t.offer_at(Candidate, 1, 1, 11, vec![2, 0]).unwrap();
    let first = t.issue().unwrap().unwrap();
    let second = t.issue().unwrap().unwrap();
    assert_eq!((first.source, first.target_depth), (0, 3));
    assert_eq!((second.source, second.target_depth), (1, 1));
    ack(&mut t, first.seq);
    ack(&mut t, second.seq);
    t.consume(first.seq).unwrap();
    t.consume(second.seq).unwrap();
    assert!(t.offer_at(Candidate, 0, 0, 12, vec![0, 1]).is_err());
}
