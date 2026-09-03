use mgbfs_runtime::{receipts::BatchReceipts, ring::StateRing};

fn permutations(items: &mut [usize], at: usize, visit: &mut impl FnMut(&[usize])) {
    if at == items.len() {
        visit(items);
        return;
    }
    for i in at..items.len() {
        items.swap(at, i);
        permutations(items, at + 1, visit);
        items.swap(at, i);
    }
}

#[test]
fn all_receipt_response_archive_orders_preserve_parent_until_last_obligation() {
    let mut schedules = 0;
    permutations(&mut [0, 1, 2, 3, 4, 5], 0, &mut |order| {
        let mut book = BatchReceipts::new(&[3, 2, 0]).unwrap();
        let mut ring = StateRing::new(8, 4).unwrap();
        let p = ring.reserve(8).unwrap();
        ring.materialized(p.id).unwrap();
        ring.publish(p.id).unwrap();
        ring.hold_origins(p.id).unwrap();
        ring.enumerated(p.id).unwrap();
        let mut released = false;
        for (position, &event) in order.iter().enumerate() {
            match event {
                0 => book.receipt(0, 3, 2).unwrap(),
                1 => book.receipt(1, 2, 1).unwrap(),
                2 => book.served(0, 11).unwrap(),
                3 => book.served(0, 12).unwrap(),
                4 => book.served(1, 21).unwrap(),
                5 => ring.archived(p.id).unwrap(),
                _ => unreachable!(),
            }
            if book.closed() && !released {
                ring.release_origins(p.id).unwrap();
                released = true;
            }
            assert_eq!(
                ring.reclaim(),
                if position == 5 { 8 } else { 0 },
                "schedule {order:?}"
            );
        }
        assert!(book.closed());
        schedules += 1;
    });
    assert_eq!(schedules, 720);
}

#[test]
fn zero_survivors_still_require_terminal_receipt() {
    let mut book = BatchReceipts::new(&[7, 0]).unwrap();
    assert!(!book.closed());
    book.receipt(0, 7, 0).unwrap();
    assert!(book.closed());
    assert!(BatchReceipts::new(&[0, 0]).unwrap().closed());
}

#[test]
fn corrupt_duplicate_or_excess_messages_poison_the_batch() {
    let mut b = BatchReceipts::new(&[2]).unwrap();
    b.served(0, 5).unwrap();
    assert!(b.served(0, 5).is_err());
    assert!(b.receipt(0, 2, 1).is_err());
    assert!(!b.closed());
    let mut b = BatchReceipts::new(&[2]).unwrap();
    b.served(0, 5).unwrap();
    assert!(b.receipt(0, 2, 0).is_err());
    assert!(!b.closed());
    for (owner, emitted, accepted) in [(1, 2, 1), (0, 3, 1), (0, 2, 3)] {
        let mut b = BatchReceipts::new(&[2]).unwrap();
        assert!(b.receipt(owner, emitted, accepted).is_err());
        assert!(!b.closed());
    }
    let mut b = BatchReceipts::new(&[2]).unwrap();
    b.receipt(0, 2, 0).unwrap();
    assert!(b.receipt(0, 2, 0).is_err());
    assert!(!b.closed());
    let mut b = BatchReceipts::new(&[1]).unwrap();
    b.receipt(0, 1, 0).unwrap();
    assert!(b.served(0, 9).is_err());
    assert!(!b.closed());
}
