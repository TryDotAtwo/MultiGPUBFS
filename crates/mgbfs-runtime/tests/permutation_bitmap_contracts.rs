use mgbfs_runtime::permutation_bitmap::{BitmapPartition, MultisetDomain};

// Lexicographic permutation enumeration is independent of rank/unrank.
fn next_permutation<T: Ord>(a: &mut [T]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let Some(i) = (0..a.len() - 1).rev().find(|&i| a[i] < a[i + 1]) else {
        return false;
    };
    let j = (i + 1..a.len()).rev().find(|&j| a[i] < a[j]).unwrap();
    a.swap(i, j);
    a[i + 1..].reverse();
    true
}

#[test]
fn exhaustive_lexicographic_roundtrips() {
    for start in [
        vec![0],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![0, 0, 1, 2, 2, 3],
        vec![-7, -7, 0, 99, 99],
        vec![4, 4, 4, 4],
    ] {
        let domain = MultisetDomain::new(&start).unwrap();
        let mut state = start.clone();
        state.sort_unstable();
        let mut rank = 0u64;
        loop {
            assert_eq!(domain.rank(&state).unwrap(), rank);
            assert_eq!(domain.unrank(rank).unwrap(), state);
            rank += 1;
            if !next_permutation(&mut state) {
                break;
            }
        }
        assert_eq!(rank, domain.size());
        assert!(domain.unrank(rank).is_err());
    }
}
#[test]
fn rejects_unrepresentable_or_nonpermutations() {
    assert!(MultisetDomain::new(&[]).is_err());
    assert!(MultisetDomain::new(&(0..21).collect::<Vec<_>>()).is_err()); // 21! > u64
    assert!(MultisetDomain::new(&vec![0; 65]).is_err());
    let d = MultisetDomain::new(&[-9, -9, 0, 7]).unwrap();
    assert!(d.rank(&[-9, 0, 0, 7]).is_err());
    assert!(d.rank(&[-9, -9, 0, 7, 8]).is_err());
}
#[test]
fn checked_large_ranks_and_repeated_symbols() {
    let d = MultisetDomain::new(&(0..20).collect::<Vec<_>>()).unwrap();
    assert_eq!(d.size(), 2432902008176640000);
    for r in [0, 1, 4294967313, d.size() - 1] {
        assert_eq!(d.rank(&d.unrank(r).unwrap()).unwrap(), r)
    }
    let mut a = vec![i64::MIN; 63];
    a.push(i64::MAX);
    let d = MultisetDomain::new(&a).unwrap();
    assert_eq!(d.size(), 64);
    for r in 0..64 {
        assert_eq!(d.rank(&d.unrank(r).unwrap()).unwrap(), r);
    }
}
#[test]
fn rank_partition_is_bijective_and_uses_bounded_bits() {
    for world in [1, 2, 3, 8, 128] {
        for total in [1u64, 15, 129, 1_000_003, u64::MAX] {
            let p = BitmapPartition::new(total, world).unwrap();
            let mut sum = 0u128;
            for owner in 0..world {
                let count = p.owner_len(owner).unwrap();
                sum += u128::from(count);
                assert_eq!(
                    p.owner_bytes(owner).unwrap(),
                    (count / 64 + u64::from(count % 64 != 0)) * 8
                );
            }
            assert_eq!(sum, u128::from(total));
            for rank in [0, total / 2, total - 1] {
                let (owner, local) = p.locate(rank).unwrap();
                assert!(local < p.owner_len(owner).unwrap());
                assert_eq!(p.reconstruct(owner, local).unwrap(), rank);
            }
            assert!(p.locate(total).is_err());
        }
    }
    assert!(BitmapPartition::new(0, 1).is_err());
    assert!(BitmapPartition::new(1, 0).is_err());
}
