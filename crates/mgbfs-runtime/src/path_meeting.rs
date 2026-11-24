use mgbfs_core::Result;

pub(crate) type LayerRange = (u32, u32);

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Meeting {
    pub(crate) left_row: u32,
    pub(crate) right_row: u32,
    pub(crate) right_depth: usize,
}

pub(crate) fn decode_meeting(
    best: u64,
    left: LayerRange,
    right_layers: &[LayerRange],
) -> Result<Option<Meeting>> {
    if best == u64::MAX {
        return Ok(None);
    }
    let left_row = best as u32;
    let right_row = (best >> 32) as u32;
    if !layer_contains(left, left_row) {
        return Err("NATIVE_PATH_MEET_LEFT_BOUNDS".into());
    }
    // Each rank's append-only ranges are ordered, with empty layers allowed.
    let right_depth = right_layers.partition_point(|&(start, count)| {
        u64::from(start) + u64::from(count) <= u64::from(right_row)
    });
    if !right_layers
        .get(right_depth)
        .is_some_and(|&range| layer_contains(range, right_row))
    {
        return Err("NATIVE_PATH_MEET_RIGHT_BOUNDS".into());
    }
    Ok(Some(Meeting {
        left_row,
        right_row,
        right_depth,
    }))
}

fn layer_contains((start, count): LayerRange, row: u32) -> bool {
    row >= start && row - start < count
}

pub(crate) fn partition_roots(
    roots: &[Vec<i64>],
    width: usize,
    world: u32,
    rank: u32,
    owner: impl Fn(&[i64]) -> u32,
) -> Result<(Vec<u32>, Vec<&[i64]>)> {
    if roots.is_empty()
        || roots.len() > u32::MAX as usize
        || roots.iter().any(|row| row.len() != width)
    {
        return Err("GENERIC_MULTI_ROOT_SHAPE".into());
    }
    let mut counts = vec![0; world as usize];
    let mut local = Vec::new();
    let mut seen = std::collections::HashSet::with_capacity(roots.len());
    for row in roots.iter().map(Vec::as_slice) {
        if seen.insert(row) {
            let destination = owner(row);
            *counts
                .get_mut(destination as usize)
                .ok_or("GENERIC_MULTI_ROOT_OWNER")? += 1;
            if destination == rank {
                local.push(row);
            }
        }
    }
    Ok((counts, local))
}

#[cfg(test)]
mod meeting_tests {
    use super::*;

    #[test]
    fn reverse_roots_preserve_exact_order_and_drop_duplicates_across_empty_ranks() {
        let goals = vec![vec![7, -1], vec![2, -1], vec![7, -1], vec![4, 1]];
        let (counts, local) =
            partition_roots(&goals, 2, 3, 0, |row| if row[0] == 7 { 2 } else { 0 }).unwrap();
        assert_eq!(counts, [2, 0, 1]);
        assert_eq!(local, [&[2, -1][..], &[4, 1][..]]);
        assert!(partition_roots(&goals, 1, 3, 0, |_| 0).is_err());
    }

    #[test]
    fn meeting_depth_uses_committed_rows_and_skips_empty_layers() {
        let prior = [(0, 0), (0, 2), (2, 0), (2, 3), (5, 0)];
        let pair = |left: u32, right: u32| (u64::from(right) << 32) | u64::from(left);
        assert_eq!(decode_meeting(u64::MAX, (9, 2), &prior).unwrap(), None);
        assert_eq!(
            decode_meeting(pair(10, 2), (9, 2), &prior).unwrap(),
            Some(Meeting {
                left_row: 10,
                right_row: 2,
                right_depth: 3
            })
        );
        assert!(decode_meeting(pair(11, 2), (9, 2), &prior).is_err());
        assert!(decode_meeting(pair(9, 5), (9, 2), &prior).is_err());
    }
}
