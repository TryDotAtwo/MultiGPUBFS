use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Debug, Eq, PartialEq)]
struct LevelCounts {
    depth: usize,
    frontier: usize,
    occurrences: usize,
    parent_returns: usize,
    visited_nonparent: usize,
    candidate_occurrences: usize,
    unique_candidates: usize,
    convergence_duplicates: usize,
    accepted_next: usize,
}

impl LevelCounts {
    fn new(
        depth: usize,
        frontier: usize,
        occurrences: usize,
        parent_returns: usize,
        visited_nonparent: usize,
        candidate_occurrences: usize,
        unique_candidates: usize,
        convergence_duplicates: usize,
        accepted_next: usize,
    ) -> Self {
        Self {
            depth,
            frontier,
            occurrences,
            parent_returns,
            visited_nonparent,
            candidate_occurrences,
            unique_candidates,
            convergence_duplicates,
            accepted_next,
        }
    }
}

fn probe<S, F>(root: S, successors: F) -> Vec<LevelCounts>
where
    S: Clone + Eq + Hash,
    F: Fn(&S) -> Vec<S>,
{
    let mut visited = HashSet::from([root.clone()]);
    let mut parent: HashMap<S, S> = HashMap::new();
    let mut frontier = vec![root];
    let mut rows = Vec::new();

    for depth in 0.. {
        let mut occurrences = 0;
        let mut parent_returns = 0;
        let mut visited_nonparent = 0;
        let mut candidate_occurrences = 0;
        let mut next_seen = HashSet::new();
        let mut next_frontier = Vec::new();

        for state in &frontier {
            for child in successors(state) {
                occurrences += 1;
                if parent.get(state) == Some(&child) {
                    parent_returns += 1;
                } else if visited.contains(&child) {
                    visited_nonparent += 1;
                } else {
                    candidate_occurrences += 1;
                    if next_seen.insert(child.clone()) {
                        parent.insert(child.clone(), state.clone());
                        next_frontier.push(child);
                    }
                }
            }
        }

        let unique_candidates = next_frontier.len();
        rows.push(LevelCounts::new(
            depth,
            frontier.len(),
            occurrences,
            parent_returns,
            visited_nonparent,
            candidate_occurrences,
            unique_candidates,
            candidate_occurrences - unique_candidates,
            unique_candidates,
        ));
        if next_frontier.is_empty() {
            break;
        }
        visited.extend(next_frontier.iter().cloned());
        frontier = next_frontier;
    }
    rows
}

fn z31_rows() -> Vec<LevelCounts> {
    probe(0_i32, |state| {
        vec![(state + 1).rem_euclid(31), (state - 1).rem_euclid(31)]
    })
}

fn z8_square_rows() -> Vec<LevelCounts> {
    probe((0_i32, 0_i32), |&(x, y)| {
        vec![
            ((x + 1).rem_euclid(8), y),
            ((x - 1).rem_euclid(8), y),
            (x, (y + 1).rem_euclid(8)),
            (x, (y - 1).rem_euclid(8)),
        ]
    })
}

fn s3_rows() -> Vec<LevelCounts> {
    probe([0_u8, 1, 2], |state| {
        (0..2)
            .map(|move_index| {
                let mut child = *state;
                child.swap(move_index, move_index + 1);
                child
            })
            .collect()
    })
}

fn nonbacktracking_words(degree: usize, depth: usize) -> usize {
    if depth == 0 {
        1
    } else {
        degree * (degree - 1).pow((depth - 1) as u32)
    }
}

fn print_rows(name: &str, degree: usize, rows: &[LevelCounts]) {
    for row in rows {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            name,
            row.depth,
            nonbacktracking_words(degree, row.depth),
            row.frontier,
            row.occurrences,
            row.parent_returns,
            row.visited_nonparent,
            row.candidate_occurrences,
            row.unique_candidates,
            row.convergence_duplicates,
            row.accepted_next,
        );
    }
}

fn main() {
    println!("graph,depth,nonbacktracking_words,frontier,occurrences,parent_returns,visited_nonparent,candidate_occurrences,unique_candidates,convergence_duplicates,accepted_next");
    print_rows("Z31", 2, &z31_rows());
    print_rows("Z8xZ8", 4, &z8_square_rows());
    print_rows("S3", 2, &s3_rows());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z31_boundary_closes_only_after_unique_geodesic_radius() {
        let rows = z31_rows();
        assert_eq!(rows.len(), 16);
        assert_eq!(rows[0], LevelCounts::new(0, 1, 2, 0, 0, 2, 2, 0, 2));
        assert_eq!(rows[14], LevelCounts::new(14, 2, 4, 2, 0, 2, 2, 0, 2));
        assert_eq!(rows[15], LevelCounts::new(15, 2, 4, 2, 2, 0, 0, 0, 0));
    }

    #[test]
    fn z8_square_relation_first_converges_at_depth_two() {
        let rows = z8_square_rows();
        assert_eq!(rows[0], LevelCounts::new(0, 1, 4, 0, 0, 4, 4, 0, 4));
        assert_eq!(rows[1], LevelCounts::new(1, 4, 16, 4, 0, 12, 8, 4, 8));
    }

    #[test]
    fn s3_braid_relation_gives_two_words_for_opposite_element() {
        let rows = s3_rows();
        assert_eq!(rows[1], LevelCounts::new(1, 2, 4, 2, 0, 2, 2, 0, 2));
        assert_eq!(rows[2], LevelCounts::new(2, 2, 4, 2, 0, 2, 1, 1, 1));
        assert_eq!(rows[3], LevelCounts::new(3, 1, 2, 1, 1, 0, 0, 0, 0));
    }
}
