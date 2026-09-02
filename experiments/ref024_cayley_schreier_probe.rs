use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

type Permutation = [u8; 3];

#[derive(Clone, Copy, Debug)]
enum Move {
    S0,
    S1,
}

const MOVES: [Move; 2] = [Move::S0, Move::S1];

#[derive(Debug, Eq, PartialEq)]
struct LevelCounts {
    depth: usize,
    frontier: usize,
    occurrences: usize,
    loops: usize,
    parent_returns: usize,
    visited_nonparent: usize,
    candidate_occurrences: usize,
    unique_candidates: usize,
    candidate_convergence: usize,
}

fn identity() -> Permutation {
    [0, 1, 2]
}

fn apply_to_point(point: u8, generator: Move) -> u8 {
    match (generator, point) {
        (Move::S0, 0) => 1,
        (Move::S0, 1) => 0,
        (Move::S1, 1) => 2,
        (Move::S1, 2) => 1,
        _ => point,
    }
}

fn apply_to_group(permutation: Permutation, generator: Move) -> Permutation {
    permutation.map(|point| apply_to_point(point, generator))
}

fn probe<S, F>(root: S, successors: F) -> Vec<LevelCounts>
where
    S: Copy + Eq + Hash,
    F: Fn(S) -> Vec<S>,
{
    let mut visited = HashSet::from([root]);
    let mut parent = HashMap::new();
    let mut frontier = vec![root];
    let mut rows = Vec::new();

    for depth in 0.. {
        let mut loops = 0;
        let mut parent_returns = 0;
        let mut visited_nonparent = 0;
        let mut candidate_occurrences = 0;
        let mut next_seen = HashSet::new();
        let mut next_frontier = Vec::new();

        for &state in &frontier {
            for child in successors(state) {
                if child == state {
                    loops += 1;
                } else if parent.get(&state) == Some(&child) {
                    parent_returns += 1;
                } else if visited.contains(&child) {
                    visited_nonparent += 1;
                } else {
                    candidate_occurrences += 1;
                    if next_seen.insert(child) {
                        parent.insert(child, state);
                        next_frontier.push(child);
                    }
                }
            }
        }

        let unique_candidates = next_frontier.len();
        rows.push(LevelCounts {
            depth,
            frontier: frontier.len(),
            occurrences: frontier.len() * MOVES.len(),
            loops,
            parent_returns,
            visited_nonparent,
            candidate_occurrences,
            unique_candidates,
            candidate_convergence: candidate_occurrences - unique_candidates,
        });
        if next_frontier.is_empty() {
            break;
        }
        visited.extend(next_frontier.iter().copied());
        frontier = next_frontier;
    }
    rows
}

fn cayley_s3_rows() -> Vec<LevelCounts> {
    probe(identity(), |state| {
        MOVES
            .iter()
            .map(|&generator| apply_to_group(state, generator))
            .collect()
    })
}

fn schreier_point_rows() -> Vec<LevelCounts> {
    probe(0, |state| {
        MOVES
            .iter()
            .map(|&generator| apply_to_point(state, generator))
            .collect()
    })
}

fn distance<S, F>(start: S, target: S, successors: F) -> Option<usize>
where
    S: Copy + Eq + Hash,
    F: Fn(S) -> Vec<S>,
{
    let mut queue = VecDeque::from([(start, 0)]);
    let mut visited = HashSet::from([start]);
    while let Some((state, depth)) = queue.pop_front() {
        if state == target {
            return Some(depth);
        }
        for child in successors(state) {
            if visited.insert(child) {
                queue.push_back((child, depth + 1));
            }
        }
    }
    None
}

fn point_distance(start: u8, target: u8) -> Option<usize> {
    distance(start, target, |state| {
        MOVES
            .iter()
            .map(|&generator| apply_to_point(state, generator))
            .collect()
    })
}

fn group_distance(start: Permutation, target: Permutation) -> Option<usize> {
    distance(start, target, |state| {
        MOVES
            .iter()
            .map(|&generator| apply_to_group(state, generator))
            .collect()
    })
}

fn representative_mapping_zero_to_two() -> Permutation {
    let after_s0 = apply_to_group(identity(), Move::S0);
    let after_s0_s1 = apply_to_group(after_s0, Move::S1);
    apply_to_group(after_s0_s1, Move::S0)
}

fn print_rows(name: &str, rows: &[LevelCounts]) {
    for row in rows {
        println!(
            "{},{},{},{},{},{},{},{},{},{}",
            name,
            row.depth,
            row.frontier,
            row.occurrences,
            row.loops,
            row.parent_returns,
            row.visited_nonparent,
            row.candidate_occurrences,
            row.unique_candidates,
            row.candidate_convergence,
        );
    }
}

fn main() {
    println!("model,depth,frontier,occurrences,loops,parent_returns,visited_nonparent,candidate_occurrences,unique_candidates,candidate_convergence");
    print_rows("Cayley-S3", &cayley_s3_rows());
    print_rows("Schreier-S3-on-point", &schreier_point_rows());
    println!(
        "distance,point-0-to-2,{},group-identity-to-chosen-representative,{}",
        point_distance(0, 2).unwrap(),
        group_distance(identity(), representative_mapping_zero_to_two()).unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_s3_action_first_converges_at_the_length_three_braid() {
        let rows = cayley_s3_rows();
        assert_eq!(
            rows.iter().map(|row| row.frontier).collect::<Vec<_>>(),
            [1, 2, 2, 1]
        );
        assert_eq!(rows[0].loops, 0);
        assert_eq!(rows[2].candidate_convergence, 1);
    }

    #[test]
    fn point_action_has_a_nonidentity_stabilizer_generator_at_the_root() {
        assert_ne!(apply_to_group(identity(), Move::S1), identity());
        assert_eq!(apply_to_point(0, Move::S1), 0);

        let rows = schreier_point_rows();
        assert_eq!(
            rows.iter().map(|row| row.frontier).collect::<Vec<_>>(),
            [1, 1, 1]
        );
        assert_eq!(rows[0].loops, 1);
        assert_eq!(rows[0].candidate_convergence, 0);
    }

    #[test]
    fn arbitrary_group_representative_can_overstate_state_distance() {
        assert_eq!(point_distance(0, 2), Some(2));
        assert_eq!(
            group_distance(identity(), representative_mapping_zero_to_two()),
            Some(3)
        );
    }
}
