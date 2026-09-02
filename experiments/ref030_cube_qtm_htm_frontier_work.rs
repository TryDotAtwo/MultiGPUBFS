use std::collections::{HashMap, HashSet};

mod cube_fixture {
    #![allow(dead_code)]

    include!("ref029_cube_qtm_relation_onset.rs");

    pub(super) fn qtm_moves() -> Vec<Vec<u8>> {
        cube_qtm_moves().1
    }

    pub(super) fn compose_same(permutation: &[u8]) -> Vec<u8> {
        apply(permutation, permutation)
    }

    pub(super) fn apply_move(state: &[u8], permutation: &[u8]) -> Vec<u8> {
        apply(state, permutation)
    }
}

#[derive(Debug)]
struct ExpansionLevel {
    depth: usize,
    degree: usize,
    frontier: usize,
    total_occurrences: usize,
    backward: usize,
    same_level: usize,
    older: usize,
    forward_candidates: usize,
    unique_next: usize,
    forward_duplicate_extras: usize,
}

#[derive(Debug)]
struct MetricProfile {
    name: &'static str,
    levels: Vec<ExpansionLevel>,
}

#[derive(Debug)]
struct MetricComparison {
    qtm: MetricProfile,
    htm: MetricProfile,
}

fn htm_moves() -> Vec<Vec<u8>> {
    let qtm = cube_fixture::qtm_moves();
    let mut htm = Vec::new();
    for face in 0..6 {
        let clockwise = &qtm[2 * face];
        htm.push(clockwise.clone());
        htm.push(qtm[2 * face + 1].clone());
        htm.push(cube_fixture::compose_same(clockwise));
    }
    htm
}

fn profile_metric(
    name: &'static str,
    moves: &[Vec<u8>],
    maximum_expanded_depth: usize,
) -> MetricProfile {
    let start: Vec<u8> = (0..54).collect();
    let mut distance = HashMap::from([(start.clone(), 0_usize)]);
    let mut frontier = vec![start];
    let mut levels = Vec::new();

    for depth in 0..=maximum_expanded_depth {
        let mut backward = 0;
        let mut same_level = 0;
        let mut older = 0;
        let mut forward_candidates = 0;
        let mut next = HashSet::new();
        for state in &frontier {
            for movement in moves {
                let child = cube_fixture::apply_move(state, movement);
                if let Some(&old_depth) = distance.get(&child) {
                    if old_depth + 1 == depth {
                        backward += 1;
                    } else if old_depth == depth {
                        same_level += 1;
                    } else if old_depth + 1 < depth {
                        older += 1;
                    } else {
                        panic!("edge reached a previously stored future layer");
                    }
                } else {
                    forward_candidates += 1;
                    next.insert(child);
                }
            }
        }
        let unique_next = next.len();
        levels.push(ExpansionLevel {
            depth,
            degree: moves.len(),
            frontier: frontier.len(),
            total_occurrences: frontier.len() * moves.len(),
            backward,
            same_level,
            older,
            forward_candidates,
            unique_next,
            forward_duplicate_extras: forward_candidates - unique_next,
        });
        for state in &next {
            distance.insert(state.clone(), depth + 1);
        }
        frontier = next.into_iter().collect();
    }
    MetricProfile { name, levels }
}

fn compare_metrics(maximum_expanded_depth: usize) -> MetricComparison {
    MetricComparison {
        qtm: profile_metric("QTM", &cube_fixture::qtm_moves(), maximum_expanded_depth),
        htm: profile_metric("HTM", &htm_moves(), maximum_expanded_depth),
    }
}

fn print_profile(profile: &MetricProfile) {
    for level in &profile.levels {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            profile.name,
            level.depth,
            level.degree,
            level.frontier,
            level.total_occurrences,
            level.backward,
            level.same_level,
            level.older,
            level.forward_candidates,
            level.unique_next,
            level.forward_duplicate_extras
        );
    }
}

fn print_profiles(profiles: &MetricComparison) {
    println!("metric,expanded_depth,degree,frontier,total_occurrences,backward,same_level,older,forward_candidates,unique_next,forward_duplicate_extras");
    print_profile(&profiles.qtm);
    print_profile(&profiles.htm);
}

fn main() {
    let profiles = compare_metrics(3);
    print_profiles(&profiles);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qtm_and_htm_prefixes_match_published_spheres() {
        let profiles = compare_metrics(3);
        let sizes = |profile: &MetricProfile| {
            std::iter::once(1)
                .chain(profile.levels.iter().map(|level| level.unique_next))
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(&profiles.qtm), vec![1, 12, 114, 1068, 10011]);
        assert_eq!(sizes(&profiles.htm), vec![1, 18, 243, 3240, 43239]);
    }

    #[test]
    fn every_expansion_profile_conserves_labeled_occurrences() {
        let profiles = compare_metrics(3);
        for profile in [&profiles.qtm, &profiles.htm] {
            for level in &profile.levels {
                assert_eq!(
                    level.total_occurrences,
                    level.backward + level.same_level + level.older + level.forward_candidates
                );
                assert_eq!(
                    level.forward_candidates,
                    level.unique_next + level.forward_duplicate_extras
                );
            }
        }
    }
}
