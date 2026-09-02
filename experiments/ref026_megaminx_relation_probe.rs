use std::collections::{HashMap, HashSet};

mod config_reader {
    #![allow(dead_code)]

    include!("ref025_megaminx_contract_probe.rs");

    pub(super) fn load(path: &str) -> Result<(Vec<u8>, Vec<String>, Vec<Vec<u8>>), String> {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let config = parse_config(&text)?;
        let names = config
            .generators
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let generators = config
            .generators
            .into_iter()
            .map(|(_, permutation)| permutation)
            .collect();
        Ok((config.central, names, generators))
    }
}

#[derive(Debug)]
struct DepthTwoWordAudit {
    reduced_word_occurrences: usize,
    unique_states: usize,
    convergence_extra: usize,
    multiplicity_two_states: usize,
    higher_multiplicity_states: usize,
    commutation_groups: usize,
    other_collision_groups: usize,
    commuting_face_pairs: usize,
    face_pairs_with_all_four_orientations: usize,
}

#[derive(Debug)]
struct TransitionProfile {
    depth: usize,
    frontier: usize,
    occurrences: usize,
    backward_to_previous: usize,
    same_level: usize,
    older_ball: usize,
    candidate_occurrences: usize,
    unique_next: usize,
    candidate_convergence: usize,
}

#[derive(Debug)]
struct SameLevelAudit {
    directed_same_level_occurrences: usize,
    order_five_power_occurrences: usize,
    other_same_level_occurrences: usize,
}

fn apply(state: &[u8], permutation: &[u8]) -> Vec<u8> {
    permutation
        .iter()
        .map(|&source| state[source as usize])
        .collect()
}

fn inverse_indices(names: &[String]) -> Result<Vec<usize>, String> {
    let by_name: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    names
        .iter()
        .map(|name| {
            let inverse_name = if let Some(base) = name.strip_prefix('-') {
                base.to_string()
            } else {
                format!("-{name}")
            };
            by_name
                .get(inverse_name.as_str())
                .copied()
                .ok_or_else(|| format!("missing inverse move {inverse_name}"))
        })
        .collect()
}

fn depth_two_word_audit(path: &str) -> Result<DepthTwoWordAudit, String> {
    let (central, names, generators) = config_reader::load(path)?;
    let inverse = inverse_indices(&names)?;
    let mut endpoint_words: HashMap<Vec<u8>, Vec<(usize, usize)>> = HashMap::new();
    for first in 0..generators.len() {
        let state_one = apply(&central, &generators[first]);
        for second in 0..generators.len() {
            if second == inverse[first] {
                continue;
            }
            endpoint_words
                .entry(apply(&state_one, &generators[second]))
                .or_default()
                .push((first, second));
        }
    }
    let reduced_word_occurrences = endpoint_words.values().map(Vec::len).sum();
    let unique_states = endpoint_words.len();
    let convergence_extra = endpoint_words.values().map(|words| words.len() - 1).sum();
    let multiplicity_two_states = endpoint_words
        .values()
        .filter(|words| words.len() == 2)
        .count();
    let higher_multiplicity_states = endpoint_words
        .values()
        .filter(|words| words.len() > 2)
        .count();
    let mut commutation_groups = 0;
    let mut other_collision_groups = 0;
    let mut face_pair_orientations: HashMap<(String, String), HashSet<(bool, bool)>> =
        HashMap::new();
    for words in endpoint_words.values().filter(|words| words.len() > 1) {
        if words.len() == 2 && words[0].0 == words[1].1 && words[0].1 == words[1].0 {
            commutation_groups += 1;
            let first_name = &names[words[0].0];
            let second_name = &names[words[0].1];
            let first_negative = first_name.starts_with('-');
            let second_negative = second_name.starts_with('-');
            let first_face = first_name.trim_start_matches('-');
            let second_face = second_name.trim_start_matches('-');
            let (pair, orientation) = if first_face < second_face {
                (
                    (first_face.to_string(), second_face.to_string()),
                    (first_negative, second_negative),
                )
            } else {
                (
                    (second_face.to_string(), first_face.to_string()),
                    (second_negative, first_negative),
                )
            };
            face_pair_orientations
                .entry(pair)
                .or_default()
                .insert(orientation);
        } else {
            other_collision_groups += 1;
        }
    }
    let commuting_face_pairs = face_pair_orientations.len();
    let face_pairs_with_all_four_orientations = face_pair_orientations
        .values()
        .filter(|orientations| orientations.len() == 4)
        .count();
    Ok(DepthTwoWordAudit {
        reduced_word_occurrences,
        unique_states,
        convergence_extra,
        multiplicity_two_states,
        higher_multiplicity_states,
        commutation_groups,
        other_collision_groups,
        commuting_face_pairs,
        face_pairs_with_all_four_orientations,
    })
}

fn transition_profiles(path: &str, max_depth: usize) -> Result<Vec<TransitionProfile>, String> {
    let (central, _, generators) = config_reader::load(path)?;
    let mut distance = HashMap::from([(central.clone(), 0_usize)]);
    let mut frontier = vec![central];
    let mut rows = Vec::new();
    for depth in 0..=max_depth {
        let mut backward_to_previous = 0;
        let mut same_level = 0;
        let mut older_ball = 0;
        let mut candidate_occurrences = 0;
        let mut next_seen = HashSet::new();
        let mut next = Vec::new();
        for state in &frontier {
            for permutation in &generators {
                let child = apply(state, permutation);
                if let Some(&child_depth) = distance.get(&child) {
                    if child_depth + 1 == depth {
                        backward_to_previous += 1;
                    } else if child_depth == depth {
                        same_level += 1;
                    } else {
                        older_ball += 1;
                    }
                } else {
                    candidate_occurrences += 1;
                    if next_seen.insert(child.clone()) {
                        next.push(child);
                    }
                }
            }
        }
        let unique_next = next.len();
        rows.push(TransitionProfile {
            depth,
            frontier: frontier.len(),
            occurrences: frontier.len() * generators.len(),
            backward_to_previous,
            same_level,
            older_ball,
            candidate_occurrences,
            unique_next,
            candidate_convergence: candidate_occurrences - unique_next,
        });
        for state in &next {
            distance.insert(state.clone(), depth + 1);
        }
        frontier = next;
    }
    Ok(rows)
}

fn f2_same_level_audit(path: &str) -> Result<SameLevelAudit, String> {
    let (central, names, generators) = config_reader::load(path)?;
    let inverse = inverse_indices(&names)?;
    let mut expected = HashSet::new();
    for generator in 0..generators.len() {
        let state_one = apply(&central, &generators[generator]);
        let state_two = apply(&state_one, &generators[generator]);
        let inverse_one = apply(&central, &generators[inverse[generator]]);
        let inverse_two = apply(&inverse_one, &generators[inverse[generator]]);
        expected.insert((state_two, inverse_two, generator));
    }

    let mut distance = HashMap::from([(central.clone(), 0_usize)]);
    let mut frontier = vec![central];
    for depth in 0..2 {
        let mut next_seen = HashSet::new();
        let mut next = Vec::new();
        for state in &frontier {
            for permutation in &generators {
                let child = apply(state, permutation);
                if !distance.contains_key(&child) && next_seen.insert(child.clone()) {
                    next.push(child);
                }
            }
        }
        for state in &next {
            distance.insert(state.clone(), depth + 1);
        }
        frontier = next;
    }

    let mut directed_same_level_occurrences = 0;
    let mut order_five_power_occurrences = 0;
    for state in &frontier {
        for (generator, permutation) in generators.iter().enumerate() {
            let child = apply(state, permutation);
            if distance.get(&child) == Some(&2) {
                directed_same_level_occurrences += 1;
                if expected.contains(&(state.clone(), child, generator)) {
                    order_five_power_occurrences += 1;
                }
            }
        }
    }
    Ok(SameLevelAudit {
        directed_same_level_occurrences,
        order_five_power_occurrences,
        other_same_level_occurrences: directed_same_level_occurrences
            - order_five_power_occurrences,
    })
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ref026_megaminx_relation_probe PUZZLE_INFO_JSON");
    let words = depth_two_word_audit(&path).expect("depth-two word audit");
    println!("section,metric,value");
    println!(
        "words,reduced_word_occurrences,{}",
        words.reduced_word_occurrences
    );
    println!("words,unique_states,{}", words.unique_states);
    println!("words,convergence_extra,{}", words.convergence_extra);
    println!(
        "words,multiplicity_two_states,{}",
        words.multiplicity_two_states
    );
    println!(
        "words,higher_multiplicity_states,{}",
        words.higher_multiplicity_states
    );
    println!("words,commutation_groups,{}", words.commutation_groups);
    println!(
        "words,other_collision_groups,{}",
        words.other_collision_groups
    );
    println!("words,commuting_face_pairs,{}", words.commuting_face_pairs);
    println!(
        "words,face_pairs_with_all_four_orientations,{}",
        words.face_pairs_with_all_four_orientations
    );
    for row in transition_profiles(&path, 2).expect("transition profiles") {
        println!(
            "depth{},frontier,{};occurrences,{};backward,{};same_level,{};older_ball,{};candidate_occurrences,{};unique_next,{};candidate_convergence,{}",
            row.depth,
            row.frontier,
            row.occurrences,
            row.backward_to_previous,
            row.same_level,
            row.older_ball,
            row.candidate_occurrences,
            row.unique_next,
            row.candidate_convergence
        );
    }
    let boundary = f2_same_level_audit(&path).expect("F2 same-level audit");
    println!(
        "f2,directed_same_level_occurrences,{}",
        boundary.directed_same_level_occurrences
    );
    println!(
        "f2,order_five_power_occurrences,{}",
        boundary.order_five_power_occurrences
    );
    println!(
        "f2,other_same_level_occurrences,{}",
        boundary.other_same_level_occurrences
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> String {
        std::env::var("REF026_PUZZLE_INFO").expect("REF026_PUZZLE_INFO must be set")
    }

    #[test]
    fn every_depth_two_convergence_is_a_two_word_commutation_square() {
        let audit = depth_two_word_audit(&fixture_path()).unwrap();
        assert_eq!(audit.reduced_word_occurrences, 552);
        assert_eq!(audit.unique_states, 408);
        assert_eq!(audit.convergence_extra, 144);
        assert_eq!(audit.multiplicity_two_states, 144);
        assert_eq!(audit.higher_multiplicity_states, 0);
        assert_eq!(audit.commutation_groups, 144);
        assert_eq!(audit.other_collision_groups, 0);
        assert_eq!(audit.commuting_face_pairs, 36);
        assert_eq!(audit.face_pairs_with_all_four_orientations, 36);
    }

    #[test]
    fn order_five_face_turns_first_show_as_same_layer_edges_at_f_two() {
        let rows = transition_profiles(&fixture_path(), 2).unwrap();
        assert_eq!(rows[1].frontier, 24);
        assert_eq!(rows[1].candidate_convergence, 144);
        assert_eq!(rows[2].frontier, 408);
        assert_eq!(rows[2].backward_to_previous, 552);
        assert_eq!(rows[2].same_level, 24);
        assert_eq!(rows[2].older_ball, 0);
        assert_eq!(rows[2].candidate_occurrences, 9216);
        assert_eq!(rows[2].unique_next, 6208);
        assert_eq!(rows[2].candidate_convergence, 3008);
    }

    #[test]
    fn every_f_two_same_level_occurrence_is_the_boundary_of_one_face_five_cycle() {
        let audit = f2_same_level_audit(&fixture_path()).unwrap();
        assert_eq!(audit.directed_same_level_occurrences, 24);
        assert_eq!(audit.order_five_power_occurrences, 24);
        assert_eq!(audit.other_same_level_occurrences, 0);
    }
}
