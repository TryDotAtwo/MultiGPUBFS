use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug)]
struct LayerAudit {
    depth: usize,
    candidate_records: usize,
    states: usize,
    shortest_words: usize,
    trace_classes: usize,
    trace_explained_extras: usize,
    extras_after_trace: usize,
    commutation_power_classes: usize,
    extras_after_commutation_power: usize,
    states_with_multiple_trace_classes: usize,
    maximum_trace_classes_per_state: usize,
    sample_cross_trace_equalities: Vec<String>,
}

#[derive(Debug)]
struct CubeQtmAudit {
    unique_sticker_layers: Vec<LayerAudit>,
    colored_sticker_layers: Vec<LayerAudit>,
}

fn permutation_from_cycles(size: usize, cycles: &[&[usize]]) -> Vec<u8> {
    let mut permutation: Vec<u8> = (0..size as u8).collect();
    for cycle in cycles {
        for index in 0..cycle.len() {
            permutation[cycle[index]] = cycle[(index + 1) % cycle.len()] as u8;
        }
    }
    permutation
}

fn inverse_permutation(permutation: &[u8]) -> Vec<u8> {
    let mut inverse = vec![0; permutation.len()];
    for (source, &destination) in permutation.iter().enumerate() {
        inverse[destination as usize] = source as u8;
    }
    inverse
}

fn cube_qtm_moves() -> (Vec<String>, Vec<Vec<u8>>, Vec<usize>) {
    let bases = [
        (
            "U",
            permutation_from_cycles(
                54,
                &[
                    &[0, 6, 8, 2],
                    &[1, 3, 7, 5],
                    &[20, 47, 29, 38],
                    &[23, 50, 32, 41],
                    &[26, 53, 35, 44],
                ],
            ),
        ),
        (
            "D",
            permutation_from_cycles(
                54,
                &[
                    &[9, 15, 17, 11],
                    &[10, 12, 16, 14],
                    &[18, 36, 27, 45],
                    &[21, 39, 30, 48],
                    &[24, 42, 33, 51],
                ],
            ),
        ),
        (
            "L",
            permutation_from_cycles(
                54,
                &[
                    &[0, 44, 9, 45],
                    &[1, 43, 10, 46],
                    &[2, 42, 11, 47],
                    &[18, 24, 26, 20],
                    &[19, 21, 25, 23],
                ],
            ),
        ),
        (
            "R",
            permutation_from_cycles(
                54,
                &[
                    &[6, 51, 15, 38],
                    &[7, 52, 16, 37],
                    &[8, 53, 17, 36],
                    &[27, 33, 35, 29],
                    &[28, 30, 34, 32],
                ],
            ),
        ),
        (
            "B",
            permutation_from_cycles(
                54,
                &[
                    &[2, 35, 15, 18],
                    &[5, 34, 12, 19],
                    &[8, 33, 9, 20],
                    &[36, 42, 44, 38],
                    &[37, 39, 43, 41],
                ],
            ),
        ),
        (
            "F",
            permutation_from_cycles(
                54,
                &[
                    &[0, 24, 17, 29],
                    &[3, 25, 14, 28],
                    &[6, 26, 11, 27],
                    &[45, 51, 53, 47],
                    &[46, 48, 52, 50],
                ],
            ),
        ),
    ];
    let mut names = Vec::new();
    let mut moves = Vec::new();
    let mut inverse = Vec::new();
    for (name, movement) in bases {
        let index = moves.len();
        names.push(name.to_string());
        moves.push(movement.clone());
        inverse.push(index + 1);
        names.push(format!("{name}'"));
        moves.push(inverse_permutation(&movement));
        inverse.push(index);
    }
    (names, moves, inverse)
}

fn apply(state: &[u8], permutation: &[u8]) -> Vec<u8> {
    permutation
        .iter()
        .map(|&source| state[source as usize])
        .collect()
}

fn trace_normal_form(word: &[usize], commuting: &[Vec<bool>]) -> Vec<usize> {
    let mut seen = HashSet::from([word.to_vec()]);
    let mut queue = VecDeque::from([word.to_vec()]);
    let mut least = word.to_vec();
    while let Some(current) = queue.pop_front() {
        least = least.min(current.clone());
        for position in 0..current.len().saturating_sub(1) {
            if current[position] != current[position + 1]
                && commuting[current[position]][current[position + 1]]
            {
                let mut swapped = current.clone();
                swapped.swap(position, position + 1);
                if seen.insert(swapped.clone()) {
                    queue.push_back(swapped);
                }
            }
        }
    }
    least
}

fn commuting_matrix(moves: &[Vec<u8>]) -> Vec<Vec<bool>> {
    let mut commuting = vec![vec![false; moves.len()]; moves.len()];
    for first in 0..moves.len() {
        for second in 0..moves.len() {
            commuting[first][second] =
                apply(&moves[first], &moves[second]) == apply(&moves[second], &moves[first]);
        }
    }
    commuting
}

fn commutation_power_normal_form(
    word: &[usize],
    commuting: &[Vec<bool>],
    inverse: &[usize],
) -> Vec<usize> {
    let mut seen = HashSet::from([word.to_vec()]);
    let mut queue = VecDeque::from([word.to_vec()]);
    let mut least = word.to_vec();
    while let Some(current) = queue.pop_front() {
        least = least.min(current.clone());
        for position in 0..current.len().saturating_sub(1) {
            if current[position] != current[position + 1]
                && commuting[current[position]][current[position + 1]]
            {
                let mut swapped = current.clone();
                swapped.swap(position, position + 1);
                if seen.insert(swapped.clone()) {
                    queue.push_back(swapped);
                }
            }
            if current[position] == current[position + 1] {
                let mut rewritten = current.clone();
                rewritten[position] = inverse[current[position]];
                rewritten[position + 1] = inverse[current[position + 1]];
                if seen.insert(rewritten.clone()) {
                    queue.push_back(rewritten);
                }
            }
        }
    }
    least
}

fn enumerate_words(
    state: &[u8],
    depth_left: usize,
    previous: Option<usize>,
    word: &mut Vec<usize>,
    moves: &[Vec<u8>],
    inverse: &[usize],
    endpoints: &HashSet<Vec<u8>>,
    endpoint_words: &mut HashMap<Vec<u8>, Vec<Vec<usize>>>,
) {
    if depth_left == 0 {
        if endpoints.contains(state) {
            endpoint_words
                .entry(state.to_vec())
                .or_default()
                .push(word.clone());
        }
        return;
    }
    for movement in 0..moves.len() {
        if previous.is_some_and(|prior| movement == inverse[prior]) {
            continue;
        }
        word.push(movement);
        let next = apply(state, &moves[movement]);
        enumerate_words(
            &next,
            depth_left - 1,
            Some(movement),
            word,
            moves,
            inverse,
            endpoints,
            endpoint_words,
        );
        word.pop();
    }
}

fn audit_start(
    start: Vec<u8>,
    names: &[String],
    moves: &[Vec<u8>],
    inverse: &[usize],
    max_depth: usize,
) -> Vec<LayerAudit> {
    let commuting = commuting_matrix(moves);

    let mut layers = vec![LayerAudit {
        depth: 0,
        candidate_records: 1,
        states: 1,
        shortest_words: 1,
        trace_classes: 1,
        trace_explained_extras: 0,
        extras_after_trace: 0,
        commutation_power_classes: 1,
        extras_after_commutation_power: 0,
        states_with_multiple_trace_classes: 0,
        maximum_trace_classes_per_state: 1,
        sample_cross_trace_equalities: Vec::new(),
    }];
    let mut visited = HashSet::from([start.clone()]);
    let mut frontier = HashSet::from([start.clone()]);

    for depth in 1..=max_depth {
        let mut candidate_records = 0;
        let mut next = HashSet::new();
        for state in &frontier {
            for movement in moves {
                let child = apply(state, movement);
                if !visited.contains(&child) {
                    candidate_records += 1;
                    next.insert(child);
                }
            }
        }

        let mut endpoint_words = HashMap::new();
        enumerate_words(
            &start,
            depth,
            None,
            &mut Vec::new(),
            moves,
            inverse,
            &next,
            &mut endpoint_words,
        );
        let shortest_words = endpoint_words.values().map(Vec::len).sum();
        let mut trace_classes = 0;
        let mut commutation_power_classes = 0;
        let mut states_with_multiple_trace_classes = 0;
        let mut maximum_trace_classes_per_state = 0;
        let mut sample_cross_trace_equalities = Vec::new();
        for words in endpoint_words.values() {
            let classes: HashSet<Vec<usize>> = words
                .iter()
                .map(|word| trace_normal_form(word, &commuting))
                .collect();
            let commutation_power: HashSet<Vec<usize>> = words
                .iter()
                .map(|word| commutation_power_normal_form(word, &commuting, inverse))
                .collect();
            trace_classes += classes.len();
            commutation_power_classes += commutation_power.len();
            maximum_trace_classes_per_state = maximum_trace_classes_per_state.max(classes.len());
            if classes.len() > 1 {
                states_with_multiple_trace_classes += 1;
                let mut representatives: Vec<Vec<usize>> = classes.into_iter().collect();
                representatives.sort();
                let render = |word: &[usize]| {
                    word.iter()
                        .map(|&index| names[index].as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                sample_cross_trace_equalities.push(format!(
                    "{} = {}",
                    render(&representatives[0]),
                    render(&representatives[1])
                ));
            }
        }
        sample_cross_trace_equalities.sort();
        sample_cross_trace_equalities.dedup();
        sample_cross_trace_equalities.truncate(12);
        let states = next.len();
        layers.push(LayerAudit {
            depth,
            candidate_records,
            states,
            shortest_words,
            trace_classes,
            trace_explained_extras: shortest_words - trace_classes,
            extras_after_trace: trace_classes - states,
            commutation_power_classes,
            extras_after_commutation_power: commutation_power_classes - states,
            states_with_multiple_trace_classes,
            maximum_trace_classes_per_state,
            sample_cross_trace_equalities,
        });
        visited.extend(next.iter().cloned());
        frontier = next;
    }
    layers
}

fn cube_qtm_audit(max_depth: usize) -> CubeQtmAudit {
    let (names, moves, inverse) = cube_qtm_moves();
    let unique_stickers: Vec<u8> = (0..54).collect();
    let colored_stickers: Vec<u8> = (0..6).flat_map(|color| [color; 9]).collect();
    CubeQtmAudit {
        unique_sticker_layers: audit_start(unique_stickers, &names, &moves, &inverse, max_depth),
        colored_sticker_layers: audit_start(colored_stickers, &names, &moves, &inverse, max_depth),
    }
}

fn print_model(model: &str, layers: &[LayerAudit], emit_samples: bool) {
    for layer in layers {
        println!(
            "{model},{},{},{},{},{},{},{},{},{},{},{}",
            layer.depth,
            layer.candidate_records,
            layer.states,
            layer.shortest_words,
            layer.trace_classes,
            layer.trace_explained_extras,
            layer.extras_after_trace,
            layer.commutation_power_classes,
            layer.extras_after_commutation_power,
            layer.states_with_multiple_trace_classes,
            layer.maximum_trace_classes_per_state
        );
        if emit_samples {
            for (index, equality) in layer.sample_cross_trace_equalities.iter().enumerate() {
                println!("sample,{model},{},{index},{equality}", layer.depth);
            }
        }
    }
}

fn print_audit(audit: &CubeQtmAudit) {
    println!("model,depth,candidate_records,states,shortest_words,trace_classes,trace_explained_extras,extras_after_trace,commutation_power_classes,extras_after_commutation_power,states_with_multiple_trace_classes,maximum_trace_classes_per_state");
    print_model("unique", &audit.unique_sticker_layers, true);
    print_model("colors", &audit.colored_sticker_layers, false);
}

fn main() {
    let audit = cube_qtm_audit(4);
    print_audit(&audit);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cayleypy_cycle_fixture_builds_six_order_four_faces_and_inverses() {
        let (names, moves, inverse) = cube_qtm_moves();
        assert_eq!(names.len(), 12);
        assert_eq!(moves.len(), 12);
        assert_eq!(inverse.len(), 12);
        let identity: Vec<u8> = (0..54).collect();
        for (index, movement) in moves.iter().enumerate() {
            assert_eq!(
                apply(
                    &apply(&apply(&apply(&identity, movement), movement), movement),
                    movement
                ),
                identity
            );
            assert_eq!(
                apply(&apply(&identity, movement), &moves[inverse[index]]),
                identity
            );
        }
    }

    #[test]
    fn first_four_unique_sticker_spheres_match_the_standard_qtm_prefix() {
        let audit = cube_qtm_audit(4);
        let sizes: Vec<usize> = audit
            .unique_sticker_layers
            .iter()
            .map(|layer| layer.states)
            .collect();
        assert_eq!(sizes, vec![1, 12, 114, 1068, 10011]);
    }

    #[test]
    fn trace_partition_conserves_geodesic_words() {
        let audit = cube_qtm_audit(4);
        for layer in &audit.unique_sticker_layers {
            assert!(layer.shortest_words >= layer.trace_classes);
            assert!(layer.trace_classes >= layer.states);
            assert_eq!(
                layer.shortest_words - layer.states,
                layer.trace_explained_extras + layer.extras_after_trace
            );
            assert!(layer.trace_classes >= layer.commutation_power_classes);
            assert!(layer.commutation_power_classes >= layer.states);
        }
    }

    #[test]
    fn order_four_half_turn_rewrite_joins_inverse_square_words() {
        let (_, moves, inverse) = cube_qtm_moves();
        let commuting = commuting_matrix(&moves);
        assert_eq!(
            commutation_power_normal_form(&[8, 8, 2], &commuting, &inverse),
            commutation_power_normal_form(&[9, 9, 2], &commuting, &inverse)
        );
        assert_ne!(
            commutation_power_normal_form(&[8, 2, 8], &commuting, &inverse),
            commutation_power_normal_form(&[9, 2, 9], &commuting, &inverse)
        );
    }
}
