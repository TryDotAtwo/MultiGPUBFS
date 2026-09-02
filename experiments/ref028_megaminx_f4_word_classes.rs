use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

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
struct F4WordClassAudit {
    candidate_records: usize,
    f4_endpoints: usize,
    shortest_word_occurrences: usize,
    commutation_classes: usize,
    commutation_explained_word_extras: usize,
    extras_after_commutation: usize,
    endpoints_with_multiple_classes: usize,
    maximum_classes_per_endpoint: usize,
    cross_class_pairs: usize,
    generator_conjugate_commutator_pairs: usize,
    word_multiplicity_histogram: BTreeMap<usize, usize>,
    class_count_histogram: BTreeMap<usize, usize>,
    sample_cross_class_equalities: Vec<String>,
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
                .ok_or_else(|| format!("missing inverse {inverse_name}"))
        })
        .collect()
}

fn trace_normal_form(word: &[usize], commuting: &[Vec<bool>]) -> Vec<usize> {
    let mut seen = HashSet::from([word.to_vec()]);
    let mut queue = VecDeque::from([word.to_vec()]);
    let mut least = word.to_vec();
    while let Some(current) = queue.pop_front() {
        least = least.min(current.clone());
        for position in 0..current.len().saturating_sub(1) {
            if commuting[current[position]][current[position + 1]] {
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

fn is_generator_conjugate_commutator(left: &[usize], right: &[usize], inverse: &[usize]) -> bool {
    if left.len() != 4 || right.len() != 4 {
        return false;
    }
    let rotate_left = right[..3] == left[1..] && right[3] == left[0];
    let rotate_right = right[0] == left[3] && right[1..] == left[..3];
    (rotate_left && left[3] == inverse[left[1]]) || (rotate_right && left[2] == inverse[left[0]])
}

fn f4_word_class_audit(path: &str) -> Result<F4WordClassAudit, String> {
    let (central, names, generators) = config_reader::load(path)?;
    let inverse = inverse_indices(&names)?;
    let move_count = generators.len();
    let mut commuting = vec![vec![false; move_count]; move_count];
    for first in 0..move_count {
        let first_state = apply(&central, &generators[first]);
        for second in 0..move_count {
            let second_state = apply(&central, &generators[second]);
            commuting[first][second] = apply(&first_state, &generators[second])
                == apply(&second_state, &generators[first]);
        }
    }

    let mut distance = HashMap::from([(central.clone(), 0_usize)]);
    let mut frontier = vec![central.clone()];
    for depth in 0..3 {
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

    let mut candidate_records = 0;
    let mut f4_states = HashSet::new();
    for state in &frontier {
        for permutation in &generators {
            let child = apply(state, permutation);
            if !distance.contains_key(&child) {
                candidate_records += 1;
                f4_states.insert(child);
            }
        }
    }

    let mut endpoint_words: HashMap<Vec<u8>, Vec<[usize; 4]>> = HashMap::new();
    for first in 0..move_count {
        let state_one = apply(&central, &generators[first]);
        for second in 0..move_count {
            if second == inverse[first] {
                continue;
            }
            let state_two = apply(&state_one, &generators[second]);
            for third in 0..move_count {
                if third == inverse[second] {
                    continue;
                }
                let state_three = apply(&state_two, &generators[third]);
                for fourth in 0..move_count {
                    if fourth == inverse[third] {
                        continue;
                    }
                    let endpoint = apply(&state_three, &generators[fourth]);
                    if f4_states.contains(&endpoint) {
                        endpoint_words
                            .entry(endpoint)
                            .or_default()
                            .push([first, second, third, fourth]);
                    }
                }
            }
        }
    }

    let f4_endpoints = endpoint_words.len();
    let shortest_word_occurrences = endpoint_words.values().map(Vec::len).sum();
    let mut commutation_classes = 0;
    let mut endpoints_with_multiple_classes = 0;
    let mut maximum_classes_per_endpoint = 0;
    let mut cross_class_pairs = 0;
    let mut generator_conjugate_commutator_pairs = 0;
    let mut word_multiplicity_histogram = BTreeMap::new();
    let mut class_count_histogram = BTreeMap::new();
    let mut sample_cross_class_equalities = Vec::new();

    for words in endpoint_words.values() {
        *word_multiplicity_histogram.entry(words.len()).or_insert(0) += 1;
        let classes: HashSet<Vec<usize>> = words
            .iter()
            .map(|word| trace_normal_form(word, &commuting))
            .collect();
        let class_count = classes.len();
        commutation_classes += class_count;
        *class_count_histogram.entry(class_count).or_insert(0) += 1;
        maximum_classes_per_endpoint = maximum_classes_per_endpoint.max(class_count);
        if class_count > 1 {
            endpoints_with_multiple_classes += 1;
            let mut representatives: Vec<Vec<usize>> = classes.into_iter().collect();
            representatives.sort();
            for first in 0..representatives.len() {
                for second in first + 1..representatives.len() {
                    cross_class_pairs += 1;
                    if is_generator_conjugate_commutator(
                        &representatives[first],
                        &representatives[second],
                        &inverse,
                    ) {
                        generator_conjugate_commutator_pairs += 1;
                    }
                }
            }
            let render = |word: &[usize]| {
                word.iter()
                    .map(|&index| names[index].as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            sample_cross_class_equalities.push(format!(
                "{} = {}",
                render(&representatives[0]),
                render(&representatives[1])
            ));
        }
    }
    sample_cross_class_equalities.sort();
    sample_cross_class_equalities.dedup();
    sample_cross_class_equalities.truncate(20);

    let commutation_explained_word_extras = shortest_word_occurrences - commutation_classes;
    let extras_after_commutation = commutation_classes - f4_endpoints;
    Ok(F4WordClassAudit {
        candidate_records,
        f4_endpoints,
        shortest_word_occurrences,
        commutation_classes,
        commutation_explained_word_extras,
        extras_after_commutation,
        endpoints_with_multiple_classes,
        maximum_classes_per_endpoint,
        cross_class_pairs,
        generator_conjugate_commutator_pairs,
        word_multiplicity_histogram,
        class_count_histogram,
        sample_cross_class_equalities,
    })
}

fn print_audit(audit: &F4WordClassAudit) {
    println!("metric,value");
    println!("candidate_records,{}", audit.candidate_records);
    println!("f4_endpoints,{}", audit.f4_endpoints);
    println!(
        "shortest_word_occurrences,{}",
        audit.shortest_word_occurrences
    );
    println!("commutation_classes,{}", audit.commutation_classes);
    println!(
        "commutation_explained_word_extras,{}",
        audit.commutation_explained_word_extras
    );
    println!(
        "extras_after_commutation,{}",
        audit.extras_after_commutation
    );
    println!(
        "endpoints_with_multiple_classes,{}",
        audit.endpoints_with_multiple_classes
    );
    println!(
        "maximum_classes_per_endpoint,{}",
        audit.maximum_classes_per_endpoint
    );
    println!("cross_class_pairs,{}", audit.cross_class_pairs);
    println!(
        "generator_conjugate_commutator_pairs,{}",
        audit.generator_conjugate_commutator_pairs
    );
    for (multiplicity, endpoints) in &audit.word_multiplicity_histogram {
        println!("word_multiplicity_{multiplicity},{endpoints}");
    }
    for (classes, endpoints) in &audit.class_count_histogram {
        println!("commutation_class_count_{classes},{endpoints}");
    }
    for (index, equality) in audit.sample_cross_class_equalities.iter().enumerate() {
        println!("sample_cross_class_equality_{index},{equality}");
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ref028_megaminx_f4_word_classes PUZZLE_INFO_JSON");
    let audit = f4_word_class_audit(&path).expect("F4 word-class audit");
    print_audit(&audit);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> String {
        std::env::var("REF028_PUZZLE_INFO").expect("REF028_PUZZLE_INFO must be set")
    }

    #[test]
    fn trace_normal_form_swaps_only_adjacent_commuting_letters() {
        let commuting = vec![
            vec![false, true, false],
            vec![true, false, false],
            vec![false, false, false],
        ];
        assert_eq!(
            trace_normal_form(&[1, 0, 2, 0], &commuting),
            vec![0, 1, 2, 0]
        );
        assert_eq!(
            trace_normal_form(&[0, 2, 1, 0], &commuting),
            vec![0, 2, 0, 1]
        );
    }

    #[test]
    fn production_f4_partition_conserves_words_classes_and_states() {
        let audit = f4_word_class_audit(&fixture_path()).unwrap();
        assert!(audit.shortest_word_occurrences >= audit.commutation_classes);
        assert!(audit.commutation_classes >= audit.f4_endpoints);
        assert_eq!(
            audit.shortest_word_occurrences - audit.f4_endpoints,
            audit.commutation_explained_word_extras + audit.extras_after_commutation
        );
    }

    #[test]
    fn recognizes_only_cyclic_commutation_with_a_conjugated_generator() {
        let inverse = vec![1, 0, 3, 2, 5, 4];
        assert!(is_generator_conjugate_commutator(
            &[0, 2, 4, 3],
            &[2, 4, 3, 0],
            &inverse
        ));
        assert!(!is_generator_conjugate_commutator(
            &[0, 2, 4, 3],
            &[2, 3, 4, 0],
            &inverse
        ));
    }
}
