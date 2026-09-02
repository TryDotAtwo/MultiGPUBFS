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
struct F3WordClassAudit {
    candidate_records: usize,
    f3_endpoints: usize,
    shortest_word_occurrences: usize,
    commutation_classes: usize,
    word_extras: usize,
    commutation_explained_word_extras: usize,
    extras_after_commutation: usize,
    endpoints_with_one_class: usize,
    endpoints_with_multiple_classes: usize,
    maximum_classes_per_endpoint: usize,
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

fn commutation_normal_form(word: [usize; 3], commuting: &[Vec<bool>]) -> [usize; 3] {
    let mut seen = HashSet::from([word]);
    let mut queue = VecDeque::from([word]);
    let mut least = word;
    while let Some(current) = queue.pop_front() {
        least = least.min(current);
        for position in 0..2 {
            if commuting[current[position]][current[position + 1]] {
                let mut swapped = current;
                swapped.swap(position, position + 1);
                if seen.insert(swapped) {
                    queue.push_back(swapped);
                }
            }
        }
    }
    least
}

fn f3_word_class_audit(path: &str) -> Result<F3WordClassAudit, String> {
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

    let mut candidate_records = 0;
    let mut f3_states = HashSet::new();
    for state in &frontier {
        for permutation in &generators {
            let child = apply(state, permutation);
            if !distance.contains_key(&child) {
                candidate_records += 1;
                f3_states.insert(child);
            }
        }
    }

    let mut endpoint_words: HashMap<Vec<u8>, Vec<[usize; 3]>> = HashMap::new();
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
                let endpoint = apply(&state_two, &generators[third]);
                if f3_states.contains(&endpoint) {
                    endpoint_words
                        .entry(endpoint)
                        .or_default()
                        .push([first, second, third]);
                }
            }
        }
    }

    let f3_endpoints = endpoint_words.len();
    let shortest_word_occurrences = endpoint_words.values().map(Vec::len).sum();
    let mut commutation_classes = 0;
    let mut endpoints_with_one_class = 0;
    let mut endpoints_with_multiple_classes = 0;
    let mut maximum_classes_per_endpoint = 0;
    let mut word_multiplicity_histogram = BTreeMap::new();
    let mut class_count_histogram = BTreeMap::new();
    let mut sample_cross_class_equalities = Vec::new();

    for words in endpoint_words.values() {
        *word_multiplicity_histogram.entry(words.len()).or_insert(0) += 1;
        let classes: HashSet<[usize; 3]> = words
            .iter()
            .map(|&word| commutation_normal_form(word, &commuting))
            .collect();
        let class_count = classes.len();
        commutation_classes += class_count;
        *class_count_histogram.entry(class_count).or_insert(0) += 1;
        maximum_classes_per_endpoint = maximum_classes_per_endpoint.max(class_count);
        if class_count == 1 {
            endpoints_with_one_class += 1;
        } else {
            endpoints_with_multiple_classes += 1;
            let mut representatives: Vec<[usize; 3]> = classes.into_iter().collect();
            representatives.sort_unstable();
            let render = |word: [usize; 3]| {
                word.iter()
                    .map(|&index| names[index].as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            sample_cross_class_equalities.push(format!(
                "{} = {}",
                render(representatives[0]),
                render(representatives[1])
            ));
        }
    }
    sample_cross_class_equalities.sort();
    sample_cross_class_equalities.dedup();
    sample_cross_class_equalities.truncate(20);

    let word_extras = shortest_word_occurrences - f3_endpoints;
    let commutation_explained_word_extras = shortest_word_occurrences - commutation_classes;
    let extras_after_commutation = commutation_classes - f3_endpoints;
    Ok(F3WordClassAudit {
        candidate_records,
        f3_endpoints,
        shortest_word_occurrences,
        commutation_classes,
        word_extras,
        commutation_explained_word_extras,
        extras_after_commutation,
        endpoints_with_one_class,
        endpoints_with_multiple_classes,
        maximum_classes_per_endpoint,
        word_multiplicity_histogram,
        class_count_histogram,
        sample_cross_class_equalities,
    })
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ref027_megaminx_f3_word_classes PUZZLE_INFO_JSON");
    let audit = f3_word_class_audit(&path).expect("F3 word-class audit");
    println!("metric,value");
    println!("candidate_records,{}", audit.candidate_records);
    println!("f3_endpoints,{}", audit.f3_endpoints);
    println!(
        "shortest_word_occurrences,{}",
        audit.shortest_word_occurrences
    );
    println!("commutation_classes,{}", audit.commutation_classes);
    println!("word_extras,{}", audit.word_extras);
    println!(
        "commutation_explained_word_extras,{}",
        audit.commutation_explained_word_extras
    );
    println!(
        "extras_after_commutation,{}",
        audit.extras_after_commutation
    );
    println!(
        "endpoints_with_one_class,{}",
        audit.endpoints_with_one_class
    );
    println!(
        "endpoints_with_multiple_classes,{}",
        audit.endpoints_with_multiple_classes
    );
    println!(
        "maximum_classes_per_endpoint,{}",
        audit.maximum_classes_per_endpoint
    );
    for (multiplicity, endpoints) in audit.word_multiplicity_histogram {
        println!("word_multiplicity_{multiplicity},{endpoints}");
    }
    for (classes, endpoints) in audit.class_count_histogram {
        println!("commutation_class_count_{classes},{endpoints}");
    }
    for (index, equality) in audit.sample_cross_class_equalities.into_iter().enumerate() {
        println!("sample_cross_class_equality_{index},{equality}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> String {
        std::env::var("REF027_PUZZLE_INFO").expect("REF027_PUZZLE_INFO must be set")
    }

    #[test]
    fn canonical_form_allows_only_adjacent_declared_commutations() {
        let commuting = vec![
            vec![false, true, false],
            vec![true, false, false],
            vec![false, false, false],
        ];
        assert_eq!(commutation_normal_form([1, 0, 2], &commuting), [0, 1, 2]);
        assert_eq!(commutation_normal_form([0, 2, 1], &commuting), [0, 2, 1]);
    }

    #[test]
    fn production_f3_word_partition_conserves_words_classes_and_endpoints() {
        let audit = f3_word_class_audit(&fixture_path()).unwrap();
        assert_eq!(audit.f3_endpoints, 6208);
        assert!(audit.shortest_word_occurrences >= audit.candidate_records);
        assert_eq!(audit.candidate_records, 9216);
        assert!(audit.commutation_classes >= audit.f3_endpoints);
        assert!(audit.commutation_classes <= audit.shortest_word_occurrences);
        assert_eq!(
            audit.word_extras,
            audit.shortest_word_occurrences - audit.f3_endpoints
        );
        assert_eq!(
            audit.extras_after_commutation,
            audit.commutation_classes - audit.f3_endpoints
        );
        assert_eq!(
            audit.commutation_explained_word_extras + audit.extras_after_commutation,
            audit.word_extras
        );
    }
}
