use std::collections::VecDeque;

fn power(base: usize, exponent: usize) -> usize {
    (0..exponent).fold(1, |value, _| value * base)
}

fn binomial(n: usize, k: usize) -> usize {
    (0..k).fold(1, |value, i| value * (n - i) / (i + 1))
}

fn weight(mut state: usize, dimensions: usize, alphabet: usize) -> usize {
    let mut out = 0;
    for _ in 0..dimensions {
        out += usize::from(state % alphabet != 0);
        state /= alphabet;
    }
    out
}

fn neighbors(state: usize, dimensions: usize, alphabet: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(dimensions * (alphabet - 1));
    let mut place = 1;
    for _ in 0..dimensions {
        let old = state / place % alphabet;
        for new in 0..alphabet {
            if new != old {
                out.push(state + new * place - old * place);
            }
        }
        place *= alphabet;
    }
    out
}

fn audit(dimensions: usize, alphabet: usize) {
    let states = power(alphabet, dimensions);
    let mut distance = vec![None; states];
    let mut shortest_paths = vec![0usize; states];
    let mut queue = VecDeque::from([0usize]);
    distance[0] = Some(0);
    shortest_paths[0] = 1;
    while let Some(state) = queue.pop_front() {
        let depth = distance[state].unwrap();
        for next in neighbors(state, dimensions, alphabet) {
            if distance[next].is_none() {
                distance[next] = Some(depth + 1);
                queue.push_back(next);
            }
            if distance[next] == Some(depth + 1) {
                shortest_paths[next] += shortest_paths[state];
            }
        }
    }

    let mut layers = vec![0usize; dimensions + 1];
    let mut expected_layers = Vec::new();
    let mut distance_mismatches = 0usize;
    let mut intersection_mismatches = 0usize;
    let mut path_mismatches = 0usize;
    for depth in 0..=dimensions {
        expected_layers.push(binomial(dimensions, depth) * power(alphabet - 1, depth));
    }
    for state in 0..states {
        let depth = distance[state].unwrap();
        layers[depth] += 1;
        distance_mismatches += usize::from(depth != weight(state, dimensions, alphabet));
        path_mismatches += usize::from(shortest_paths[state] != (1..=depth).product());
        let mut inward = 0;
        let mut same = 0;
        let mut outward = 0;
        for next in neighbors(state, dimensions, alphabet) {
            match distance[next].unwrap().cmp(&depth) {
                std::cmp::Ordering::Less => inward += 1,
                std::cmp::Ordering::Equal => same += 1,
                std::cmp::Ordering::Greater => outward += 1,
            }
        }
        intersection_mismatches += usize::from(
            inward != depth
                || same != depth * (alphabet - 2)
                || outward != (dimensions - depth) * (alphabet - 1),
        );
    }
    println!(
        "H({dimensions},{alphabet}) states={states} degree={} diameter={} layers={layers:?} layer_formula_match={} distance_mismatches={distance_mismatches} intersection_mismatches={intersection_mismatches} path_mismatches={path_mismatches}",
        dimensions * (alphabet - 1),
        dimensions,
        layers == expected_layers
    );
}

fn main() {
    for alphabet in 2..=4 {
        for dimensions in 1..=5 {
            audit(dimensions, alphabet);
        }
    }
}
