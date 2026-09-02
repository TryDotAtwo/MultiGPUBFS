use std::collections::VecDeque;

fn binomial(n: usize, k: usize) -> usize {
    (0..k).fold(1, |value, i| value * (n - i) / (i + 1))
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn vertices(n: usize, k: usize) -> Vec<usize> {
    (0..1usize << n)
        .filter(|state| state.count_ones() as usize == k)
        .collect()
}

fn neighbors(state: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for removed in 0..n {
        if state & (1 << removed) == 0 {
            continue;
        }
        for added in 0..n {
            if state & (1 << added) == 0 {
                out.push(state ^ (1 << removed) ^ (1 << added));
            }
        }
    }
    out
}

fn audit(n: usize, k: usize) {
    let root = (1usize << k) - 1;
    let universe = vertices(n, k);
    let mut distance = vec![None; 1usize << n];
    let mut shortest_paths = vec![0usize; 1usize << n];
    let mut queue = VecDeque::from([root]);
    distance[root] = Some(0);
    shortest_paths[root] = 1;
    while let Some(state) = queue.pop_front() {
        let depth = distance[state].unwrap();
        for next in neighbors(state, n) {
            if distance[next].is_none() {
                distance[next] = Some(depth + 1);
                queue.push_back(next);
            }
            if distance[next] == Some(depth + 1) {
                shortest_paths[next] += shortest_paths[state];
            }
        }
    }

    let diameter = k.min(n - k);
    let mut layers = vec![0usize; diameter + 1];
    let expected_layers: Vec<_> = (0..=diameter)
        .map(|depth| binomial(k, depth) * binomial(n - k, depth))
        .collect();
    let mut distance_mismatches = 0usize;
    let mut intersection_mismatches = 0usize;
    let mut path_mismatches = 0usize;
    for state in universe.iter().copied() {
        let depth = distance[state].unwrap();
        layers[depth] += 1;
        let expected_depth = k - (state & root).count_ones() as usize;
        distance_mismatches += usize::from(depth != expected_depth);
        let expected_paths = factorial(depth) * factorial(depth);
        path_mismatches += usize::from(shortest_paths[state] != expected_paths);

        let mut inward = 0;
        let mut same = 0;
        let mut outward = 0;
        for next in neighbors(state, n) {
            match distance[next].unwrap().cmp(&depth) {
                std::cmp::Ordering::Less => inward += 1,
                std::cmp::Ordering::Equal => same += 1,
                std::cmp::Ordering::Greater => outward += 1,
            }
        }
        intersection_mismatches += usize::from(
            inward != depth * depth
                || same != depth * (n - 2 * depth)
                || outward != (k - depth) * (n - k - depth),
        );
    }
    println!(
        "J({n},{k}) states={} expected={} degree={} diameter={diameter} layers={layers:?} layer_formula_match={} distance_mismatches={distance_mismatches} intersection_mismatches={intersection_mismatches} path_mismatches={path_mismatches}",
        universe.len(),
        binomial(n, k),
        k * (n - k),
        layers == expected_layers
    );
}

fn main() {
    for n in 2..=12 {
        for k in 1..=n / 2 {
            audit(n, k);
        }
    }
}
