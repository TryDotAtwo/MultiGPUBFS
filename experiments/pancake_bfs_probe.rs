use std::collections::{HashMap, VecDeque};

fn successors(permutation: &[u8]) -> impl Iterator<Item = Vec<u8>> + '_ {
    (2..=permutation.len()).map(|prefix| {
        let mut next = permutation.to_vec();
        next[..prefix].reverse();
        next
    })
}

fn bfs(size: usize) -> HashMap<Vec<u8>, usize> {
    let identity: Vec<u8> = (0..size as u8).collect();
    let mut distance = HashMap::from([(identity.clone(), 0usize)]);
    let mut queue = VecDeque::from([identity]);
    while let Some(permutation) = queue.pop_front() {
        let next_distance = distance[&permutation] + 1;
        for next in successors(&permutation) {
            if !distance.contains_key(&next) {
                distance.insert(next.clone(), next_distance);
                queue.push_back(next);
            }
        }
    }
    distance
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn audit(size: usize) {
    let distance = bfs(size);
    let diameter = *distance.values().max().unwrap();
    let mut layers = vec![0usize; diameter + 1];
    for depth in distance.values() {
        layers[*depth] += 1;
    }
    let expected_s1 = size.saturating_sub(1);
    let expected_s2 = size.saturating_sub(1) * size.saturating_sub(2);
    let expected_s3 = if size >= 3 {
        (size - 1) * (size - 2) * (size - 2) - 1
    } else {
        0
    };
    let early_match = layers.get(1).copied().unwrap_or(0) == expected_s1
        && layers.get(2).copied().unwrap_or(0) == expected_s2
        && (size < 3 || layers.get(3).copied().unwrap_or(0) == expected_s3);
    println!(
        "P({size}) states={} expected={} degree={} diameter={diameter} early_formula_match={early_match} layers={layers:?}",
        distance.len(),
        factorial(size),
        size.saturating_sub(1)
    );
}

fn main() {
    for size in 2..=8 {
        audit(size);
    }
}
