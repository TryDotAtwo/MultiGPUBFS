use std::collections::{HashMap, VecDeque};

fn successors(permutation: &[u8]) -> impl Iterator<Item = Vec<u8>> + '_ {
    (1..permutation.len()).map(|position| {
        let mut next = permutation.to_vec();
        next.swap(0, position);
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

fn star_length(permutation: &[u8]) -> usize {
    let mut seen = vec![false; permutation.len()];
    let mut support = 0usize;
    let mut nontrivial_cycles = 0usize;
    let mut center_in_nontrivial_cycle = false;
    for start in 0..permutation.len() {
        if seen[start] {
            continue;
        }
        let mut current = start;
        let mut length = 0usize;
        loop {
            seen[current] = true;
            length += 1;
            current = permutation[current] as usize;
            if seen[current] {
                break;
            }
        }
        if length > 1 {
            support += length;
            nontrivial_cycles += 1;
            center_in_nontrivial_cycle |= start == 0;
        }
    }
    support + nontrivial_cycles - 2 * usize::from(center_in_nontrivial_cycle)
}

fn odd(permutation: &[u8]) -> bool {
    let mut inversions = 0usize;
    for i in 0..permutation.len() {
        for j in i + 1..permutation.len() {
            inversions += usize::from(permutation[i] > permutation[j]);
        }
    }
    inversions % 2 == 1
}

fn audit(size: usize) {
    let distance = bfs(size);
    let diameter = *distance.values().max().unwrap();
    let mut layers = vec![0usize; diameter + 1];
    let mut metric_mismatches = 0usize;
    let mut parity_mismatches = 0usize;
    for (permutation, &depth) in &distance {
        layers[depth] += 1;
        metric_mismatches += usize::from(depth != star_length(permutation));
        parity_mismatches += usize::from((depth % 2 == 1) != odd(permutation));
    }
    let expected_diameter = 3 * (size - 1) / 2;
    println!(
        "ST({size}) states={} expected={} degree={} diameter={diameter} expected_diameter={expected_diameter} metric_mismatches={metric_mismatches} parity_mismatches={parity_mismatches} layers={layers:?}",
        distance.len(),
        factorial(size),
        size - 1
    );
}

fn main() {
    for size in 2..=8 {
        audit(size);
    }
}
