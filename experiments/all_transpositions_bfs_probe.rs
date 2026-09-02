use std::collections::{HashMap, VecDeque};

fn successors(permutation: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for i in 0..permutation.len() {
        for j in i + 1..permutation.len() {
            let mut next = permutation.to_vec();
            next.swap(i, j);
            out.push(next);
        }
    }
    out
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

fn cycle_count(permutation: &[u8]) -> usize {
    let mut seen = vec![false; permutation.len()];
    let mut cycles = 0usize;
    for start in 0..permutation.len() {
        if seen[start] {
            continue;
        }
        cycles += 1;
        let mut current = start;
        while !seen[current] {
            seen[current] = true;
            current = permutation[current] as usize;
        }
    }
    cycles
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

fn stirling_layers(size: usize) -> Vec<usize> {
    let mut row = vec![0usize; size + 1];
    row[0] = 1;
    for n in 1..=size {
        let mut next = vec![0usize; size + 1];
        for cycles in 1..=n {
            next[cycles] = row[cycles - 1] + (n - 1) * row[cycles];
        }
        row = next;
    }
    (0..size).map(|depth| row[size - depth]).collect()
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn audit(size: usize) {
    let distance = bfs(size);
    let diameter = *distance.values().max().unwrap();
    let mut layers = vec![0usize; diameter + 1];
    let mut metric_mismatches = 0usize;
    let mut parity_mismatches = 0usize;
    for (permutation, &depth) in &distance {
        layers[depth] += 1;
        metric_mismatches += usize::from(depth != size - cycle_count(permutation));
        parity_mismatches += usize::from((depth % 2 == 1) != odd(permutation));
    }
    let expected_layers = stirling_layers(size);
    println!(
        "T({size}) states={} expected={} degree={} diameter={diameter} metric_mismatches={metric_mismatches} parity_mismatches={parity_mismatches} stirling_match={} layers={layers:?}",
        distance.len(),
        factorial(size),
        size * (size - 1) / 2,
        layers == expected_layers
    );
}

fn main() {
    for size in 2..=8 {
        audit(size);
    }
}
