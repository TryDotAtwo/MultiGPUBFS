use std::collections::{HashSet, VecDeque};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn index_below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for end in (1..values.len()).rev() {
            let index = self.index_below(end + 1);
            values.swap(index, end);
        }
    }
}

fn random_regular(n: usize, degree: usize, seed: u64) -> (Vec<Vec<usize>>, usize) {
    assert_eq!(n * degree % 2, 0);
    let original: Vec<_> = (0..n)
        .flat_map(|vertex| std::iter::repeat(vertex).take(degree))
        .collect();
    let mut rng = Rng(seed.max(1));
    for attempt in 1usize.. {
        let mut stubs = original.clone();
        rng.shuffle(&mut stubs);
        let mut edges = HashSet::with_capacity(n * degree / 2);
        let mut valid = true;
        for pair in stubs.chunks_exact(2) {
            let (left, right) = if pair[0] < pair[1] {
                (pair[0], pair[1])
            } else {
                (pair[1], pair[0])
            };
            if left == right || !edges.insert((left, right)) {
                valid = false;
                break;
            }
        }
        if valid {
            let mut graph = vec![Vec::with_capacity(degree); n];
            for (left, right) in edges {
                graph[left].push(right);
                graph[right].push(left);
            }
            return (graph, attempt);
        }
    }
    unreachable!()
}

fn distances(graph: &[Vec<usize>], root: usize) -> Vec<Option<usize>> {
    let mut distance = vec![None; graph.len()];
    let mut queue = VecDeque::from([root]);
    distance[root] = Some(0);
    while let Some(vertex) = queue.pop_front() {
        let next_depth = distance[vertex].unwrap() + 1;
        for &next in &graph[vertex] {
            if distance[next].is_none() {
                distance[next] = Some(next_depth);
                queue.push_back(next);
            }
        }
    }
    distance
}

fn layers(distance: &[Option<usize>]) -> Vec<usize> {
    let maximum = distance.iter().flatten().copied().max().unwrap();
    let mut out = vec![0usize; maximum + 1];
    for depth in distance.iter().flatten().copied() {
        out[depth] += 1;
    }
    out
}

fn outward_occurrences_per_new(graph: &[Vec<usize>], distance: &[Option<usize>]) -> Vec<f64> {
    let layer_sizes = layers(distance);
    let mut outward = vec![0usize; layer_sizes.len().saturating_sub(1)];
    for vertex in 0..graph.len() {
        let Some(depth) = distance[vertex] else {
            continue;
        };
        if depth + 1 >= layer_sizes.len() {
            continue;
        }
        outward[depth] += graph[vertex]
            .iter()
            .filter(|&&next| distance[next] == Some(depth + 1))
            .count();
    }
    outward
        .into_iter()
        .enumerate()
        .map(|(depth, count)| count as f64 / layer_sizes[depth + 1] as f64)
        .collect()
}

fn intersection_ranges(
    graph: &[Vec<usize>],
    distance: &[Option<usize>],
) -> Vec<((usize, usize), (usize, usize), (usize, usize))> {
    let layer_count = layers(distance).len();
    let mut ranges = vec![((usize::MAX, 0), (usize::MAX, 0), (usize::MAX, 0)); layer_count];
    for vertex in 0..graph.len() {
        let depth = distance[vertex].unwrap();
        let mut counts = [0usize; 3];
        for &next in &graph[vertex] {
            let next_depth = distance[next].unwrap();
            counts[usize::from(next_depth >= depth) + usize::from(next_depth > depth)] += 1;
        }
        let entry = &mut ranges[depth];
        entry.0 .0 = entry.0 .0.min(counts[0]);
        entry.0 .1 = entry.0 .1.max(counts[0]);
        entry.1 .0 = entry.1 .0.min(counts[1]);
        entry.1 .1 = entry.1 .1.max(counts[1]);
        entry.2 .0 = entry.2 .0.min(counts[2]);
        entry.2 .1 = entry.2 .1.max(counts[2]);
    }
    ranges
}

fn main() {
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    for degree in [3usize, 4] {
        let mut attempts = Vec::new();
        let mut eccentricities = Vec::new();
        let mut connected = 0usize;
        let mut representative = None;
        for sample in 0..SAMPLES {
            let (graph, generation_attempts) = random_regular(
                N,
                degree,
                0xd1b54a32d192ed03u64 ^ ((sample as u64 + 1) * 0x9e3779b1),
            );
            assert!(graph.iter().all(|neighbors| neighbors.len() == degree));
            let distance = distances(&graph, 0);
            let profile = layers(&distance);
            attempts.push(generation_attempts);
            eccentricities.push(profile.len() - 1);
            connected += usize::from(profile.iter().sum::<usize>() == N);
            if sample == 0 {
                let multiplicity = outward_occurrences_per_new(&graph, &distance);
                let ranges = intersection_ranges(&graph, &distance);
                representative = Some((profile, multiplicity, ranges));
            }
        }
        let attempts_mean = attempts.iter().sum::<usize>() as f64 / SAMPLES as f64;
        let eccentricity_mean = eccentricities.iter().sum::<usize>() as f64 / SAMPLES as f64;
        let (profile, multiplicity, ranges) = representative.unwrap();
        let tree_bounds: Vec<_> = (0..profile.len())
            .map(|depth| {
                if depth == 0 {
                    1
                } else {
                    degree * (degree - 1).pow((depth - 1) as u32)
                }
            })
            .collect();
        println!(
            "random_{degree}_regular n={N} samples={SAMPLES} connected={connected}/{SAMPLES} rejection_attempts_mean={attempts_mean:.2} rejection_attempts_range=[{},{}] root_eccentricity_mean={eccentricity_mean:.2} root_eccentricity_range=[{},{}]",
            attempts.iter().min().unwrap(),
            attempts.iter().max().unwrap(),
            eccentricities.iter().min().unwrap(),
            eccentricities.iter().max().unwrap(),
        );
        println!("  representative_layers={profile:?}");
        println!("  regular_tree_layer_bounds={tree_bounds:?}");
        println!("  representative_outward_occurrences_per_new={multiplicity:?}");
        println!("  representative_inward_same_outward_ranges={ranges:?}");
    }
}
