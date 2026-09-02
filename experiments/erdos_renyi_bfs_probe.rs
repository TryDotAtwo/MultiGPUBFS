use std::collections::VecDeque;

struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        ((value >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn erdos_renyi(n: usize, mean_degree: f64, seed: u64) -> Vec<Vec<usize>> {
    let probability = mean_degree / (n - 1) as f64;
    let mut rng = Rng(seed.max(1));
    let mut graph = vec![Vec::new(); n];
    for left in 0..n {
        for right in left + 1..n {
            if rng.next_f64() < probability {
                graph[left].push(right);
                graph[right].push(left);
            }
        }
    }
    graph
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

fn largest_component_root(graph: &[Vec<usize>]) -> (usize, usize) {
    let mut seen = vec![false; graph.len()];
    let mut best = (0usize, 0usize);
    for root in 0..graph.len() {
        if seen[root] {
            continue;
        }
        let mut size = 0;
        let mut queue = VecDeque::from([root]);
        seen[root] = true;
        while let Some(vertex) = queue.pop_front() {
            size += 1;
            for &next in &graph[vertex] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        if size > best.1 {
            best = (root, size);
        }
    }
    best
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
        .map(|(depth, occurrences)| occurrences as f64 / layer_sizes[depth + 1] as f64)
        .collect()
}

fn predicted_giant_fraction(mean_degree: f64) -> f64 {
    if mean_degree <= 1.0 {
        return 0.0;
    }
    let mut fraction = 1.0 - (-mean_degree).exp();
    for _ in 0..100 {
        fraction = 1.0 - (-mean_degree * fraction).exp();
    }
    fraction
}

fn main() {
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    for mean_degree in [0.8, 1.0, 1.2, 4.0] {
        let mut largest_fractions = Vec::new();
        let mut root_fractions = Vec::new();
        let mut root_reaches_depth_five = 0usize;
        let mut representative = None;
        for sample in 0..SAMPLES {
            let graph = erdos_renyi(
                N,
                mean_degree,
                0x9e3779b97f4a7c15u64 ^ ((sample as u64 + 1) * 0x100000001b3),
            );
            let (largest_root, largest_size) = largest_component_root(&graph);
            let root_distance = distances(&graph, 0);
            let root_layers = layers(&root_distance);
            largest_fractions.push(largest_size as f64 / N as f64);
            root_fractions.push(root_layers.iter().sum::<usize>() as f64 / N as f64);
            root_reaches_depth_five += usize::from(root_layers.len() > 5);
            if sample == 0 {
                let largest_distance = distances(&graph, largest_root);
                representative = Some((
                    layers(&largest_distance),
                    outward_occurrences_per_new(&graph, &largest_distance),
                ));
            }
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let (representative_layers, representative_multiplicity) = representative.unwrap();
        println!(
            "G(n,c/n) n={N} c={mean_degree:.1} samples={SAMPLES} predicted_giant={:.4} largest_fraction_mean={:.4} largest_fraction_range=[{:.4},{:.4}] root0_fraction_mean={:.4} root0_reaches_depth5={root_reaches_depth_five}/{SAMPLES}",
            predicted_giant_fraction(mean_degree),
            mean(&largest_fractions),
            largest_fractions.iter().copied().fold(f64::INFINITY, f64::min),
            largest_fractions.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean(&root_fractions),
        );
        println!("  representative_largest_layers={representative_layers:?}");
        println!("  representative_outward_occurrences_per_new={representative_multiplicity:?}");
    }
}
