use std::collections::VecDeque;

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

    fn bounded(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for right in (1..values.len()).rev() {
            let left = self.bounded(right + 1);
            values.swap(left, right);
        }
    }
}

fn degree_sequence(name: &str, n: usize) -> Vec<usize> {
    match name {
        "regular-4" => vec![4; n],
        "half-2-half-6" => (0..n)
            .map(|index| if index < n / 2 { 2 } else { 6 })
            .collect(),
        "half-1-half-7" => (0..n)
            .map(|index| if index < n / 2 { 1 } else { 7 })
            .collect(),
        _ => unreachable!(),
    }
}

fn sample_configuration(mut degrees: Vec<usize>, seed: u64) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut rng = Rng(seed.max(1));
    rng.shuffle(&mut degrees);
    let mut stubs = Vec::with_capacity(degrees.iter().sum());
    for (vertex, &degree) in degrees.iter().enumerate() {
        stubs.extend(std::iter::repeat(vertex).take(degree));
    }
    assert_eq!(stubs.len() % 2, 0);
    rng.shuffle(&mut stubs);

    let mut graph = vec![Vec::new(); degrees.len()];
    for pair in stubs.chunks_exact(2) {
        graph[pair[0]].push(pair[1]);
        graph[pair[1]].push(pair[0]);
    }
    for (neighbors, &degree) in graph.iter().zip(&degrees) {
        assert_eq!(neighbors.len(), degree);
    }
    (graph, degrees)
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

fn largest_component(graph: &[Vec<usize>]) -> usize {
    let mut seen = vec![false; graph.len()];
    let mut largest = 0;
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
        largest = largest.max(size);
    }
    largest
}

fn layer_profile(
    graph: &[Vec<usize>],
    degrees: &[usize],
    distance: &[Option<usize>],
) -> Vec<String> {
    let maximum = distance.iter().flatten().copied().max().unwrap();
    let mut out = Vec::new();
    for depth in 0..=maximum {
        let vertices: Vec<_> = distance
            .iter()
            .enumerate()
            .filter_map(|(vertex, &found)| (found == Some(depth)).then_some(vertex))
            .collect();
        let degree_sum: usize = vertices.iter().map(|&vertex| degrees[vertex]).sum();
        let mut previous = 0usize;
        let mut same = 0usize;
        let mut outward = 0usize;
        for &vertex in &vertices {
            for &next in &graph[vertex] {
                match distance[next] {
                    Some(next_depth) if next_depth + 1 == depth => previous += 1,
                    Some(next_depth) if next_depth == depth => same += 1,
                    Some(next_depth) if next_depth == depth + 1 => outward += 1,
                    _ => {}
                }
            }
        }
        let next_size = distance
            .iter()
            .filter(|&&found| found == Some(depth + 1))
            .count();
        out.push(format!(
            "(d={depth},n={},mean_deg={:.2},prev={previous},same={same},out={outward},new={next_size},out/new={:.2})",
            vertices.len(),
            degree_sum as f64 / vertices.len() as f64,
            if next_size == 0 {
                0.0
            } else {
                outward as f64 / next_size as f64
            }
        ));
    }
    out
}

fn main() {
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    for name in ["regular-4", "half-2-half-6", "half-1-half-7"] {
        let template = degree_sequence(name, N);
        let mean = template.iter().sum::<usize>() as f64 / N as f64;
        let second_factorial = template
            .iter()
            .map(|&degree| degree * degree.saturating_sub(1))
            .sum::<usize>() as f64
            / N as f64;
        let excess = second_factorial / mean;
        let edge_endpoint_degree = excess + 1.0;

        let mut largest_sum = 0.0;
        let mut root_sum = 0.0;
        let mut root_degree_sum = 0.0;
        let mut root_giant_hits = 0usize;
        let mut representative = Vec::new();
        for sample in 0..SAMPLES {
            let seed = 0x9e3779b97f4a7c15u64 ^ ((sample as u64 + 1) * 0xbf58476d1ce4e5b9);
            let (graph, degrees) = sample_configuration(template.clone(), seed);
            let distance = distances(&graph, 0);
            let largest_size = largest_component(&graph);
            let root_size = distance.iter().filter(|entry| entry.is_some()).count();
            largest_sum += largest_size as f64 / N as f64;
            root_sum += root_size as f64 / N as f64;
            root_degree_sum += degrees[0] as f64;
            root_giant_hits += usize::from(root_size == largest_size);
            if sample == 0 {
                representative = layer_profile(&graph, &degrees, &distance);
            }
        }
        println!(
            "configuration name={name} n={N} samples={SAMPLES} mean_degree={mean:.2} excess_mean={excess:.2} edge_endpoint_degree_mean={edge_endpoint_degree:.2} root_degree_mean={:.2} root_giant_hits={root_giant_hits}/{SAMPLES} largest_fraction_mean={:.4} root0_fraction_mean={:.4}",
            root_degree_sum / SAMPLES as f64,
            largest_sum / SAMPLES as f64,
            root_sum / SAMPLES as f64,
        );
        println!("  representative_layers={}", representative.join(" "));
    }
}
