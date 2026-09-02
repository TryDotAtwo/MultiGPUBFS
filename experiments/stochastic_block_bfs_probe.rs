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

fn vertex_type(vertex: usize, n: usize) -> usize {
    usize::from(vertex >= n / 2)
}

fn sample_sbm(n: usize, within: f64, across: f64, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Rng(seed.max(1));
    let mut graph = vec![Vec::new(); n];
    for left in 0..n {
        for right in left + 1..n {
            let coefficient = if vertex_type(left, n) == vertex_type(right, n) {
                within
            } else {
                across
            };
            if rng.next_f64() < coefficient / n as f64 {
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

fn typed_layers(distance: &[Option<usize>]) -> Vec<(usize, usize)> {
    let maximum = distance.iter().flatten().copied().max().unwrap();
    let mut out = vec![(0usize, 0usize); maximum + 1];
    for (vertex, depth) in distance.iter().enumerate() {
        if let Some(depth) = depth {
            if vertex_type(vertex, distance.len()) == 0 {
                out[*depth].0 += 1;
            } else {
                out[*depth].1 += 1;
            }
        }
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
        let mut size = 0usize;
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

fn remote_fractions(graph: &[Vec<usize>]) -> (f64, f64) {
    let mut edges = 0usize;
    let mut block_remote = 0usize;
    let mut striped_remote = 0usize;
    for left in 0..graph.len() {
        for &right in &graph[left] {
            if left >= right {
                continue;
            }
            edges += 1;
            block_remote +=
                usize::from(vertex_type(left, graph.len()) != vertex_type(right, graph.len()));
            striped_remote += usize::from(left % 2 != right % 2);
        }
    }
    (
        block_remote as f64 / edges as f64,
        striped_remote as f64 / edges as f64,
    )
}

fn main() {
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    for (name, within, across) in [
        ("segregated", 8.0, 0.0),
        ("assortative", 7.5, 0.5),
        ("neutral", 4.0, 4.0),
        ("disassortative", 0.5, 7.5),
    ] {
        let mut largest_fractions = Vec::new();
        let mut root_fractions = Vec::new();
        let mut block_remote = Vec::new();
        let mut striped_remote = Vec::new();
        let mut representative = None;
        for sample in 0..SAMPLES {
            let graph = sample_sbm(
                N,
                within,
                across,
                0xa0761d6478bd642fu64 ^ ((sample as u64 + 1) * 0xe7037ed1),
            );
            let (_, largest_size) = largest_component_root(&graph);
            let root_distance = distances(&graph, 0);
            let root_layers = typed_layers(&root_distance);
            let (by_block, striped) = remote_fractions(&graph);
            largest_fractions.push(largest_size as f64 / N as f64);
            root_fractions.push(
                root_layers
                    .iter()
                    .map(|(left, right)| left + right)
                    .sum::<usize>() as f64
                    / N as f64,
            );
            block_remote.push(by_block);
            striped_remote.push(striped);
            if sample == 0 {
                representative = Some(root_layers);
            }
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        println!(
            "sbm name={name} n={N} samples={SAMPLES} within={within:.1} across={across:.1} mean_degree~{:.1} branching_eigenvalues=({:.1},{:.1}) largest_fraction_mean={:.4} root0_fraction_mean={:.4} block_owner_remote_mean={:.4} striped_owner_remote_mean={:.4}",
            (within + across) / 2.0,
            (within + across) / 2.0,
            (within - across) / 2.0,
            mean(&largest_fractions),
            mean(&root_fractions),
            mean(&block_remote),
            mean(&striped_remote),
        );
        println!(
            "  representative_root0_typed_layers={:?}",
            representative.unwrap()
        );
    }
}
