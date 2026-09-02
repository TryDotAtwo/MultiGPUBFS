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

fn sample_digraph(n: usize, c: f64, seed: u64) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut rng = Rng(seed.max(1));
    let mut graph = vec![Vec::new(); n];
    let mut transpose = vec![Vec::new(); n];
    for source in 0..n {
        for target in 0..n {
            if source != target && rng.next_f64() < c / n as f64 {
                graph[source].push(target);
                transpose[target].push(source);
            }
        }
    }
    (graph, transpose)
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
    let Some(maximum) = distance.iter().flatten().copied().max() else {
        return Vec::new();
    };
    let mut out = vec![0; maximum + 1];
    for &depth in distance.iter().flatten() {
        out[depth] += 1;
    }
    out
}

fn finish_order(graph: &[Vec<usize>]) -> Vec<usize> {
    let mut seen = vec![false; graph.len()];
    let mut order = Vec::with_capacity(graph.len());
    for root in 0..graph.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root, 0usize)];
        while let Some((vertex, next_index)) = stack.pop() {
            if next_index == graph[vertex].len() {
                order.push(vertex);
                continue;
            }
            stack.push((vertex, next_index + 1));
            let next = graph[vertex][next_index];
            if !seen[next] {
                seen[next] = true;
                stack.push((next, 0));
            }
        }
    }
    order
}

fn strongly_connected_components(
    graph: &[Vec<usize>],
    transpose: &[Vec<usize>],
) -> (Vec<usize>, Vec<usize>) {
    let order = finish_order(graph);
    let mut component = vec![usize::MAX; graph.len()];
    let mut sizes = Vec::new();
    for &root in order.iter().rev() {
        if component[root] != usize::MAX {
            continue;
        }
        let label = sizes.len();
        let mut size = 0;
        let mut stack = vec![root];
        component[root] = label;
        while let Some(vertex) = stack.pop() {
            size += 1;
            for &next in &transpose[vertex] {
                if component[next] == usize::MAX {
                    component[next] = label;
                    stack.push(next);
                }
            }
        }
        sizes.push(size);
    }
    (component, sizes)
}

fn reached(distance: &[Option<usize>]) -> usize {
    distance.iter().filter(|entry| entry.is_some()).count()
}

fn validate_scc_oracle() {
    for (case, c) in [0.5, 1.0, 2.0, 5.0].into_iter().enumerate() {
        let (graph, transpose) = sample_digraph(24, c, 0x243f6a8885a308d3 ^ case as u64);
        let (component, sizes) = strongly_connected_components(&graph, &transpose);
        assert_eq!(sizes.iter().sum::<usize>(), graph.len());
        for root in 0..graph.len() {
            let forward = distances(&graph, root);
            let reverse = distances(&transpose, root);
            for vertex in 0..graph.len() {
                let mutually_reachable = forward[vertex].is_some() && reverse[vertex].is_some();
                assert_eq!(component[root] == component[vertex], mutually_reachable);
            }
        }
    }
}

fn main() {
    validate_scc_oracle();
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    for c in [0.8, 1.0, 1.2, 4.0] {
        let mut largest_scc_sum = 0.0;
        let mut giant_in_sum = 0.0;
        let mut giant_out_sum = 0.0;
        let mut root_forward_sum = 0.0;
        let mut root_reverse_sum = 0.0;
        let mut root_scc_sum = 0.0;
        let mut root_in_gin = 0usize;
        let mut root_in_gout = 0usize;
        let mut root_in_gscc = 0usize;
        let mut representative = None;

        for sample in 0..SAMPLES {
            let seed = 0xd1b54a32d192ed03u64 ^ ((sample as u64 + 1) * 0x94d049bb133111eb);
            let (graph, transpose) = sample_digraph(N, c, seed);
            let (component, sizes) = strongly_connected_components(&graph, &transpose);
            let largest_label = sizes
                .iter()
                .enumerate()
                .max_by_key(|&(_, size)| size)
                .unwrap()
                .0;
            let core_root = component
                .iter()
                .position(|&label| label == largest_label)
                .unwrap();

            let root_forward = distances(&graph, 0);
            let root_reverse = distances(&transpose, 0);
            let core_forward = distances(&graph, core_root);
            let core_reverse = distances(&transpose, core_root);
            let root_scc = root_forward
                .iter()
                .zip(&root_reverse)
                .filter(|(forward, reverse)| forward.is_some() && reverse.is_some())
                .count();

            largest_scc_sum += sizes[largest_label] as f64 / N as f64;
            giant_in_sum += reached(&core_reverse) as f64 / N as f64;
            giant_out_sum += reached(&core_forward) as f64 / N as f64;
            root_forward_sum += reached(&root_forward) as f64 / N as f64;
            root_reverse_sum += reached(&root_reverse) as f64 / N as f64;
            root_scc_sum += root_scc as f64 / N as f64;
            root_in_gin += usize::from(root_forward[core_root].is_some());
            root_in_gout += usize::from(core_forward[0].is_some());
            root_in_gscc += usize::from(component[0] == largest_label);

            if sample == 0 {
                representative = Some((layers(&root_forward), layers(&root_reverse)));
            }
        }

        let denominator = SAMPLES as f64;
        let (forward_layers, reverse_layers) = representative.unwrap();
        println!(
            "directed_er c={c:.1} n={N} samples={SAMPLES} largest_scc_fraction_mean={:.4} gin_fraction_mean={:.4} gout_fraction_mean={:.4} root_forward_fraction_mean={:.4} root_reverse_fraction_mean={:.4} root_scc_fraction_mean={:.4} root_in_gin={root_in_gin}/{SAMPLES} root_in_gout={root_in_gout}/{SAMPLES} root_in_gscc={root_in_gscc}/{SAMPLES}",
            largest_scc_sum / denominator,
            giant_in_sum / denominator,
            giant_out_sum / denominator,
            root_forward_sum / denominator,
            root_reverse_sum / denominator,
            root_scc_sum / denominator,
        );
        println!("  representative_root0_forward_layers={forward_layers:?}");
        println!("  representative_root0_reverse_layers={reverse_layers:?}");
    }
}
