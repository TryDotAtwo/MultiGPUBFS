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

    fn bounded(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }
}

fn canonical(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn add_edge(
    graph: &mut [Vec<usize>],
    edges: &mut HashSet<(usize, usize)>,
    left: usize,
    right: usize,
) {
    assert!(left != right);
    assert!(edges.insert(canonical(left, right)));
    graph[left].push(right);
    graph[right].push(left);
}

fn sample_small_world(
    n: usize,
    shortcut_count: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, HashSet<(usize, usize)>, usize) {
    let mut graph = vec![Vec::new(); n];
    let mut edges = HashSet::new();
    for vertex in 0..n {
        for offset in 1..=2 {
            let next = (vertex + offset) % n;
            if !edges.contains(&canonical(vertex, next)) {
                add_edge(&mut graph, &mut edges, vertex, next);
            }
        }
    }

    let mut shortcuts = HashSet::new();
    let mut attempts = 0usize;
    let mut rng = Rng(seed.max(1));
    while shortcuts.len() < shortcut_count {
        attempts += 1;
        let left = rng.bounded(n);
        let right = rng.bounded(n);
        if left == right || edges.contains(&canonical(left, right)) {
            continue;
        }
        add_edge(&mut graph, &mut edges, left, right);
        shortcuts.insert(canonical(left, right));
    }
    (graph, shortcuts, attempts)
}

fn distances(graph: &[Vec<usize>], root: usize) -> Vec<usize> {
    let mut distance = vec![usize::MAX; graph.len()];
    let mut queue = VecDeque::from([root]);
    distance[root] = 0;
    while let Some(vertex) = queue.pop_front() {
        let next_depth = distance[vertex] + 1;
        for &next in &graph[vertex] {
            if distance[next] == usize::MAX {
                distance[next] = next_depth;
                queue.push_back(next);
            }
        }
    }
    assert!(distance.iter().all(|&item| item != usize::MAX));
    distance
}

fn ring_distance(n: usize, root: usize, vertex: usize) -> usize {
    let clockwise = root.abs_diff(vertex);
    let cyclic = clockwise.min(n - clockwise);
    cyclic.div_ceil(2)
}

fn frontier(distance: &[usize]) -> Vec<usize> {
    let maximum = *distance.iter().max().unwrap();
    let mut out = vec![0; maximum + 1];
    for &depth in distance {
        out[depth] += 1;
    }
    out
}

fn owner_profile(distance: &[usize], split: usize) -> Vec<(usize, usize)> {
    let maximum = *distance.iter().max().unwrap();
    let mut out = vec![(0, 0); maximum + 1];
    for (vertex, &depth) in distance.iter().enumerate() {
        if vertex < split {
            out[depth].0 += 1;
        } else {
            out[depth].1 += 1;
        }
    }
    out
}

fn remote_fractions(graph: &[Vec<usize>], shortcuts: &HashSet<(usize, usize)>) -> (f64, f64, f64) {
    let split = graph.len() / 2;
    let mut edges = 0usize;
    let mut contiguous_remote = 0usize;
    let mut striped_remote = 0usize;
    let mut shortcut_remote = 0usize;
    for left in 0..graph.len() {
        for &right in &graph[left] {
            if left >= right {
                continue;
            }
            edges += 1;
            contiguous_remote += usize::from((left < split) != (right < split));
            striped_remote += usize::from(left % 2 != right % 2);
            if shortcuts.contains(&(left, right)) {
                shortcut_remote += usize::from((left < split) != (right < split));
            }
        }
    }
    (
        contiguous_remote as f64 / edges as f64,
        striped_remote as f64 / edges as f64,
        if shortcuts.is_empty() {
            0.0
        } else {
            shortcut_remote as f64 / shortcuts.len() as f64
        },
    )
}

fn clipped_profile(values: &[usize]) -> String {
    if values.len() <= 24 {
        return format!("{values:?}");
    }
    let peak = values
        .iter()
        .enumerate()
        .max_by_key(|&(_, value)| value)
        .unwrap()
        .0;
    let start = peak.saturating_sub(3);
    let end = (peak + 4).min(values.len());
    format!(
        "head={:?} peak_window@{}={:?} tail={:?}",
        &values[..8],
        start,
        &values[start..end],
        &values[values.len() - 8..]
    )
}

fn main() {
    const N: usize = 4096;
    const SAMPLES: usize = 20;
    const ROOT: usize = N / 4;

    for shortcut_count in [0, 4, 16, 64, 256, 1024] {
        let mut eccentricity_sum = 0.0;
        let mut mean_distance_sum = 0.0;
        let mut peak_sum = 0.0;
        let mut beneficial_fraction_sum = 0.0;
        let mut mean_saving_sum = 0.0;
        let mut maximum_saving = 0usize;
        let mut first_mixed_depth_sum = 0.0;
        let mut contiguous_remote_sum = 0.0;
        let mut striped_remote_sum = 0.0;
        let mut shortcut_remote_sum = 0.0;
        let mut attempts_sum = 0usize;
        let mut representative = None;

        for sample in 0..SAMPLES {
            let seed = 0x6a09e667f3bcc909u64 ^ ((sample as u64 + 1) * 0xbb67ae8584caa73b);
            let (graph, shortcuts, attempts) = sample_small_world(N, shortcut_count, seed);
            let distance = distances(&graph, ROOT);
            let frontiers = frontier(&distance);
            let owners = owner_profile(&distance, N / 2);
            let first_mixed = owners
                .iter()
                .position(|&(left, right)| left > 0 && right > 0)
                .unwrap();

            let mut beneficial = 0usize;
            let mut saving_sum = 0usize;
            for (vertex, &hop) in distance.iter().enumerate() {
                let baseline = ring_distance(N, ROOT, vertex);
                assert!(hop <= baseline);
                if hop < baseline {
                    beneficial += 1;
                    saving_sum += baseline - hop;
                    maximum_saving = maximum_saving.max(baseline - hop);
                }
            }

            let (contiguous, striped, shortcut_remote) = remote_fractions(&graph, &shortcuts);
            eccentricity_sum += *distance.iter().max().unwrap() as f64;
            mean_distance_sum += distance.iter().sum::<usize>() as f64 / N as f64;
            peak_sum += *frontiers.iter().max().unwrap() as f64;
            beneficial_fraction_sum += beneficial as f64 / N as f64;
            mean_saving_sum += saving_sum as f64 / N as f64;
            first_mixed_depth_sum += first_mixed as f64;
            contiguous_remote_sum += contiguous;
            striped_remote_sum += striped;
            shortcut_remote_sum += shortcut_remote;
            attempts_sum += attempts;

            if sample == 0 {
                representative = Some((frontiers, owners));
            }
        }

        let denominator = SAMPLES as f64;
        let (frontiers, owners) = representative.unwrap();
        println!(
            "small_world n={N} samples={SAMPLES} shortcuts={shortcut_count} mean_degree={:.3} eccentricity_mean={:.2} root_mean_distance={:.2} frontier_peak_mean={:.2} beneficial_fraction_mean={:.4} mean_hop_saving_all_vertices={:.2} maximum_hop_saving={maximum_saving} first_mixed_owner_depth_mean={:.2} contiguous_remote_mean={:.4} striped_remote_mean={:.4} shortcut_remote_mean={:.4} shortcut_attempts_mean={:.2}",
            4.0 + 2.0 * shortcut_count as f64 / N as f64,
            eccentricity_sum / denominator,
            mean_distance_sum / denominator,
            peak_sum / denominator,
            beneficial_fraction_sum / denominator,
            mean_saving_sum / denominator,
            first_mixed_depth_sum / denominator,
            contiguous_remote_sum / denominator,
            striped_remote_sum / denominator,
            shortcut_remote_sum / denominator,
            attempts_sum as f64 / denominator,
        );
        println!("  representative_frontiers={}", clipped_profile(&frontiers));
        let owner_sizes: Vec<_> = owners
            .iter()
            .map(|&(left, right)| left.abs_diff(right))
            .collect();
        println!(
            "  representative_owner_imbalance={}",
            clipped_profile(&owner_sizes)
        );
    }
}
