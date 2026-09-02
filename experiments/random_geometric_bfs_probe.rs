use std::collections::VecDeque;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

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

fn distance(left: Point, right: Point, torus: bool) -> f64 {
    let mut dx = (left.x - right.x).abs();
    let mut dy = (left.y - right.y).abs();
    if torus {
        dx = dx.min(1.0 - dx);
        dy = dy.min(1.0 - dy);
    }
    (dx * dx + dy * dy).sqrt()
}

fn sample_graph(points: &[Point], radius: f64, torus: bool) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); points.len()];
    for left in 0..points.len() {
        for right in left + 1..points.len() {
            if distance(points[left], points[right], torus) <= radius {
                graph[left].push(right);
                graph[right].push(left);
            }
        }
    }
    graph
}

fn distances(graph: &[Vec<usize>], root: usize) -> Vec<Option<usize>> {
    let mut out = vec![None; graph.len()];
    let mut queue = VecDeque::from([root]);
    out[root] = Some(0);
    while let Some(vertex) = queue.pop_front() {
        let next_depth = out[vertex].unwrap() + 1;
        for &next in &graph[vertex] {
            if out[next].is_none() {
                out[next] = Some(next_depth);
                queue.push_back(next);
            }
        }
    }
    out
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

fn nearest(points: &[Point], target: Point, torus: bool) -> usize {
    (0..points.len())
        .min_by(|&left, &right| {
            distance(points[left], target, torus).total_cmp(&distance(points[right], target, torus))
        })
        .unwrap()
}

fn remote_fractions(graph: &[Vec<usize>], points: &[Point]) -> (f64, f64) {
    let mut edges = 0usize;
    let mut spatial = 0usize;
    let mut striped = 0usize;
    for left in 0..graph.len() {
        for &right in &graph[left] {
            if left >= right {
                continue;
            }
            edges += 1;
            spatial += usize::from((points[left].x < 0.5) != (points[right].x < 0.5));
            striped += usize::from(left % 2 != right % 2);
        }
    }
    (spatial as f64 / edges as f64, striped as f64 / edges as f64)
}

fn stretch_summary(
    points: &[Point],
    root: usize,
    found: &[Option<usize>],
    radius: f64,
    torus: bool,
) -> (f64, usize, usize) {
    let mut ratio_sum = 0.0;
    let mut count = 0usize;
    let mut maximum_excess = 0usize;
    for (vertex, &hop) in found.iter().enumerate() {
        let Some(hop) = hop else { continue };
        if vertex == root {
            continue;
        }
        let euclidean = distance(points[root], points[vertex], torus);
        let lower = (euclidean / radius).ceil() as usize;
        assert!(hop >= lower);
        if euclidean >= radius {
            ratio_sum += hop as f64 * radius / euclidean;
            count += 1;
        }
        maximum_excess = maximum_excess.max(hop - lower);
    }
    (ratio_sum, count, maximum_excess)
}

fn layer_profile(graph: &[Vec<usize>], points: &[Point], found: &[Option<usize>]) -> Vec<String> {
    let maximum = found.iter().flatten().copied().max().unwrap();
    (0..=maximum)
        .map(|depth| {
            let mut count = 0usize;
            let mut left = 0usize;
            let mut degree_sum = 0usize;
            for (vertex, &item) in found.iter().enumerate() {
                if item == Some(depth) {
                    count += 1;
                    left += usize::from(points[vertex].x < 0.5);
                    degree_sum += graph[vertex].len();
                }
            }
            format!(
                "(d={depth},n={count},owners={left}/{},mean_deg={:.1})",
                count - left,
                degree_sum as f64 / count as f64
            )
        })
        .collect()
}

fn main() {
    const N: usize = 2000;
    const SAMPLES: usize = 20;
    let threshold_scale = ((N as f64).ln() / (std::f64::consts::PI * N as f64)).sqrt();

    for multiplier in [0.8, 1.0, 1.3, 2.0] {
        let radius = multiplier * threshold_scale;
        for torus in [false, true] {
            let topology = if torus { "torus" } else { "square" };
            let mut connected = 0usize;
            let mut largest_sum = 0.0;
            let mut degree_sum = 0.0;
            let mut center_eccentricity_sum = 0.0;
            let mut corner_eccentricity_sum = 0.0;
            let mut center_stretch_sum = 0.0;
            let mut center_stretch_pairs = 0usize;
            let mut center_max_excess = 0usize;
            let mut spatial_remote_sum = 0.0;
            let mut striped_remote_sum = 0.0;
            let mut representative = None;

            for sample in 0..SAMPLES {
                let mut rng =
                    Rng(0xa0761d6478bd642fu64 ^ ((sample as u64 + 1) * 0xe7037ed1a0b428db));
                let points: Vec<_> = (0..N)
                    .map(|_| Point {
                        x: rng.next_f64(),
                        y: rng.next_f64(),
                    })
                    .collect();
                let graph = sample_graph(&points, radius, torus);
                let largest = largest_component(&graph);
                connected += usize::from(largest == N);
                largest_sum += largest as f64 / N as f64;
                degree_sum += graph.iter().map(Vec::len).sum::<usize>() as f64 / N as f64;

                let center = nearest(&points, Point { x: 0.5, y: 0.5 }, torus);
                let corner = nearest(&points, Point { x: 0.0, y: 0.0 }, torus);
                let center_distance = distances(&graph, center);
                let corner_distance = distances(&graph, corner);
                center_eccentricity_sum +=
                    center_distance.iter().flatten().copied().max().unwrap() as f64;
                corner_eccentricity_sum +=
                    corner_distance.iter().flatten().copied().max().unwrap() as f64;
                let (stretch_sum, stretch_pairs, excess) =
                    stretch_summary(&points, center, &center_distance, radius, torus);
                center_stretch_sum += stretch_sum;
                center_stretch_pairs += stretch_pairs;
                center_max_excess = center_max_excess.max(excess);
                let (spatial, striped) = remote_fractions(&graph, &points);
                spatial_remote_sum += spatial;
                striped_remote_sum += striped;

                if sample == 0 && multiplier == 1.3 {
                    representative = Some((
                        layer_profile(&graph, &points, &center_distance),
                        layer_profile(&graph, &points, &corner_distance),
                    ));
                }
            }

            println!(
                "rgg topology={topology} n={N} samples={SAMPLES} multiplier={multiplier:.1} radius={radius:.5} connected={connected}/{SAMPLES} largest_fraction_mean={:.4} mean_degree={:.2} center_eccentricity_mean={:.2} corner_eccentricity_mean={:.2} center_stretch_pair_mean={:.3} center_stretch_pairs={center_stretch_pairs} center_max_hop_excess={center_max_excess} spatial_remote_mean={:.4} striped_remote_mean={:.4}",
                largest_sum / SAMPLES as f64,
                degree_sum / SAMPLES as f64,
                center_eccentricity_sum / SAMPLES as f64,
                corner_eccentricity_sum / SAMPLES as f64,
                center_stretch_sum / center_stretch_pairs as f64,
                spatial_remote_sum / SAMPLES as f64,
                striped_remote_sum / SAMPLES as f64,
            );
            if let Some((center, corner)) = representative {
                println!("  representative_center_layers={}", center.join(" "));
                println!("  representative_corner_layers={}", corner.join(" "));
            }
        }
    }
}
