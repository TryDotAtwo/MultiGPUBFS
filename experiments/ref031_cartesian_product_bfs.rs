use std::collections::VecDeque;

#[derive(Clone, Copy)]
enum Product {
    Cartesian,
    Strong,
}

fn path(n: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); n];
    for vertex in 0..n - 1 {
        graph[vertex].push(vertex + 1);
        graph[vertex + 1].push(vertex);
    }
    graph
}

fn cycle(n: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); n];
    for vertex in 0..n {
        let next = (vertex + 1) % n;
        graph[vertex].push(next);
        graph[next].push(vertex);
    }
    for neighbors in &mut graph {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    graph
}

fn product(left: &[Vec<usize>], right: &[Vec<usize>], kind: Product) -> Vec<Vec<usize>> {
    let right_size = right.len();
    let mut graph = vec![Vec::new(); left.len() * right_size];
    let index = |u: usize, v: usize| u * right_size + v;

    for u in 0..left.len() {
        for v in 0..right.len() {
            let source = index(u, v);
            for &next_u in &left[u] {
                graph[source].push(index(next_u, v));
            }
            for &next_v in &right[v] {
                graph[source].push(index(u, next_v));
            }
            if matches!(kind, Product::Strong) {
                for &next_u in &left[u] {
                    for &next_v in &right[v] {
                        graph[source].push(index(next_u, next_v));
                    }
                }
            }
            graph[source].sort_unstable();
            graph[source].dedup();
        }
    }
    graph
}

fn bfs(graph: &[Vec<usize>], source: usize) -> (Vec<usize>, Vec<u64>) {
    let mut distance = vec![usize::MAX; graph.len()];
    let mut shortest_paths = vec![0_u64; graph.len()];
    let mut queue = VecDeque::from([source]);
    distance[source] = 0;
    shortest_paths[source] = 1;

    while let Some(vertex) = queue.pop_front() {
        for &child in &graph[vertex] {
            if distance[child] == usize::MAX {
                distance[child] = distance[vertex] + 1;
                queue.push_back(child);
            }
            if distance[child] == distance[vertex] + 1 {
                shortest_paths[child] += shortest_paths[vertex];
            }
        }
    }
    (distance, shortest_paths)
}

fn spheres(distance: &[usize]) -> Vec<usize> {
    let diameter = *distance.iter().max().unwrap();
    let mut counts = vec![0; diameter + 1];
    for &depth in distance {
        counts[depth] += 1;
    }
    counts
}

fn convolution(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut result = vec![0; left.len() + right.len() - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            result[i + j] += a * b;
        }
    }
    result
}

fn main() {
    let p3 = path(3);
    let c4 = cycle(4);
    let cartesian = product(&p3, &c4, Product::Cartesian);
    let strong = product(&p3, &c4, Product::Strong);
    let (p3_distance, _) = bfs(&p3, 0);
    let (c4_distance, _) = bfs(&c4, 0);
    let (cartesian_distance, cartesian_paths) = bfs(&cartesian, 0);
    let (strong_distance, _) = bfs(&strong, 0);

    let p3_spheres = spheres(&p3_distance);
    let c4_spheres = spheres(&c4_distance);
    let predicted = convolution(&p3_spheres, &c4_spheres);
    let observed = spheres(&cartesian_distance);

    println!("P3 spheres: {p3_spheres:?}");
    println!("C4 spheres: {c4_spheres:?}");
    println!("Cartesian convolution: {predicted:?}");
    println!("Cartesian observed:    {observed:?}");
    println!(
        "far endpoint (2,2): Cartesian distance={}, shortest paths={}",
        cartesian_distance[10], cartesian_paths[10]
    );
    println!(
        "diagonal (1,1): Cartesian distance={}, strong distance={}",
        cartesian_distance[5], strong_distance[5]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartesian_distances_are_coordinate_sums() {
        let left = path(3);
        let right = cycle(4);
        let graph = product(&left, &right, Product::Cartesian);
        let (left_distance, _) = bfs(&left, 0);
        let (right_distance, _) = bfs(&right, 0);
        let (distance, _) = bfs(&graph, 0);
        for u in 0..left.len() {
            for v in 0..right.len() {
                assert_eq!(
                    distance[u * right.len() + v],
                    left_distance[u] + right_distance[v]
                );
            }
        }
    }

    #[test]
    fn cartesian_spheres_are_a_convolution() {
        let left = path(3);
        let right = cycle(4);
        let graph = product(&left, &right, Product::Cartesian);
        let (left_distance, _) = bfs(&left, 0);
        let (right_distance, _) = bfs(&right, 0);
        let (distance, _) = bfs(&graph, 0);
        assert_eq!(
            spheres(&distance),
            convolution(&spheres(&left_distance), &spheres(&right_distance))
        );
        assert_eq!(spheres(&distance), vec![1, 3, 4, 3, 1]);
    }

    #[test]
    fn shortest_paths_include_coordinate_interleavings() {
        let graph = product(&path(3), &cycle(4), Product::Cartesian);
        let (distance, paths) = bfs(&graph, 0);
        let far = 2 * 4 + 2;
        assert_eq!(distance[far], 4);
        assert_eq!(paths[far], 12);
    }

    #[test]
    fn the_same_sum_formula_fails_for_the_strong_product() {
        let cartesian = product(&path(3), &cycle(4), Product::Cartesian);
        let strong = product(&path(3), &cycle(4), Product::Strong);
        let (cartesian_distance, _) = bfs(&cartesian, 0);
        let (strong_distance, _) = bfs(&strong, 0);
        let diagonal = 4 + 1;
        assert_eq!(cartesian_distance[diagonal], 2);
        assert_eq!(strong_distance[diagonal], 1);
    }
}
