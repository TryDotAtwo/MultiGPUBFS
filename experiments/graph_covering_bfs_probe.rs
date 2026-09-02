use std::collections::VecDeque;

fn add_edge(graph: &mut [Vec<usize>], a: usize, b: usize) {
    graph[a].push(b);
    graph[b].push(a);
}

fn multi_source_distances(graph: &[Vec<usize>], sources: &[usize]) -> Vec<Option<usize>> {
    let mut distance = vec![None; graph.len()];
    let mut queue = VecDeque::new();
    for &source in sources {
        if distance[source].is_none() {
            distance[source] = Some(0);
            queue.push_back(source);
        }
    }
    while let Some(vertex) = queue.pop_front() {
        let next_distance = distance[vertex].unwrap() + 1;
        for &next in &graph[vertex] {
            if distance[next].is_none() {
                distance[next] = Some(next_distance);
                queue.push_back(next);
            }
        }
    }
    distance
}

fn covering_radius(graph: &[Vec<usize>], centers: &[usize]) -> Option<usize> {
    multi_source_distances(graph, centers)
        .into_iter()
        .try_fold(0usize, |radius, distance| {
            distance.map(|value| radius.max(value))
        })
}

fn subsets_of_size(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(start: usize, n: usize, k: usize, chosen: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if chosen.len() == k {
            out.push(chosen.clone());
            return;
        }
        for vertex in start..n {
            chosen.push(vertex);
            visit(vertex + 1, n, k, chosen, out);
            chosen.pop();
        }
    }
    let mut out = Vec::new();
    visit(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn exact_k_center(graph: &[Vec<usize>], k: usize) -> (usize, Vec<Vec<usize>>) {
    let scored: Vec<_> = subsets_of_size(graph.len(), k)
        .into_iter()
        .map(|centers| (covering_radius(graph, &centers).unwrap(), centers))
        .collect();
    let optimum = scored.iter().map(|(radius, _)| *radius).min().unwrap();
    let witnesses = scored
        .into_iter()
        .filter_map(|(radius, centers)| (radius == optimum).then_some(centers))
        .collect();
    (optimum, witnesses)
}

fn farthest_first(graph: &[Vec<usize>], first: usize, k: usize) -> Vec<usize> {
    let mut centers = vec![first];
    while centers.len() < k {
        let distances = multi_source_distances(graph, &centers);
        let farthest = (0..graph.len())
            .max_by_key(|&vertex| (distances[vertex].unwrap(), vertex))
            .unwrap();
        centers.push(farthest);
    }
    centers
}

fn path(n: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); n];
    for vertex in 0..n - 1 {
        add_edge(&mut graph, vertex, vertex + 1);
    }
    graph
}

fn main() {
    let mut degree_trap = path(5);
    degree_trap.resize(8, Vec::new());
    for leaf in 5..8 {
        add_edge(&mut degree_trap, 0, leaf);
    }
    let eccentricities: Vec<_> = (0..degree_trap.len())
        .map(|vertex| covering_radius(&degree_trap, &[vertex]).unwrap())
        .collect();
    let degrees: Vec<_> = degree_trap.iter().map(Vec::len).collect();
    let (one_center_radius, one_center_witnesses) = exact_k_center(&degree_trap, 1);
    println!(
        "degree_trap degrees={degrees:?} eccentricities={eccentricities:?} highest_degree=0 highest_degree_radius={} optimum_radius={one_center_radius} optimum_centers={one_center_witnesses:?}",
        eccentricities[0]
    );

    let path_six = path(6);
    let greedy = farthest_first(&path_six, 0, 2);
    let greedy_radius = covering_radius(&path_six, &greedy).unwrap();
    let (optimum_radius, optimum_centers) = exact_k_center(&path_six, 2);
    println!(
        "path6 start=0 greedy_centers={greedy:?} greedy_radius={greedy_radius} optimum_radius={optimum_radius} optimum_centers={optimum_centers:?}"
    );

    let covered_at_one = multi_source_distances(&path_six, &[1, 4])
        .into_iter()
        .all(|distance| distance.is_some_and(|value| value <= 1));
    println!("path6 centers=[1, 4] radius=1 covered_at_one={covered_at_one}");

    let disconnected = vec![vec![1], vec![0], vec![]];
    println!(
        "disconnected centers=[0] covering_radius={:?}",
        covering_radius(&disconnected, &[0])
    );
}
