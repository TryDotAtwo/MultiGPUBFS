use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
struct Edge {
    to: usize,
    cost: u32,
}

fn bfs_first_parent(
    graph: &[Vec<Edge>],
    source: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut distance = vec![None; graph.len()];
    let mut parent = vec![None; graph.len()];
    let mut queue = VecDeque::from([source]);
    distance[source] = Some(0);
    while let Some(vertex) = queue.pop_front() {
        let next_depth = distance[vertex].unwrap() + 1;
        for edge in &graph[vertex] {
            if distance[edge.to].is_none() {
                distance[edge.to] = Some(next_depth);
                parent[edge.to] = Some(vertex);
                queue.push_back(edge.to);
            }
        }
    }
    (distance, parent)
}

fn path_cost(graph: &[Vec<Edge>], parent: &[Option<usize>], mut target: usize) -> u32 {
    let mut cost = 0;
    while let Some(previous) = parent[target] {
        cost += graph[previous]
            .iter()
            .find(|edge| edge.to == target)
            .unwrap()
            .cost;
        target = previous;
    }
    cost
}

fn secondary_cost_on_shortest_dag(
    graph: &[Vec<Edge>],
    distance: &[Option<usize>],
    source: usize,
) -> Vec<Option<u32>> {
    let maximum_depth = distance.iter().flatten().copied().max().unwrap();
    let mut best = vec![None; graph.len()];
    best[source] = Some(0);
    for depth in 0..maximum_depth {
        for vertex in 0..graph.len() {
            if distance[vertex] != Some(depth) || best[vertex].is_none() {
                continue;
            }
            for edge in &graph[vertex] {
                if distance[edge.to] == Some(depth + 1) {
                    let candidate = best[vertex].unwrap() + edge.cost;
                    best[edge.to] = Some(best[edge.to].map_or(candidate, |old| old.min(candidate)));
                }
            }
        }
    }
    best
}

fn enumerate_simple_paths(
    graph: &[Vec<Edge>],
    vertex: usize,
    target: usize,
    seen: &mut [bool],
    hops: usize,
    cost: u32,
    out: &mut Vec<(usize, u32)>,
) {
    if vertex == target {
        out.push((hops, cost));
        return;
    }
    seen[vertex] = true;
    for edge in &graph[vertex] {
        if !seen[edge.to] {
            enumerate_simple_paths(
                graph,
                edge.to,
                target,
                seen,
                hops + 1,
                cost + edge.cost,
                out,
            );
        }
    }
    seen[vertex] = false;
}

fn main() {
    let graph = vec![
        vec![
            Edge { to: 1, cost: 100 },
            Edge { to: 2, cost: 1 },
            Edge { to: 4, cost: 0 },
        ],
        vec![Edge { to: 3, cost: 100 }],
        vec![Edge { to: 3, cost: 1 }],
        vec![],
        vec![Edge { to: 5, cost: 0 }],
        vec![Edge { to: 3, cost: 0 }],
    ];
    let (distance, parent) = bfs_first_parent(&graph, 0);
    let first_parent_cost = path_cost(&graph, &parent, 3);
    let best_shortest_cost = secondary_cost_on_shortest_dag(&graph, &distance, 0)[3].unwrap();

    let mut paths = Vec::new();
    enumerate_simple_paths(
        &graph,
        0,
        3,
        &mut vec![false; graph.len()],
        0,
        0,
        &mut paths,
    );
    paths.sort_unstable();
    let pareto: Vec<_> = paths
        .iter()
        .copied()
        .filter(|&(hops, cost)| {
            !paths.iter().any(|&(other_hops, other_cost)| {
                other_hops <= hops && other_cost <= cost && (other_hops < hops || other_cost < cost)
            })
        })
        .collect();

    println!(
        "target=3 bfs_hops={:?} first_parent={:?} first_parent_cost={first_parent_cost}",
        distance[3], parent[3]
    );
    println!("shortest_dag_secondary_cost={best_shortest_cost}");
    println!("all_simple_path_pairs={paths:?} pareto_pairs={pareto:?}");
}
