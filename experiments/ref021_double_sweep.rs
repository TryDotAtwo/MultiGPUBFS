use std::collections::VecDeque;

fn distances(adjacency: &[Vec<usize>], source: usize) -> Vec<usize> {
    let mut distance = vec![usize::MAX; adjacency.len()];
    let mut queue = VecDeque::from([source]);
    distance[source] = 0;

    while let Some(vertex) = queue.pop_front() {
        for &next in &adjacency[vertex] {
            if distance[next] == usize::MAX {
                distance[next] = distance[vertex] + 1;
                queue.push_back(next);
            }
        }
    }
    distance
}

fn main() {
    for vertex_count in 4..=7 {
        let edges: Vec<(usize, usize)> = (0..vertex_count)
            .flat_map(|left| ((left + 1)..vertex_count).map(move |right| (left, right)))
            .collect();

        for mask in 0_u64..(1_u64 << edges.len()) {
            let mut adjacency = vec![Vec::new(); vertex_count];
            for (bit, &(left, right)) in edges.iter().enumerate() {
                if mask & (1_u64 << bit) != 0 {
                    adjacency[left].push(right);
                    adjacency[right].push(left);
                }
            }

            let all_distances: Vec<Vec<usize>> = (0..vertex_count)
                .map(|source| distances(&adjacency, source))
                .collect();
            if all_distances[0].contains(&usize::MAX) {
                continue;
            }

            let eccentricities: Vec<usize> = all_distances
                .iter()
                .map(|row| *row.iter().max().unwrap())
                .collect();
            let diameter = *eccentricities.iter().max().unwrap();

            for start in 0..vertex_count {
                let first_radius = eccentricities[start];
                let farthest: Vec<usize> = (0..vertex_count)
                    .filter(|&vertex| all_distances[start][vertex] == first_radius)
                    .collect();
                if farthest.len() != 1 {
                    continue;
                }

                let pivot = farthest[0];
                if eccentricities[pivot] < diameter {
                    let selected_second: Vec<usize> = (0..vertex_count)
                        .filter(|&vertex| all_distances[pivot][vertex] == eccentricities[pivot])
                        .collect();
                    let present_edges: Vec<String> = edges
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| mask & (1_u64 << bit) != 0)
                        .map(|(_, (left, right))| format!("{left}-{right}"))
                        .collect();

                    println!("REF021_DOUBLE_SWEEP_COUNTEREXAMPLE");
                    println!("vertices={vertex_count}");
                    println!("edges={}", present_edges.join(","));
                    println!("start={start}");
                    println!("unique_first_farthest={pivot}");
                    println!("first_distance={first_radius}");
                    println!("second_farthest={selected_second:?}");
                    println!("double_sweep_value={}", eccentricities[pivot]);
                    println!("true_diameter={diameter}");
                    println!("eccentricities={eccentricities:?}");
                    return;
                }
            }
        }
    }

    println!("NO_COUNTEREXAMPLE_THROUGH_7_VERTICES");
}
