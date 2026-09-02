use std::collections::VecDeque;

#[derive(Debug, PartialEq, Eq)]
struct QueueStats {
    distances: Vec<Option<usize>>,
    enqueued: usize,
    expanded: usize,
    stale_pops: usize,
    peak_queue: usize,
}

fn complete_bipartite_layers(left: usize, right: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); 1 + left + right];
    graph[0].extend(1..=left);
    for parent in 1..=left {
        graph[parent].extend((1 + left)..(1 + left + right));
    }
    graph
}

fn two_vertex_layered_dag(depth: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); 1 + 2 * depth];
    if depth == 0 {
        return graph;
    }
    graph[0].extend([1, 2]);
    for layer in 1..depth {
        let current = [2 * layer - 1, 2 * layer];
        let next = [2 * layer + 1, 2 * layer + 2];
        for vertex in current {
            graph[vertex].extend(next);
        }
    }
    graph
}

fn claim_before_enqueue(graph: &[Vec<usize>], source: usize) -> QueueStats {
    let mut claimed = vec![false; graph.len()];
    let mut distances = vec![None; graph.len()];
    let mut queue = VecDeque::from([(source, 0_usize)]);
    let mut enqueued = 1;
    let mut expanded = 0;
    let mut peak_queue = 1;
    claimed[source] = true;
    distances[source] = Some(0);

    while let Some((vertex, depth)) = queue.pop_front() {
        expanded += 1;
        for &child in &graph[vertex] {
            if !claimed[child] {
                claimed[child] = true;
                distances[child] = Some(depth + 1);
                queue.push_back((child, depth + 1));
                enqueued += 1;
                peak_queue = peak_queue.max(queue.len());
            }
        }
    }

    QueueStats {
        distances,
        enqueued,
        expanded,
        stale_pops: 0,
        peak_queue,
    }
}

fn settle_on_dequeue(graph: &[Vec<usize>], source: usize) -> QueueStats {
    let mut settled = vec![false; graph.len()];
    let mut distances = vec![None; graph.len()];
    let mut queue = VecDeque::from([(source, 0_usize)]);
    let mut enqueued = 1;
    let mut expanded = 0;
    let mut stale_pops = 0;
    let mut peak_queue = 1;

    while let Some((vertex, depth)) = queue.pop_front() {
        if settled[vertex] {
            stale_pops += 1;
            continue;
        }
        settled[vertex] = true;
        distances[vertex] = Some(depth);
        expanded += 1;
        for &child in &graph[vertex] {
            if !settled[child] {
                queue.push_back((child, depth + 1));
                enqueued += 1;
                peak_queue = peak_queue.max(queue.len());
            }
        }
    }

    QueueStats {
        distances,
        enqueued,
        expanded,
        stale_pops,
        peak_queue,
    }
}

fn expand_every_occurrence(graph: &[Vec<usize>], source: usize) -> Vec<usize> {
    let mut queue = VecDeque::from([(source, 0_usize)]);
    let mut occurrences = Vec::new();

    while let Some((vertex, depth)) = queue.pop_front() {
        if occurrences.len() == depth {
            occurrences.push(0);
        }
        occurrences[depth] += 1;
        for &child in &graph[vertex] {
            queue.push_back((child, depth + 1));
        }
    }
    occurrences
}

fn main() {
    let boundary = complete_bipartite_layers(100, 100);
    let unique = claim_before_enqueue(&boundary, 0);
    let duplicate = settle_on_dequeue(&boundary, 0);
    let layered = two_vertex_layered_dag(12);
    let suppressed = settle_on_dequeue(&layered, 0);
    let walks = expand_every_occurrence(&layered, 0);

    println!("schedule,enqueued,expanded,stale_pops,peak_queue");
    println!(
        "claim_before_enqueue,{},{},{},{}",
        unique.enqueued, unique.expanded, unique.stale_pops, unique.peak_queue
    );
    println!(
        "settle_on_dequeue,{},{},{},{}",
        duplicate.enqueued, duplicate.expanded, duplicate.stale_pops, duplicate.peak_queue
    );
    println!(
        "layered_stale_suppressed,{},{},{},{}",
        suppressed.enqueued, suppressed.expanded, suppressed.stale_pops, suppressed.peak_queue
    );
    println!("layered_expand_every_occurrence={walks:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_bipartite_boundary_separates_unique_and_occurrence_queues() {
        let graph = complete_bipartite_layers(100, 100);
        let unique = claim_before_enqueue(&graph, 0);
        let duplicate = settle_on_dequeue(&graph, 0);

        assert_eq!(unique.peak_queue, 199);
        assert_eq!(unique.enqueued, 201);
        assert_eq!(unique.stale_pops, 0);
        assert_eq!(duplicate.peak_queue, 10_000);
        assert_eq!(duplicate.enqueued, 10_101);
        assert_eq!(duplicate.expanded, 201);
        assert_eq!(duplicate.stale_pops, 9_900);
        assert_eq!(unique.distances, duplicate.distances);
        assert!(duplicate
            .distances
            .iter()
            .skip(101)
            .all(|&depth| depth == Some(2)));
    }

    #[test]
    fn stale_suppression_keeps_unique_expansion_linear() {
        let depth = 12;
        let graph = two_vertex_layered_dag(depth);
        let duplicate = settle_on_dequeue(&graph, 0);

        assert_eq!(duplicate.expanded, 1 + 2 * depth);
        assert_eq!(duplicate.stale_pops, 2 * (depth - 1));
        assert_eq!(duplicate.peak_queue, 6);
    }

    #[test]
    fn expanding_every_occurrence_enumerates_exponentially_many_walks() {
        let depth = 12;
        let graph = two_vertex_layered_dag(depth);
        let occurrences = expand_every_occurrence(&graph, 0);
        let expected: Vec<usize> = (0..=depth).map(|level| 1_usize << level).collect();

        assert_eq!(occurrences, expected);
        assert_eq!(occurrences[depth], 4_096);
        assert_eq!(graph.len(), 25);
    }
}
