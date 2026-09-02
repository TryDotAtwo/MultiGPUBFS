use std::collections::{HashMap, HashSet, VecDeque};

fn q_power(base: usize, exponent: usize) -> usize {
    (0..exponent).fold(1, |value, _| value * base)
}

fn q_integer(n: usize, q: usize) -> usize {
    (0..n).map(|exponent| q_power(q, exponent)).sum()
}

fn q_binomial(n: usize, k: usize, q: usize) -> usize {
    let mut row = vec![0usize; k + 1];
    row[0] = 1;
    for current_n in 1..=n {
        for current_k in (1..=k.min(current_n)).rev() {
            row[current_k] += q_power(q, current_n - current_k) * row[current_k - 1];
        }
    }
    row[k]
}

fn span_mask(basis: &[usize]) -> u128 {
    let mut vectors = vec![0usize];
    for &basis_vector in basis {
        let previous = vectors.clone();
        vectors.extend(previous.into_iter().map(|vector| vector ^ basis_vector));
    }
    vectors
        .into_iter()
        .fold(0u128, |mask, vector| mask | (1u128 << vector))
}

fn enumerate_subspaces(n: usize, k: usize) -> Vec<u128> {
    fn choose(
        next: usize,
        end: usize,
        k: usize,
        basis: &mut Vec<usize>,
        subspaces: &mut HashSet<u128>,
    ) {
        if basis.len() == k {
            let mask = span_mask(basis);
            if mask.count_ones() as usize == 1usize << k {
                subspaces.insert(mask);
            }
            return;
        }
        for vector in next..end {
            basis.push(vector);
            choose(vector + 1, end, k, basis, subspaces);
            basis.pop();
        }
    }

    let mut subspaces = HashSet::new();
    choose(1, 1usize << n, k, &mut Vec::new(), &mut subspaces);
    let mut out: Vec<_> = subspaces.into_iter().collect();
    out.sort_unstable();
    out
}

fn intersection_dimension(left: u128, right: u128) -> usize {
    (left & right).count_ones().ilog2() as usize
}

fn audit(n: usize, k: usize) {
    const Q: usize = 2;
    let subspaces = enumerate_subspaces(n, k);
    let index: HashMap<_, _> = subspaces
        .iter()
        .copied()
        .enumerate()
        .map(|(index, subspace)| (subspace, index))
        .collect();
    let root = span_mask(
        &(0..k)
            .map(|coordinate| 1usize << coordinate)
            .collect::<Vec<_>>(),
    );
    let root_index = index[&root];
    let mut graph = vec![Vec::new(); subspaces.len()];
    for left in 0..subspaces.len() {
        for right in left + 1..subspaces.len() {
            if intersection_dimension(subspaces[left], subspaces[right]) == k - 1 {
                graph[left].push(right);
                graph[right].push(left);
            }
        }
    }

    let diameter = k.min(n - k);
    let mut distance = vec![None; subspaces.len()];
    let mut shortest_paths = vec![0usize; subspaces.len()];
    let mut queue = VecDeque::from([root_index]);
    distance[root_index] = Some(0);
    shortest_paths[root_index] = 1;
    while let Some(vertex) = queue.pop_front() {
        let depth = distance[vertex].unwrap();
        for &next in &graph[vertex] {
            if distance[next].is_none() {
                distance[next] = Some(depth + 1);
                queue.push_back(next);
            }
            if distance[next] == Some(depth + 1) {
                shortest_paths[next] += shortest_paths[vertex];
            }
        }
    }

    let expected_degree = Q * q_integer(k, Q) * q_integer(n - k, Q);
    let expected_layers: Vec<_> = (0..=diameter)
        .map(|depth| {
            q_power(Q, depth * depth) * q_binomial(k, depth, Q) * q_binomial(n - k, depth, Q)
        })
        .collect();
    let mut layers = vec![0usize; diameter + 1];
    let mut degree_mismatches = 0usize;
    let mut distance_mismatches = 0usize;
    let mut intersection_mismatches = 0usize;
    let mut path_mismatches = 0usize;
    for vertex in 0..subspaces.len() {
        let depth = distance[vertex].unwrap();
        layers[depth] += 1;
        degree_mismatches += usize::from(graph[vertex].len() != expected_degree);
        distance_mismatches +=
            usize::from(depth != k - intersection_dimension(root, subspaces[vertex]));
        let mut inward = 0;
        let mut same = 0;
        let mut outward = 0;
        for &next in &graph[vertex] {
            match distance[next].unwrap().cmp(&depth) {
                std::cmp::Ordering::Less => inward += 1,
                std::cmp::Ordering::Equal => same += 1,
                std::cmp::Ordering::Greater => outward += 1,
            }
        }
        let expected_inward = q_integer(depth, Q) * q_integer(depth, Q);
        let expected_outward =
            q_power(Q, 2 * depth + 1) * q_integer(k - depth, Q) * q_integer(n - k - depth, Q);
        let expected_same = expected_degree - expected_inward - expected_outward;
        intersection_mismatches += usize::from(
            inward != expected_inward || same != expected_same || outward != expected_outward,
        );
        let q_factorial = (1..=depth).fold(1usize, |value, term| value * q_integer(term, Q));
        path_mismatches += usize::from(shortest_paths[vertex] != q_factorial * q_factorial);
    }
    println!(
        "J_2({n},{k}) states={} expected={} degree={expected_degree} diameter={diameter} layers={layers:?} layer_formula_match={} degree_mismatches={degree_mismatches} distance_mismatches={distance_mismatches} intersection_mismatches={intersection_mismatches} path_mismatches={path_mismatches}",
        subspaces.len(),
        q_binomial(n, k, Q),
        layers == expected_layers
    );
}

fn main() {
    for n in 2..=6 {
        for k in 1..=n / 2 {
            audit(n, k);
        }
    }
}
