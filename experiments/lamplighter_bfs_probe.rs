use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct State {
    lamps: usize,
    pos: usize,
}

fn id(state: State, cycle: usize) -> usize {
    state.lamps * cycle + state.pos
}

fn neighbors(state: State, cycle: usize) -> [State; 3] {
    [
        State {
            lamps: state.lamps ^ (1 << state.pos),
            pos: state.pos,
        },
        State {
            lamps: state.lamps,
            pos: (state.pos + cycle - 1) % cycle,
        },
        State {
            lamps: state.lamps,
            pos: (state.pos + 1) % cycle,
        },
    ]
}

fn lamplighter_distances(cycle: usize) -> Vec<usize> {
    let state_count = (1 << cycle) * cycle;
    let mut distance = vec![usize::MAX; state_count];
    let start = State { lamps: 0, pos: 0 };
    distance[id(start, cycle)] = 0;
    let mut queue = VecDeque::from([start]);
    while let Some(state) = queue.pop_front() {
        let next_distance = distance[id(state, cycle)] + 1;
        for next in neighbors(state, cycle) {
            let next_id = id(next, cycle);
            if distance[next_id] == usize::MAX {
                distance[next_id] = next_distance;
                queue.push_back(next);
            }
        }
    }
    distance
}

fn route_length(cycle: usize, required: usize, target: usize) -> usize {
    let all_required = required | 1;
    let state_count = (1 << cycle) * cycle;
    let mut distance = vec![usize::MAX; state_count];
    let start_id = id(
        State {
            lamps: 1,
            pos: 0,
        },
        cycle,
    );
    distance[start_id] = 0;
    let mut queue = VecDeque::from([(1usize, 0usize)]);
    while let Some((visited, pos)) = queue.pop_front() {
        let here = id(
            State {
                lamps: visited,
                pos,
            },
            cycle,
        );
        if pos == target && visited & all_required == all_required {
            return distance[here];
        }
        for next_pos in [(pos + cycle - 1) % cycle, (pos + 1) % cycle] {
            let next_visited = visited | (1 << next_pos);
            let next = id(
                State {
                    lamps: next_visited,
                    pos: next_pos,
                },
                cycle,
            );
            if distance[next] == usize::MAX {
                distance[next] = distance[here] + 1;
                queue.push_back((next_visited, next_pos));
            }
        }
    }
    unreachable!()
}

fn audit(cycle: usize) {
    let distance = lamplighter_distances(cycle);
    let diameter = *distance.iter().max().unwrap();
    let mut layers = vec![0usize; diameter + 1];
    let mut decomposition_mismatches = 0usize;
    let mut interior_dead_ends = Vec::new();

    for lamps in 0..(1 << cycle) {
        for pos in 0..cycle {
            let state = State { lamps, pos };
            let depth = distance[id(state, cycle)];
            layers[depth] += 1;
            let predicted = lamps.count_ones() as usize + route_length(cycle, lamps, pos);
            decomposition_mismatches += usize::from(depth != predicted);
            if depth < diameter
                && neighbors(state, cycle)
                    .iter()
                    .all(|&next| distance[id(next, cycle)] <= depth)
            {
                interior_dead_ends.push(state);
            }
        }
    }

    println!(
        "C2wrC{cycle} states={} diameter={diameter} decomposition_mismatches={decomposition_mismatches} layers={layers:?}",
        distance.len()
    );
    println!(
        "C2wrC{cycle} interior_dead_ends={} examples={:?}",
        interior_dead_ends.len(),
        &interior_dead_ends[..interior_dead_ends.len().min(4)]
    );
}

fn main() {
    audit(4);
    audit(5);
}
