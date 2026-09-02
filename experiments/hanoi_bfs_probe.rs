use std::collections::VecDeque;

fn state_count(disks: usize) -> usize {
    3usize.pow(disks as u32)
}

fn decode(mut state: usize, disks: usize) -> Vec<usize> {
    let mut pegs = Vec::with_capacity(disks);
    for _ in 0..disks {
        pegs.push(state % 3);
        state /= 3;
    }
    pegs
}

fn encode(pegs: &[usize]) -> usize {
    pegs.iter()
        .rev()
        .fold(0usize, |state, &peg| state * 3 + peg)
}

fn labeled_successors(state: usize, disks: usize) -> [usize; 3] {
    let mut pegs = decode(state, disks);
    let mut top = [None; 3];
    for (disk, &peg) in pegs.iter().enumerate() {
        if top[peg].is_none() {
            top[peg] = Some(disk);
        }
    }
    let pairs = [(0, 1), (0, 2), (1, 2)];
    let mut out = [state; 3];
    for (label, (left, right)) in pairs.into_iter().enumerate() {
        let moving = match (top[left], top[right]) {
            (None, None) => None,
            (Some(disk), None) => Some((disk, right)),
            (None, Some(disk)) => Some((disk, left)),
            (Some(a), Some(b)) if a < b => Some((a, right)),
            (Some(_), Some(b)) => Some((b, left)),
        };
        if let Some((disk, target)) = moving {
            let old = pegs[disk];
            pegs[disk] = target;
            out[label] = encode(&pegs);
            pegs[disk] = old;
        }
    }
    out
}

fn bfs(source: usize, disks: usize) -> Vec<usize> {
    let mut distance = vec![usize::MAX; state_count(disks)];
    distance[source] = 0;
    let mut queue = VecDeque::from([source]);
    while let Some(state) = queue.pop_front() {
        let next_distance = distance[state] + 1;
        for next in labeled_successors(state, disks) {
            if next != state && distance[next] == usize::MAX {
                distance[next] = next_distance;
                queue.push_back(next);
            }
        }
    }
    distance
}

fn audit(disks: usize) {
    let states = state_count(disks);
    let corner_0 = 0;
    let corner_1 = (0..disks).fold(0usize, |state, _| state * 3 + 1);
    let distance = bfs(corner_0, disks);
    let corner_distance = distance[corner_1];
    let eccentricity = *distance.iter().max().unwrap();
    let mut layers = vec![0usize; eccentricity + 1];
    for &depth in &distance {
        layers[depth] += 1;
    }

    let mut diameter = 0usize;
    let mut loops = 0usize;
    let mut degree_two = 0usize;
    let mut degree_three = 0usize;
    for state in 0..states {
        let successors = labeled_successors(state, disks);
        loops += successors.iter().filter(|&&next| next == state).count();
        let degree = successors.iter().filter(|&&next| next != state).count();
        degree_two += usize::from(degree == 2);
        degree_three += usize::from(degree == 3);
        diameter = diameter.max(*bfs(state, disks).iter().max().unwrap());
    }

    println!(
        "Hanoi3({disks}) states={states} corner_distance={corner_distance} eccentricity={eccentricity} diameter={diameter} loops={loops} degree2={degree_two} degree3={degree_three} layers={layers:?}"
    );
}

fn main() {
    for disks in 1..=6 {
        audit(disks);
    }
}
