use std::collections::{HashMap, VecDeque};

fn words(alphabet: u8, len: usize, kautz: bool) -> Vec<Vec<u8>> {
    fn rec(out: &mut Vec<Vec<u8>>, cur: &mut Vec<u8>, alphabet: u8, len: usize, kautz: bool) {
        if cur.len() == len {
            out.push(cur.clone());
            return;
        }
        for symbol in 0..alphabet {
            if kautz && cur.last() == Some(&symbol) {
                continue;
            }
            cur.push(symbol);
            rec(out, cur, alphabet, len, kautz);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(&mut out, &mut Vec::new(), alphabet, len, kautz);
    out
}

fn successors(word: &[u8], alphabet: u8, kautz: bool) -> Vec<Vec<u8>> {
    (0..alphabet)
        .filter(|&symbol| !kautz || word.last() != Some(&symbol))
        .map(|symbol| {
            let mut next = word[1..].to_vec();
            next.push(symbol);
            next
        })
        .collect()
}

fn bfs(root: &[u8], alphabet: u8, kautz: bool) -> HashMap<Vec<u8>, usize> {
    let mut distance = HashMap::from([(root.to_vec(), 0)]);
    let mut queue = VecDeque::from([root.to_vec()]);
    while let Some(word) = queue.pop_front() {
        let next_distance = distance[&word] + 1;
        for next in successors(&word, alphabet, kautz) {
            if !distance.contains_key(&next) {
                distance.insert(next.clone(), next_distance);
                queue.push_back(next);
            }
        }
    }
    distance
}

fn overlap_distance(source: &[u8], target: &[u8]) -> usize {
    let n = source.len();
    let overlap = (0..=n)
        .rev()
        .find(|&k| source[n - k..] == target[..k])
        .unwrap();
    n - overlap
}

fn audit(name: &str, alphabet: u8, len: usize, kautz: bool, roots: &[&[u8]]) {
    let states = words(alphabet, len, kautz);
    let mut global_diameter = 0;
    let mut mismatches = 0;
    for source in &states {
        let distance = bfs(source, alphabet, kautz);
        for target in &states {
            global_diameter = global_diameter.max(distance[target]);
            mismatches += usize::from(distance[target] != overlap_distance(source, target));
        }
    }
    println!(
        "{name} states={} diameter={} overlap_mismatches={mismatches}",
        states.len(), global_diameter
    );
    for root in roots {
        let distance = bfs(root, alphabet, kautz);
        let max_distance = distance.values().copied().max().unwrap();
        let mut layers = vec![0; max_distance + 1];
        for depth in distance.values() {
            layers[*depth] += 1;
        }
        println!("{name} root={root:?} layers={layers:?}");
    }
}

fn main() {
    audit("B(2,3)", 2, 3, false, &[&[0, 0, 0], &[0, 1, 0]]);
    audit("K(2,3)", 3, 3, true, &[&[0, 1, 0], &[0, 1, 2]]);
}
