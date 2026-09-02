use std::collections::{BTreeMap, HashSet};
use std::convert::TryFrom;

#[derive(Debug)]
struct Config {
    central: Vec<u8>,
    generators: Vec<(String, Vec<u8>)>,
}

#[derive(Debug)]
struct Audit {
    state_len: usize,
    move_count: usize,
    central_is_identity: bool,
    inverse_pairs: usize,
    depth_one_loops: usize,
    depth_one_unique: usize,
    generator_orders: Vec<(usize, usize)>,
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

fn parse_string(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    skip_ws(bytes, pos);
    if bytes.get(*pos) != Some(&b'"') {
        return Err(format!("expected string at byte {}", pos));
    }
    *pos += 1;
    let begin = *pos;
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        if bytes[*pos] == b'\\' {
            return Err("escaped generator names are outside this probe's format".to_string());
        }
        *pos += 1;
    }
    if *pos == bytes.len() {
        return Err("unterminated string".to_string());
    }
    let value = std::str::from_utf8(&bytes[begin..*pos])
        .map_err(|error| error.to_string())?
        .to_string();
    *pos += 1;
    Ok(value)
}

fn parse_u8_array(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
    while *pos < bytes.len() && bytes[*pos] != b'[' {
        *pos += 1;
    }
    if *pos == bytes.len() {
        return Err("missing array".to_string());
    }
    *pos += 1;
    let mut values = Vec::new();
    loop {
        skip_ws(bytes, pos);
        if bytes.get(*pos) == Some(&b']') {
            *pos += 1;
            return Ok(values);
        }
        let begin = *pos;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
        if begin == *pos {
            return Err(format!("expected unsigned integer at byte {}", pos));
        }
        let value: u16 = std::str::from_utf8(&bytes[begin..*pos])
            .map_err(|error| error.to_string())?
            .parse::<u16>()
            .map_err(|error| error.to_string())?;
        values.push(u8::try_from(value).map_err(|_| format!("state value {value} exceeds u8"))?);
        skip_ws(bytes, pos);
        match bytes.get(*pos) {
            Some(b',') => *pos += 1,
            Some(b']') => {}
            _ => return Err(format!("expected comma or array end at byte {}", pos)),
        }
    }
}

fn parse_config(text: &str) -> Result<Config, String> {
    let bytes = text.as_bytes();
    let central_key = text
        .find("\"central_state\"")
        .ok_or_else(|| "missing central_state".to_string())?;
    let mut central_pos = central_key;
    let central = parse_u8_array(bytes, &mut central_pos)?;

    let generators_key = text
        .find("\"generators\"")
        .ok_or_else(|| "missing generators".to_string())?;
    let mut pos = text[generators_key..]
        .find('{')
        .map(|offset| generators_key + offset + 1)
        .ok_or_else(|| "missing generators object".to_string())?;
    let mut generators = Vec::new();
    loop {
        skip_ws(bytes, &mut pos);
        if bytes.get(pos) == Some(&b'}') {
            break;
        }
        if bytes.get(pos) == Some(&b',') {
            pos += 1;
            skip_ws(bytes, &mut pos);
        }
        let name = parse_string(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        if bytes.get(pos) != Some(&b':') {
            return Err(format!("missing colon after generator {name}"));
        }
        pos += 1;
        let permutation = parse_u8_array(bytes, &mut pos)?;
        generators.push((name, permutation));
    }
    Ok(Config {
        central,
        generators,
    })
}

fn apply(state: &[u8], permutation: &[u8]) -> Vec<u8> {
    permutation
        .iter()
        .map(|&source| state[source as usize])
        .collect()
}

fn is_permutation(permutation: &[u8]) -> bool {
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    sorted
        .iter()
        .enumerate()
        .all(|(index, &value)| value as usize == index)
}

fn are_inverse(first: &[u8], second: &[u8]) -> bool {
    first.len() == second.len()
        && (0..first.len()).all(|position| first[second[position] as usize] as usize == position)
}

fn permutation_order(permutation: &[u8]) -> Result<usize, String> {
    let identity: Vec<u8> = (0..permutation.len()).map(|value| value as u8).collect();
    let mut state = identity.clone();
    for order in 1..=10_000 {
        state = apply(&state, permutation);
        if state == identity {
            return Ok(order);
        }
    }
    Err("generator order exceeded probe limit".to_string())
}

fn audit_config(config: &Config) -> Result<Audit, String> {
    let state_len = config.central.len();
    if state_len > 256 {
        return Err("u8 position representation supports at most 256 entries".to_string());
    }
    for (name, permutation) in &config.generators {
        if permutation.len() != state_len || !is_permutation(permutation) {
            return Err(format!(
                "generator {name} is not a permutation of state positions"
            ));
        }
    }
    let identity: Vec<u8> = (0..state_len).map(|value| value as u8).collect();
    let depth_one: Vec<Vec<u8>> = config
        .generators
        .iter()
        .map(|(_, permutation)| apply(&config.central, permutation))
        .collect();
    let depth_one_unique = depth_one.iter().cloned().collect::<HashSet<_>>().len();
    let depth_one_loops = depth_one
        .iter()
        .filter(|state| *state == &config.central)
        .count();

    let by_name: BTreeMap<&str, &[u8]> = config
        .generators
        .iter()
        .map(|(name, permutation)| (name.as_str(), permutation.as_slice()))
        .collect();
    let mut inverse_pairs = 0;
    for (name, permutation) in &config.generators {
        if !name.starts_with('-') {
            let inverse_name = format!("-{name}");
            let inverse = by_name
                .get(inverse_name.as_str())
                .ok_or_else(|| format!("missing named inverse {inverse_name}"))?;
            if !are_inverse(permutation, inverse) {
                return Err(format!(
                    "{name} and {inverse_name} are not inverse permutations"
                ));
            }
            inverse_pairs += 1;
        }
    }

    let mut order_counts = BTreeMap::new();
    for (_, permutation) in &config.generators {
        *order_counts
            .entry(permutation_order(permutation)?)
            .or_insert(0) += 1;
    }
    Ok(Audit {
        state_len,
        move_count: config.generators.len(),
        central_is_identity: config.central == identity,
        inverse_pairs,
        depth_one_loops,
        depth_one_unique,
        generator_orders: order_counts.into_iter().collect(),
    })
}

fn exact_frontier_sizes(config: &Config, max_depth: usize) -> Vec<usize> {
    let mut visited = HashSet::from([config.central.clone()]);
    let mut frontier = vec![config.central.clone()];
    let mut sizes = vec![1];
    for _ in 0..max_depth {
        let mut next_seen = HashSet::new();
        let mut next = Vec::new();
        for state in &frontier {
            for (_, permutation) in &config.generators {
                let child = apply(state, permutation);
                if !visited.contains(&child) && next_seen.insert(child.clone()) {
                    next.push(child);
                }
            }
        }
        visited.extend(next.iter().cloned());
        sizes.push(next.len());
        frontier = next;
    }
    sizes
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ref025_megaminx_contract_probe PUZZLE_INFO_JSON");
    let text = std::fs::read_to_string(path).expect("read puzzle_info.json");
    let config = parse_config(&text).expect("parse puzzle_info.json");
    let audit = audit_config(&config).expect("audit puzzle_info.json");
    println!("metric,value");
    println!("state_len,{}", audit.state_len);
    println!("move_count,{}", audit.move_count);
    println!("central_is_identity,{}", audit.central_is_identity);
    println!("inverse_pairs,{}", audit.inverse_pairs);
    println!("depth_one_loops,{}", audit.depth_one_loops);
    println!("depth_one_unique,{}", audit.depth_one_unique);
    for (order, count) in audit.generator_orders {
        println!("generator_order_{order},{count}");
    }
    for (depth, size) in exact_frontier_sizes(&config, 4).into_iter().enumerate() {
        println!("frontier_{depth},{size}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_json_parser_reads_identity_and_named_permutations() {
        let text = r#"{"central_state":[0,1,2],"generators":{"a":[1,0,2],"-a":[1,0,2]}}"#;
        let config = parse_config(text).unwrap();
        assert_eq!(config.central, [0, 1, 2]);
        assert_eq!(config.generators[0], ("a".to_string(), vec![1, 0, 2]));
        assert_eq!(config.generators[1], ("-a".to_string(), vec![1, 0, 2]));
    }

    #[test]
    fn production_megaminx_contract_and_first_four_layers_are_exact() {
        let path = std::env::var("REF025_PUZZLE_INFO").expect("REF025_PUZZLE_INFO must be set");
        let text = std::fs::read_to_string(path).unwrap();
        let config = parse_config(&text).unwrap();
        let audit = audit_config(&config).unwrap();

        assert_eq!(audit.state_len, 120);
        assert_eq!(audit.move_count, 24);
        assert!(audit.central_is_identity);
        assert_eq!(audit.inverse_pairs, 12);
        assert_eq!(audit.depth_one_loops, 0);
        assert_eq!(audit.depth_one_unique, 24);
        assert_eq!(audit.generator_orders, vec![(5, 24)]);
        assert_eq!(
            exact_frontier_sizes(&config, 4),
            vec![1, 24, 408, 6208, 90144]
        );
    }
}
