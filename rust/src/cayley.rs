use std::ffi::{c_char, c_void, CStr};
use std::io::Write;

const N: usize = 8;
const STATES: usize = 40_320;
const GENERATORS: usize = N - 1;

type Permutation = [u8; N];
type Handle = *mut c_void;

#[link(name = "multigpubfs_cuda")]
extern "C" {
    fn mgbfs_cayley_create(
        n: i32,
        frontier_capacity: usize,
        handle: *mut Handle,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_cayley_reset(handle: Handle, error: *mut c_char, error_capacity: usize) -> i32;
    fn mgbfs_cayley_step(
        handle: Handle,
        variant: i32,
        layout: i32,
        next_frontier_count: *mut usize,
        kernel_milliseconds: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_cayley_copy_frontier(
        handle: Handle,
        host_frontier: *mut u64,
        host_capacity: usize,
        copied: *mut usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_cayley_destroy(handle: Handle);
}

struct GpuCayley {
    handle: Handle,
    host_frontier: Vec<u64>,
}

impl GpuCayley {
    fn new(n: usize, capacity: usize) -> Result<Self, String> {
        let mut error = vec![0 as c_char; 512];
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            mgbfs_cayley_create(
                n as i32,
                capacity,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Self {
            handle,
            host_frontier: vec![0; capacity],
        })
    }

    fn reset(&mut self) -> Result<(), String> {
        let mut error = vec![0 as c_char; 512];
        let status = unsafe { mgbfs_cayley_reset(self.handle, error.as_mut_ptr(), error.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        }
    }

    fn step(&mut self, variant: i32, layout: i32) -> Result<(usize, f64), String> {
        let mut error = vec![0 as c_char; 512];
        let mut count = 0;
        let mut milliseconds = 0.0_f32;
        let status = unsafe {
            mgbfs_cayley_step(
                self.handle,
                variant,
                layout,
                &mut count,
                &mut milliseconds,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok((count, milliseconds.into()))
        } else {
            Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        }
    }

    fn copy_frontier(&mut self) -> Result<Vec<u64>, String> {
        let mut error = vec![0 as c_char; 512];
        let mut copied = 0;
        let status = unsafe {
            mgbfs_cayley_copy_frontier(
                self.handle,
                self.host_frontier.as_mut_ptr(),
                self.host_frontier.len(),
                &mut copied,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(self.host_frontier[..copied].to_vec())
        } else {
            Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        }
    }
}

impl Drop for GpuCayley {
    fn drop(&mut self) {
        unsafe { mgbfs_cayley_destroy(self.handle) };
    }
}

fn rank(permutation: &Permutation) -> usize {
    let mut result = 0;
    for index in 0..N {
        let smaller = permutation[index + 1..]
            .iter()
            .filter(|&&value| value < permutation[index])
            .count();
        result = result * (N - index) + smaller;
    }
    result
}

fn successor(mut state: Permutation, generator: usize) -> Permutation {
    state.swap(generator, generator + 1);
    state
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn candidates(frontier: &[Permutation], generator_major: bool) -> Vec<(Permutation, usize)> {
    let mut output = Vec::with_capacity(frontier.len() * GENERATORS);
    if generator_major {
        for generator in 0..GENERATORS {
            for &state in frontier {
                let child = successor(state, generator);
                output.push((child, rank(&child)));
            }
        }
    } else {
        for &state in frontier {
            for generator in 0..GENERATORS {
                let child = successor(state, generator);
                output.push((child, rank(&child)));
            }
        }
    }
    output
}

struct Locality {
    equal_key_warp_savings: usize,
    equal_key_block_savings: usize,
    bitmap_word_warp_collisions: usize,
    warps_with_equal_keys: usize,
    blocks_with_equal_keys: usize,
    maximum_equal_key_lanes: usize,
}

fn locality(ranks: impl Iterator<Item = usize>, count: usize) -> Locality {
    let ranks: Vec<usize> = ranks.collect();
    debug_assert_eq!(ranks.len(), count);
    let mut equal_key_warp_savings = 0;
    let mut bitmap_word_warp_collisions = 0;
    let mut warps_with_equal_keys = 0;
    let mut maximum_equal_key_lanes = 0;
    for warp in ranks.chunks(32) {
        let mut key_counts = [0_u8; 32];
        let mut unique_keys = 0;
        let mut unique_words = 0;
        let mut words = [usize::MAX; 32];
        for (lane, &key) in warp.iter().enumerate() {
            let previous = warp[..lane].iter().position(|&other| other == key);
            if let Some(previous) = previous {
                key_counts[previous] += 1;
            } else {
                key_counts[lane] = 1;
                unique_keys += 1;
            }
            let word = key >> 5;
            if !words[..unique_words].contains(&word) {
                words[unique_words] = word;
                unique_words += 1;
            }
        }
        let savings = warp.len() - unique_keys;
        equal_key_warp_savings += savings;
        bitmap_word_warp_collisions += warp.len() - unique_words;
        warps_with_equal_keys += usize::from(savings != 0);
        maximum_equal_key_lanes = maximum_equal_key_lanes
            .max(key_counts.iter().copied().max().unwrap_or_default() as usize);
    }

    let mut equal_key_block_savings = 0;
    let mut blocks_with_equal_keys = 0;
    let mut seen = vec![false; STATES];
    let mut touched = Vec::with_capacity(256);
    for block in ranks.chunks(256) {
        touched.clear();
        for &key in block {
            if !seen[key] {
                seen[key] = true;
                touched.push(key);
            }
        }
        let savings = block.len() - touched.len();
        equal_key_block_savings += savings;
        blocks_with_equal_keys += usize::from(savings != 0);
        for &key in &touched {
            seen[key] = false;
        }
    }

    Locality {
        equal_key_warp_savings,
        equal_key_block_savings,
        bitmap_word_warp_collisions,
        warps_with_equal_keys,
        blocks_with_equal_keys,
        maximum_equal_key_lanes,
    }
}

fn emit(line: &str) -> Result<(), String> {
    println!("{line}");
    if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
        let mut output = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        writeln!(output, "{line}").map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub fn locality_sweep() -> Result<(), String> {
    if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
        std::fs::File::create(path).map_err(|error| error.to_string())?;
    }
    let identity: Permutation = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut visited = vec![false; STATES];
    visited[rank(&identity)] = true;
    let mut frontier = vec![identity];
    let mut discovery_frontier = frontier.clone();
    let mut depth = 0;
    let mut total_generated = 0;
    let mut total_batch_duplicates = 0;
    let mut total_unique_visited = 0;
    let mut total_accepted = 0;
    let mut peak_frontier = 0;

    loop {
        peak_frontier = peak_frontier.max(frontier.len());
        let parent_major = candidates(&frontier, false);
        let generated = parent_major.len();
        total_generated += generated;
        let mut frequencies = vec![0_u16; STATES];
        let mut representative: Vec<Option<Permutation>> = vec![None; STATES];
        for &(state, key) in &parent_major {
            frequencies[key] += 1;
            representative[key].get_or_insert(state);
        }
        let unique_candidates = frequencies.iter().filter(|&&count| count != 0).count();
        let batch_duplicates = generated - unique_candidates;
        let unique_visited = frequencies
            .iter()
            .enumerate()
            .filter(|&(key, &count)| count != 0 && visited[key])
            .count();
        let accepted = unique_candidates - unique_visited;
        let repeated_keys = frequencies.iter().filter(|&&count| count > 1).count();
        let maximum_multiplicity = frequencies.iter().copied().max().unwrap_or_default();
        total_batch_duplicates += batch_duplicates;
        total_unique_visited += unique_visited;
        total_accepted += accepted;

        let mut shuffled_frontier = frontier.clone();
        shuffled_frontier.sort_by_key(|state| mix64(rank(state) as u64));
        for (frontier_order, ordered_frontier) in [
            ("rank-sorted", frontier.as_slice()),
            ("discovery", discovery_frontier.as_slice()),
            ("hash-shuffled", shuffled_frontier.as_slice()),
        ] {
            let ordered_parent_major = candidates(ordered_frontier, false);
            let ordered_generator_major = candidates(ordered_frontier, true);
            for (layout, batch) in [
                ("parent-major", &ordered_parent_major),
                ("generator-major", &ordered_generator_major),
            ] {
                let metrics = locality(batch.iter().map(|&(_, key)| key), generated);
                emit(&format!(
                    "{{\"status\":\"pass\",\"experiment\":\"cayley-s8-locality-v2\",\"frontier_order\":\"{}\",\"layout\":\"{}\",\"depth\":{},\"frontier_count\":{},\"generators\":{},\"generated\":{},\"unique_candidates\":{},\"batch_duplicate_occurrences\":{},\"unique_visited_hits\":{},\"accepted\":{},\"repeated_keys\":{},\"maximum_multiplicity\":{},\"equal_key_warp_savings\":{},\"equal_key_block_savings\":{},\"bitmap_word_warp_collisions\":{},\"warps_with_equal_keys\":{},\"blocks_with_equal_keys\":{},\"maximum_equal_key_lanes\":{}}}",
                    frontier_order, layout, depth, frontier.len(), GENERATORS, generated,
                    unique_candidates, batch_duplicates, unique_visited, accepted,
                    repeated_keys, maximum_multiplicity, metrics.equal_key_warp_savings,
                    metrics.equal_key_block_savings, metrics.bitmap_word_warp_collisions,
                    metrics.warps_with_equal_keys, metrics.blocks_with_equal_keys,
                    metrics.maximum_equal_key_lanes
                ))?;
            }
        }

        if generated != batch_duplicates + unique_visited + accepted {
            return Err(format!("depth {depth} conservation failure"));
        }
        let discovery_candidates = candidates(&discovery_frontier, false);
        let mut next_discovery_frontier = Vec::with_capacity(accepted);
        let mut discovery_seen = vec![false; STATES];
        for (state, key) in discovery_candidates {
            if !visited[key] && !discovery_seen[key] {
                discovery_seen[key] = true;
                next_discovery_frontier.push(state);
            }
        }
        if next_discovery_frontier.len() != accepted {
            return Err(format!("depth {depth} discovery frontier mismatch"));
        }

        let mut next_frontier = Vec::with_capacity(accepted);
        for key in 0..STATES {
            if frequencies[key] != 0 && !visited[key] {
                visited[key] = true;
                next_frontier.push(representative[key].expect("representative must exist"));
            }
        }
        if next_frontier.len() != accepted {
            return Err(format!("depth {depth} frontier mismatch"));
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
        discovery_frontier = next_discovery_frontier;
        depth += 1;
    }

    let visited_count = visited.iter().filter(|&&value| value).count();
    if visited_count != STATES
        || depth != 28
        || peak_frontier != 3_836
        || total_generated != 282_240
        || total_batch_duplicates != 201_602
        || total_unique_visited != 40_319
        || total_accepted != 40_319
    {
        return Err(format!(
            "S8 oracle mismatch: visited={visited_count}, depth={depth}, peak={peak_frontier}, generated={total_generated}, duplicates={total_batch_duplicates}, visited_hits={total_unique_visited}, accepted={total_accepted}"
        ));
    }
    eprintln!(
        "{{\"status\":\"pass\",\"validator\":\"rust-cayley-s8-oracle-v1\",\"visited\":{},\"diameter\":{},\"peak_frontier\":{},\"generated\":{}}}",
        visited_count, depth, peak_frontier, total_generated
    );
    Ok(())
}

fn json_field<'a>(line: &'a str, name: &str) -> Result<&'a str, String> {
    let marker = format!("\"{name}\":");
    let value = line
        .split_once(&marker)
        .ok_or_else(|| format!("missing field {name}"))?
        .1;
    if let Some(value) = value.strip_prefix('"') {
        return value
            .split_once('"')
            .map(|(field, _)| field)
            .ok_or_else(|| format!("unterminated string field {name}"));
    }
    Ok(value
        .split([',', '}'])
        .next()
        .ok_or_else(|| format!("empty field {name}"))?)
}

fn usize_field(line: &str, name: &str) -> Result<usize, String> {
    json_field(line, name)?
        .parse::<usize>()
        .map_err(|error| error.to_string())
}

pub fn validate_locality_artifact() -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};

    let path = std::env::var("MGBFS_INPUT_PATH")
        .unwrap_or_else(|_| "/input/REF-016-cayley-s8-locality.jsonl".into());
    let contents = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let orders = ["rank-sorted", "discovery", "hash-shuffled"];
    let layouts = ["parent-major", "generator-major"];
    let mut combinations = BTreeSet::new();
    let mut depth_counts = BTreeMap::new();
    let mut totals = BTreeMap::new();
    let mut rows = 0;

    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        rows += 1;
        if json_field(line, "status")? != "pass"
            || json_field(line, "experiment")? != "cayley-s8-locality-v2"
        {
            return Err("non-passing or wrong-version REF-016 row".into());
        }
        let order = json_field(line, "frontier_order")?.to_owned();
        let layout = json_field(line, "layout")?.to_owned();
        let depth = usize_field(line, "depth")?;
        if !orders.contains(&order.as_str()) || !layouts.contains(&layout.as_str()) || depth > 28 {
            return Err(format!("unexpected dimension {order}/{layout}/{depth}"));
        }
        if !combinations.insert((order.clone(), layout.clone(), depth)) {
            return Err(format!("duplicate row {order}/{layout}/{depth}"));
        }
        let generated = usize_field(line, "generated")?;
        let duplicates = usize_field(line, "batch_duplicate_occurrences")?;
        let visited = usize_field(line, "unique_visited_hits")?;
        let accepted = usize_field(line, "accepted")?;
        if generated != duplicates + visited + accepted {
            return Err(format!("conservation failure {order}/{layout}/{depth}"));
        }
        let invariant = (
            usize_field(line, "frontier_count")?,
            generated,
            usize_field(line, "unique_candidates")?,
            duplicates,
            visited,
            accepted,
        );
        match depth_counts.entry(depth) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(invariant);
            }
            std::collections::btree_map::Entry::Occupied(entry) if *entry.get() != invariant => {
                return Err(format!(
                    "order/layout changed exact BFS counts at depth {depth}"
                ));
            }
            _ => {}
        }
        let total = totals.entry((order, layout)).or_insert((0_usize, 0_usize));
        total.0 += generated;
        total.1 += duplicates;
    }
    let expected_rows = orders.len() * layouts.len() * 29;
    if rows != expected_rows || combinations.len() != expected_rows || depth_counts.len() != 29 {
        return Err(format!(
            "expected {expected_rows} unique rows, found {rows}"
        ));
    }
    if totals.values().any(|&value| value != (282_240, 201_602)) {
        return Err("aggregate S8 counts differ between layouts/orders".into());
    }
    println!(
        "{{\"status\":\"pass\",\"validator\":\"rust-ref016-artifact-v1\",\"rows\":{},\"exact_depth_groups\":{},\"layout_order_groups\":{}}}",
        rows,
        depth_counts.len(),
        totals.len()
    );
    Ok(())
}

fn factorial_dynamic(n: usize) -> usize {
    (2..=n).product()
}

fn unpack_state(state: u64, n: usize) -> Result<Vec<u8>, String> {
    let mut permutation = Vec::with_capacity(n);
    let mut seen = 0_u16;
    for index in 0..n {
        let value = ((state >> (4 * index)) & 0xf) as u8;
        if value as usize >= n || seen & (1 << value) != 0 {
            return Err(format!("invalid packed permutation 0x{state:x}"));
        }
        seen |= 1 << value;
        permutation.push(value);
    }
    Ok(permutation)
}

fn rank_dynamic(permutation: &[u8]) -> usize {
    let n = permutation.len();
    let mut result = 0;
    for index in 0..n {
        let smaller = permutation[index + 1..]
            .iter()
            .filter(|&&value| value < permutation[index])
            .count();
        result = result * (n - index) + smaller;
    }
    result
}

fn inversion_count(permutation: &[u8]) -> usize {
    let mut inversions = 0;
    for left in 0..permutation.len() {
        for right in left + 1..permutation.len() {
            inversions += usize::from(permutation[left] > permutation[right]);
        }
    }
    inversions
}

fn mahonian_counts(n: usize) -> Vec<usize> {
    let mut counts = vec![1_usize];
    for size in 2..=n {
        let mut next = vec![0; counts.len() + size - 1];
        for (inversions, &count) in counts.iter().enumerate() {
            for added in 0..size {
                next[inversions + added] += count;
            }
        }
        counts = next;
    }
    counts
}

fn packed_successor_rank(state: u64, generator: usize, n: usize) -> usize {
    let left_shift = 4 * generator;
    let right_shift = left_shift + 4;
    let left = (state >> left_shift) & 0xf;
    let right = (state >> right_shift) & 0xf;
    let mask = (0xf_u64 << left_shift) | (0xf_u64 << right_shift);
    let child = (state & !mask) | (left << right_shift) | (right << left_shift);
    let mut permutation = [0_u8; 10];
    for (index, value) in permutation[..n].iter_mut().enumerate() {
        *value = ((child >> (4 * index)) & 0xf) as u8;
    }
    rank_dynamic(&permutation[..n])
}

fn packed_candidate_locality(frontier: &[u64], n: usize, layout: i32) -> (usize, usize) {
    let candidate_count = frontier.len() * (n - 1);
    let mut warp_savings = 0;
    let mut word_collisions = 0;
    for warp_start in (0..candidate_count).step_by(32) {
        let warp_end = (warp_start + 32).min(candidate_count);
        let mut keys = [usize::MAX; 32];
        let mut unique_keys = 0;
        let mut words = [usize::MAX; 32];
        let mut unique_words = 0;
        for candidate_index in warp_start..warp_end {
            let (parent, generator) = if layout == 0 {
                (candidate_index / (n - 1), candidate_index % (n - 1))
            } else {
                (
                    candidate_index % frontier.len(),
                    candidate_index / frontier.len(),
                )
            };
            let key = packed_successor_rank(frontier[parent], generator, n);
            if !keys[..unique_keys].contains(&key) {
                keys[unique_keys] = key;
                unique_keys += 1;
            }
            let word = key >> 5;
            if !words[..unique_words].contains(&word) {
                words[unique_words] = word;
                unique_words += 1;
            }
        }
        let warp_len = warp_end - warp_start;
        warp_savings += warp_len - unique_keys;
        word_collisions += warp_len - unique_words;
    }
    (warp_savings, word_collisions)
}

fn validate_gpu_frontier(
    frontier: &[u64],
    n: usize,
    depth: usize,
    expected_count: usize,
    seen: &mut [bool],
) -> Result<(u64, u64), String> {
    if frontier.len() != expected_count {
        return Err(format!(
            "depth {depth}: frontier count {} != Mahonian {expected_count}",
            frontier.len()
        ));
    }
    let mut sum = 0_u64;
    let mut xor = 0_u64;
    let mut touched = Vec::with_capacity(frontier.len());
    for &state in frontier {
        let permutation = unpack_state(state, n)?;
        if inversion_count(&permutation) != depth {
            return Err(format!("depth {depth}: inversion count mismatch"));
        }
        let key = rank_dynamic(&permutation);
        if seen[key] {
            return Err(format!("depth {depth}: duplicate rank {key}"));
        }
        seen[key] = true;
        touched.push(key);
        let mixed = mix64(key as u64);
        sum = sum.wrapping_add(mixed);
        xor ^= mixed;
    }
    for key in touched {
        seen[key] = false;
    }
    Ok((sum, xor))
}

fn run_gpu_traversal(
    n: usize,
    variant: i32,
    layout: i32,
    repetition: usize,
    emit_rows: bool,
) -> Result<(f64, usize), String> {
    let state_count = factorial_dynamic(n);
    let expected = mahonian_counts(n);
    let mut context = GpuCayley::new(n, state_count)?;
    context.reset()?;
    let mut frontier = context.copy_frontier()?;
    let mut seen = vec![false; state_count];
    validate_gpu_frontier(&frontier, n, 0, expected[0], &mut seen)?;
    let mut total_kernel_ms = 0.0;
    let mut total_generated = 0;
    for depth in 0..expected.len() {
        let generated = frontier.len() * (n - 1);
        let (warp_savings, word_collisions) = packed_candidate_locality(&frontier, n, layout);
        let (next_count, kernel_ms) = context.step(variant, layout)?;
        total_kernel_ms += kernel_ms;
        total_generated += generated;
        let next_frontier = context.copy_frontier()?;
        let expected_next = expected.get(depth + 1).copied().unwrap_or(0);
        let fingerprint =
            validate_gpu_frontier(&next_frontier, n, depth + 1, expected_next, &mut seen)?;
        if next_count != expected_next {
            return Err(format!("depth {depth}: device count drift"));
        }
        if emit_rows {
            emit(&format!(
                "{{\"status\":\"pass\",\"experiment\":\"gpu-cayley-levels-v1\",\"n\":{},\"repetition\":{},\"variant\":\"{}\",\"layout\":\"{}\",\"depth\":{},\"frontier_count\":{},\"generated\":{},\"equal_key_warp_savings\":{},\"bitmap_word_warp_collisions\":{},\"next_frontier_count\":{},\"kernel_ms\":{:.6},\"next_validation_sum\":{},\"next_validation_xor\":{}}}",
                n,
                repetition,
                if variant == 0 { "baseline" } else { "warp-aggregate" },
                if layout == 0 { "parent-major" } else { "generator-major" },
                depth, frontier.len(), generated, warp_savings, word_collisions,
                next_count, kernel_ms, fingerprint.0, fingerprint.1
            ))?;
        }
        frontier = next_frontier;
    }
    if !frontier.is_empty() || total_generated != state_count * (n - 1) {
        return Err("full GPU Cayley traversal accounting mismatch".into());
    }
    Ok((total_kernel_ms, total_generated))
}

fn run_gpu_timed(n: usize, variant: i32, layout: i32) -> Result<(f64, f64), String> {
    use std::time::Instant;

    let state_count = factorial_dynamic(n);
    let expected = mahonian_counts(n);
    let mut context = GpuCayley::new(n, state_count)?;
    context.reset()?;
    let started = Instant::now();
    let mut kernel_ms = 0.0;
    for depth in 0..expected.len() {
        let (next_count, level_kernel_ms) = context.step(variant, layout)?;
        let expected_next = expected.get(depth + 1).copied().unwrap_or(0);
        if next_count != expected_next {
            return Err(format!("timed depth {depth}: Mahonian count mismatch"));
        }
        kernel_ms += level_kernel_ms;
    }
    let traversal_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((kernel_ms, traversal_ms))
}

pub fn gpu_self_test() -> Result<(), String> {
    for variant in 0..=1 {
        for layout in 0..=1 {
            run_gpu_traversal(8, variant, layout, 0, false)?;
        }
    }
    Ok(())
}

pub fn gpu_s9_sweep() -> Result<(), String> {
    if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
        std::fs::File::create(path).map_err(|error| error.to_string())?;
    }
    for variant in 0..=1 {
        for layout in 0..=1 {
            run_gpu_traversal(9, variant, layout, 0, true)?;
            for _ in 0..2 {
                run_gpu_timed(9, variant, layout)?;
            }
            for repetition in 0..10 {
                let (kernel_ms, traversal_ms) = run_gpu_timed(9, variant, layout)?;
                emit(&format!(
                    "{{\"status\":\"pass\",\"experiment\":\"gpu-cayley-traversal-v1\",\"n\":9,\"repetition\":{},\"variant\":\"{}\",\"layout\":\"{}\",\"levels\":37,\"generated\":2903040,\"kernel_ms_sum\":{:.6},\"traversal_ms\":{:.6},\"kernel_billion_transitions_per_s\":{:.6},\"traversal_billion_transitions_per_s\":{:.6}}}",
                    repetition,
                    if variant == 0 { "baseline" } else { "warp-aggregate" },
                    if layout == 0 { "parent-major" } else { "generator-major" },
                    kernel_ms,
                    traversal_ms,
                    2_903_040.0 / (kernel_ms / 1_000.0) / 1e9,
                    2_903_040.0 / (traversal_ms / 1_000.0) / 1e9
                ))?;
            }
        }
    }
    Ok(())
}

pub fn validate_gpu_s9_artifact() -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};

    let path = std::env::var("MGBFS_INPUT_PATH")
        .unwrap_or_else(|_| "/input/REF-017-gpu-cayley-s9-levels.jsonl".into());
    let contents = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let expected = mahonian_counts(9);
    let variants = ["baseline", "warp-aggregate"];
    let layouts = ["parent-major", "generator-major"];
    let mut level_keys = BTreeSet::new();
    let mut traversal_keys = BTreeSet::new();
    let mut depth_outcomes = BTreeMap::new();
    let mut level_rows = 0;
    let mut traversal_rows = 0;

    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        if json_field(line, "status")? != "pass" || usize_field(line, "n")? != 9 {
            return Err("non-passing or wrong-n REF-017 row".into());
        }
        let experiment = json_field(line, "experiment")?;
        let variant = json_field(line, "variant")?.to_owned();
        let layout = json_field(line, "layout")?.to_owned();
        if !variants.contains(&variant.as_str()) || !layouts.contains(&layout.as_str()) {
            return Err("unexpected REF-017 variant/layout".into());
        }
        match experiment {
            "gpu-cayley-levels-v1" => {
                level_rows += 1;
                let depth = usize_field(line, "depth")?;
                if depth >= expected.len()
                    || !level_keys.insert((variant.clone(), layout.clone(), depth))
                {
                    return Err(format!(
                        "duplicate/invalid level row {variant}/{layout}/{depth}"
                    ));
                }
                let frontier = usize_field(line, "frontier_count")?;
                let generated = usize_field(line, "generated")?;
                let next = usize_field(line, "next_frontier_count")?;
                let expected_next = expected.get(depth + 1).copied().unwrap_or(0);
                if frontier != expected[depth] || generated != frontier * 8 || next != expected_next
                {
                    return Err(format!("Mahonian mismatch at depth {depth}"));
                }
                let outcome = (
                    next,
                    json_field(line, "next_validation_sum")?
                        .parse::<u64>()
                        .map_err(|error| error.to_string())?,
                    json_field(line, "next_validation_xor")?
                        .parse::<u64>()
                        .map_err(|error| error.to_string())?,
                );
                match depth_outcomes.entry(depth) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(outcome);
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if *entry.get() != outcome =>
                    {
                        return Err(format!("cross-config frontier mismatch at depth {depth}"));
                    }
                    _ => {}
                }
            }
            "gpu-cayley-traversal-v1" => {
                traversal_rows += 1;
                let repetition = usize_field(line, "repetition")?;
                if repetition >= 10
                    || !traversal_keys.insert((variant.clone(), layout.clone(), repetition))
                {
                    return Err(format!(
                        "duplicate/invalid traversal row {variant}/{layout}/{repetition}"
                    ));
                }
                if usize_field(line, "levels")? != 37
                    || usize_field(line, "generated")? != 2_903_040
                    || json_field(line, "kernel_ms_sum")?
                        .parse::<f64>()
                        .map_err(|error| error.to_string())?
                        <= 0.0
                    || json_field(line, "traversal_ms")?
                        .parse::<f64>()
                        .map_err(|error| error.to_string())?
                        <= 0.0
                {
                    return Err("invalid traversal summary".into());
                }
            }
            _ => return Err(format!("unknown REF-017 experiment {experiment}")),
        }
    }
    if level_rows != 148
        || traversal_rows != 40
        || level_keys.len() != 148
        || traversal_keys.len() != 40
        || depth_outcomes.len() != 37
    {
        return Err(format!(
            "REF-017 coverage mismatch: levels={level_rows}, traversals={traversal_rows}"
        ));
    }
    println!(
        "{{\"status\":\"pass\",\"validator\":\"rust-ref017-artifact-v1\",\"level_rows\":{},\"traversal_rows\":{},\"exact_depth_groups\":{}}}",
        level_rows,
        traversal_rows,
        depth_outcomes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lehmer_rank_is_a_bijection_on_s8() {
        fn enumerate(prefix: &mut Vec<u8>, unused: &mut Vec<u8>, seen: &mut [bool]) {
            if unused.is_empty() {
                let permutation: Permutation = prefix.as_slice().try_into().unwrap();
                let key = rank(&permutation);
                assert!(!seen[key]);
                seen[key] = true;
                return;
            }
            for index in 0..unused.len() {
                let value = unused.remove(index);
                prefix.push(value);
                enumerate(prefix, unused, seen);
                prefix.pop();
                unused.insert(index, value);
            }
        }
        let mut seen = vec![false; STATES];
        enumerate(&mut Vec::new(), &mut (0..N as u8).collect(), &mut seen);
        assert!(seen.into_iter().all(|value| value));
    }
}
