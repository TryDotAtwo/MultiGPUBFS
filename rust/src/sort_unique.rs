use std::ffi::{c_char, c_void, CStr};

type Handle = *mut c_void;

#[link(name = "multigpubfs_cuda")]
extern "C" {
    fn mgbfs_sort_unique_create(
        universe_size: u64,
        candidate_capacity: usize,
        output_capacity: usize,
        handle: *mut Handle,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_sort_unique_seed(
        handle: Handle,
        host_keys: *const u32,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_sort_unique_upload(
        handle: Handle,
        host_candidates: *const u32,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_sort_unique_run(
        handle: Handle,
        count: usize,
        host_output: *mut u32,
        host_output_capacity: usize,
        unique_count: *mut usize,
        accepted_count: *mut usize,
        output_written: *mut usize,
        overflow: *mut i32,
        sort_milliseconds: *mut f32,
        unique_milliseconds: *mut f32,
        claim_milliseconds: *mut f32,
        total_milliseconds: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_sort_unique_destroy(handle: Handle);
    fn mgbfs_sort_unique_memory(
        handle: Handle,
        temporary_bytes: *mut usize,
        allocated_bytes: *mut usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

pub struct SortUniqueVisited {
    handle: Handle,
    capacity: usize,
    output: Vec<u32>,
    temporary_bytes: usize,
    allocated_bytes: usize,
}

pub struct Stats {
    pub unique_count: usize,
    pub accepted_count: usize,
    pub output_written: usize,
    pub overflow: bool,
    pub sort_ms: f64,
    pub unique_ms: f64,
    pub claim_ms: f64,
    pub total_ms: f64,
}

fn ffi_error(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl SortUniqueVisited {
    pub fn new(universe: u64, capacity: usize, output_capacity: usize) -> Result<Self, String> {
        let mut error = vec![0 as c_char; 512];
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            mgbfs_sort_unique_create(
                universe,
                capacity,
                output_capacity,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(&error));
        }
        let mut temporary_bytes = 0;
        let mut allocated_bytes = 0;
        let status = unsafe {
            mgbfs_sort_unique_memory(
                handle,
                &mut temporary_bytes,
                &mut allocated_bytes,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            unsafe { mgbfs_sort_unique_destroy(handle) };
            return Err(ffi_error(&error));
        }
        Ok(Self {
            handle,
            capacity,
            output: vec![0; output_capacity],
            temporary_bytes,
            allocated_bytes,
        })
    }

    pub fn seed(&mut self, keys: &[u32]) -> Result<(), String> {
        let mut error = vec![0 as c_char; 512];
        let status = unsafe {
            mgbfs_sort_unique_seed(
                self.handle,
                keys.as_ptr(),
                keys.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (status == 0).then_some(()).ok_or_else(|| ffi_error(&error))
    }

    pub fn upload(&mut self, candidates: &[u32]) -> Result<(), String> {
        if candidates.len() > self.capacity {
            return Err("candidate capacity exceeded".into());
        }
        let mut error = vec![0 as c_char; 512];
        let status = unsafe {
            mgbfs_sort_unique_upload(
                self.handle,
                candidates.as_ptr(),
                candidates.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (status == 0).then_some(()).ok_or_else(|| ffi_error(&error))
    }

    pub fn run(&mut self, count: usize) -> Result<Stats, String> {
        let mut error = vec![0 as c_char; 512];
        let mut unique_count = 0;
        let mut accepted_count = 0;
        let mut output_written = 0;
        let mut overflow = 0;
        let mut sort_ms = 0.0_f32;
        let mut unique_ms = 0.0_f32;
        let mut claim_ms = 0.0_f32;
        let mut total_ms = 0.0_f32;
        let status = unsafe {
            mgbfs_sort_unique_run(
                self.handle,
                count,
                self.output.as_mut_ptr(),
                self.output.len(),
                &mut unique_count,
                &mut accepted_count,
                &mut output_written,
                &mut overflow,
                &mut sort_ms,
                &mut unique_ms,
                &mut claim_ms,
                &mut total_ms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(&error));
        }
        Ok(Stats {
            unique_count,
            accepted_count,
            output_written,
            overflow: overflow != 0,
            sort_ms: sort_ms.into(),
            unique_ms: unique_ms.into(),
            claim_ms: claim_ms.into(),
            total_ms: total_ms.into(),
        })
    }

    pub fn output(&self, count: usize) -> &[u32] {
        &self.output[..count]
    }
}

impl Drop for SortUniqueVisited {
    fn drop(&mut self) {
        unsafe { mgbfs_sort_unique_destroy(self.handle) };
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn fingerprint(values: impl Iterator<Item = u32>) -> (u64, u64) {
    values.fold((0, 0), |(sum, xor), value| {
        let mixed = mix64(value as u64);
        (sum.wrapping_add(mixed), xor ^ mixed)
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

pub fn self_test() -> Result<(), String> {
    let mut backend = SortUniqueVisited::new(64, 16, 16)?;
    backend.seed(&[2, 9])?;
    let candidates = [9, 4, 4, 63, 2, 5, 5, 0];
    backend.upload(&candidates)?;
    let stats = backend.run(candidates.len())?;
    let mut actual = backend.output(stats.output_written).to_vec();
    actual.sort_unstable();
    if stats.unique_count != 6 || stats.accepted_count != 4 || stats.overflow {
        return Err("self-test count mismatch".into());
    }
    if actual != [0, 4, 5, 63] {
        return Err(format!("self-test set mismatch: {actual:?}"));
    }
    backend.upload(&candidates)?;
    let repeated = backend.run(candidates.len())?;
    if repeated.unique_count != 6 || repeated.accepted_count != 0 || repeated.overflow {
        return Err("self-test persistent visited mismatch".into());
    }
    backend.upload(&[])?;
    let empty = backend.run(0)?;
    if empty.unique_count != 0 || empty.accepted_count != 0 || empty.overflow {
        return Err("self-test empty batch mismatch".into());
    }

    let mut overflow_backend = SortUniqueVisited::new(64, 8, 2)?;
    overflow_backend.seed(&[])?;
    overflow_backend.upload(&[1, 2, 2, 3, 4])?;
    let overflow = overflow_backend.run(5)?;
    if overflow.unique_count != 4 || overflow.accepted_count != 4 || !overflow.overflow {
        return Err("self-test overflow mismatch".into());
    }
    overflow_backend.seed(&[])?;
    overflow_backend.upload(&[64])?;
    if overflow_backend.run(1).is_ok() {
        return Err("self-test accepted an out-of-range key".into());
    }
    Ok(())
}

pub fn sweep() -> Result<(), String> {
    use std::io::Write;

    const UNIVERSE: u64 = 1 << 24;
    const WARMUPS: usize = 3;
    const REPETITIONS: usize = 10;
    let sizes = [1_usize << 16, 1 << 20, 1 << 22, 1 << 24];
    let patterns = ["all-new", "half-seeded-fourfold", "all-seen", "single-key"];
    if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
        std::fs::File::create(path).map_err(|error| error.to_string())?;
    }

    for &candidate_count in &sizes {
        for pattern in patterns {
            let key_space = match pattern {
                "half-seeded-fourfold" => candidate_count / 4,
                "all-seen" => candidate_count.min(1 << 20),
                _ => candidate_count,
            };
            let candidates: Vec<u32> = match pattern {
                "all-new" => (0..candidate_count as u32).collect(),
                "half-seeded-fourfold" | "all-seen" => (0..candidate_count)
                    .map(|index| {
                        (index as u32).wrapping_mul(2_654_435_761) & (key_space as u32 - 1)
                    })
                    .collect(),
                "single-key" => vec![7; candidate_count],
                _ => unreachable!(),
            };
            let seed: Vec<u32> = match pattern {
                "half-seeded-fourfold" => (0..key_space as u32).step_by(2).collect(),
                "all-seen" => (0..key_space as u32).collect(),
                _ => Vec::new(),
            };
            let expected_unique = match pattern {
                "all-new" => candidate_count,
                "half-seeded-fourfold" | "all-seen" => key_space,
                "single-key" => 1,
                _ => unreachable!(),
            };
            let expected_accepted = match pattern {
                "all-new" => candidate_count,
                "half-seeded-fourfold" => key_space / 2,
                "all-seen" => 0,
                "single-key" => 1,
                _ => unreachable!(),
            };
            let expected_fingerprint = match pattern {
                "all-new" => fingerprint((0..candidate_count).map(|value| value as u32)),
                "half-seeded-fourfold" => {
                    fingerprint((0..expected_accepted).map(|value| value as u32 * 2 + 1))
                }
                "all-seen" => (0, 0),
                "single-key" => fingerprint(std::iter::once(7)),
                _ => unreachable!(),
            };
            let mut backend = SortUniqueVisited::new(UNIVERSE, candidate_count, candidate_count)?;
            for _ in 0..WARMUPS {
                backend.seed(&seed)?;
                backend.upload(&candidates)?;
                let stats = backend.run(candidate_count)?;
                if stats.unique_count != expected_unique
                    || stats.accepted_count != expected_accepted
                    || stats.overflow
                {
                    return Err(format!("{pattern}/{candidate_count} warmup mismatch"));
                }
            }
            let mut sort_times = Vec::with_capacity(REPETITIONS);
            let mut unique_times = Vec::with_capacity(REPETITIONS);
            let mut claim_times = Vec::with_capacity(REPETITIONS);
            let mut total_times = Vec::with_capacity(REPETITIONS);
            let mut last_written = 0;
            for _ in 0..REPETITIONS {
                backend.seed(&seed)?;
                backend.upload(&candidates)?;
                let stats = backend.run(candidate_count)?;
                if stats.unique_count != expected_unique
                    || stats.accepted_count != expected_accepted
                    || stats.overflow
                {
                    return Err(format!("{pattern}/{candidate_count} measured mismatch"));
                }
                last_written = stats.output_written;
                sort_times.push(stats.sort_ms);
                unique_times.push(stats.unique_ms);
                claim_times.push(stats.claim_ms);
                total_times.push(stats.total_ms);
            }
            let actual_fingerprint = fingerprint(backend.output(last_written).iter().copied());
            if last_written != expected_accepted || actual_fingerprint != expected_fingerprint {
                return Err(format!("{pattern}/{candidate_count} fingerprint mismatch"));
            }
            let sort_ms = median(&mut sort_times);
            let unique_ms = median(&mut unique_times);
            let claim_ms = median(&mut claim_times);
            let total_ms = median(&mut total_times);
            let line = format!(
                "{{\"status\":\"pass\",\"benchmark\":\"sort-unique-visited-v1\",\"pattern\":\"{}\",\"universe\":{},\"candidate_count\":{},\"key_space\":{},\"seed_count\":{},\"unique_count\":{},\"accepted_count\":{},\"device_temporary_bytes\":{},\"device_allocated_bytes\":{},\"warmups\":{},\"repetitions\":{},\"sort_ms_median\":{:.6},\"unique_ms_median\":{:.6},\"claim_ms_median\":{:.6},\"pipeline_ms_median\":{:.6},\"pipeline_billion_candidates_per_s\":{:.6},\"validation_sum\":{},\"validation_xor\":{}}}",
                pattern, UNIVERSE, candidate_count, key_space, seed.len(), expected_unique,
                expected_accepted, backend.temporary_bytes, backend.allocated_bytes,
                WARMUPS, REPETITIONS, sort_ms, unique_ms, claim_ms, total_ms,
                candidate_count as f64 / (total_ms / 1_000.0) / 1e9,
                actual_fingerprint.0, actual_fingerprint.1
            );
            println!("{line}");
            if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
                let mut output = std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .map_err(|error| error.to_string())?;
                writeln!(output, "{line}").map_err(|error| error.to_string())?;
            }
        }
    }
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

pub fn validate_artifact() -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};

    let directory = std::env::var("MGBFS_INPUT_DIR").unwrap_or_else(|_| "/input".into());
    let sort_contents =
        std::fs::read_to_string(format!("{directory}/REF-015-sort-unique-sweep.jsonl"))
            .map_err(|error| error.to_string())?;
    let bitmap_contents =
        std::fs::read_to_string(format!("{directory}/REF-014-bitmap-variant-sweep.jsonl"))
            .map_err(|error| error.to_string())?;
    let patterns = ["all-new", "half-seeded-fourfold", "all-seen", "single-key"];
    let sizes = [1_usize << 16, 1 << 20, 1 << 22, 1 << 24];
    let mut sort_outcomes = BTreeMap::new();
    let mut sort_rows = 0;
    for line in sort_contents.lines().filter(|line| !line.trim().is_empty()) {
        sort_rows += 1;
        if json_field(line, "status")? != "pass"
            || json_field(line, "benchmark")? != "sort-unique-visited-v1"
        {
            return Err("non-passing REF-015 row".into());
        }
        let pattern = json_field(line, "pattern")?.to_owned();
        let size = json_field(line, "candidate_count")?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if !patterns.contains(&pattern.as_str()) || !sizes.contains(&size) {
            return Err(format!("unexpected REF-015 dimension {pattern}/{size}"));
        }
        let outcome = (
            json_field(line, "accepted_count")?
                .parse::<usize>()
                .map_err(|error| error.to_string())?,
            json_field(line, "validation_sum")?
                .parse::<u64>()
                .map_err(|error| error.to_string())?,
            json_field(line, "validation_xor")?
                .parse::<u64>()
                .map_err(|error| error.to_string())?,
        );
        if sort_outcomes
            .insert((pattern.clone(), size), outcome)
            .is_some()
        {
            return Err(format!("duplicate REF-015 row {pattern}/{size}"));
        }
    }
    if sort_rows != 16 || sort_outcomes.len() != 16 {
        return Err(format!("expected 16 REF-015 rows, found {sort_rows}"));
    }

    let mut compared = BTreeSet::new();
    for line in bitmap_contents
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        if json_field(line, "variant")? != "baseline" {
            continue;
        }
        let pattern = json_field(line, "pattern")?.to_owned();
        let size = json_field(line, "candidate_count")?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let bitmap_outcome = (
            json_field(line, "accepted_count")?
                .parse::<usize>()
                .map_err(|error| error.to_string())?,
            json_field(line, "validation_sum")?
                .parse::<u64>()
                .map_err(|error| error.to_string())?,
            json_field(line, "validation_xor")?
                .parse::<u64>()
                .map_err(|error| error.to_string())?,
        );
        if sort_outcomes.get(&(pattern.clone(), size)) != Some(&bitmap_outcome) {
            return Err(format!("REF-014/015 outcome mismatch {pattern}/{size}"));
        }
        compared.insert((pattern, size));
    }
    if compared.len() != 16 {
        return Err(format!(
            "expected 16 cross-backend comparisons, found {}",
            compared.len()
        ));
    }
    println!(
        "{{\"status\":\"pass\",\"validator\":\"rust-ref015-artifact-v1\",\"rows\":{},\"cross_backend_outcomes\":{}}}",
        sort_rows,
        compared.len()
    );
    Ok(())
}
