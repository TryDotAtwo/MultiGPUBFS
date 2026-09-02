use std::ffi::{c_char, c_void, CStr};

type Handle = *mut c_void;

#[link(name = "multigpubfs_cuda")]
extern "C" {
    fn mgbfs_bitmap_create(
        universe_size: u64,
        candidate_capacity: usize,
        output_capacity: usize,
        handle: *mut Handle,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_bitmap_seed(
        handle: Handle,
        host_keys: *const u32,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_bitmap_upload(
        handle: Handle,
        host_candidates: *const u32,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_bitmap_run_variant(
        handle: Handle,
        variant: i32,
        count: usize,
        host_output: *mut u32,
        host_output_capacity: usize,
        accepted_count: *mut usize,
        output_written: *mut usize,
        overflow: *mut i32,
        kernel_milliseconds: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_bitmap_destroy(handle: Handle);
}

pub struct BitmapVisited {
    handle: Handle,
    candidate_capacity: usize,
    host_output: Vec<u32>,
}

pub struct FilterStats {
    pub accepted_count: usize,
    pub output_written: usize,
    pub overflow: bool,
    pub kernel_milliseconds: f32,
}

#[derive(Clone, Copy)]
#[repr(i32)]
pub enum BitmapVariant {
    Baseline = 0,
    WarpAggregate = 1,
    BlockCompact = 2,
    WarpBlock = 3,
}

impl BitmapVariant {
    pub const ALL: [Self; 4] = [
        Self::Baseline,
        Self::WarpAggregate,
        Self::BlockCompact,
        Self::WarpBlock,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::WarpAggregate => "warp-aggregate",
            Self::BlockCompact => "block-compact",
            Self::WarpBlock => "warp-block",
        }
    }
}

impl BitmapVisited {
    pub fn new(
        universe_size: u64,
        candidate_capacity: usize,
        output_capacity: usize,
    ) -> Result<Self, String> {
        let mut error = vec![0 as c_char; 512];
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            mgbfs_bitmap_create(
                universe_size,
                candidate_capacity,
                output_capacity,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        Ok(Self {
            handle,
            candidate_capacity,
            host_output: vec![0; output_capacity],
        })
    }

    pub fn seed(&mut self, keys: &[u32]) -> Result<(), String> {
        if keys.len() > self.candidate_capacity {
            return Err("seed exceeds candidate capacity".into());
        }
        let mut error = vec![0 as c_char; 512];
        let status = unsafe {
            mgbfs_bitmap_seed(
                self.handle,
                keys.as_ptr(),
                keys.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)
    }

    pub fn upload(&mut self, candidates: &[u32]) -> Result<(), String> {
        if candidates.len() > self.candidate_capacity {
            return Err("upload exceeds candidate capacity".into());
        }
        let mut error = vec![0 as c_char; 512];
        let status = unsafe {
            mgbfs_bitmap_upload(
                self.handle,
                candidates.as_ptr(),
                candidates.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)
    }

    pub fn run_in_place(&mut self, count: usize) -> Result<FilterStats, String> {
        self.run_in_place_variant(BitmapVariant::Baseline, count)
    }

    pub fn run_in_place_variant(
        &mut self,
        variant: BitmapVariant,
        count: usize,
    ) -> Result<FilterStats, String> {
        if count > self.candidate_capacity {
            return Err("run count exceeds candidate capacity".into());
        }
        let mut accepted_count = 0;
        let mut output_written = 0;
        let mut overflow = 0;
        let mut kernel_milliseconds = 0.0;
        let mut error = vec![0 as c_char; 512];
        let status = unsafe {
            mgbfs_bitmap_run_variant(
                self.handle,
                variant as i32,
                count,
                self.host_output.as_mut_ptr(),
                self.host_output.len(),
                &mut accepted_count,
                &mut output_written,
                &mut overflow,
                &mut kernel_milliseconds,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        Ok(FilterStats {
            accepted_count,
            output_written,
            overflow: overflow != 0,
            kernel_milliseconds,
        })
    }

    pub fn output(&self, written: usize) -> &[u32] {
        &self.host_output[..written]
    }
}

impl Drop for BitmapVisited {
    fn drop(&mut self) {
        unsafe { mgbfs_bitmap_destroy(self.handle) };
    }
}

fn check(status: i32, error: &[c_char]) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(if message.is_empty() {
        format!("CUDA bitmap call failed with status {status}")
    } else {
        message
    })
}

pub fn self_test() -> Result<(), String> {
    let candidates = [0, 1, 2, 2, 63, 62, 0];
    let mut bitmap = BitmapVisited::new(64, candidates.len(), candidates.len())?;
    bitmap.seed(&[1, 63])?;
    bitmap.upload(&candidates)?;
    let result = bitmap.run_in_place(candidates.len())?;
    let mut output = bitmap.output(result.output_written).to_vec();
    output.sort_unstable();
    if output != [0, 2, 62] || result.accepted_count != 3 || result.overflow {
        return Err(format!(
            "dedup fixture mismatch: output={:?} accepted={} overflow={}",
            output, result.accepted_count, result.overflow
        ));
    }

    for variant in BitmapVariant::ALL {
        let mut candidate = BitmapVisited::new(64, candidates.len(), candidates.len())?;
        candidate.seed(&[1, 63])?;
        candidate.upload(&candidates)?;
        let stats = candidate.run_in_place_variant(variant, candidates.len())?;
        let mut actual = candidate.output(stats.output_written).to_vec();
        actual.sort_unstable();
        if actual != [0, 2, 62] || stats.accepted_count != 3 || stats.overflow {
            return Err(format!("{} fixture mismatch", variant.name()));
        }
    }

    let second = bitmap.run_in_place(candidates.len())?;
    if second.accepted_count != 0 || second.output_written != 0 || second.overflow {
        return Err("visited persistence fixture failed".into());
    }

    bitmap.seed(&[])?;
    bitmap.upload(&[])?;
    let empty = bitmap.run_in_place(0)?;
    if empty.accepted_count != 0 || empty.output_written != 0 || empty.overflow {
        return Err("zero-candidate fixture failed".into());
    }

    let overflow_candidates = [3, 4, 5];
    for variant in BitmapVariant::ALL {
        let mut small = BitmapVisited::new(64, 3, 2)?;
        small.seed(&[])?;
        small.upload(&overflow_candidates)?;
        let overflow = small.run_in_place_variant(variant, 3)?;
        if overflow.accepted_count != 3 || overflow.output_written != 2 || !overflow.overflow {
            return Err(format!("{} overflow fixture failed", variant.name()));
        }
    }

    let mut invalid = BitmapVisited::new(64, 1, 1)?;
    invalid.seed(&[])?;
    invalid.upload(&[64])?;
    if invalid.run_in_place(1).is_ok() {
        return Err("out-of-range candidate was silently accepted".into());
    }
    if invalid.seed(&[64]).is_ok() {
        return Err("out-of-range seed was silently accepted".into());
    }
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub fn benchmark() -> Result<(), String> {
    use std::time::Instant;

    const UNIVERSE: u64 = 1 << 24;
    const KEY_SPACE: u32 = 1 << 20;
    const CANDIDATE_COUNT: usize = 1 << 22;
    const EXPECTED_ACCEPTED: usize = (KEY_SPACE as usize) / 2;
    const WARMUPS: usize = 5;
    const REPETITIONS: usize = 20;

    let candidates: Vec<u32> = (0..CANDIDATE_COUNT)
        .map(|index| (index as u32).wrapping_mul(2_654_435_761) & (KEY_SPACE - 1))
        .collect();
    let seed: Vec<u32> = (0..KEY_SPACE).step_by(2).collect();
    let mut bitmap = BitmapVisited::new(UNIVERSE, CANDIDATE_COUNT, CANDIDATE_COUNT)?;

    bitmap.seed(&seed)?;
    bitmap.upload(&candidates)?;
    let validation = bitmap.run_in_place(CANDIDATE_COUNT)?;
    if validation.accepted_count != EXPECTED_ACCEPTED || validation.overflow {
        return Err(format!(
            "benchmark validation count mismatch: got {}, expected {}, overflow={}",
            validation.accepted_count, EXPECTED_ACCEPTED, validation.overflow
        ));
    }
    let mut accepted = bitmap.output(validation.output_written).to_vec();
    accepted.sort_unstable();
    let expected: Vec<u32> = (1..KEY_SPACE).step_by(2).collect();
    if accepted != expected {
        return Err("benchmark full accepted-set validation failed".into());
    }

    for _ in 0..WARMUPS {
        bitmap.seed(&seed)?;
        bitmap.upload(&candidates)?;
        let result = bitmap.run_in_place(CANDIDATE_COUNT)?;
        if result.accepted_count != EXPECTED_ACCEPTED || result.overflow {
            return Err("warmup correctness drift".into());
        }
    }

    let mut kernel_ms = Vec::with_capacity(REPETITIONS);
    let mut iteration_ms = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        bitmap.seed(&seed)?;
        bitmap.upload(&candidates)?;
        let result = bitmap.run_in_place(CANDIDATE_COUNT)?;
        iteration_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        if result.accepted_count != EXPECTED_ACCEPTED || result.overflow {
            return Err("measured repetition correctness drift".into());
        }
        kernel_ms.push(result.kernel_milliseconds as f64);
    }
    let kernel_min = kernel_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let kernel_max = kernel_ms.iter().copied().fold(0.0, f64::max);
    let kernel_median = median(&mut kernel_ms);
    let iteration_median = median(&mut iteration_ms);
    let billion_candidates_per_second = CANDIDATE_COUNT as f64 / (kernel_median / 1_000.0) / 1e9;
    println!(
        "{{\"status\":\"pass\",\"benchmark\":\"bitmap-visited-v1\",\"universe\":{},\"candidate_count\":{},\"key_space\":{},\"seed_count\":{},\"accepted_count\":{},\"warmups\":{},\"repetitions\":{},\"kernel_ms_min\":{:.6},\"kernel_ms_median\":{:.6},\"kernel_ms_max\":{:.6},\"iteration_ms_median\":{:.6},\"kernel_billion_candidates_per_s\":{:.6}}}",
        UNIVERSE,
        CANDIDATE_COUNT,
        KEY_SPACE,
        seed.len(),
        EXPECTED_ACCEPTED,
        WARMUPS,
        REPETITIONS,
        kernel_min,
        kernel_median,
        kernel_max,
        iteration_median,
        billion_candidates_per_second
    );
    Ok(())
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn fingerprint(values: impl Iterator<Item = u32>) -> (u64, u64) {
    values.fold((0_u64, 0_u64), |(sum, xor), value| {
        let mixed = mix64(value as u64);
        (sum.wrapping_add(mixed), xor ^ mixed)
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    sorted[index]
}

pub fn sweep() -> Result<(), String> {
    sweep_impl(&[BitmapVariant::Baseline], false)
}

pub fn variant_sweep() -> Result<(), String> {
    if let Ok(path) = std::env::var("MGBFS_OUTPUT_PATH") {
        std::fs::File::create(path).map_err(|error| error.to_string())?;
    }
    sweep_impl(&BitmapVariant::ALL, true)
}

fn emit_variant_line(line: String) -> Result<(), String> {
    use std::io::Write;

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

fn sweep_impl(variants: &[BitmapVariant], include_variant: bool) -> Result<(), String> {
    use std::time::Instant;

    const UNIVERSE: u64 = 1 << 24;
    const WARMUPS: usize = 3;
    const REPETITIONS: usize = 10;
    let sizes = [1_usize << 16, 1 << 20, 1 << 22, 1 << 24];
    let patterns = ["all-new", "half-seeded-fourfold", "all-seen", "single-key"];

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
            let expected_count = match pattern {
                "all-new" => candidate_count,
                "half-seeded-fourfold" => key_space / 2,
                "all-seen" => 0,
                "single-key" => 1,
                _ => unreachable!(),
            };
            let expected_fingerprint = match pattern {
                "all-new" => fingerprint((0..candidate_count).map(|value| value as u32)),
                "half-seeded-fourfold" => {
                    fingerprint((0..expected_count).map(|value| value as u32 * 2 + 1))
                }
                "all-seen" => (0, 0),
                "single-key" => fingerprint(std::iter::once(7)),
                _ => unreachable!(),
            };

            for &variant in variants {
                let mut bitmap = BitmapVisited::new(UNIVERSE, candidate_count, candidate_count)?;
                for _ in 0..WARMUPS {
                    bitmap.seed(&seed)?;
                    bitmap.upload(&candidates)?;
                    let result = bitmap.run_in_place_variant(variant, candidate_count)?;
                    if result.accepted_count != expected_count || result.overflow {
                        return Err(format!("{pattern}/{candidate_count} warmup mismatch"));
                    }
                }

                let mut kernel_ms = Vec::with_capacity(REPETITIONS);
                let mut iteration_ms = Vec::with_capacity(REPETITIONS);
                let mut last_written = 0;
                for _ in 0..REPETITIONS {
                    let started = Instant::now();
                    bitmap.seed(&seed)?;
                    bitmap.upload(&candidates)?;
                    let result = bitmap.run_in_place_variant(variant, candidate_count)?;
                    iteration_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                    if result.accepted_count != expected_count || result.overflow {
                        return Err(format!("{pattern}/{candidate_count} measured mismatch"));
                    }
                    last_written = result.output_written;
                    kernel_ms.push(result.kernel_milliseconds as f64);
                }
                let actual_fingerprint = fingerprint(bitmap.output(last_written).iter().copied());
                if actual_fingerprint != expected_fingerprint || last_written != expected_count {
                    return Err(format!(
                        "{pattern}/{candidate_count} fingerprint mismatch: {:?} != {:?}",
                        actual_fingerprint, expected_fingerprint
                    ));
                }
                kernel_ms.sort_by(|left, right| left.total_cmp(right));
                iteration_ms.sort_by(|left, right| left.total_cmp(right));
                let kernel_median = median(&mut kernel_ms.clone());
                let iteration_median = median(&mut iteration_ms.clone());
                let kernel_throughput = candidate_count as f64 / (kernel_median / 1_000.0) / 1e9;
                let iteration_throughput =
                    candidate_count as f64 / (iteration_median / 1_000.0) / 1e9;
                if include_variant {
                    emit_variant_line(format!(
                "{{\"status\":\"pass\",\"benchmark\":\"bitmap-variant-sweep-v1\",\"variant\":\"{}\",\"pattern\":\"{}\",\"universe\":{},\"candidate_count\":{},\"key_space\":{},\"seed_count\":{},\"accepted_count\":{},\"accept_fraction\":{:.9},\"warmups\":{},\"repetitions\":{},\"kernel_ms_min\":{:.6},\"kernel_ms_median\":{:.6},\"kernel_ms_p95\":{:.6},\"kernel_ms_max\":{:.6},\"iteration_ms_median\":{:.6},\"kernel_billion_candidates_per_s\":{:.6},\"iteration_billion_candidates_per_s\":{:.6},\"validation_sum\":{},\"validation_xor\":{}}}",
                variant.name(), pattern, UNIVERSE, candidate_count, key_space,
                seed.len(), expected_count,
                expected_count as f64 / candidate_count as f64,
                WARMUPS, REPETITIONS, kernel_ms[0], kernel_median,
                percentile(&kernel_ms, 0.95), kernel_ms[kernel_ms.len() - 1],
                iteration_median, kernel_throughput, iteration_throughput,
                actual_fingerprint.0, actual_fingerprint.1
                ))?;
                } else {
                    println!(
                "{{\"status\":\"pass\",\"benchmark\":\"bitmap-sweep-v1\",\"pattern\":\"{}\",\"universe\":{},\"candidate_count\":{},\"key_space\":{},\"seed_count\":{},\"accepted_count\":{},\"accept_fraction\":{:.9},\"warmups\":{},\"repetitions\":{},\"kernel_ms_min\":{:.6},\"kernel_ms_median\":{:.6},\"kernel_ms_p95\":{:.6},\"kernel_ms_max\":{:.6},\"iteration_ms_median\":{:.6},\"kernel_billion_candidates_per_s\":{:.6},\"iteration_billion_candidates_per_s\":{:.6},\"validation_sum\":{},\"validation_xor\":{}}}",
                pattern,
                UNIVERSE,
                candidate_count,
                key_space,
                seed.len(),
                expected_count,
                expected_count as f64 / candidate_count as f64,
                WARMUPS,
                REPETITIONS,
                kernel_ms[0],
                kernel_median,
                percentile(&kernel_ms, 0.95),
                kernel_ms[kernel_ms.len() - 1],
                iteration_median,
                kernel_throughput,
                iteration_throughput,
                actual_fingerprint.0,
                actual_fingerprint.1
            );
                }
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

pub fn validate_variant_artifact() -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};

    let path = std::env::var("MGBFS_INPUT_PATH")
        .unwrap_or_else(|_| "/input/REF-014-bitmap-variant-sweep.jsonl".into());
    let contents = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let expected_variants = ["baseline", "warp-aggregate", "block-compact", "warp-block"];
    let expected_patterns = ["all-new", "half-seeded-fourfold", "all-seen", "single-key"];
    let expected_sizes = [1_usize << 16, 1 << 20, 1 << 22, 1 << 24];
    let mut combinations = BTreeSet::new();
    let mut outcomes = BTreeMap::new();
    let mut rows = 0_usize;

    for (line_index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        if json_field(line, "status")? != "pass"
            || json_field(line, "benchmark")? != "bitmap-variant-sweep-v1"
        {
            return Err(format!(
                "line {} is not a passing REF-014 row",
                line_index + 1
            ));
        }
        let variant = json_field(line, "variant")?.to_owned();
        let pattern = json_field(line, "pattern")?.to_owned();
        let size = json_field(line, "candidate_count")?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if !expected_variants.contains(&variant.as_str())
            || !expected_patterns.contains(&pattern.as_str())
            || !expected_sizes.contains(&size)
        {
            return Err(format!(
                "line {} has an unexpected dimension",
                line_index + 1
            ));
        }
        if !combinations.insert((variant.clone(), pattern.clone(), size)) {
            return Err(format!("duplicate row for {variant}/{pattern}/{size}"));
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
        match outcomes.entry((pattern.clone(), size)) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(outcome);
            }
            std::collections::btree_map::Entry::Occupied(entry) if *entry.get() != outcome => {
                return Err(format!("variant outcome mismatch for {pattern}/{size}"));
            }
            _ => {}
        }
    }
    let expected_rows = expected_variants.len() * expected_patterns.len() * expected_sizes.len();
    if rows != expected_rows || combinations.len() != expected_rows {
        return Err(format!(
            "expected {expected_rows} unique rows, found {rows}"
        ));
    }
    println!(
        "{{\"status\":\"pass\",\"validator\":\"rust-ref014-artifact-v1\",\"rows\":{},\"outcome_groups\":{}}}",
        rows,
        outcomes.len()
    );
    Ok(())
}
