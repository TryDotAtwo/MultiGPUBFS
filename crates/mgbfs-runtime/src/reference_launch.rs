use mgbfs_core::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchPhase { Warmup, Measure }

pub fn bench_warmup_for_launch(value: Option<&str>, stream: Option<&str>) -> Result<bool> {
    let warmup = match value {
        Some("1") => true,
        Some("0") | None => false,
        _ => return Err("BENCH_WARMUP_CONFIG".into()),
    };
    if warmup && stream == Some("1") {
        return Err("BENCH_WARMUP_REQUIRES_FILE_ARCHIVE".into());
    }
    Ok(warmup)
}

pub fn bench_archive_for_launch(skip: Option<&str>, search_only: bool) -> Result<bool> {
    match skip {
        None | Some("0") => Ok(true),
        Some("1") if search_only => Ok(false),
        _ => Err("CLI_BENCH_ARCHIVE_REQUIRED".into()),
    }
}

/// Keep the first rendezvous identical across ranks even if their warmup
/// settings disagree; the configuration vote then rejects that disagreement.
pub fn bench_phase_paths(args: &[&str], warmup: bool, phase: BenchPhase) -> Result<Vec<String>> {
    if args.len() != 6 || (phase == BenchPhase::Warmup && !warmup) {
        return Err("BENCH_PHASE_ARGS".into());
    }
    let mut paths: Vec<String> = args.iter().map(|x| (*x).to_owned()).collect();
    match phase {
        BenchPhase::Warmup => {
            paths[4].push_str(".warmup");
            paths[5].push_str(".warmup");
        }
        BenchPhase::Measure if warmup => paths[3].push_str(".measure"),
        BenchPhase::Measure => (),
    }
    Ok(paths)
}

/// Select the existing single-rank macro path, or reject unsupported
/// multi-rank macro settings while all ranks still participate in admission.
pub fn macro_depth_for_launch(value: Option<&str>, world: u32) -> Result<bool> {
    let depth = value.unwrap_or("1").parse::<u32>()
        .map_err(|_| "ENV_MGBFS_MACRO_DEPTH")?;
    if depth == 0 {
        return Err("ENV_MGBFS_MACRO_DEPTH".into());
    }
    if world > 1 && depth > 1 {
        return Err("MACRO_MULTI_GPU_UNSUPPORTED".into());
    }
    Ok(depth > 1)
}

pub fn macro_depth_from_env(world: u32) -> Result<bool> {
    let value = match std::env::var("MGBFS_MACRO_DEPTH") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("ENV_MGBFS_MACRO_DEPTH".into()),
    };
    macro_depth_for_launch(value.as_deref(), world)
}
