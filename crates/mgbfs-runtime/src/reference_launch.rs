use mgbfs_core::Result;

/// Optional calibration stop at an existing globally completed layer boundary.
pub fn calibration_layers(value: Option<&str>) -> Result<Option<u32>> {
    match value {
        None => Ok(None),
        Some(value) => {
            let layers = value.parse::<u32>().map_err(|_| "ENV_MGBFS_CALIBRATION_LAYERS")?;
            if layers == 0 { return Err("ENV_MGBFS_CALIBRATION_LAYERS".into()); }
            Ok(Some(layers))
        }
    }
}

/// Startup-only reserve; native admission still checks warmed free memory.
pub fn vram_reserve_bytes(value: Option<&str>) -> Result<u64> {
    let bytes = value.unwrap_or("1073741824").parse::<u64>()
        .map_err(|_| "ENV_MGBFS_VRAM_RESERVE_BYTES")?;
    if bytes < 64 << 20 {
        return Err("ENV_MGBFS_VRAM_RESERVE_BYTES".into());
    }
    Ok(bytes)
}

/// Explicit full-window capture policy. Legacy submissions remain the default
/// until the multi-rank graph/archive hardware acceptance gates pass.
pub fn graph_batches(value: Option<&str>) -> Result<u32> {
    match value.unwrap_or("0") {
        "0" => Ok(0), "32" => Ok(32),
        _ => Err("ENV_MGBFS_CUDA_GRAPH_BATCHES".into()),
    }
}

/// Enough archive credits for in-flight graph windows plus the next window's
/// copies. Depths shorter than 32 batches need only their actual window width.
pub fn graph_archive_credits(rounds: u32, batch: u32, archive_rows: u32,
                             inflight: usize) -> Result<usize> {
    if rounds == 0 || batch == 0 || archive_rows == 0 || inflight == 0 {
        return Err("GRAPH_ARCHIVE_CREDIT_SHAPE".into());
    }
    let width = u64::from(archive_rows.min(batch));
    let frames = u64::from(batch)/width + u64::from(u64::from(batch)%width != 0);
    let windows = u64::try_from(inflight).map_err(|_| "GRAPH_ARCHIVE_CREDIT_OVERFLOW")?
        .checked_add(1).ok_or("GRAPH_ARCHIVE_CREDIT_OVERFLOW")?;
    let credits = u64::from(rounds.min(32)).checked_mul(frames)
        .and_then(|x| x.checked_mul(windows)).ok_or("GRAPH_ARCHIVE_CREDIT_OVERFLOW")?;
    usize::try_from(credits).map_err(|_| "GRAPH_ARCHIVE_CREDIT_OVERFLOW".into())
}

/// Setup-only bound for device-count epochs. Does not enlarge data buffers.
pub fn inflight_batches(value: Option<&str>) -> Result<usize> {
    let count=value.unwrap_or("2").parse::<usize>()
        .map_err(|_| "ENV_MGBFS_INFLIGHT_BATCHES")?;
    if !(1..=32).contains(&count) {
        return Err("ENV_MGBFS_INFLIGHT_BATCHES".into());
    }
    Ok(count)
}

/// Number of completion credits, not a count of payload receive buffers.
pub fn epoch_window_for_launch(value: Option<&str>) -> Result<usize> {
    let count = value
        .unwrap_or("2")
        .parse::<u32>()
        .map_err(|_| "ENV_MGBFS_EPOCH_WINDOW")?;
    if count < 2 {
        return Err("ENV_MGBFS_EPOCH_WINDOW".into());
    }
    usize::try_from(count).map_err(|_| "ENV_MGBFS_EPOCH_WINDOW".into())
}

/// Decode and validate before device admission. Paths are not graph identity.
pub fn load_matrix_manifest(
    path: &std::path::Path,
) -> Result<(String, mgbfs_core::matrix::MatrixGroup)> {
    use sha2::{Digest, Sha256};
    let file = std::fs::File::open(path).map_err(|e| format!("MATRIX_MANIFEST_OPEN: {e}"))?;
    let graph: mgbfs_core::matrix::MatrixGroup =
        serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| format!("MATRIX_MANIFEST_PARSE: {e}"))?;
    graph.validate()?;
    let digest = Sha256::digest(serde_json::to_vec(&graph).map_err(|e| e.to_string())?);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Ok((format!("matrix-{hex}"), graph))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchPhase {
    Warmup,
    Measure,
}

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
    let depth = value
        .unwrap_or("1")
        .parse::<u32>()
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
