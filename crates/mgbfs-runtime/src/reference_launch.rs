use mgbfs_core::Result;

/// Select the existing single-rank macro path, or reject unsupported
/// multi-rank macro settings while all ranks still participate in admission.
pub fn macro_depth_for_launch(value: Option<&str>, world: u32) -> Result<bool> {
    let depth = value.unwrap_or("1").parse::<u32>()
        .map_err(|_| "ENV_MGBFS_MACRO_DEPTH")?;
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
