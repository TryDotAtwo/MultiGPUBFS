//! Startup-only loading admission. No healthy-batch synchronization/readback.
use std::{
    ffi::OsStr,
    sync::atomic::{AtomicI32, Ordering},
};
static VERIFIED_DRIVER_MODE: AtomicI32 = AtomicI32::new(0);

fn validate_value(name: &str, value: Option<&OsStr>) -> Result<(), String> {
    match value {
        None => Ok(()),
        Some(value) if value == OsStr::new("EAGER") => Ok(()),
        Some(value) => Err(format!(
            "CUDA_LOADING_POLICY: {name}={value:?}; EAGER required"
        )),
    }
}

pub fn validate_requested() -> Result<(), String> {
    for name in ["CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING"] {
        validate_value(name, std::env::var_os(name).as_deref())?;
    }
    Ok(())
}

/// CLI startup only, before worker threads/CUDA initialization. Preserve explicit
/// settings: conflicting requests are errors, never silently changed/fallbacks.
pub fn configure_cli_before_cuda() -> Result<bool, String> {
    let mut needs_exec = false;
    for name in ["CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING"] {
        if std::env::var_os(name).is_none() {
            std::env::set_var(name, "EAGER");
            needs_exec = true;
        }
    }
    validate_requested()?;
    Ok(needs_exec)
}

/// After cudaSetDevice, before persistent allocations/communicator creation.
/// A direct library caller must select EAGER before initializing CUDA. Runtime
/// admission rejects a driver already initialized LAZY even if env says EAGER.
pub fn verify_driver_before_allocations() -> Result<(), String> {
    validate_requested()?;
    unsafe {
        let handle = libc::dlopen(
            b"libcuda.so.1\0".as_ptr().cast(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if handle.is_null() {
            return Err("CUDA_LOADING_QUERY: libcuda unavailable".into());
        }
        let symbol = libc::dlsym(handle, b"cuModuleGetLoadingMode\0".as_ptr().cast());
        if symbol.is_null() {
            libc::dlclose(handle);
            return Err("CUDA_LOADING_QUERY: cuModuleGetLoadingMode unavailable".into());
        }
        let query: unsafe extern "C" fn(*mut i32) -> i32 = std::mem::transmute(symbol);
        let mut mode = 0;
        let status = query(&mut mode);
        libc::dlclose(handle);
        if status != 0 {
            return Err(format!("CUDA_LOADING_QUERY: driver status {status}"));
        }
        if mode != 1 {
            return Err(format!(
                "CUDA_LOADING_POLICY: actual driver mode {mode}; initialize EAGER before CUDA"
            ));
        }
        VERIFIED_DRIVER_MODE.store(mode, Ordering::Release);
        Ok(())
    }
}

pub fn evidence() -> serde_json::Value {
    let mode = VERIFIED_DRIVER_MODE.load(Ordering::Acquire);
    serde_json::json!({
        "requested_module_loading": std::env::var("CUDA_MODULE_LOADING").ok(),
        "requested_data_loading": std::env::var("CUDA_MODULE_DATA_LOADING").ok(),
        "verified_driver_mode": mode,
        "actual_driver_mode_verified": mode == 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lazy_unknown_and_empty_requests_are_not_silently_normalized() {
        for value in ["LAZY", "lazy", "", "AUTO"] {
            assert!(validate_value("CUDA_MODULE_LOADING", Some(OsStr::new(value))).is_err());
        }
        assert!(validate_value("CUDA_MODULE_LOADING", None).is_ok());
        assert!(validate_value("CUDA_MODULE_LOADING", Some(OsStr::new("EAGER"))).is_ok());
        assert!(validate_value("CUDA_MODULE_DATA_LOADING", Some(OsStr::new("LAZY"))).is_err());
    }
}
