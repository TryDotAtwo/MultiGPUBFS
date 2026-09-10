use std::{fs::File, io::BufReader, path::PathBuf};

fn execute() -> Result<(), (i32, String)> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.first().and_then(|x| x.to_str()) {
        Some("--help") | Some("-h") if args.len() == 1 => {
            println!("mgbfs verify <archive>\nmgbfs preflight --offline <config.json>\nmgbfs bench --reference <sN> <batch> <bootstrap> <archive-prefix> <output-dir>\nReference bench requires a Linux CUDA build and torchrun topology; archive is mandatory.\nOffline preflight validates only the configuration, not device memory or hardware readiness.\nProduction run/preflight/calibrate commands are not connected yet.");
        }
        Some("bench") if args.len() == 7 && args[1] == "--reference" => {
            match std::env::var("MGBFS_BENCH_SKIP_ARCHIVE") {
                Err(std::env::VarError::NotPresent) => (),
                Ok(value) if value == "0" => (),
                _ => return Err((2, "CLI_BENCH_ARCHIVE_REQUIRED".into())),
            }
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            {
                let launch = std::iter::once("mgbfs-bench".to_string())
                    .map(Ok)
                    .chain(args[2..].iter().map(|s| s.clone().into_string()
                        .map_err(|_| (2, "CLI_BENCH_ARGUMENT_ENCODING".into()))))
                    .collect::<Result<Vec<_>, _>>()?;
                mgbfs_runtime::reference_bench::run(launch).map_err(|e| (1, e))?;
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            return Err((2, "CLI_BENCH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("bench") => return Err((2,
            "CLI_USAGE: mgbfs bench --reference <sN> <batch> <bootstrap> <archive-prefix> <output-dir>".into()
        )),
        Some("preflight") if args.len() == 3 && args[1] == "--offline" => {
            let file = File::open(PathBuf::from(&args[2]))
                .map_err(|e| (1, format!("CONFIG_OPEN: {e}")))?;
            let config: mgbfs_core::config::RunConfigV1 =
                serde_json::from_reader(BufReader::new(file))
                    .map_err(|e| (1, format!("CONFIG_PARSE: {e}")))?;
            let digest = config.digest().map_err(|e| (1, e))?;
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            println!(
                "{}",
                serde_json::json!({
                    "status": "CONFIG_VALIDATED", "scope": "offline_config_only",
                    "hardware_ready": false, "config_digest": hex,
                    "unchecked": ["device_and_build_queries", "free_vram", "pinned_ram", "disk_extents", "runtime_backend_support"]
                })
            );
        }
        Some("preflight") => return Err((
            2,
            "CLI_USAGE: mgbfs preflight --offline <config.json>; hardware preflight unavailable"
                .into(),
        )),
        Some("verify") if args.len() == 2 => {
            let path = PathBuf::from(&args[1]);
            let file = File::open(&path).map_err(|e| (1, format!("ARCHIVE_OPEN: {e}")))?;
            mgbfs_runtime::archive::verify_reader(&mut BufReader::with_capacity(65536, file))
                .map_err(|e| (1, e))?;
            println!(
                "{}",
                serde_json::json!({"status": "VERIFIED", "archive": path,
                "scope": "committed_archive_checksums_and_counts"})
            );
        }
        Some("verify") | None => return Err((2, "CLI_USAGE: mgbfs verify <archive>".into())),
        _ => return Err((2, "CLI_COMMAND_UNAVAILABLE".into())),
    }
    Ok(())
}

fn main() {
    if let Err((code, error)) = execute() {
        eprintln!("{}", serde_json::json!({"status": "ERROR", "error": error}));
        std::process::exit(code);
    }
}
