use std::{fs::File, io::BufReader, path::PathBuf};

fn execute() -> Result<(), (i32, String)> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if matches!(
        args.first().and_then(|x| x.to_str()),
        Some("run") | Some("bench") | Some("session") | Some("graph") | Some("graph-info") | Some("graph-plan") | Some("graph-local-plan") | Some("graph-rank") | Some("graph-session") | Some("graph-count")
    ) {
        match mgbfs_runtime::cuda_loading::configure_cli_before_cuda() {
            Ok(true) => {
                // CUDA-linked constructors may run before main. Replace this
                // process once so EAGER is inherited before those constructors.
                // exec preserves the launch PID/rank; no initialized BFS exists.
                use std::os::unix::process::CommandExt;
                let executable =
                    std::env::current_exe().map_err(|e| (2, format!("CUDA_LOADING_EXEC: {e}")))?;
                let error = std::process::Command::new(executable).args(&args).exec();
                return Err((2, format!("CUDA_LOADING_EXEC: {error}")));
            }
            Ok(false) => (),
            Err(error) => {
                // Ranks report rejected settings through configuration agreement.
                if std::env::var_os("RANK").is_none() {
                    return Err((2, error));
                }
            }
        }
    }

    match args.first().and_then(|x| x.to_str()) {
        Some("graph-count") if args.len()==1 => {
            #[cfg(all(feature="cuda",target_os="linux"))]
            mgbfs_runtime::generic_distributed_run::visible_count().map_err(|e|(1,e))?;
            #[cfg(not(all(feature="cuda",target_os="linux")))]
            return Err((2,"CLI_GRAPH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("graph-session") if args.len()==4 => {
            #[cfg(all(feature="cuda",target_os="linux"))]
            {let rank=args[2].to_str().ok_or((2,"SESSION_RANK_ENCODING".into()))?.parse::<u32>().map_err(|_|(2,"SESSION_RANK".into()))?;
             let device=args[3].to_str().ok_or((2,"SESSION_DEVICE_ENCODING".into()))?.parse::<u32>().map_err(|_|(2,"SESSION_DEVICE".into()))?;
             mgbfs_runtime::generic_session::run(&PathBuf::from(&args[1]),rank,device).map_err(|e|(1,e))?;}
            #[cfg(not(all(feature="cuda",target_os="linux")))]
            return Err((2,"CLI_GRAPH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("key-info") if args.len()==1 => {println!("{}",serde_json::json!({"schema":1,"lossless_bitpack128_feistel_v1":cfg!(all(feature="cuda",target_os="linux")),"scope":"compiled implementation; graph domain and GPU correctness checked separately"}));}

        Some(command @ ("graph-info" | "graph-plan" | "graph-local-plan" | "graph-rank")) => {
            #[cfg(all(feature="cuda",target_os="linux"))]
            {let paths=args[1..].iter().map(|v|v.clone().into_string().map_err(|_|(2,"CLI_GRAPH_ARGUMENT_ENCODING".into()))).collect::<Result<Vec<_>,_>>()?;
             if command=="graph-info"{mgbfs_runtime::generic_distributed_run::info(&paths)}else if command=="graph-local-plan"{mgbfs_runtime::generic_distributed_run::local_info(&paths)}else if command=="graph-plan"{mgbfs_runtime::generic_distributed_run::global_info(&paths)}else{mgbfs_runtime::generic_distributed_run::run(&paths)}.map_err(|e|(1,e))?;}
            #[cfg(not(all(feature="cuda",target_os="linux")))]
            return Err((2,"CLI_GRAPH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("graph") => {
            #[cfg(all(feature="cuda",target_os="linux"))]
            {let paths=args[1..].iter().map(|v|v.clone().into_string().map_err(|_|(2,"CLI_GRAPH_ARGUMENT_ENCODING".into()))).collect::<Result<Vec<_>,_>>()?;
             mgbfs_runtime::generic_run::run(&paths).map_err(|e|(1,e))?;}
            #[cfg(not(all(feature="cuda",target_os="linux")))]
            return Err((2,"CLI_GRAPH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("session") if args.len() == 2 => {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            mgbfs_runtime::session_worker::run(&PathBuf::from(&args[1])).map_err(|e| (1, e))?;
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            return Err((2, "CLI_SESSION_REQUIRES_LINUX_CUDA".into()));
        }
        Some("run") if args.len() == 5 => {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            {
                let paths = args[1..].iter().map(|x| x.clone().into_string()
                    .map_err(|_| (2, "CLI_RUN_ARGUMENT_ENCODING".into())))
                    .collect::<Result<Vec<_>, _>>()?;
                mgbfs_runtime::reference_bench::run_config(paths[0].clone(), paths[1].clone(),
                    paths[2].clone(), paths[3].clone()).map_err(|e| (1, e))?;
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            return Err((2, "CLI_RUN_REQUIRES_LINUX_CUDA".into()));
        }
        Some("run") => return Err((2,
            "CLI_USAGE: mgbfs run <config.json> <bootstrap> <archive-prefix> <output-dir>".into())),
        Some("--help") | Some("-h") if args.len() == 1 => {
            println!("mgbfs graph <graph.json> <output-dir> [--device N] [--capacity N] [--seconds N]\nGeneral exact GPU path currently supports one device; directed graphs retain all visited states.\n");
            println!("mgbfs bench --manifest <matrix.json> <batch> <bootstrap> <archive-prefix> <output-dir> [--search-only]\nManifest input uses the same benchmark runtime; it is not the production RunConfigV1 dispatcher.");
            println!("mgbfs run <config.json> <bootstrap> <archive-prefix> <output-dir>\nRun currently supports unit-depth matrix states, two producer banks, exact candidate slot capacity and aligned pinned slots; other contracts fail explicitly.\nmgbfs verify <archive>\nmgbfs preflight --offline <config.json>\nmgbfs bench --reference <sN|uNmM> <batch> <bootstrap> <archive-prefix> <output-dir> [--search-only]\nReference bench requires a Linux CUDA build and torchrun topology; archive is enabled unless --search-only is explicit.\nOffline preflight validates only the configuration, not device memory or hardware readiness.\nHardware preflight/calibrate are not connected yet.");
        }
        Some("bench") if (args.len() == 7 || (args.len() == 8 && args[7] == "--search-only"))
            && (args[1] == "--reference" || args[1] == "--manifest") => {
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            match std::env::var("MGBFS_MACRO_DEPTH").as_deref() {
                Ok("1") | Err(std::env::VarError::NotPresent) => (),
                Ok(value) if value.parse::<u32>().is_ok_and(|depth| depth > 1)
                    && std::env::var("WORLD_SIZE").as_deref() == Ok("1") => (),
                _ => return Err((2, "CLI_BENCH_MACRO_DEPTH_UNAVAILABLE".into())),
            }
            if args.len() == 8 {
                // Explicit alternative output contract, never an implicit fallback.
                std::env::set_var("MGBFS_BENCH_SKIP_ARCHIVE", "1");
                std::env::set_var("MGBFS_SEARCH_ONLY", "1");
                std::env::set_var("MGBFS_ARCHIVE_STREAM", "0");
            } else {
              std::env::remove_var("MGBFS_SEARCH_ONLY");
              #[cfg(not(all(feature = "cuda", target_os = "linux")))]
              match std::env::var("MGBFS_BENCH_SKIP_ARCHIVE") {
                Err(std::env::VarError::NotPresent) => (),
                Ok(value) if value == "0" => (),
                _ => return Err((2, "CLI_BENCH_ARCHIVE_REQUIRED".into())),
              }
            }
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            {
                let launch = std::iter::once("mgbfs-bench".to_string())
                    .map(Ok)
                    .chain(args[2..7].iter().map(|s| s.clone().into_string()
                        .map_err(|_| (2, "CLI_BENCH_ARGUMENT_ENCODING".into()))))
                    .collect::<Result<Vec<_>, _>>()?;
                if args[1] == "--manifest" {
                    mgbfs_runtime::reference_bench::run_manifest(launch).map_err(|e| (1, e))?;
                } else {
                    mgbfs_runtime::reference_bench::run(launch).map_err(|e| (1, e))?;
                }
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            return Err((2, "CLI_BENCH_REQUIRES_LINUX_CUDA".into()));
        }
        Some("bench") => return Err((2,
            "CLI_USAGE: mgbfs bench --reference <sN|uNmM> <batch> <bootstrap> <archive-prefix> <output-dir>".into()
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
        // Formatting Value directly into stderr issues several small writes.
        // torchrun ranks share the pipe, so their JSON tokens can interleave.
        // Serialize first; short terminal records fit one atomic pipe write.
        // Leading newline also isolates JSON from a peer's partial trace line.
        let line=format!("\n{}\n",serde_json::json!({"status":"ERROR","error":error}));
        let _=std::io::Write::write_all(&mut std::io::stderr(),line.as_bytes());
        std::process::exit(code);
    }
}
