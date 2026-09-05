use std::{fs::File, io::BufReader, path::PathBuf};

fn execute() -> Result<(), (i32, String)> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.first().and_then(|x| x.to_str()) {
        Some("--help") | Some("-h") if args.len() == 1 => {
            println!("mgbfs verify <archive>\nVerify the committed archive prefix with bounded memory.\nGPU run/preflight/calibrate/bench commands are not connected yet.");
        }
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
