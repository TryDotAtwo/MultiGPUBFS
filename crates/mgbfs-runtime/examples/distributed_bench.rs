#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = mgbfs_runtime::reference_bench::run(std::env::args().collect()) {
        eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: {e}");
        std::process::exit(1)
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: REQUIRES_LINUX_CUDA");
    std::process::exit(1)
}
