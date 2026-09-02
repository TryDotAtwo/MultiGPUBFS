use std::env;

fn main() {
    let library_dir = env::var("MULTIGPUBFS_CUDA_LIB_DIR")
        .expect("MULTIGPUBFS_CUDA_LIB_DIR must point to the CUDA library");
    println!("cargo:rustc-link-search=native={library_dir}");
    println!("cargo:rustc-link-lib=dylib=multigpubfs_cuda");
    println!("cargo:rerun-if-env-changed=MULTIGPUBFS_CUDA_LIB_DIR");
    println!("cargo:rerun-if-changed=gpu/device_smoke.cu");
    println!("cargo:rerun-if-changed=gpu/bitmap_visited.cu");
    println!("cargo:rerun-if-changed=gpu/sort_unique_visited.cu");
    println!("cargo:rerun-if-changed=gpu/cayley_bfs.cu");
    println!("cargo:rerun-if-changed=gpu/multigpubfs_cuda.h");
}
