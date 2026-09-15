fn main() {
    println!("cargo:rerun-if-env-changed=MGBFS_CUDA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MGBFS_LIBRARY_OWNER_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_LIBRARY_OWNER").is_some() {
        let dir = std::env::var("MGBFS_LIBRARY_OWNER_LIB_DIR")
            .expect("build experiments/library_owner and set MGBFS_LIBRARY_OWNER_LIB_DIR");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=mgbfs_library_owner");
    }
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let dir =
            std::env::var("MGBFS_CUDA_LIB_DIR").expect("build cuda/ and set MGBFS_CUDA_LIB_DIR");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=mgbfs_cuda");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
