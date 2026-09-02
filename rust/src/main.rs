use std::ffi::{c_char, CStr};

mod bitmap;
mod cayley;
mod sort_unique;

#[link(name = "multigpubfs_cuda")]
extern "C" {
    fn mgbfs_cuda_device_info(
        name: *mut c_char,
        name_capacity: usize,
        major: *mut i32,
        minor: *mut i32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn mgbfs_cuda_affine(
        host_input: *const u32,
        host_output: *mut u32,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

fn ffi_error(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn affine_smoke() {
    const COUNT: usize = 1 << 20;
    let mut error = vec![0 as c_char; 512];
    let mut name = vec![0 as c_char; 256];
    let mut major = 0;
    let mut minor = 0;
    let info_status = unsafe {
        mgbfs_cuda_device_info(
            name.as_mut_ptr(),
            name.len(),
            &mut major,
            &mut minor,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if info_status != 0 {
        eprintln!("cuda device query failed: {}", ffi_error(&error));
        std::process::exit(info_status);
    }

    let input: Vec<u32> = (0..COUNT as u32).collect();
    let mut output = vec![0_u32; COUNT];
    let status = unsafe {
        mgbfs_cuda_affine(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        eprintln!("cuda affine smoke failed: {}", ffi_error(&error));
        std::process::exit(status);
    }
    for (index, &actual) in output.iter().enumerate() {
        let expected = (index as u32).wrapping_mul(3).wrapping_add(1);
        if actual != expected {
            eprintln!("validation mismatch at {index}: got {actual}, expected {expected}");
            std::process::exit(3);
        }
    }
    let gpu_name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
    println!(
        "{{\"status\":\"pass\",\"host\":\"rust\",\"gpu_code\":\"cuda_c_abi\",\"gpu\":\"{}\",\"compute_capability\":\"{}.{}\",\"elements\":{}}}",
        gpu_name, major, minor, COUNT
    );
}

fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "smoke".into());
    match command.as_str() {
        "smoke" => affine_smoke(),
        "bitmap-self-test" => match bitmap::self_test() {
            Ok(()) => println!(
                "{{\"status\":\"pass\",\"test\":\"bitmap-self-test\",\"host\":\"rust\",\"backend\":\"cuda-atomic-bitmap\"}}"
            ),
            Err(error) => {
                eprintln!("bitmap self-test failed: {error}");
                std::process::exit(4);
            }
        },
        "bitmap-benchmark" => {
            if let Err(error) = bitmap::benchmark() {
                eprintln!("bitmap benchmark failed: {error}");
                std::process::exit(5);
            }
        }
        "bitmap-sweep" => {
            if let Err(error) = bitmap::sweep() {
                eprintln!("bitmap sweep failed: {error}");
                std::process::exit(6);
            }
        }
        "bitmap-variant-sweep" => {
            if let Err(error) = bitmap::variant_sweep() {
                eprintln!("bitmap variant sweep failed: {error}");
                std::process::exit(7);
            }
        }
        "validate-bitmap-variant-artifact" => {
            if let Err(error) = bitmap::validate_variant_artifact() {
                eprintln!("bitmap variant artifact validation failed: {error}");
                std::process::exit(8);
            }
        }
        "sort-unique-self-test" => match sort_unique::self_test() {
            Ok(()) => println!(
                "{{\"status\":\"pass\",\"test\":\"sort-unique-self-test\",\"host\":\"rust\",\"backend\":\"cuda-cub-sort-unique\"}}"
            ),
            Err(error) => {
                eprintln!("sort-unique self-test failed: {error}");
                std::process::exit(9);
            }
        },
        "sort-unique-sweep" => {
            if let Err(error) = sort_unique::sweep() {
                eprintln!("sort-unique sweep failed: {error}");
                std::process::exit(10);
            }
        }
        "validate-sort-unique-artifact" => {
            if let Err(error) = sort_unique::validate_artifact() {
                eprintln!("sort-unique artifact validation failed: {error}");
                std::process::exit(11);
            }
        }
        "cayley-locality-sweep" => {
            if let Err(error) = cayley::locality_sweep() {
                eprintln!("Cayley locality sweep failed: {error}");
                std::process::exit(12);
            }
        }
        "validate-cayley-locality-artifact" => {
            if let Err(error) = cayley::validate_locality_artifact() {
                eprintln!("Cayley locality artifact validation failed: {error}");
                std::process::exit(13);
            }
        }
        "cayley-gpu-self-test" => match cayley::gpu_self_test() {
            Ok(()) => println!(
                "{{\"status\":\"pass\",\"test\":\"cayley-gpu-self-test\",\"host\":\"rust\",\"backend\":\"fused-cuda-cayley-bfs\"}}"
            ),
            Err(error) => {
                eprintln!("Cayley GPU self-test failed: {error}");
                std::process::exit(14);
            }
        },
        "cayley-gpu-s9-sweep" => {
            if let Err(error) = cayley::gpu_s9_sweep() {
                eprintln!("Cayley GPU S9 sweep failed: {error}");
                std::process::exit(15);
            }
        }
        "validate-cayley-gpu-s9-artifact" => {
            if let Err(error) = cayley::validate_gpu_s9_artifact() {
                eprintln!("Cayley GPU S9 artifact validation failed: {error}");
                std::process::exit(16);
            }
        }
        other => {
            eprintln!("unknown command: {other}");
            std::process::exit(64);
        }
    }
}
