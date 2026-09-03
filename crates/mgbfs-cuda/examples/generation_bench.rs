//! Generation stage benchmark; Rust orchestration, native CUDA data plane.
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_cuda::ffi::*;
use std::ffi::{c_void, CStr};
extern "C" {
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}
fn check(code: i32) {
    assert_eq!(code, 0);
}
fn allocation(n: usize) -> *mut c_void {
    let mut p = std::ptr::null_mut();
    check(unsafe { cudaMalloc(&mut p, n) });
    p
}
fn event() -> *mut c_void {
    let mut p = std::ptr::null_mut();
    check(unsafe { cudaEventCreateWithFlags(&mut p, 0) });
    p
}
fn used() -> usize {
    let (mut free, mut total) = (0, 0);
    check(unsafe { cudaMemGetInfo(&mut free, &mut total) });
    total - free
}
fn elapsed(a: *mut c_void, b: *mut c_void) -> f32 {
    let mut ms = 0.;
    check(unsafe { cudaEventElapsedTime(&mut ms, a, b) });
    ms
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let variant: u32 = args[1].parse().unwrap();
    let count: u32 = args[2].parse().unwrap();
    let g = MatrixGroup::unitriangular(4, 256).unwrap();
    let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
    let parents: Vec<u8> = (0..count as usize * 16)
        .map(|i| ((i / 16 * 173 + i % 16 * 97) % 256) as u8)
        .collect();
    unsafe {
        let mut stream = std::ptr::null_mut();
        check(cudaStreamCreateWithFlags(&mut stream, 1));
        let before = used();
        let input = allocation(parents.len());
        let output = allocation(count as usize * 96);
        let mut plan = std::ptr::null_mut();
        let mut error = [0i8; 512];
        assert_eq!(
            mgbfs_generate_create_variant(
                4,
                6,
                256,
                count,
                generators.as_ptr(),
                variant,
                &mut plan,
                error.as_mut_ptr(),
                512
            ),
            0,
            "{}",
            CStr::from_ptr(error.as_ptr()).to_string_lossy()
        );
        let begin = event();
        let end = event();
        let marks = [event(), event(), event(), event()];
        let bytes = used().saturating_sub(before);
        check(cudaMemcpy(input, parents.as_ptr().cast(), parents.len(), 1));
        let warm_start = std::time::Instant::now();
        while warm_start.elapsed() < std::time::Duration::from_millis(200) {
            check(mgbfs_generate_run(
                plan,
                input.cast(),
                output.cast(),
                count,
                stream,
            ));
            check(cudaStreamSynchronize(stream));
        }
        check(cudaStreamSynchronize(stream));
        // Independent arithmetic at beginning, middle and tail, outside timing.
        for parent in [0, count as usize / 2, count as usize - 1] {
            let mut actual = [0u8; 96];
            check(cudaMemcpy(
                actual.as_mut_ptr().cast(),
                output.cast::<u8>().add(parent * 96).cast(),
                96,
                2,
            ));
            for m in 0..6 {
                assert_eq!(
                    &actual[m * 16..m * 16 + 16],
                    g.successor(&parents[parent * 16..parent * 16 + 16], m)
                        .unwrap()
                );
            }
        }
        let mut uninstrumented_ms = Vec::new();
        for _ in 0..7 {
            check(cudaEventRecord(begin, stream));
            for _ in 0..50 {
                check(mgbfs_generate_run(
                    plan,
                    input.cast(),
                    output.cast(),
                    count,
                    stream,
                ));
            }
            check(cudaEventRecord(end, stream));
            check(cudaStreamSynchronize(stream));
            uninstrumented_ms.push(elapsed(begin, end) / 50.);
        }
        let mut stage_ms = Vec::new();
        for _ in 0..7 {
            check(mgbfs_generate_profile_run(
                plan,
                input.cast(),
                output.cast(),
                count,
                stream,
                marks.as_ptr(),
            ));
            check(cudaStreamSynchronize(stream));
            stage_ms.push([
                elapsed(marks[0], marks[1]),
                elapsed(marks[1], marks[2]),
                elapsed(marks[2], marks[3]),
            ]);
        }
        println!("{{\"status\":\"COMPLETE\",\"variant\":{variant},\"parents\":{count},\"device_allocation_delta_bytes\":{bytes},\"uninstrumented_generation_ms\":{uninstrumented_ms:?},\"profiled_pack_gemm_write_ms\":{stage_ms:?}}}");
        for e in marks.into_iter().chain([begin, end]) {
            check(cudaEventDestroy(e));
        }
        mgbfs_generate_destroy(plan);
        check(cudaFree(output));
        check(cudaFree(input));
        check(cudaStreamDestroy(stream));
    }
}
