#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;

#[test]
fn two_devices_scatter_exact_bytes_from_each_source_and_drain_empty_epochs() {
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let workers: Vec<_> = (0..2u32)
        .map(|rank| {
            std::thread::spawn(move || unsafe {
                let mut comm = std::ptr::null_mut();
                let mut error = [0i8; 512];
                assert_eq!(
                    mgbfs_nccl_create(
                        rank,
                        2,
                        rank,
                        id.as_ptr().cast(),
                        &mut comm,
                        error.as_mut_ptr(),
                        error.len()
                    ),
                    0
                );
                let mut stream = std::ptr::null_mut();
                assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
                let mut send = std::ptr::null_mut();
                let mut recv = std::ptr::null_mut();
                assert_eq!(cudaMalloc(&mut send, 8), 0);
                assert_eq!(cudaMalloc(&mut recv, 4), 0);
                for source in 0..2u32 {
                    let payload = [11u8, 12, 13, 14, 21, 22, 23, 24];
                    assert_eq!(cudaMemcpy(send, payload.as_ptr().cast(), 8, 1), 0);
                    let sizes = [4u64, 4];
                    assert_eq!(
                        mgbfs_nccl_scatter(comm, source, send, 8, sizes.as_ptr(), recv, 4, 4, stream),
                        0
                    );
                    assert_eq!(cudaStreamSynchronize(stream), 0);
                    assert_eq!(mgbfs_nccl_poll(comm), 0);
                    let mut actual = [0u8; 4];
                    let selected = if rank == source {
                        send.cast::<u8>().add(rank as usize * 4).cast()
                    } else {
                        recv
                    };
                    assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), selected, 4, 2), 0);
                    assert_eq!(
                        actual,
                        if rank == 0 {
                            [11, 12, 13, 14]
                        } else {
                            [21, 22, 23, 24]
                        }
                    );
                    let zero = [0u64; 2];
                    assert_eq!(
                        mgbfs_nccl_scatter(comm, source, send, 8, zero.as_ptr(), recv, 0, 4, stream),
                        0
                    );
                    assert_eq!(cudaStreamSynchronize(stream), 0);
                }
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                mgbfs_nccl_destroy(comm);
                assert_eq!(cudaFree(send), 0);
                assert_eq!(cudaFree(recv), 0);
                assert_eq!(cudaStreamDestroy(stream), 0);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
