use mgbfs_core::library_memory::CandidateSoaLayout;
use mgbfs_core::library_memory::LibraryMemoryBudget;

#[test]
fn soa_buffer_accounts_for_plane_padding_and_source_indices() {
    for (rows, stride, bytes) in [
        (1, 256, 1280),
        (64, 256, 1280),
        (65, 512, 2560),
        (65536, 262144, 1310720),
    ] {
        let layout = CandidateSoaLayout::plan(rows).unwrap();
        assert_eq!(layout.plane_stride_bytes, stride);
        assert_eq!(layout.allocation_bytes, bytes);
    }
}

#[test]
fn soa_rejects_capacity_outside_libcudf_row_domain() {
    for rows in [0, i32::MAX as u64 + 1, u64::MAX] {
        assert!(CandidateSoaLayout::plan(rows).is_err());
    }
    let largest = CandidateSoaLayout::plan(i32::MAX as u64).unwrap();
    assert_eq!(largest.plane_stride_bytes, 8589934592);
    assert_eq!(largest.allocation_bytes, 42949672960);
}

const GIB: u64 = 1 << 30;

#[test]
fn counts_the_entire_reserved_pool_not_just_live_library_allocations() {
    let budget = LibraryMemoryBudget {
        pool_bytes: 3 * GIB,
        fixed_device_bytes: 4 * GIB,
        untouched_reserve_bytes: GIB,
        free_after_warmup_bytes: 8 * GIB,
    };
    assert_eq!(budget.validate().unwrap(), 7 * GIB);
    assert!(LibraryMemoryBudget {
        free_after_warmup_bytes: 8 * GIB - 1,
        ..budget
    }
    .validate()
    .is_err());
}

#[test]
fn refuses_unaligned_or_empty_pools_before_gpu_allocation() {
    for pool_bytes in [0, 255, 257] {
        assert!(LibraryMemoryBudget {
            pool_bytes,
            fixed_device_bytes: 0,
            untouched_reserve_bytes: GIB,
            free_after_warmup_bytes: 16 * GIB,
        }
        .validate()
        .is_err());
    }
}

#[test]
fn refuses_accounting_overflow_and_missing_t4_reserve() {
    assert!(LibraryMemoryBudget {
        pool_bytes: 256,
        fixed_device_bytes: u64::MAX,
        untouched_reserve_bytes: GIB,
        free_after_warmup_bytes: u64::MAX,
    }
    .validate()
    .is_err());
    assert!(LibraryMemoryBudget {
        pool_bytes: 256,
        fixed_device_bytes: 0,
        untouched_reserve_bytes: GIB - 1,
        free_after_warmup_bytes: 16 * GIB,
    }
    .validate()
    .is_err());
}
