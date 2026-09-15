use mgbfs_core::library_memory::LibraryMemoryBudget;

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
