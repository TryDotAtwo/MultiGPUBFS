use mgbfs_core::owner_job::{admit, BucketJob, JobLimits, Range};

fn limits() -> JobLimits {
    JobLimits {
        incoming: 8,
        touched_buckets: 2,
        bucket_capacity: 4,
        buckets_per_shard: 4,
        shard_count: 2,
        lane: 1,
        generation: 7,
        prev_arena_count: 1000,
        curr_arena_count: 2000,
    }
}
fn jobs() -> [BucketJob; 2] {
    [
        BucketJob {
            bucket: 4,
            lane: 1,
            incoming: Range { begin: 0, count: 5 },
            prev: Range {
                begin: 996,
                count: 4,
            },
            curr: Range {
                begin: 1996,
                count: 4,
            },
            accepted_count: 4,
            generation: 7,
        },
        BucketJob {
            bucket: 5,
            lane: 1,
            incoming: Range { begin: 5, count: 3 },
            prev: Range {
                begin: 1000,
                count: 0,
            },
            curr: Range {
                begin: 2000,
                count: 0,
            },
            accepted_count: 0,
            generation: 7,
        },
    ]
}

#[test]
fn bounded_job_accepts_large_arenas_but_only_small_bucket_ranges() {
    // Incoming may exceed K: duplicates can still leave <=K survivors.
    let input = jobs();
    assert_eq!(admit(&input, limits()).unwrap(), 8);
    assert_eq!(input, jobs());
}

#[test]
fn admission_rejects_invalid_ranges_and_identity_without_mutation() {
    for (field, expected) in [
        (0, "OWNER_INCOMING_RANGE"),
        (1, "OWNER_PREV_RANGE"),
        (2, "OWNER_CURR_RANGE"),
        (3, "OWNER_BUCKET_CAPACITY"),
        (4, "OWNER_LANE"),
        (5, "OWNER_GENERATION"),
        (6, "OWNER_SHARD"),
        (7, "OWNER_BUCKET_ORDER"),
    ] {
        let mut input = jobs();
        match field {
            0 => input[1].incoming.begin = 4,
            1 => input[0].prev.begin = u64::MAX,
            2 => input[0].curr.begin = 1997,
            3 => input[0].accepted_count = 5,
            4 => input[1].lane = 0,
            5 => input[1].generation = 6,
            6 => input[1].bucket = 3,
            7 => input[1].bucket = 4,
            _ => unreachable!(),
        }
        let before = input;
        assert_eq!(admit(&input, limits()).unwrap_err(), expected);
        assert_eq!(input, before);
    }
}

#[test]
fn rejects_capacity_and_empty_job() {
    let mut l = limits();
    l.incoming = 7;
    assert_eq!(admit(&jobs(), l).unwrap_err(), "OWNER_INCOMING_CAPACITY");
    l = limits();
    l.touched_buckets = 1;
    assert_eq!(admit(&jobs(), l).unwrap_err(), "OWNER_JOB_CAPACITY");
    assert_eq!(admit(&[], limits()).unwrap_err(), "OWNER_EMPTY_JOB");
    for plane in 0..2 {
        let mut input = jobs();
        if plane == 0 {
            input[0].prev.count = 5;
        } else {
            input[0].curr.count = 5;
        }
        assert_eq!(
            admit(&input, limits()).unwrap_err(),
            "OWNER_BUCKET_CAPACITY"
        );
    }
}

#[test]
fn abi_is_64_bytes_and_offsets_are_frozen() {
    assert_eq!(std::mem::size_of::<BucketJob>(), 64);
    assert_eq!(std::mem::align_of::<BucketJob>(), 64);
    assert_eq!(std::mem::size_of::<Range>(), 16);
    let b = BucketJob::default();
    let base = &b as *const _ as usize;
    let offsets = [
        &b.bucket as *const _ as usize - base,
        &b.lane as *const _ as usize - base,
        &b.incoming as *const _ as usize - base,
        &b.prev as *const _ as usize - base,
        &b.curr as *const _ as usize - base,
        &b.accepted_count as *const _ as usize - base,
        &b.generation as *const _ as usize - base,
    ];
    assert_eq!(offsets, [0, 4, 8, 24, 40, 56, 60]);
}

#[test]
fn rejects_invalid_limits_before_division_or_launch() {
    for field in 0..8 {
        let mut l = limits();
        match field {
            0 => l.incoming = 0,
            1 => l.incoming = i32::MAX as u32 + 1,
            2 => l.touched_buckets = 0,
            3 => l.bucket_capacity = 0,
            4 => l.buckets_per_shard = 0,
            5 => l.buckets_per_shard = 3,
            6 => l.shard_count = 3,
            7 => {
                l.shard_count = 1 << 31;
                l.buckets_per_shard = 4;
            }
            _ => unreachable!(),
        }
        assert_eq!(admit(&jobs(), l).unwrap_err(), "OWNER_JOB_LIMITS");
    }
}

#[test]
fn rejects_zero_rows_gaps_overflow_and_out_of_rank_buckets() {
    for field in 0..5 {
        let mut input = jobs();
        let expected = match field {
            0 => {
                input[1].incoming.count = 0;
                "OWNER_INCOMING_RANGE"
            }
            1 => {
                input[1].incoming.begin = 6;
                "OWNER_INCOMING_RANGE"
            }
            2 => {
                input[1].incoming.count = u64::MAX;
                "OWNER_INCOMING_CAPACITY"
            }
            3 => {
                input[0].bucket = 8;
                "OWNER_SHARD"
            }
            4 => {
                input[1].prev.begin = 1001;
                "OWNER_PREV_RANGE"
            }
            _ => unreachable!(),
        };
        assert_eq!(admit(&input, limits()).unwrap_err(), expected);
    }
}

#[test]
fn incoming_limit_is_not_the_live_count() {
    let input = [jobs()[0]];
    assert_eq!(admit(&input, limits()).unwrap(), 5);
}
