use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::simulation::{run_concurrent, Config, Profile};
fn config(profile: Profile, seed: u64, map: Vec<usize>) -> Config {
    Config {
        profile,
        prededup: seed % 2 == 0,
        rank_map: map,
        buckets: 4,
        bucket_capacity: 128,
        ring_records: 256,
        seed: (seed as u128).to_le_bytes(),
        schedule: seed,
        delayed_archive: true,
    }
}
#[test]
fn overlapping_batches_tickets_and_archive_match_full_layers() {
    let mut peak_batches = 0;
    let mut peak_tickets = 0;
    for (n, m) in [(3, 3), (4, 2)] {
        let g = MatrixGroup::unitriangular(n, m).unwrap();
        let mut oracle = g.exact_layers(128).unwrap();
        for l in &mut oracle {
            l.sort();
        }
        for p in [Profile::Dense, Profile::HashFirst] {
            for map in [vec![0], vec![0, 1], vec![1, 0]] {
                for seed in 1..=12 {
                    let r = run_concurrent(&g, &config(p, seed, map.clone()), 3).unwrap();
                    assert_eq!(r.result.layers, oracle, "{n}/{m} {p:?} {seed}");
                    assert_eq!(
                        r.result.generated,
                        g.expected_max_unique_states * g.generators.len() as u64
                    );
                    assert_eq!(r.result.committed, g.expected_max_unique_states - 1);
                    assert_eq!(r.result.requests, r.result.responses);
                    assert_eq!(
                        r.result.requests,
                        if p == Profile::HashFirst {
                            r.result.committed
                        } else {
                            0
                        }
                    );
                    assert!(r.peak_batches <= 3);
                    assert_eq!(r.state_peak_records.len(), map.len());
                    assert!(r.state_peak_records.iter().all(|&n| n <= 256));
                    assert!(r.state_peak_records.iter().sum::<u64>() > 0);
                    assert!(r.steps > 0);
                    peak_batches = peak_batches.max(r.peak_batches);
                    peak_tickets = peak_tickets.max(r.peak_tickets);
                }
            }
        }
    }
    assert_eq!(peak_batches, 3);
    assert!(peak_tickets >= 2);
}
#[test]
fn concurrent_capacity_failure_is_not_a_partial_success() {
    let g = MatrixGroup::unitriangular(4, 2).unwrap();
    let mut c = config(Profile::HashFirst, 1, vec![0, 1]);
    assert!(run_concurrent(&g, &c, 0).is_err());
    c.ring_records = 1;
    assert!(run_concurrent(&g, &c, 3).is_err());
    c.ring_records = 256;
    c.bucket_capacity = 1;
    assert!(run_concurrent(&g, &c, 3).is_err());
}
