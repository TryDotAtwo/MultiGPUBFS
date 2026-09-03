use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::simulation::{run, Config, Profile};
fn config(profile: Profile) -> Config {
    Config {
        profile,
        prededup: false,
        rank_map: vec![0, 1],
        buckets: 4,
        bucket_capacity: 128,
        ring_records: 256,
        seed: [0; 16],
        schedule: 1,
        delayed_archive: false,
    }
}
#[test]
fn both_profiles_match_full_state_oracle_under_schedules_and_rank_maps() {
    for (n, m) in [(3, 2), (3, 3), (4, 2)] {
        let g = MatrixGroup::unitriangular(n, m).unwrap();
        let mut oracle = g.exact_layers(128).unwrap();
        for l in &mut oracle {
            l.sort();
        }
        for profile in [Profile::Dense, Profile::HashFirst] {
            for pre in [false, true] {
                for rank_map in [vec![0], vec![0, 1], vec![1, 0]] {
                    for schedule in 1..=6 {
                        let mut c = config(profile);
                        c.prededup = pre;
                        c.schedule = schedule;
                        c.delayed_archive = schedule % 2 == 0;
                        c.seed[0] = schedule as u8;
                        c.rank_map = rank_map.clone();
                        let r = run(&g, &c).unwrap();
                        assert_eq!(r.layers, oracle, "{n}/{m} {profile:?} {schedule}");
                        assert_eq!(
                            r.generated,
                            g.expected_max_unique_states * g.generators.len() as u64
                        );
                        assert_eq!(r.committed, g.expected_max_unique_states - 1);
                        assert_eq!(r.requests, r.responses);
                        assert_eq!(
                            r.requests,
                            if profile == Profile::HashFirst {
                                r.committed
                            } else {
                                0
                            }
                        );
                        assert!(r.tickets > oracle.len() as u64);
                    }
                }
            }
        }
    }
}

#[test]
fn dense_reuses_packed_parent_but_hash_first_and_archive_lease_prevent_reuse() {
    let g = MatrixGroup::unitriangular(2, 2).unwrap();
    let mut c = config(Profile::Dense);
    c.rank_map = vec![0];
    c.ring_records = 1;
    assert_eq!(run(&g, &c).unwrap().layers.len(), 2);
    c.profile = Profile::HashFirst;
    assert!(run(&g, &c).is_err());
    c.profile = Profile::Dense;
    c.delayed_archive = true;
    assert!(run(&g, &c).is_err());
}
#[test]
fn capacity_errors_never_return_a_completed_simulation() {
    let g = MatrixGroup::unitriangular(4, 2).unwrap();
    for profile in [Profile::Dense, Profile::HashFirst] {
        let mut c = config(profile);
        c.ring_records = 1;
        assert!(run(&g, &c).is_err());
        let mut c = config(profile);
        c.bucket_capacity = 1;
        assert!(run(&g, &c).is_err());
    }
    let mut c = config(Profile::Dense);
    c.rank_map = vec![0, 0];
    assert!(run(&g, &c).is_err());
}
