use mgbfs_runtime::benchmark::{run_phases, Phase};

#[test]
fn full_warmup_precedes_measurement_and_failure_prevents_measurement() {
    let mut phases = Vec::new();
    run_phases(true, |phase| {
        phases.push(phase);
        Ok(())
    })
    .unwrap();
    assert_eq!(phases, [Phase::Warmup, Phase::Measure]);
    phases.clear();
    let result = run_phases(true, |phase| {
        phases.push(phase);
        Err("warmup failed".into())
    });
    assert_eq!(result.unwrap_err(), "warmup failed");
    assert_eq!(phases, [Phase::Warmup]);
    phases.clear();
    run_phases(false, |phase| {
        phases.push(phase);
        Ok(())
    })
    .unwrap();
    assert_eq!(phases, [Phase::Measure]);
}
