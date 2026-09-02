#[derive(Clone, Copy, Debug)]
pub enum Protocol {
    BlindDrop,
    HelpableDescriptor,
    LoggedIntent,
}

#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub visited: bool,
    pub published: bool,
    pub publication_attempts: usize,
    pub unique_publications: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ChainOutcome {
    pub target_reached: bool,
}

#[derive(Clone, Copy)]
enum Event {
    StopClaimant,
    ClaimantPublish,
    Recover,
}

pub fn enumerate_single_stop_schedules(protocol: Protocol) -> Vec<Outcome> {
    const SCHEDULES: [[Event; 3]; 6] = [
        [Event::StopClaimant, Event::ClaimantPublish, Event::Recover],
        [Event::StopClaimant, Event::Recover, Event::ClaimantPublish],
        [Event::ClaimantPublish, Event::StopClaimant, Event::Recover],
        [Event::ClaimantPublish, Event::Recover, Event::StopClaimant],
        [Event::Recover, Event::StopClaimant, Event::ClaimantPublish],
        [Event::Recover, Event::ClaimantPublish, Event::StopClaimant],
    ];

    SCHEDULES
        .iter()
        .map(|schedule| run_schedule(protocol, schedule))
        .collect()
}

pub fn enumerate_chain_single_stop_schedules(
    protocol: Protocol,
    edge_count: usize,
) -> Vec<ChainOutcome> {
    (0..edge_count)
        .flat_map(|_| enumerate_single_stop_schedules(protocol))
        .map(|outcome| ChainOutcome {
            target_reached: outcome.published,
        })
        .collect()
}

fn run_schedule(protocol: Protocol, schedule: &[Event; 3]) -> Outcome {
    let mut claimant_alive = true;
    let mut outcome = Outcome {
        visited: true,
        published: false,
        publication_attempts: 0,
        unique_publications: 0,
    };

    for event in schedule {
        match event {
            Event::StopClaimant => claimant_alive = false,
            Event::ClaimantPublish if claimant_alive => publish(&mut outcome),
            Event::ClaimantPublish => {}
            Event::Recover => match protocol {
                Protocol::BlindDrop => {}
                Protocol::HelpableDescriptor | Protocol::LoggedIntent => publish(&mut outcome),
            },
        }
    }

    outcome
}

fn publish(outcome: &mut Outcome) {
    outcome.publication_attempts += 1;
    if !outcome.published {
        outcome.published = true;
        outcome.unique_publications = 1;
    }
}
