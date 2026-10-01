/// Publish constructor failure before an earlier-declared communicator drops.
pub(crate) struct ConstructorFailureReport {
    report: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    armed: bool,
}
impl ConstructorFailureReport {
    pub(crate) fn new(report: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>) -> Self {
        Self { report, armed: true }
    }
    pub(crate) fn disarm(&mut self) { self.armed = false; }
}
impl Drop for ConstructorFailureReport {
    fn drop(&mut self) {
        if self.armed {
            if let Some(report) = &self.report {
                report.store(2, std::sync::atomic::Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod constructor_report_tests {
    use super::ConstructorFailureReport;
    use std::sync::{Arc, atomic::{AtomicU8, Ordering}};
    #[test]
    fn constructor_error_is_reported_before_communicator_cleanup() {
        struct Cleanup(Arc<AtomicU8>);
        impl Drop for Cleanup {
            fn drop(&mut self) { assert_eq!(self.0.load(Ordering::Acquire), 2); }
        }
        let report = Arc::new(AtomicU8::new(0));
        let failed = || -> Result<(), ()> {
            let _communicator = Cleanup(report.clone());
            let _notification = ConstructorFailureReport::new(Some(report.clone()));
            Err(())
        };
        assert!(failed().is_err());
    }
    #[test]
    fn successful_constructor_does_not_publish_failure() {
        let report = Arc::new(AtomicU8::new(0));
        {
            let mut notification = ConstructorFailureReport::new(Some(report.clone()));
            notification.disarm();
        }
        assert_eq!(report.load(Ordering::Acquire), 0);
    }
}

/// Attempt every rank-safe operation and preserve the first local failure.
///
/// Distributed callers can then enter the same failure collective even when
/// an earlier local operation failed, instead of abandoning a peer in NCCL.
pub fn attempt_all<I, F, E>(items: I, mut attempt: F) -> Result<(), E>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<(), E>,
{
    let mut first = None;
    for item in items {
        if let Err(error) = attempt(item) {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    first.map_or(Ok(()), Err)
}

/// Poison before cleanup; preserve the originating failure rather than replacing
/// it with an abort status. The caller must serialize communicator access.
pub fn abort_on_error<T, E>(
    result: Result<T, E>,
    failed: &mut bool,
    abort: impl FnOnce(),
) -> Result<T, E> {
    if result.is_err() {
        *failed = true;
        abort();
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerFailurePolicy {
    /// Legacy collective agreement needs both groups to reach the vote.
    CollectiveVote,
    /// The dispatcher reports failure out-of-band and aborts the communicator.
    CancelGroup,
}

/// Process independent local input, then establish remote readiness before
/// consuming remote input. Fatal cancellation may skip readiness, but never
/// authorizes reuse of any source or receive lease.
pub fn process_owner_pair<T, E>(
    policy: OwnerFailurePolicy,
    local: T,
    remote: T,
    mut process: impl FnMut(T) -> Result<(), E>,
    ready: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    let first = process(local);
    if policy == OwnerFailurePolicy::CancelGroup && first.is_err() {
        return first;
    }
    if let Err(error) = ready() {
        return first.and(Err(error));
    }
    let second = process(remote);
    first.and(second)
}

/// Enter the same failure vote on every rank before irreversible owner work.
/// A local error must not skip the collective; a remote error stops this rank.
pub fn vote_group_error<E>(
    local: Result<(), E>,
    vote: impl FnOnce(bool) -> Result<bool, E>,
    remote_error: E,
) -> Result<(), E> {
    let failed = vote(local.is_err())?;
    if failed {
        Err(local.err().unwrap_or(remote_error))
    } else {
        local
    }
}

/// Publish host/API failure before cleanup can block on a live GPU reader.
/// The cleanup still owns all memory leases; notification never authorizes reuse.
/// The dispatcher performs local abort before waiting for those readers.
pub fn report_and_abort_on_error<T, E>(
    result: Result<T, E>,
    failed: &mut bool,
    report: Option<&std::sync::atomic::AtomicU8>,
    abort: impl FnOnce(),
    cleanup: impl FnOnce(),
) -> Result<T, E> {
    abort_on_error(result, failed, || {
        if let Some(report) = report {
            report.store(2, std::sync::atomic::Ordering::Release);
        }
        abort();
        cleanup();
    })
}
