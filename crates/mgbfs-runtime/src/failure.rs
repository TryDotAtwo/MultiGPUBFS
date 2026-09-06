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
