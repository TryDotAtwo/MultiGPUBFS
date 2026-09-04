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
