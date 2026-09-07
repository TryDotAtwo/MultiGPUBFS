//! DENSE completion metadata recovered from consumed job descriptor slots.
use crate::jobs::JobSpan;
use mgbfs_core::Result;
use mgbfs_cuda::native_owner::Extent;

// GPU result capture overwrites one fully consumed BucketJob, never adjacent
// descriptors. Both Rust/CUDA ABI records must remain exactly 64 bytes.
const _: [(); 64] = [(); std::mem::size_of::<Extent>()];
const _: [(); 64] = [(); std::mem::size_of::<mgbfs_core::owner_job::BucketJob>()];

pub fn append_dense_results(
    records: &[Extent],
    spans: &[JobSpan],
    next: &mut Vec<Extent>,
) -> Result<()> {
    let mut previous = None;
    for span in spans {
        let e = records.get(span.first).ok_or("OWNER_RESULT_SLOT")?;
        if previous.is_some_and(|p| span.first <= p)
            || e.ready != 1
            || e.count != u64::from(e.granted_rows)
            || e.begin.checked_add(e.count).is_none()
            || e.sequence.checked_add(e.count).is_none()
        {
            return Err("OWNER_RESULT_INVALID".into());
        }
        previous = Some(span.first);
    }
    for span in spans {
        let mut e = records[span.first];
        if e.count == 0 {
            continue;
        }
        e.padding[1] = e.descriptor;
        if let Some(last) = next.last_mut().filter(|last| {
            last.begin.checked_add(last.count) == Some(e.begin)
                && last.sequence.checked_add(last.count) == Some(e.sequence)
        }) {
            let count = last
                .count
                .checked_add(e.count)
                .ok_or("EXTENT_COUNT_OVERFLOW")?;
            let rows = u32::try_from(count).map_err(|_| "EXTENT_COUNT_OVERFLOW")?;
            last.count = count;
            last.granted_rows = rows;
            last.padding[1] = e.descriptor;
        } else {
            if next.len() == next.capacity() {
                return Err("HOST_EXTENT_CAPACITY".into());
            }
            next.push(e);
        }
    }
    Ok(())
}
