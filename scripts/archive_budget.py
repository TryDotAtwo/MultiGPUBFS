"""Explicit archive RAM budget and diagnostic fluid-queue replay, not a bound."""
import math

def replay(counts, seconds, records_per_second):
    if (len(counts) != len(seconds) or not math.isfinite(records_per_second)
            or records_per_second <= 0 or any(n < 0 for n in counts)
            or any(not math.isfinite(t) or t < 0 for t in seconds)):
        raise ValueError('ARCHIVE_REPLAY_SHAPE')
    queued = peak = 0.0
    for count, duration in zip(counts, seconds):
        queued = max(0.0, queued + count - records_per_second * duration)
        peak = max(peak, queued)
    return dict(peak_records=math.ceil(peak), remaining_records=math.ceil(queued))

def host_budget(available_bytes, ranks, rows, slots, state_bytes,
                upload_bytes, scratch_bytes, reserve_bytes):
    values = (available_bytes, ranks, rows, slots, state_bytes,
              upload_bytes, scratch_bytes, reserve_bytes)
    if any(not isinstance(x, int) or x < 0 for x in values) or min(ranks, rows, slots, state_bytes) == 0:
        raise ValueError('HOST_RAM_SHAPE')
    pinned = ranks * rows * slots * (state_bytes + 16)
    required = pinned + upload_bytes + scratch_bytes + reserve_bytes
    if available_bytes < required:
        raise ValueError(f'HOST_RAM_PREFLIGHT required={required} available={available_bytes}')
    return dict(available_bytes=available_bytes, pinned_bytes=pinned,
                upload_bytes=upload_bytes, scratch_bytes=scratch_bytes,
                reserve_bytes=reserve_bytes, required_bytes=required)
