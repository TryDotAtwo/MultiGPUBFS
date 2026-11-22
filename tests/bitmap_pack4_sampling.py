"""Optional, explicitly selected GPU-memory samples for benchmark runs."""
from contextlib import contextmanager
import threading


def sample_pattern(raw, count):
    """A comma-separated 0/1 per run; unset means no sampling."""
    if raw is None:
        return [False] * count
    values = raw.split(",")
    if len(values) != count or any(value not in ("0", "1") for value in values):
        raise ValueError("MGBFS_BENCH_SAMPLE_PATTERN must contain one 0 or 1 per run")
    return [value == "1" for value in values]


@contextmanager
def collect_vram_samples(enabled, sampler):
    samples = []
    stop = threading.Event()
    thread = threading.Thread(target=sampler, args=(stop, samples), daemon=True) if enabled else None
    if thread is not None:
        thread.start()
    try:
        yield samples
    finally:
        stop.set()
        if thread is not None:
            thread.join()


def vram_summary(enabled, samples, devices):
    valid = [sample["gpus"] for sample in samples if "gpus" in sample]
    peaks = None
    if enabled and valid:
        peaks = {str(device): max((sample[device] for sample in valid if device in sample), default=None)
                 for device in devices}
    return {"nvml_enabled": enabled,
            "nvml_sample_interval_seconds": .5 if enabled else None,
            "nvml_max_mib_by_gpu": peaks,
            "nvml_samples": len(valid)}
