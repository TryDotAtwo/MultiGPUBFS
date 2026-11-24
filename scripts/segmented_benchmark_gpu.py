"""CUDA ordinal provenance and optional physical pressure for a local BFS pair."""

import csv
import ctypes as ct
from contextlib import ExitStack, contextmanager
import subprocess
import threading


MIB = 1 << 20


class _NvmlMemory(ct.Structure):
    _fields_ = [("total", ct.c_ulonglong), ("free", ct.c_ulonglong), ("used", ct.c_ulonglong)]


def _nvidia_smi(*args):
    return subprocess.run(
        ["nvidia-smi", *args], check=True, capture_output=True, text=True, timeout=10,
    ).stdout


def _pci_id(value):
    domain, location = value.strip().lower().split(":", 1)
    return f"{int(domain, 16):04x}:{location}"


class BenchmarkGpus:
    def __init__(self, ordinals):
        self.driver = ct.CDLL("libcuda.so.1")
        self._call("cuInit", 0)
        self.devices = {}
        self.pci_by_ordinal = {}
        for ordinal in ordinals:
            device = ct.c_int()
            pci = ct.create_string_buffer(32)
            self._call("cuDeviceGet", ct.byref(device), ordinal)
            self._call("cuDeviceGetPCIBusId", pci, len(pci), device)
            self.devices[ordinal] = device
            self.pci_by_ordinal[ordinal] = _pci_id(pci.value.decode("ascii"))
        if len(set(self.pci_by_ordinal.values())) != len(ordinals):
            raise ValueError("selected CUDA ordinals must identify distinct physical GPUs")

    def _call(self, name, *args):
        code = getattr(self.driver, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with CUDA driver status {code}")

    def _release(self, context, address):
        try:
            self._call("cuCtxSetCurrent", context)
            self._call("cuMemFree_v2", address)
        finally:
            self._call("cuCtxSetCurrent", None)

    @contextmanager
    def pressure(self, count):
        with ExitStack() as cleanup:
            if count:
                for device in self.devices.values():
                    context = ct.c_void_p()
                    address = ct.c_uint64()
                    self._call("cuDevicePrimaryCtxRetain", ct.byref(context), device)
                    cleanup.callback(self._call, "cuDevicePrimaryCtxRelease_v2", device)
                    self._call("cuCtxSetCurrent", context)
                    try:
                        self._call("cuMemAlloc_v2", ct.byref(address), ct.c_size_t(count))
                        cleanup.callback(self._release, context, address)
                        self._call("cuMemsetD8_v2", address, 0, ct.c_size_t(count))
                        self._call("cuCtxSynchronize")
                    finally:
                        self._call("cuCtxSetCurrent", None)
            yield

    def snapshot(self):
        inventory = _nvidia_smi(
            "--query-gpu=pci.bus_id,uuid,name,driver_version,memory.total,memory.free,mig.mode.current",
            "--format=csv,noheader,nounits",
        )
        by_bus = {}
        for line in csv.reader(inventory.splitlines(), skipinitialspace=True):
            pci, uuid, name, driver, total, free, mig = (field.strip() for field in line)
            bus = _pci_id(pci)
            if bus not in self.pci_by_ordinal.values():
                continue
            if mig == "Enabled":
                raise ValueError("MIG GPU identity and pressure require an instance-aware probe")
            by_bus[bus] = {
                "pci_bus_id": bus, "uuid": uuid, "name": name, "driver": driver,
                "total_bytes": int(total) * MIB, "free_bytes": int(free) * MIB,
            }
        topology = _nvidia_smi("topo", "-m").strip()
        if not topology:
            raise RuntimeError("GPU topology is missing")
        return {
            "devices": [
                {"ordinal": ordinal, **by_bus[pci]}
                for ordinal, pci in self.pci_by_ordinal.items()
            ],
            "topology": topology,
        }

    @contextmanager
    def sample_memory(self, snapshot, *, interval=0.05):
        """Sample physical device-wide VRAM in the admitted CUDA ordinal order."""
        nvml = ct.CDLL("libnvidia-ml.so.1")

        def call(name, *args):
            code = getattr(nvml, name)(*args)
            if code:
                raise RuntimeError(f"{name} failed with NVML status {code}")

        def read():
            used = []
            for handle in handles:
                info = _NvmlMemory()
                call("nvmlDeviceGetMemoryInfo", handle, ct.byref(info))
                used.append(info.used)
            return used

        call("nvmlInit_v2")
        worker = None
        stop = threading.Event()
        try:
            handles = [ct.c_void_p() for _ in snapshot["devices"]]
            for gpu, handle in zip(snapshot["devices"], handles):
                call("nvmlDeviceGetHandleByUUID", gpu["uuid"].encode("ascii"), ct.byref(handle))
            baseline = read()
            result = {
                "scope": "device-wide NVML used bytes, including the pressure holder and CUDA contexts",
                "sample_interval_seconds": interval,
                "samples": 0,
                "baseline_bytes_by_rank": baseline,
                "peak_bytes_by_rank": [0] * len(baseline),
                "peak_sum_bytes_in_one_poll": 0,
            }

            def sample():
                while not stop.wait(interval):
                    try:
                        used = read()
                    except RuntimeError as error:
                        result["error"] = str(error)
                        return
                    if stop.is_set():
                        return
                    result["samples"] += 1
                    result["peak_bytes_by_rank"] = [max(a, b) for a, b in zip(result["peak_bytes_by_rank"], used)]
                    result["peak_sum_bytes_in_one_poll"] = max(result["peak_sum_bytes_in_one_poll"], sum(used))

            worker = threading.Thread(target=sample, name="mgbfs-memory-sampling", daemon=True)
            worker.start()
            yield result
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=2)
            if worker is not None and worker.is_alive():
                result["error"] = "NVML memory sampler did not stop"
            else:
                call("nvmlShutdown")
