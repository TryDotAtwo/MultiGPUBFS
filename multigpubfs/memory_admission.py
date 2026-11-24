"""Storage policy shared by local and external BFS sessions."""

from collections import Counter
from dataclasses import dataclass
from math import comb


def bitmap_pack4_bits(graph):
    elements = graph.state_elements
    if not 1 <= elements <= 64:
        return None
    distinct = len(set(graph.start))
    bits = 4 if elements <= 16 else max(1, (distinct - 1).bit_length())
    return bits if distinct <= 16 and elements * bits <= 64 else None


def _default_pack4(graph, preset, capabilities, env, options):
    return (
        preset == "maxmemory"
        and bitmap_pack4_bits(graph) is not None
        and capabilities.get("bitmap_pack4_experimental_v1") is True
        and (graph.state_elements <= 16 or capabilities.get("bitmap_pack4_adaptive_v2") is True)
        and "bitmap_pack4" not in options
        and "MGBFS_SEGMENT_PACK4" not in env
    )


def bitmap_frontier_row_bytes(elements, state_bytes, pack4):
    # Adaptive Pack4 changes the dictionary slot width, but all versions
    # store one u64 per row.
    return 8 if pack4 else elements * state_bytes


def bitmap_frontier_banks(env, *, maxmemory=False):
    banks = env.get("MGBFS_BITMAP_BANKS", "2" if maxmemory else "3")
    if banks not in ("2", "3"):
        raise ValueError("BITMAP_BANK_COUNT")
    return int(banks)


def bitmap_domain(graph, *, position_orbits=False):
    if position_orbits:
        parent = list(range(graph.state_elements))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for generator in graph.action["generators"]:
            for index, source in enumerate(generator):
                parent[root(index)] = root(source)
        by_orbit = {}
        for index, symbol in enumerate(graph.start):
            by_orbit.setdefault(root(index), Counter())[symbol] += 1
        groups = by_orbit.values()
    else:
        groups = (Counter(graph.start),)
    domain = 1
    for counts in groups:
        used = 0
        for frequency in counts.values():
            domain *= comb(used + frequency, frequency)
            if domain >= 1 << 64:
                return None
            used += frequency
    return domain


def _hash_frontier_exceeds_budget(graph, bound, total_budget):
    if bound is None:
        return False
    if bound["kind"] == "CERTIFIED_INDEPENDENT_SWAPS":
        # History needs three equal rolling banks or an append-only orbit.
        # Even a byte codec and sorted runs need at least this much memory.
        rows = min(3 * bound["maximum_frontier"], bound["order"])
        return rows * (graph.state_elements + 12) > total_budget
    return bound["maximum_live_states"] * (20 + graph.state_elements) > total_budget


def bitmap_bytes_by_rank(domain, ranks):
    rows, tail = divmod(domain, ranks)
    return [8 * max(1, (rows + (rank < tail) + 63) // 64) for rank in range(ranks)]


def _bitmap_fits(bitmap_bytes, budgets):
    # The seed owner can be moved to physical rank 0, so check the weakest GPU
    # against the largest logical allocation.
    return max(bitmap_bytes) <= min(budgets) // 4


def _bitmap_reason(graph, preset, capabilities, env, options, load_inventory):
    if (capabilities.get("permutation_bitmap_v1") is not True
            or graph.action["kind"] != "permutation" or graph.state_elements > 64):
        return None
    from .workload_geometry import coxeter_frontier_bound, independent_swap_frontier_bound

    full_domain = bitmap_domain(graph)
    domain = (
        bitmap_domain(graph, position_orbits=True)
        if capabilities.get("permutation_bitmap_position_orbits_v1") is True else full_domain
    )
    if domain is None:
        return None
    bound = coxeter_frontier_bound(graph) or independent_swap_frontier_bound(graph)
    if preset != "maxmemory" and bound is None:
        return None
    budgets = [max(0, item["free_bytes"] - item.get("additional_reserve_bytes", 0))
               for item in load_inventory()]
    bitmap_bytes = bitmap_bytes_by_rank(domain, len(budgets))
    if not _bitmap_fits(bitmap_bytes, budgets):
        return None
    hash_impossible = _hash_frontier_exceeds_budget(graph, bound, sum(budgets))
    if bound is not None and bound["kind"] == "CERTIFIED_INDEPENDENT_SWAPS":
        # This lower bound assumes perfect balance. Native admission still
        # accounts for rank skew, transport, scratch, and allocation granularity.
        pack4_preset = "maxmemory" if hash_impossible else preset
        explicit = options.get("bitmap_pack4")
        requested = explicit if explicit is not None else env.get("MGBFS_SEGMENT_PACK4") == "1"
        packed = options.get("segment_strategy", "atomic") == "atomic" and (
            requested or _default_pack4(graph, pack4_preset, capabilities, env, options)
        )
        state_bytes = 1 if all(0 <= value <= 255 for value in graph.start) else 8
        row_bytes = bitmap_frontier_row_bytes(graph.state_elements, state_bytes, packed)
        banks = bitmap_frontier_banks(env, maxmemory=preset == "maxmemory")
        if bound["maximum_frontier"] * banks * row_bytes > sum(budgets) - sum(bitmap_bytes):
            return None
    if hash_impossible:
        return "HASH_FRONTIER_EXCEEDS_AVAILABLE_VRAM"
    if preset == "maxmemory" and full_domain is not None:
        if _bitmap_fits(bitmap_bytes_by_rank(full_domain, len(budgets)), budgets):
            return "BITMAP_DOMAIN_FITS_CONSERVATIVE_BUDGET"
    return None


@dataclass(frozen=True)
class TraversalMemory:
    backend: str
    strategy: str
    reason: str
    launch_options: dict


def bounded_hash_available(capabilities, env):
    supported = capabilities.get("segmented_strategies_v1")
    return (
        isinstance(supported, dict)
        and supported.get("batch") is True
        and capabilities.get("segmented_hash_series_v1") is True
        and env.get("MGBFS_SEGMENT_HASH_SERIES") != "0"
    )


def resolve_traversal_memory(graph, preset, capabilities, env, options, load_inventory, *,
                             default_backend="segmented24"):
    """Select storage and pass the same representation options to native admission."""
    options = dict(options)
    backend = options.pop("memory_backend", None)
    reason = "EXPLICIT_NATIVE_OPTIONS"
    if backend is None:
        bitmap_reason = _bitmap_reason(graph, preset, capabilities, env, options, load_inventory)
        backend = "permutation_bitmap" if bitmap_reason else default_backend
        reason = bitmap_reason or ("LEGACY_COMPATIBILITY" if backend == "legacy" else "SEGMENTED_EXACT_HISTORY")
    default_strategy = (
        "warp" if backend == "segmented24" and capabilities.get("segmented_warp_hash_safe_v1") is True
        else "atomic"
    )
    if ("segment_strategy" not in options and backend == "segmented24"
            and bounded_hash_available(capabilities, env)):
        default_strategy = "batch"
        reason = "SEGMENTED_BOUNDED_HASH"
    strategy = options.pop("segment_strategy", default_strategy)
    pack4_preset = "maxmemory" if reason == "HASH_FRONTIER_EXCEEDS_AVAILABLE_VRAM" else preset
    if backend == "permutation_bitmap" and strategy == "atomic" and graph.action["kind"] == "permutation":
        if _default_pack4(graph, pack4_preset, capabilities, env, options):
            options["bitmap_pack4"] = True
    if (preset == "maxmemory" and backend == "segmented24" and graph.action["kind"] == "permutation"
            and graph.state_elements > 24 and capabilities.get("segmented_resident_alphabet8_v1") is True
            and "resident_alphabet8" not in options and "MGBFS_SEGMENT_RESIDENT_ALPHABET8" not in env
            and any(not 0 <= value <= 255 for value in graph.start) and len(set(graph.start)) <= 256):
        options["resident_alphabet8"] = True
    return TraversalMemory(backend, strategy, reason, options)
