"""Map address-only Nsight CUDA callchains to rank ELF offsets.

The input maps must be captured from the *same processes* as the callchains.
This script does not infer a Rust function or critical-path duration.
"""

import argparse
import json
from pathlib import Path


def parse_maps(path):
    mappings = []
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        start, end = fields[0].split("-", 1)
        mappings.append((int(start, 16), int(end, 16),
                         int(fields[2], 16), fields[5]))
    return mappings


def containing(mappings, address):
    return next((entry for entry in mappings
                 if entry[0] <= address < entry[1]), None)


def is_executable(path):
    return Path(path).name == "mgbfs"


def project_module(path):
    name = Path(path).name
    return name == "mgbfs" or name.startswith("libmgbfs_")


def attribute(callsites, maps):
    rows = []
    for row in callsites["rows"]:
        addresses = [int(symbol, 16) for symbol in row["symbols"]
                     if isinstance(symbol, str) and symbol.startswith("0x")]
        matching_ranks = [rank for rank, mappings in maps.items()
                          if any((mapping := containing(mappings, address))
                                 and is_executable(mapping[3])
                                 for address in addresses)]
        if len(matching_ranks) > 1:
            raise ValueError("NSYS_AMBIGUOUS_RANK")
        rank = matching_ranks[0] if matching_ranks else None
        frames = []
        if rank is not None:
            for address in addresses:
                mapping = containing(maps[rank], address)
                if mapping is not None and project_module(mapping[3]):
                    start, _, file_offset, module = mapping
                    frames.append({"module": module,
                                   "offset": hex(address - start + file_offset)})
        rows.append({"api": row["api"], "calls": row["calls"],
                     "api_duration_ns": row["api_duration_ns"],
                     "rank": rank, "project_frames": frames})
    return {"scope": "same-process ELF offsets; not symbolized or critical-path time",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("callsites", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--map", action="append", required=True, metavar="RANK:PATH")
    args = parser.parse_args()
    maps = {}
    for item in args.map:
        rank_text, path_text = item.split(":", 1)
        rank = int(rank_text)
        if rank in maps:
            raise ValueError("NSYS_DUPLICATE_RANK_MAP")
        maps[rank] = parse_maps(Path(path_text))
    result = attribute(json.loads(args.callsites.read_text()), maps)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
