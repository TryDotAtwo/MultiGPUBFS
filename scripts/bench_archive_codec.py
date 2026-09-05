"""Bounded same-host Parquet replay. Run on Kaggle, not on local graph data."""
import argparse
import json
import platform
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

METADATA = {'run_id', 'group_id', 'config_digest', 'rank', 'depth'}

def bench_table(table, repeats=3, slot_bytes=128 * 1024**2):
    if repeats < 1 or slot_bytes < 1 or not 0 < table.num_rows <= 1_000_000:
        raise ValueError('BENCH_CAPACITY')
    buffer = pa.allocate_buffer(slot_bytes)
    results = []
    projections = [('all', table)] + [(name, table.select([name])) for name in table.column_names]
    variants = [('zstd_dictionary', 'zstd', True),
                ('zstd_selective', 'zstd', False),
                ('snappy_selective', 'snappy', False),
                ('lz4_selective', 'lz4', False)]
    for column, projected in projections:
        # Alternate order between repeats to reduce systematic warm-cache bias.
        for repeat in range(repeats):
            for name, codec, dictionary in variants[::1 if repeat % 2 == 0 else -1]:
                writer = pa.FixedSizeBufferWriter(buffer)
                started = time.perf_counter()
                try:
                    pq.write_table(projected, writer, compression=codec,
                                   use_dictionary=True if dictionary else
                                   [c for c in projected.column_names if c in METADATA],
                                   row_group_size=min(projected.num_rows, 131072))
                    size = writer.tell()
                finally:
                    writer.close()
                elapsed = time.perf_counter() - started
                decoded = pq.read_table(pa.BufferReader(buffer.slice(0, size)))
                equal = projected.equals(decoded)
                if not equal:
                    raise RuntimeError('PARQUET_ROUNDTRIP_MISMATCH')
                results.append(dict(column=column, variant=name, repeat=repeat,
                                    rows=projected.num_rows, encode_seconds=elapsed,
                                    parquet_bytes=size, roundtrip_equal=equal,
                                    fixed_writer_bytes=slot_bytes,
                                    arrow_pool_bytes_observed=pa.total_allocated_bytes()))
                del decoded
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    parquet = pq.ParquetFile(args.input)
    if parquet.metadata.num_rows > 1_000_000 or args.input.stat().st_size > 128 * 1024**2:
        raise ValueError('INPUT_CAPACITY')
    table = parquet.read()
    result = dict(pyarrow=pa.__version__, platform=platform.platform(),
                  input_bytes=args.input.stat().st_size, table_bytes=table.nbytes,
                  measurements=bench_table(table))
    # Arrow allocator observations are not whole-process peak RSS. Fixed writer
    # capacity also excludes Parquet encoder scratch and the decoded validation table.
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
