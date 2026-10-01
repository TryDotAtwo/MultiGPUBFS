"""Checksummed incremental native archive reader with CPU nibble packing."""
import hashlib
import struct
from pathlib import Path
import numpy as np


def read_exact(stream, size):
    chunks, left = [], size
    while left:
        part = stream.read(left)
        if not part:
            raise EOFError('truncated archive')
        chunks.append(part)
        left -= len(part)
    return b''.join(chunks)


def pack_batch(raw, count, n):
    if not 1 <= n <= 32 or len(raw) != count * n:
        raise ValueError('word shape')
    symbols = np.frombuffer(raw, dtype=np.uint8).reshape(count, n)
    if np.any(symbols > 15):
        raise ValueError('alphabet exceeds four bits')
    width = 8 if n <= 16 else 16
    result = np.zeros((count, width), dtype=np.uint8)
    for i in range(n):
        result[:, i // 2] |= symbols[:, i] << (4 * (i % 2))
    return result.tobytes()


def consume(stream, root, n, on_layer, *, max_frame_bytes=64*1024*1024):
    """Callback only after checksummed layer commit; returns run receipt.

    Incomplete final layer is removed on EOF/error. Root belongs to one rank.
    Callback must consume/delete each committed file to bound spool storage.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    header = read_exact(stream, 48)
    if header[:8] != b'MGBFSAR1' or struct.unpack_from('<Q', header, 8)[0] != n:
        raise ValueError('archive width/header mismatch')
    config = header[16:48].hex()
    chain = hashlib.sha256(header).digest()
    seq = depth = rows = total = 0
    path = None
    output = None
    try:
        while True:
            frame = read_exact(stream, 80)
            kind, at, count, size, sequence = struct.unpack_from('<QQQQQ', frame, 8)
            if frame[:8] != b'MGBFSFR1' or at != depth or sequence != seq or frame[48:] != chain:
                raise ValueError('archive chain/order')
            if size > max_frame_bytes:
                raise ValueError('frame memory bound')
            payload = read_exact(stream, size)
            chain = hashlib.sha256(frame + payload).digest()
            if read_exact(stream, 32) != chain:
                raise ValueError('archive checksum')
            if kind == 1:
                if count == 0 or size != count * (n + 16):
                    raise ValueError('record shape')
                if output is None:
                    path = root / f'layer-{depth:06d}.bin'
                    output = path.open('xb')
                output.write(pack_batch(payload[:count*n], count, n))
                rows += count
            elif kind == 2:
                if size != 0 or count != rows:
                    raise ValueError('layer count')
                if output is None:
                    path = root / f'layer-{depth:06d}.bin'
                    output = path.open('xb')
                output.close()
                output = None
                on_layer(depth, rows, path, config)
                path = None
                total += rows
                rows = 0
                depth += 1
            elif kind == 3:
                if size or rows or depth == 0 or count != total:
                    raise ValueError('run count')
                return dict(depths=depth, states=total, config_digest=config)
            else:
                raise ValueError('frame kind')
            seq += 1
    finally:
        if output:
            output.close()
        if path:
            path.unlink(missing_ok=True)
