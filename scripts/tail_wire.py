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


def pack_batch(raw, count, n, bits_per_symbol=4):
    if not 1 <= n <= 128 or len(raw) != count * n or bits_per_symbol not in (4,8):
        raise ValueError('word shape')
    symbols = np.frombuffer(raw, dtype=np.uint8).reshape(count, n)
    if np.any(symbols >= 1 << bits_per_symbol):
        raise ValueError('alphabet exceeds four bits')
    width = ((n*bits_per_symbol+63)//64)*8
    result = np.zeros((count, width), dtype=np.uint8)
    if bits_per_symbol == 8:
        result[:, :n] = symbols
    else:
        for i in range(n):
            result[:, i // 2] |= symbols[:, i] << (4 * (i % 2))
    return result.tobytes()


def consume(stream, root, n, on_layer, *, max_frame_bytes=64*1024*1024, bits_per_symbol=4, on_packed_layer=None):
    """Callback only after checksummed layer commit; returns run receipt.

    Incomplete final layer is removed on EOF/error. Root belongs to one rank.
    Callback must consume/delete each committed file to bound spool storage.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    header = read_exact(stream, 48)
    state_only = header[:8] == b'MGBFSAS3'
    if header[:8] not in (b'MGBFSAR1', b'MGBFSAS3') or struct.unpack_from('<Q', header, 8)[0] != n:
        raise ValueError('archive width/header mismatch')
    config = header[16:48].hex()
    chain = hashlib.sha256(header).digest()
    packed_digest = hashlib.sha256()
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
                if count == 0 or size != count * (n if state_only else n + 16):
                    raise ValueError('record shape')
                if output is None:
                    path = root / f'layer-{depth:06d}.bin'
                    output = path.open('xb')
                packed=pack_batch(payload[:count*n], count, n, bits_per_symbol)
                output.write(packed)
                packed_digest.update(packed)
                rows += count
            elif kind == 2:
                if size != 0 or count != rows:
                    raise ValueError('layer count')
                if output is None:
                    path = root / f'layer-{depth:06d}.bin'
                    output = path.open('xb')
                output.close()
                output = None
                if on_packed_layer is None:
                    on_layer(depth, rows, path, config)
                else:
                    on_packed_layer(depth, rows, path, config, packed_digest.hexdigest())
                packed_digest = hashlib.sha256()
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


def consume_selected(stream, root, n, on_layer, *, bits_per_symbol=4,
                     max_frame_bytes=64*1024*1024, on_packed_layer=None):
    """Only a terminal replacement touches SSD. Samples stay bounded in RAM.

    on_layer(depth, rows, bytes_or_path, config_digest, replacement).
    The GPU producer sends canonical symbols; packing/checksums run here.
    """
    root = Path(root)
    header = read_exact(stream, 48)
    if header[:8] != b'MGBFSAS2' or struct.unpack_from('<Q', header, 8)[0] != n:
        raise ValueError('selected archive header')
    config = header[16:48].hex(); chain = hashlib.sha256(header).digest()
    seq = depth = rows = total = previous_rows = 0
    replacement = replaced = False
    buffer = bytearray(); path = output = None
    packed_digest = hashlib.sha256()
    try:
        while True:
            frame = read_exact(stream, 80)
            kind, at, count, size, sequence = struct.unpack_from('<QQQQQ', frame, 8)
            expected = depth-1 if kind == 4 else depth
            if (frame[:8] != b'MGBFSFR1' or at != expected or sequence != seq
                    or frame[48:] != chain or size > max_frame_bytes):
                raise ValueError('selected archive chain/order')
            payload = read_exact(stream, size)
            chain = hashlib.sha256(frame+payload).digest()
            if read_exact(stream, 32) != chain:
                raise ValueError('selected archive checksum')
            if kind == 1:
                if not count or size != count*n or (not replacement and rows+count > 1000):
                    raise ValueError('selected record shape')
                packed = pack_batch(payload, count, n, bits_per_symbol)
                if replacement:
                    if output is None:
                        root.mkdir(parents=True, exist_ok=True)
                        path = root/f'layer-{depth:06d}.bin'; output = path.open('xb')
                    output.write(packed)
                    packed_digest.update(packed)
                else:
                    buffer.extend(packed)
                rows += count
            elif kind == 2:
                if size or count != rows:
                    raise ValueError('selected layer count')
                if replacement:
                    if output is None:
                        raise ValueError('empty terminal replacement')
                    output.close(); output = None
                    if on_packed_layer is None:
                        on_layer(depth, rows, path, config, True)
                    else:
                        on_packed_layer(depth, rows, path, config, packed_digest.hexdigest())
                    path = None
                else:
                    on_layer(depth, rows, bytes(buffer), config, False); buffer.clear()
                total += rows; previous_rows = rows; rows = 0; depth += 1
            elif kind == 4:
                if size or count or rows or not depth or replaced:
                    raise ValueError('selected replacement phase')
                replacement = replaced = True; depth -= 1; total -= previous_rows
            elif kind == 3:
                if size or rows or not depth or count != total:
                    raise ValueError('selected run count')
                return dict(depths=depth, states=total, config_digest=config,
                            selected=True, terminal_replaced=replaced)
            else:
                raise ValueError('selected frame kind')
            if replacement and kind == 2:
                # A terminal replacement must be followed by RunCommit only.
                replacement = False
            elif replaced and not replacement and kind != 3:
                raise ValueError('records after terminal replacement')
            seq += 1
    finally:
        if output: output.close()
        if path: path.unlink(missing_ok=True)
