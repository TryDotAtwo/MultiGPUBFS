"""Runtime layouts: no generated code or per-pair compilation.

Symbols occupy 4/8/16/... bits; storage consists of aligned 64-bit words.
Orbit size is metadata, not a CUCO key or a device allocation requirement.
"""
import math


def symbol_bits(alphabet):
    if type(alphabet) is not int or alphabet < 1:
        raise ValueError('positive alphabet required')
    required = max(1, (alphabet-1).bit_length())
    bits = 4
    while bits < required:
        bits *= 2
    return bits


def state_layout(n, alphabet):
    if type(n) is not int or not 1 <= n <= 128:
        raise ValueError('state degree must be in 1..128')
    bits = symbol_bits(alphabet)
    words = (n*bits+63)//64
    return dict(bits_per_symbol=bits, bytes_per_state=words*8,
                alignment_bytes=8, uint64_words=words)


def orbit_layout(n, r):
    if not 2 <= n <= 128 or not 1 <= r <= n:
        raise ValueError('LRX shape')
    order = math.prod(range(r+1, n+1))
    bits = max(1, (order-1).bit_length())
    words = max(1, (order.bit_length()+63)//64)
    return dict(states=str(order), ordinal_bits=bits, uint64_words=words,
                bytes=8*words,
                order_words=[(order>>(64*i))&((1<<64)-1) for i in range(words)])
