"""Byte accounting for eager and deferred distributed BFS wire records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WireFormat:
    """Payload widths; transport/message headers are intentionally excluded."""

    key_bytes: int
    parent_bytes: int
    move_bytes: int

    def __post_init__(self):
        if self.key_bytes <= 0 or self.parent_bytes <= 0 or self.move_bytes < 0:
            raise ValueError("key/parent widths must be positive and move nonnegative")


@dataclass(frozen=True)
class WireByteEstimate:
    eager_bytes: int
    two_phase_key_bytes: int
    two_phase_control_bytes: int
    two_phase_metadata_bytes: int
    two_phase_total_bytes: int
    two_phase_reduction_fraction: float


def estimate_wire_bytes(
    *,
    remote_candidates: int,
    remote_only_accepted: int,
    accept_bitmap_bytes: int,
    wire_format: WireFormat,
) -> WireByteEstimate:
    """Estimate payload bytes for eager-full and key/bitmap/metadata exchange.

    Two-phase exchange keeps source buffers alive, sends compact keys, receives
    one acceptance bitmap per source/destination buffer, and sends parent/move
    metadata only for accepted states that have no local producer at the owner.
    """

    values = (remote_candidates, remote_only_accepted, accept_bitmap_bytes)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("wire counts must be nonnegative integers")
    if remote_only_accepted > remote_candidates:
        raise ValueError("remote accepted count exceeds remote candidate count")
    if accept_bitmap_bytes > remote_candidates:
        raise ValueError("bitmap bytes exceed one byte per remote candidate")

    metadata_width = wire_format.parent_bytes + wire_format.move_bytes
    eager = remote_candidates * (wire_format.key_bytes + metadata_width)
    key_bytes = remote_candidates * wire_format.key_bytes
    metadata_bytes = remote_only_accepted * metadata_width
    two_phase = key_bytes + accept_bitmap_bytes + metadata_bytes
    reduction = 0.0 if eager == 0 else 1.0 - two_phase / eager
    return WireByteEstimate(
        eager_bytes=eager,
        two_phase_key_bytes=key_bytes,
        two_phase_control_bytes=accept_bitmap_bytes,
        two_phase_metadata_bytes=metadata_bytes,
        two_phase_total_bytes=two_phase,
        two_phase_reduction_fraction=reduction,
    )
