"""Time a depth range using runtime's per-created-layer timing contract."""

def measured_layer_window(layer_seconds, layer_count, first_depth, last_depth):
    """Return time for inclusive depths; item zero records creation of depth 1."""
    if len(layer_seconds) != layer_count - 1:
        raise ValueError("per-layer timing count disagrees with layer count")
    if not 1 <= first_depth <= last_depth < layer_count:
        raise ValueError("timed depths must be a nonempty range of completed layers")
    return sum(layer_seconds[first_depth-1:last_depth])
