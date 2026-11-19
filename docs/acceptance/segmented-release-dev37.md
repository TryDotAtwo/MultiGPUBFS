# Experimental segmented32 and bitmap BFS dev37

The configurable per-bank segment maximum defaults to 2^32. It is a maximum,
not an allocation promise. The separate segment and 32-bit local row fields,
64-bit counts and strides, and overflow behavior were checked on RTX 3060.
Both modes remain opt-in; legacy remains the compatibility default.

Measured correctness and performance cover 2x RTX 3060 and particular graph
inputs. No installed dev37 package, Blackwell GPU, full S14, or single rank
holding more than 2^32 real resident states was accepted at release assembly.
A virtual-memory boundary probe validates the last row without allocating all
the surrounding state array. Capacity estimates are not runtime guarantees.

See docs/experimental-memory-backends.md and the acceptance JSON and logs.
