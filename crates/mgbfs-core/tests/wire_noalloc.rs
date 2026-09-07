use mgbfs_core::wire::{ExpectedFrame, FrameHeader, FrameKind};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[test]
fn frame_header_encode_and_decode_do_not_allocate_in_transport_loop() {
    let h = FrameHeader {
        kind: FrameKind::Dense,
        run_tag: 1,
        sequence: 2,
        batch: 3,
        depth: 4,
        source: 0,
        destination: 1,
        count: 17,
    };
    let e = ExpectedFrame {
        run_tag: 1,
        sequence: 2,
        batch: 3,
        depth: 4,
        source: 0,
        destination: 1,
        world: 2,
        kind: FrameKind::Dense,
        max_records: 17,
        max_payload: 1536,
        state_stride: 32,
    };
    ALLOCS.with(|n| n.set(0));
    TRACK.with(|t| t.set(true));
    let bytes = h.encode(32).unwrap();
    let decoded = FrameHeader::decode(&bytes, &e).unwrap();
    TRACK.with(|t| t.set(false));
    assert_eq!(decoded, h);
    assert_eq!(ALLOCS.with(Cell::get), 0);
}
