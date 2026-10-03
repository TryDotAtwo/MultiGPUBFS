//! Full-state archive assertions shared by physical route-bank GPU gates.
use mgbfs_core::hash::GemmHash;
use mgbfs_runtime::archive::{verify, Extent};
use std::sync::{Arc, Mutex};

pub struct MemoryExtent(pub Arc<Mutex<Vec<u8>>>);
impl Extent for MemoryExtent {
    fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().resize(bytes as usize, 0);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap()[offset as usize..offset as usize + bytes.len()]
            .copy_from_slice(bytes);
        Ok(bytes.len())
    }
    fn sync(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub fn assert_layers(bytes: &[u8], expected: &[Vec<Vec<u8>>], seed: [u8; 16]) {
    verify(bytes).unwrap();
    let width = expected[0][0].len();
    let hash = GemmHash::from_seed(width, seed).unwrap();
    let mut actual = vec![Vec::new(); expected.len()];
    let mut offset = 48;
    loop {
        let word = |at| u64::from_le_bytes(bytes[offset + at..offset + at + 8]
            .try_into().unwrap()) as usize;
        let (kind, depth, count, size) = (word(8), word(16), word(24), word(32));
        if kind == 3 { break; }
        if kind == 1 {
            let payload = &bytes[offset + 80..offset + 80 + size];
            for row in 0..count {
                let state = &payload[row * width..(row + 1) * width];
                assert_eq!(hash.hash(state).unwrap().to_le_bytes(),
                    payload[count * width + row * 16..count * width + (row + 1) * 16]);
                actual[depth].push(state.to_vec());
            }
        }
        offset += 112 + size;
    }
    for layer in &mut actual { layer.sort(); }
    assert_eq!(actual, expected);
}
