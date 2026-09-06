use mgbfs_runtime::{
    control_outbox::ControlOutbox,
    control_wire::{Action, ControlFrame, Plane},
};
use std::io::{self, Write};
struct BudgetWriter {
    bytes: Vec<u8>,
    budget: usize,
}
impl Write for BudgetWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.budget == 0 {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let n = bytes.len().min(self.budget).min(7);
        self.bytes.extend_from_slice(&bytes[..n]);
        self.budget -= n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn ready(slot: u64) -> ControlFrame {
    ControlFrame {
        action: Action::Ready,
        rank: 1,
        depth: 0,
        epoch: 0,
        slot,
        plane: Plane::Candidate,
        source_rank: 0,
        fatal_code: 0,
        destination_rank: 0,
        payload_bytes: 0,
    }
}
#[test]
fn partial_writes_keep_queue_credit_and_exact_frame_order() {
    let mut queue = ControlOutbox::new(2, 2).unwrap();
    queue.enqueue(ready(3)).unwrap();
    queue.enqueue(ready(9)).unwrap();
    let mut writer = BudgetWriter {
        bytes: Vec::new(),
        budget: 17,
    };
    assert!(!queue.poll(&mut writer).unwrap());
    assert_eq!(writer.bytes.len(), 17);
    assert!(queue.enqueue(ready(11)).is_err());
    writer.budget = 111;
    assert!(
        !queue.poll(&mut writer).unwrap(),
        "one poll retires at most one frame"
    );
    assert_eq!(writer.bytes.len(), 64);
    queue.enqueue(ready(11)).unwrap();
    assert!(!queue.poll(&mut writer).unwrap());
    assert_eq!(writer.bytes.len(), 128);
    writer.budget = 64;
    assert!(queue.poll(&mut writer).unwrap());
    let slots: Vec<_> = writer
        .bytes
        .chunks_exact(64)
        .map(|x| ControlFrame::decode(x, 2).unwrap().slot)
        .collect();
    assert_eq!(slots, [3, 9, 11]);
}
#[test]
fn invalid_capacity_and_write_zero_do_not_silently_drop_frames() {
    assert!(ControlOutbox::new(2, 0).is_err());
    assert!(ControlOutbox::new(0, 1).is_err());
    assert!(ControlOutbox::new(2, usize::MAX).is_err());
    let mut queue = ControlOutbox::new(2, 1).unwrap();
    queue.enqueue(ready(0)).unwrap();
    let mut no_space: &mut [u8] = &mut [];
    assert!(queue.poll(&mut no_space).is_err());
    assert!(queue.enqueue(ready(1)).is_err());
    assert!(queue.poll(&mut Vec::new()).is_err());
}
