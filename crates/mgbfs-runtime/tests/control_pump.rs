use mgbfs_runtime::{control_connection::ControlConnection, control_wire::ControlFrame};
use mgbfs_runtime::{
    control_pump::ControlPump,
    control_wire::{Action, Plane},
};
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};
fn pair() -> [ControlPump; 2] {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    [
        ControlPump::new(
            2,
            0,
            2,
            vec![None, Some(ControlConnection::new(server, 2, 0, 1).unwrap())],
        )
        .unwrap(),
        ControlPump::new(
            2,
            1,
            2,
            vec![Some(ControlConnection::new(client, 2, 1, 0).unwrap()), None],
        )
        .unwrap(),
    ]
}
fn commands(pumps: &mut [ControlPump; 2]) -> [ControlFrame; 2] {
    let mut out = [None; 2];
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        for (rank, pump) in pumps.iter_mut().enumerate() {
            pump.poll().unwrap();
            if out[rank].is_none() {
                out[rank] = pump.command().unwrap();
            }
        }
        if let [Some(a), Some(b)] = out {
            return [a, b];
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
#[test]
fn tcp_pump_drives_two_depths_with_alternating_source_and_empty_peer() {
    let mut pumps = pair();
    for depth in 0..2 {
        pumps[(1 - depth) as usize]
            .offer(Plane::Candidate, 7)
            .unwrap();
        for pump in &mut pumps {
            pump.close_source().unwrap();
        }
        let frames = commands(&mut pumps);
        assert!(frames
            .iter()
            .all(|x| x.action == Action::Begin && x.depth == depth && x.epoch == depth * 2));
        for pump in &mut pumps {
            pump.transfer_complete(depth * 2).unwrap();
            pump.consumed(depth * 2).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|x| x.action == Action::Finalize));
        for pump in &mut pumps {
            pump.finalized(true).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|x| x.action == Action::Publish && x.depth == depth + 1));
    }
}
#[test]
fn single_rank_pump_preserves_full_control_lifecycle_without_sockets() {
    let mut pump = ControlPump::new(1, 0, 2, vec![None]).unwrap();
    for depth in 0..2 {
        pump.offer(Plane::Candidate, 7).unwrap();
        pump.poll().unwrap();
        let begin = pump.command().unwrap().unwrap();
        assert_eq!(
            (begin.action, begin.depth, begin.epoch),
            (Action::Begin, depth, depth * 2)
        );
        pump.transfer_complete(begin.epoch).unwrap();
        pump.consumed(begin.epoch).unwrap();
        pump.close_source().unwrap();
        pump.poll().unwrap();
        assert_eq!(pump.command().unwrap().unwrap().action, Action::Finalize);
        pump.finalized(true).unwrap();
        pump.poll().unwrap();
        assert_eq!(pump.command().unwrap().unwrap().action, Action::Publish);
        assert!(pump.command().unwrap().is_none());
    }
}
