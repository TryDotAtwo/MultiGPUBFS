//! Bounded setup handshake, before nonblocking control traffic.
use crate::control_connection::ControlConnection;
use mgbfs_core::Result;
use std::{
    io::{Read, Write},
    net::TcpStream,
    time::{Duration, Instant},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunIdentity {
    pub config_digest: [u8; 32],
    pub run_id: [u8; 16],
}
impl ControlConnection {
    pub fn accept_peer(
        mut stream: TcpStream,
        world: u32,
        identity: RunIdentity,
        timeout: Duration,
    ) -> Result<(u32, Self)> {
        let deadline = setup(&stream, world, timeout)?;
        let mut bytes = [0; 80];
        transfer(&mut stream, &mut bytes, deadline, false)?;
        let rank = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        if rank == 0 || rank >= world || bytes != hello(world, rank, identity) {
            return Err("CONTROL_HANDSHAKE_IDENTITY".into());
        }
        transfer(&mut stream, &mut hello(world, 0, identity), deadline, true)?;
        Ok((rank, Self::new(stream, world, 0, rank)?))
    }
    pub fn connect_peer(
        mut stream: TcpStream,
        world: u32,
        rank: u32,
        identity: RunIdentity,
        timeout: Duration,
    ) -> Result<Self> {
        if rank == 0 || rank >= world {
            return Err("CONTROL_HANDSHAKE_RANK".into());
        }
        let deadline = setup(&stream, world, timeout)?;
        transfer(
            &mut stream,
            &mut hello(world, rank, identity),
            deadline,
            true,
        )?;
        let mut bytes = [0; 80];
        transfer(&mut stream, &mut bytes, deadline, false)?;
        if bytes != hello(world, 0, identity) {
            return Err("CONTROL_HANDSHAKE_IDENTITY".into());
        }
        Self::new(stream, world, rank, 0)
    }
}

fn hello(world: u32, rank: u32, identity: RunIdentity) -> [u8; 80] {
    let mut bytes = [0; 80];
    bytes[..8].copy_from_slice(b"MGBHEL01");
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&world.to_le_bytes());
    bytes[16..20].copy_from_slice(&rank.to_le_bytes());
    bytes[24..56].copy_from_slice(&identity.config_digest);
    bytes[56..72].copy_from_slice(&identity.run_id);
    bytes
}

fn setup(stream: &TcpStream, world: u32, timeout: Duration) -> Result<Instant> {
    if world < 2 || timeout.is_zero() {
        return Err("CONTROL_HANDSHAKE_CONFIG".into());
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or("CONTROL_HANDSHAKE_TIMEOUT")?;
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    Ok(deadline)
}

// One shared deadline covers both directions, including fragmented transfers.
fn transfer(
    stream: &mut TcpStream,
    bytes: &mut [u8],
    deadline: Instant,
    write: bool,
) -> Result<()> {
    let mut offset = 0;
    while offset < bytes.len() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or("CONTROL_HANDSHAKE_TIMEOUT")?;
        let operation = if write {
            stream
                .set_write_timeout(Some(remaining))
                .map_err(|e| e.to_string())?;
            stream.write(&bytes[offset..])
        } else {
            stream
                .set_read_timeout(Some(remaining))
                .map_err(|e| e.to_string())?;
            stream.read(&mut bytes[offset..])
        };
        match operation {
            Ok(0) => return Err("CONTROL_HANDSHAKE_CLOSED".into()),
            Ok(n) => offset += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                return Err("CONTROL_HANDSHAKE_TIMEOUT".into());
            }
            Err(e) => return Err(format!("CONTROL_HANDSHAKE_IO: {e}")),
        }
    }
    if Instant::now() >= deadline {
        return Err("CONTROL_HANDSHAKE_TIMEOUT".into());
    }
    Ok(())
}
