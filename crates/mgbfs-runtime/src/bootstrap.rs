//! Single-node bootstrap record; publication and dispatcher integration pending.
use crate::control_handshake::RunIdentity;
use mgbfs_core::Result;
use std::net::{Ipv4Addr, SocketAddrV4};

pub struct BootstrapGroup {
    pub nccl_id: [u8; 128],
    pub peers: Vec<Option<crate::control_connection::ControlConnection>>,
}
pub fn rendezvous(
    path: &std::path::Path,
    rank: u32,
    world: u32,
    identity: RunIdentity,
    timeout: std::time::Duration,
    create_id: impl FnOnce() -> Result<[u8; 128]>,
) -> Result<BootstrapGroup> {
    if world == 0 || rank >= world || timeout.is_zero() {
        return Err("BOOTSTRAP_CONFIG".into());
    }
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .ok_or("BOOTSTRAP_TIMEOUT")?;
    let remaining = || {
        deadline
            .checked_duration_since(std::time::Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| "BOOTSTRAP_TIMEOUT".to_string())
    };
    if rank == 0 {
        let nccl_id = create_id()?;
        remaining()?;
        if world == 1 {
            return Ok(BootstrapGroup {
                nccl_id,
                peers: vec![None],
            });
        }
        let mut listener = BootstrapListener::bind(world, identity, nccl_id)?;
        listener.record().publish(path)?;
        let peers = listener.accept_all(remaining()?)?;
        return Ok(BootstrapGroup { nccl_id, peers });
    }
    loop {
        let left = remaining()?;
        match std::fs::metadata(path) {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::thread::sleep(left.min(std::time::Duration::from_millis(1)))
            }
            Err(e) => return Err(format!("BOOTSTRAP_METADATA: {e}")),
        }
    }
    let record = BootstrapRecord::read(path, world, identity)?;
    let mut peers = Vec::new();
    peers
        .try_reserve_exact(world as usize)
        .map_err(|_| "BOOTSTRAP_CAPACITY")?;
    peers.resize_with(world as usize, || None);
    peers[0] = Some(record.connect(rank, remaining()?)?);
    Ok(BootstrapGroup {
        nccl_id: record.nccl_id,
        peers,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapRecord {
    pub world: u32,
    pub identity: RunIdentity,
    pub endpoint: SocketAddrV4,
    pub nccl_id: [u8; 128],
}

pub struct BootstrapListener {
    listener: std::net::TcpListener,
    record: BootstrapRecord,
    admitted: Vec<bool>,
    failed: bool,
}
impl BootstrapListener {
    pub fn accept_all(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<Vec<Option<crate::control_connection::ControlConnection>>> {
        if self.failed || self.admitted[1..].iter().any(|x| *x) {
            return Err("BOOTSTRAP_GROUP_ALREADY_STARTED".into());
        }
        let result = (|| {
            let deadline = std::time::Instant::now()
                .checked_add(timeout)
                .ok_or("BOOTSTRAP_ACCEPT_TIMEOUT")?;
            let mut peers = Vec::new();
            peers
                .try_reserve_exact(self.record.world as usize)
                .map_err(|_| "BOOTSTRAP_CAPACITY")?;
            peers.resize_with(self.record.world as usize, || None);
            for _ in 1..self.record.world {
                let remaining = deadline
                    .checked_duration_since(std::time::Instant::now())
                    .filter(|d| !d.is_zero())
                    .ok_or("BOOTSTRAP_ACCEPT_TIMEOUT")?;
                let (rank, connection) = self.accept_next(remaining)?;
                peers[rank as usize] = Some(connection);
            }
            Ok(peers)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn bind(world: u32, identity: RunIdentity, nccl_id: [u8; 128]) -> Result<Self> {
        if world < 2 {
            return Err("BOOTSTRAP_WORLD".into());
        }
        let listener =
            std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let endpoint = match listener.local_addr().map_err(|e| e.to_string())? {
            std::net::SocketAddr::V4(addr) => addr,
            _ => return Err("BOOTSTRAP_ENDPOINT".into()),
        };
        let mut admitted = Vec::new();
        admitted
            .try_reserve_exact(world as usize)
            .map_err(|_| "BOOTSTRAP_CAPACITY")?;
        admitted.resize(world as usize, false);
        admitted[0] = true;
        Ok(Self {
            listener,
            record: BootstrapRecord {
                world,
                identity,
                endpoint,
                nccl_id,
            },
            admitted,
            failed: false,
        })
    }
    pub fn record(&self) -> &BootstrapRecord {
        &self.record
    }
    pub fn accept_next(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<(u32, crate::control_connection::ControlConnection)> {
        if self.failed {
            return Err("BOOTSTRAP_LISTENER_FAILED".into());
        }
        let result = (|| {
            let deadline = std::time::Instant::now()
                .checked_add(timeout)
                .ok_or("BOOTSTRAP_ACCEPT_TIMEOUT")?;
            loop {
                let remaining = deadline
                    .checked_duration_since(std::time::Instant::now())
                    .filter(|d| !d.is_zero())
                    .ok_or("BOOTSTRAP_ACCEPT_TIMEOUT")?;
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        let remaining = deadline
                            .checked_duration_since(std::time::Instant::now())
                            .filter(|d| !d.is_zero())
                            .ok_or("BOOTSTRAP_ACCEPT_TIMEOUT")?;
                        let (rank, connection) =
                            crate::control_connection::ControlConnection::accept_peer_admitted(
                                stream,
                                self.record.world,
                                self.record.identity,
                                remaining,
                                |rank| {
                                    if self.admitted[rank as usize] {
                                        Err("BOOTSTRAP_DUPLICATE_RANK".into())
                                    } else {
                                        Ok(())
                                    }
                                },
                            )?;
                        self.admitted[rank as usize] = true;
                        return Ok((rank, connection));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(remaining.min(std::time::Duration::from_millis(1)));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(format!("BOOTSTRAP_ACCEPT: {e}")),
                }
            }
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}
impl BootstrapRecord {
    pub fn connect(
        &self,
        rank: u32,
        timeout: std::time::Duration,
    ) -> Result<crate::control_connection::ControlConnection> {
        self.encode()?;
        if rank == 0 || rank >= self.world || timeout.is_zero() {
            return Err("BOOTSTRAP_CONNECT_CONFIG".into());
        }
        let deadline = std::time::Instant::now()
            .checked_add(timeout)
            .ok_or("BOOTSTRAP_CONNECT_TIMEOUT")?;
        let stream =
            std::net::TcpStream::connect_timeout(&std::net::SocketAddr::V4(self.endpoint), timeout)
                .map_err(|e| format!("BOOTSTRAP_CONNECT: {e}"))?;
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or("BOOTSTRAP_CONNECT_TIMEOUT")?;
        crate::control_connection::ControlConnection::connect_peer(
            stream,
            self.world,
            rank,
            self.identity,
            remaining,
        )
    }
    pub fn read(path: &std::path::Path, world: u32, identity: RunIdentity) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("BOOTSTRAP_OPEN: {e}"))?;
        let metadata = file
            .metadata()
            .map_err(|e| format!("BOOTSTRAP_METADATA: {e}"))?;
        if !metadata.is_file() || metadata.len() != 200 {
            return Err("BOOTSTRAP_LENGTH".into());
        }
        let mut bytes = [0; 200];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("BOOTSTRAP_READ: {e}"))?;
        if file
            .read(&mut [0; 1])
            .map_err(|e| format!("BOOTSTRAP_READ: {e}"))?
            != 0
        {
            return Err("BOOTSTRAP_LENGTH".into());
        }
        Self::decode(&bytes, world, identity)
    }
    /// Same-filesystem hard link publishes only a complete file, without replacing
    /// a prior run. No fallback on filesystems without hard-link support.
    /// The staging file is retained for diagnosis if publication fails.
    pub fn publish(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let bytes = self.encode()?;
        let mut name = path.as_os_str().to_os_string();
        name.push(".rank0.staging");
        let staging = std::path::PathBuf::from(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
            .map_err(|e| format!("BOOTSTRAP_STAGE: {e}"))?;
        file.write_all(&bytes)
            .map_err(|e| format!("BOOTSTRAP_WRITE: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("BOOTSTRAP_SYNC: {e}"))?;
        drop(file);
        std::fs::hard_link(&staging, path).map_err(|e| format!("BOOTSTRAP_PUBLISH: {e}"))?;
        std::fs::remove_file(staging).map_err(|e| format!("BOOTSTRAP_STAGE_REMOVE: {e}"))?;
        Ok(())
    }
    pub fn encode(&self) -> Result<[u8; 200]> {
        if self.world == 0 || !self.endpoint.ip().is_loopback() || self.endpoint.port() == 0 {
            return Err("BOOTSTRAP_CONFIG".into());
        }
        let mut bytes = [0; 200];
        bytes[..8].copy_from_slice(b"MGBBOOT1");
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.world.to_le_bytes());
        bytes[16..48].copy_from_slice(&self.identity.config_digest);
        bytes[48..64].copy_from_slice(&self.identity.run_id);
        bytes[64..68].copy_from_slice(&self.endpoint.ip().octets());
        bytes[68..70].copy_from_slice(&self.endpoint.port().to_le_bytes());
        bytes[72..].copy_from_slice(&self.nccl_id);
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8], world: u32, identity: RunIdentity) -> Result<Self> {
        if bytes.len() != 200
            || &bytes[..8] != b"MGBBOOT1"
            || bytes[8..12] != 1u32.to_le_bytes()
            || bytes[12..16] != world.to_le_bytes()
            || bytes[16..48] != identity.config_digest
            || bytes[48..64] != identity.run_id
            || bytes[70..72] != [0, 0]
        {
            return Err("BOOTSTRAP_FORMAT_OR_IDENTITY".into());
        }
        let result = Self {
            world,
            identity,
            endpoint: SocketAddrV4::new(
                Ipv4Addr::new(bytes[64], bytes[65], bytes[66], bytes[67]),
                u16::from_le_bytes(bytes[68..70].try_into().unwrap()),
            ),
            nccl_id: bytes[72..].try_into().unwrap(),
        };
        result.encode()?;
        Ok(result)
    }
}
