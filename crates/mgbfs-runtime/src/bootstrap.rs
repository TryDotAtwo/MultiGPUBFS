//! Single-node bootstrap record; publication and dispatcher integration pending.
use crate::control_handshake::RunIdentity;
use mgbfs_core::Result;
use std::net::{Ipv4Addr, SocketAddrV4};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapRecord {
    pub world: u32,
    pub identity: RunIdentity,
    pub endpoint: SocketAddrV4,
    pub nccl_id: [u8; 128],
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
