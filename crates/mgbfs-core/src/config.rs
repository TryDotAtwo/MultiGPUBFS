use crate::{hash::Hash128, matrix::MatrixGroup, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FrontierProfile {
    Dense,
    HashFirst,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OwnerBackend {
    CubSortMerge,
    BmmaBucket,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GenerationBackend {
    #[serde(rename = "CUTLASS_U8_SM75_V1")]
    CutlassU8Sm75V1,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HashBackend {
    #[serde(rename = "GEMM_U8_P32X4_V1")]
    GemmU8P32x4V1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub world_size: u32,
    pub shards_per_rank: u32,
    pub buckets_per_shard: u32,
    pub logical_owner_to_rank: Vec<u32>,
}
impl Topology {
    pub fn validate(&self) -> Result<()> {
        if !self.world_size.is_power_of_two()
            || !self.shards_per_rank.is_power_of_two()
            || !self.buckets_per_shard.is_power_of_two()
        {
            return Err("TOPOLOGY_POWER_OF_TWO".into());
        }
        if self.world_size.ilog2() + self.shards_per_rank.ilog2() + self.buckets_per_shard.ilog2()
            > 64
        {
            return Err("TOPOLOGY_PREFIX_WIDTH".into());
        }
        let mut ranks = self.logical_owner_to_rank.clone();
        ranks.sort_unstable();
        if ranks.len() != self.world_size as usize
            || ranks.iter().enumerate().any(|(i, &r)| i != r as usize)
        {
            return Err("RANK_MAP_PERMUTATION".into());
        }
        Ok(())
    }
    pub fn locate(&self, hash: Hash128) -> Result<(u32, u32, u32)> {
        self.validate()?;
        let owner_bits = self.world_size.ilog2();
        let shard_bits = self.shards_per_rank.ilog2();
        let bucket_bits = self.buckets_per_shard.ilog2();
        let prefix = hash.prefix(owner_bits + shard_bits + bucket_bits);
        let bucket = (prefix & (self.buckets_per_shard as u64 - 1)) as u32;
        let shard = ((prefix >> bucket_bits) & (self.shards_per_rank as u64 - 1)) as u32;
        let owner = (prefix >> (shard_bits + bucket_bits)) as usize;
        Ok((self.logical_owner_to_rank[owner], shard, bucket))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capacities {
    pub state_ring_records: u64,
    pub state_extent_descriptors: u64,
    pub layer_hash_records_per_arena: u64,
    pub next_bucket_capacity_records: u64,
    pub route_slot_records: u64,
    pub route_slot_count: u32,
    pub pinned_archive_slots: u32,
    pub pinned_archive_slot_bytes: u64,
    pub disk_extent_bytes_per_rank: u64,
    pub untouched_vram_reserve_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfigV1 {
    pub schema: u32,
    pub graph: MatrixGroup,
    pub seed: [u8; 16],
    pub topology: Topology,
    pub frontier_profile: FrontierProfile,
    pub local_pre_dedup: bool,
    pub owner_backend: OwnerBackend,
    pub generation_backend: GenerationBackend,
    pub hash_backend: HashBackend,
    pub parent_batch: u64,
    pub capacities: Capacities,
}
impl RunConfigV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema != 1 {
            return Err("CONFIG_SCHEMA".into());
        }
        self.graph.validate()?;
        self.topology.validate()?;
        let c = &self.capacities;
        let generated = self
            .parent_batch
            .checked_mul(self.graph.generators.len() as u64)
            .ok_or("CANDIDATE_COUNT_OVERFLOW")?;
        if self.parent_batch == 0 || generated > c.route_slot_records {
            return Err("ROUTE_SLOT_CAPACITY".into());
        }
        if c.state_ring_records == 0
            || c.state_extent_descriptors == 0
            || c.layer_hash_records_per_arena == 0
            || c.next_bucket_capacity_records == 0
            || c.route_slot_count < 2
            || c.pinned_archive_slots == 0
            || c.pinned_archive_slot_bytes == 0
            || c.disk_extent_bytes_per_rank == 0
        {
            return Err("ZERO_CAPACITY".into());
        }
        for (count, stride) in [
            (c.state_ring_records, self.graph.start.len() as u64),
            (c.state_extent_descriptors, 64),
            (c.layer_hash_records_per_arena, 16),
            (c.route_slot_records, 32),
            (c.pinned_archive_slot_bytes, c.pinned_archive_slots as u64),
        ] {
            count.checked_mul(stride).ok_or("BYTE_OVERFLOW")?;
        }
        (self.topology.shards_per_rank as u64)
            .checked_mul(self.topology.buckets_per_shard as u64)
            .and_then(|b| b.checked_mul(c.next_bucket_capacity_records))
            .and_then(|r| r.checked_mul(24))
            .ok_or("BUCKET_BYTE_OVERFLOW")?;
        Ok(())
    }
    pub fn digest(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(Sha256::digest(bytes).into())
    }
    pub fn fixture(modulus: u16) -> Result<Self> {
        Ok(Self {
            schema: 1,
            graph: MatrixGroup::unitriangular(4, modulus)?,
            seed: [0; 16],
            topology: Topology {
                world_size: 2,
                shards_per_rank: 64,
                buckets_per_shard: 256,
                logical_owner_to_rank: vec![0, 1],
            },
            frontier_profile: FrontierProfile::Dense,
            local_pre_dedup: true,
            owner_backend: OwnerBackend::CubSortMerge,
            generation_backend: GenerationBackend::CutlassU8Sm75V1,
            hash_backend: HashBackend::GemmU8P32x4V1,
            parent_batch: 16384,
            capacities: Capacities {
                state_ring_records: 1 << 20,
                state_extent_descriptors: 65536,
                layer_hash_records_per_arena: 1 << 20,
                next_bucket_capacity_records: 128,
                route_slot_records: 6 * 16384,
                route_slot_count: 3,
                pinned_archive_slots: 4,
                pinned_archive_slot_bytes: 1 << 24,
                disk_extent_bytes_per_rank: 1 << 30,
                untouched_vram_reserve_bytes: 1 << 30,
            },
        })
    }
}
