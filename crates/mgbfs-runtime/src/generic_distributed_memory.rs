//! Byte-exact admission for retained-history routing. This is admission,
//! not a measured throughput tuner. Skew is handled by immutable-source batch retry before owner mutation.
use crate::generic_memory::MAX_ARENA_CAPACITY;
use mgbfs_core::Result;
const MAX_ROLLING_CAPACITY: u32 = (u32::MAX - 3) / 3;
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GenericDistributedMemoryPlan {
    #[serde(default = "cuda_generator")]
    pub generator_backend: String,
    #[serde(default = "hash_history_algorithm")]
    pub history_algorithm: String,
    #[serde(default)]
    pub owner_lanes: u32,
    #[serde(default)]
    pub sorted_owner_bytes: u64,
    #[serde(default)]
    pub driver_graph_reserve_bytes: u64,
    #[serde(default)]
    pub parent_transport: bool,
    #[serde(default)]
    pub sort_candidates: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub compact_transport: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub compact_route_lookahead: bool,
    #[serde(
        default = "one_packet_slot",
        skip_serializing_if = "uses_one_packet_slot"
    )]
    pub compact_packet_slots: u32,
    #[serde(
        default = "one_packet_slot",
        skip_serializing_if = "uses_one_packet_slot"
    )]
    pub compact_receive_slots: u32,
    #[serde(default = "retained_history_layers")]
    pub history_layers: u32,
    pub state_bytes: u32,
    pub elements: u32,
    pub world: u32,
    pub shards: u32,
    pub capacity: u32,
    pub batch: u32,
    pub table_layout: String,
    pub queue_capacity: u32,
    pub table_slots: u64,
    pub generator_bytes: u64,
    pub device_bytes: u64,
}
fn retained_history_layers() -> u32 {
    1
}
fn is_false(value: &bool) -> bool {
    !*value
}
fn one_packet_slot() -> u32 {
    1
}
fn uses_one_packet_slot(value: &u32) -> bool {
    *value == 1
}
fn compact_transport_requested(world: u32) -> bool {
    world > 1
        && std::env::var_os("MGBFS_SEGMENT_PACKET_CONTROL_ADDR").is_some()
        && (std::env::var("MGBFS_EXPERIMENTAL_SEGMENTED24").as_deref() == Ok("1")
            || compact_path_requested())
}
pub(crate) fn compact_path_requested() -> bool {
    std::env::var("MGBFS_GENERIC_PATH_COMPACT").as_deref() == Ok("1")
        && std::env::var("MGBFS_GENERIC_PATH_APPEND_ONLY").as_deref() == Ok("1")
}
pub(crate) fn compact_producer_factor(world: u32) -> u32 {
    world.clamp(2, 4)
}
fn dense_queue_bank_count(world: u32) -> usize {
    if world == 1 {
        2
    } else {
        3
    }
}
fn transport_queue_bytes(
    world: u32,
    shards: u32,
    rows: u32,
    payload: u64,
    compact: bool,
    tx_slots: u32,
    rx_slots: u32,
) -> Option<u64> {
    let boxes = u64::from(world) * u64::from(shards);
    let row_bytes = 32 + payload;
    let rows = u64::from(rows);
    if compact {
        // Charge full wire widths and reserve decoded rows for each additional RX
        // window. Some codecs use the wire payload directly and need less memory.
        let wire_windows = u64::from(tx_slots) + u64::from(rx_slots);
        let extra_decode_windows = u64::from(rx_slots.checked_sub(1)?);
        let wire_headers = 8 * wire_windows * u64::from(shards).min(rows);
        ((wire_windows + extra_decode_windows + u64::from(compact_producer_factor(world))) * rows)
            .checked_mul(row_bytes)?
            .checked_add(wire_headers + 4 * boxes + 8 + 4 * u64::from(world))
    } else {
        let banks = dense_queue_bank_count(world) as u64;
        (banks * boxes).checked_mul(rows.checked_mul(row_bytes)?.checked_add(4)?)
    }
}
fn cuda_generator() -> String {
    "cuda".into()
}
fn hash_history_algorithm() -> String {
    "HASH".into()
}

#[derive(Clone, Copy)]
pub(crate) struct HistoryMemory {
    pub(crate) extra_row_bytes: u64,
    pub(crate) shared_path_queues: bool,
}
impl HistoryMemory {
    fn one_history(extra_row_bytes: u64) -> Self {
        Self {
            extra_row_bytes,
            shared_path_queues: false,
        }
    }

    pub(crate) fn required_bytes(self, plan: &GenericDistributedMemoryPlan) -> Result<u64> {
        let one = plan
            .device_bytes
            .checked_add(
                u64::from(plan.capacity)
                    .checked_mul(self.extra_row_bytes)
                    .ok_or("GENERIC_HISTORY_ADMISSION_BYTES")?,
            )
            .ok_or("GENERIC_HISTORY_ADMISSION_BYTES")?;
        if !self.shared_path_queues {
            return Ok(one);
        }
        let queues = plan.shared_path_queue_bytes()?;
        one.checked_sub(queues)
            .and_then(|other| one.checked_add(other))
            .ok_or_else(|| "NATIVE_PATH_DUAL_ARENA_BYTES".into())
    }
}

impl GenericDistributedMemoryPlan {
    pub fn new(
        elements: u32,
        world: u32,
        shards: u32,
        capacity: u32,
        batch: u32,
        generators: u32,
        generator_bytes: u64,
    ) -> Result<Self> {
        Self::with_storage(
            elements,
            world,
            shards,
            capacity,
            batch,
            generators,
            generator_bytes,
            8,
        )
    }
    pub fn with_storage(
        elements: u32,
        world: u32,
        shards: u32,
        capacity: u32,
        batch: u32,
        generators: u32,
        generator_bytes: u64,
        state_bytes: u32,
    ) -> Result<Self> {
        Self::with_storage_layout(
            elements,
            world,
            shards,
            capacity,
            batch,
            generators,
            generator_bytes,
            state_bytes,
            compact_transport_requested(world),
        )?
        .with_requested_compact_pipeline()
    }
    fn with_storage_layout(
        elements: u32,
        world: u32,
        shards: u32,
        capacity: u32,
        batch: u32,
        generators: u32,
        generator_bytes: u64,
        state_bytes: u32,
        compact_transport: bool,
    ) -> Result<Self> {
        if ![1, 8].contains(&state_bytes)
            || elements == 0
            || !(1..=128).contains(&world)
            || !(1..=4096).contains(&shards)
            || capacity == 0
            || capacity > MAX_ARENA_CAPACITY
            || batch == 0
            || batch > capacity
            || generators == 0
        {
            return Err("GENERIC_DISTRIBUTED_SHAPE".into());
        }
        if crate::generic_gemm::requested()?
            && u64::from(batch) * u64::from(if state_bytes == 1 { 1 } else { elements }) > 1048576
        {
            return Err("GEMM_BATCH_EXCEEDS_ADMISSION".into());
        }
        let full = batch
            .checked_mul(generators)
            .filter(|v| *v < 0x80000000)
            .ok_or("GENERIC_QUEUE_INDEX_RANGE")?;
        let boxes = u64::from(world) * u64::from(shards);
        let full = u64::from(full);
        // Compact queues hold a recipient's complete packet. Shards occupy
        // ranges inside that packet and do not divide its row capacity.
        let queue_target = if compact_transport {
            (full + u64::from(world) - 1) / u64::from(world)
        } else {
            ((full + boxes - 1) / boxes) * 4
        };
        let queue_capacity = full.min(queue_target.max(u64::from(generators))) as u32;
        if !compact_transport && u64::from(queue_capacity) * boxes >= 0x7fffffff {
            return Err("GENERIC_QUEUE_INDEX_RANGE".into());
        }
        // Logical shard hash regions share one exact table and overflow reserve.
        // Immutable pending origins include the source queue AND shard.
        let table_slots = u64::from(capacity)
            .checked_mul(2)
            .and_then(u64::checked_next_power_of_two)
            .ok_or("GENERIC_SHARD_SLOTS")?;
        let checked = || -> Option<u64> {
            let gather_rows = if crate::generic_memory::nearest_direct_rows() {
                1
            } else {
                capacity
            };
            let arena = u64::from(elements)
                .checked_mul(u64::from(capacity) + u64::from(gather_rows))?
                .checked_mul(u64::from(state_bytes))?
                .checked_add(u64::from(capacity) * 8)?;
            let tables = u64::from(table_slots).checked_mul(8)?;
            let payload = if state_bytes == 1 && elements <= 24 {
                0
            } else {
                u64::from(elements) * u64::from(state_bytes)
            };
            let queues = transport_queue_bytes(
                world,
                shards,
                queue_capacity,
                payload,
                compact_transport,
                1,
                1,
            )?;
            arena
                .checked_add(tables)?
                .checked_add(queues)?
                .checked_add(generator_bytes)?
                .checked_add(
                    98 + u64::from(world) * 20
                        + if state_bytes == 1 && elements <= 24 {
                            if world == 1 {
                                2
                            } else {
                                3
                            }
                        } else {
                            0
                        },
                )
        };
        let device_bytes = checked()
            .ok_or("GENERIC_DISTRIBUTED_BYTES_OVERFLOW")?
            .checked_add(crate::generic_gemm::workspace(elements, generators, batch)?)
            .ok_or("GEMM_WORKSPACE_OVERFLOW")?;
        Self {
            generator_backend: if crate::generic_gemm::requested()? {
                "gemm".into()
            } else {
                cuda_generator()
            },
            history_algorithm: hash_history_algorithm(),
            owner_lanes: 0,
            sorted_owner_bytes: 0,
            driver_graph_reserve_bytes: 0,
            parent_transport: false,
            sort_candidates: false,
            compact_transport,
            compact_route_lookahead: false,
            compact_packet_slots: 1,
            compact_receive_slots: 1,
            history_layers: 1,
            table_layout: "SHARD_HASH_REGIONS_SHARED_OVERFLOW".into(),
            state_bytes,
            elements,
            world,
            shards,
            capacity,
            batch,
            queue_capacity,
            table_slots,
            generator_bytes,
            device_bytes,
        }
        .with_parent_transport(match std::env::var("MGBFS_GENERIC_TRANSPORT").as_deref() {
            Ok("parent") => true,
            Ok("full") | Err(_) => false,
            _ => return Err("GENERIC_TRANSPORT_MODE".into()),
        })?
        .with_sort_candidates(match std::env::var("MGBFS_GENERIC_SORT").as_deref() {
            Ok("radix") => true,
            Ok("none") | Err(_) => false,
            _ => return Err("GENERIC_SORT_MODE".into()),
        })
    }
    pub fn automatic(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
    ) -> Result<Self> {
        Self::automatic_storage(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            8,
        )
    }
    pub fn automatic_storage(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
    ) -> Result<Self> {
        let reserve = crate::generic_memory::admission_reserve_bytes()?;
        let budget = free_bytes
            .checked_sub(reserve)
            .ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
        let ceiling = upper_bound
            .unwrap_or(u64::from(MAX_ARENA_CAPACITY))
            .min(u64::from(MAX_ARENA_CAPACITY)) as u32;
        if ceiling == 0 {
            return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());
        }
        // Reserve at most a quarter for predictable transport, then admit the
        // largest history arena with the actual bounded queues included.
        let ceiling_batch = ceiling
            .min((1 << 20) / generators.max(1))
            .min(if crate::generic_gemm::requested()? {
                (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
            } else {
                u32::MAX
            })
            .max(1);
        let minimum = Self::with_storage(
            elements,
            world,
            shards,
            1,
            1,
            generators,
            generator_bytes,
            state_bytes,
        )?;
        if minimum.device_bytes > budget {
            return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());
        }
        let queue_budget = (budget / 4).max(minimum.transport_bytes());
        let mut lo = 1;
        let mut hi = ceiling_batch;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            let p = Self::with_storage(
                elements,
                world,
                shards,
                mid,
                mid,
                generators,
                generator_bytes,
                state_bytes,
            )?;
            if p.transport_bytes() <= queue_budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Self::automatic_storage_fixed_batch(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            lo,
        )
    }
    pub fn automatic_storage_fixed_batch(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        batch_target: u32,
    ) -> Result<Self> {
        if batch_target == 0 {
            return Err("GENERIC_BATCH_TARGET".into());
        }
        let reserve = crate::generic_memory::admission_reserve_bytes()?;
        let budget = free_bytes
            .checked_sub(reserve)
            .ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
        let ceiling = upper_bound
            .unwrap_or(u64::from(MAX_ARENA_CAPACITY))
            .min(u64::from(MAX_ARENA_CAPACITY)) as u32;
        if ceiling == 0 {
            return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());
        }
        if Self::with_storage(
            elements,
            world,
            shards,
            1,
            1,
            generators,
            generator_bytes,
            state_bytes,
        )?
        .device_bytes
            > budget
        {
            return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());
        }
        let mut low = 1;
        let mut high = ceiling;
        while low < high {
            let mid = low + (high - low + 1) / 2;
            let p = Self::with_storage(
                elements,
                world,
                shards,
                mid,
                mid.min(batch_target)
                    .min(if crate::generic_gemm::requested()? {
                        (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
                    } else {
                        u32::MAX
                    }),
                generators,
                generator_bytes,
                state_bytes,
            )?;
            if p.device_bytes <= budget {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        Self::with_storage(
            elements,
            world,
            shards,
            low,
            low.min(batch_target)
                .min(if crate::generic_gemm::requested()? {
                    (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
                } else {
                    u32::MAX
                }),
            generators,
            generator_bytes,
            state_bytes,
        )
    }
    fn with_requested_compact_pipeline(self) -> Result<Self> {
        let enabled = self.compact_transport
            && std::env::var("MGBFS_SEGMENT_PACKET_ROUTE_LOOKAHEAD").as_deref() == Ok("1");
        let slots = match std::env::var("MGBFS_SEGMENT_PACKET_RING_SLOTS") {
            Err(std::env::VarError::NotPresent) => 1,
            Ok(value) if value == "1" => 1,
            Ok(value) if value == "3" => 3,
            _ => return Err("MGBFS_SEGMENT_PACKET_RING_SLOTS_REQUIRES_1_OR_3".into()),
        };
        let receive_slots = match std::env::var("MGBFS_SEGMENT_PACKET_RX_SLOTS") {
            Err(std::env::VarError::NotPresent) => 1,
            Ok(value) if value == "1" => 1,
            Ok(value) if value == "3" => 3,
            _ => return Err("MGBFS_SEGMENT_PACKET_RX_SLOTS_REQUIRES_1_OR_3".into()),
        };
        self.with_compact_lookahead(enabled)?
            .with_compact_packet_slots(slots)?
            .with_compact_receive_slots(receive_slots)
    }
    fn with_compact_packet_slots(mut self, slots: u32) -> Result<Self> {
        if !matches!(slots, 1 | 3) {
            return Err("MGBFS_SEGMENT_PACKET_RING_SLOTS_REQUIRES_1_OR_3".into());
        }
        if slots == 3 && (!self.compact_transport || self.world < 2) {
            return Err("SEGMENT_PACKET_RING_REQUIRES_COMPACT_TRANSPORT".into());
        }
        if slots == self.compact_packet_slots {
            return Ok(self);
        }
        let old = self.transport_bytes();
        self.compact_packet_slots = slots;
        self.device_bytes = self
            .device_bytes
            .checked_sub(old)
            .and_then(|other| other.checked_add(self.transport_bytes()))
            .ok_or("SEGMENT_PACKET_RING_TRANSPORT_BYTES")?;
        Ok(self)
    }
    fn with_compact_receive_slots(mut self, slots: u32) -> Result<Self> {
        if !matches!(slots, 1 | 3) {
            return Err("MGBFS_SEGMENT_PACKET_RX_SLOTS_REQUIRES_1_OR_3".into());
        }
        if slots == 3 && self.compact_packet_slots != 3 {
            return Err("SEGMENT_PACKET_RX_RING_REQUIRES_TX_RING".into());
        }
        if slots == self.compact_receive_slots {
            return Ok(self);
        }
        let old = self.transport_bytes();
        self.compact_receive_slots = slots;
        self.device_bytes = self
            .device_bytes
            .checked_sub(old)
            .and_then(|other| other.checked_add(self.transport_bytes()))
            .ok_or("SEGMENT_PACKET_RX_RING_TRANSPORT_BYTES")?;
        Ok(self)
    }
    fn with_compact_lookahead(mut self, enabled: bool) -> Result<Self> {
        if enabled && !self.compact_transport {
            return Err("GENERIC_LOOKAHEAD_REQUIRES_COMPACT_TRANSPORT".into());
        }
        if enabled == self.compact_route_lookahead {
            return Ok(self);
        }
        let old = self.transport_bytes();
        self.compact_route_lookahead = enabled;
        self.device_bytes = self
            .device_bytes
            .checked_sub(old)
            .and_then(|bytes| bytes.checked_add(self.transport_bytes()))
            .ok_or("GENERIC_LOOKAHEAD_TRANSPORT_BYTES")?;
        Ok(self)
    }
    fn alternate_producer_bytes(&self) -> u64 {
        if !self.compact_route_lookahead {
            return 0;
        }
        let rows = u64::from(self.queue_capacity) * u64::from(compact_producer_factor(self.world));
        rows * (32 + self.queue_payload_bytes() as u64)
            + 4 * u64::from(self.world) * u64::from(self.shards)
            + 8
    }
    pub fn transport_bytes(&self) -> u64 {
        transport_queue_bytes(
            self.world,
            self.shards,
            self.queue_capacity,
            self.queue_payload_bytes() as u64,
            self.compact_transport,
            self.compact_packet_slots,
            self.compact_receive_slots,
        )
        .unwrap_or(u64::MAX)
        .saturating_add(if self.packed_candidates() || self.parent_transport {
            if self.world == 1 {
                2
            } else {
                3
            }
        } else {
            0
        })
        .saturating_add(self.parent_cache_bytes())
        .saturating_add(self.sort_cache_bytes())
        .saturating_add(self.alternate_producer_bytes())
    }
    pub fn packed_candidates(&self) -> bool {
        self.state_bytes == 1 && self.elements <= 24
    }
    pub fn queue_payload_bytes(&self) -> usize {
        if self.packed_candidates() || self.parent_transport {
            0
        } else {
            self.elements as usize * self.state_bytes as usize
        }
    }
    pub(crate) fn dense_queue_bank_count(&self) -> usize {
        dense_queue_bank_count(self.world)
    }
    pub(crate) fn dense_queue_bank_extents(&self) -> [usize; 3] {
        let boxes = self.world as usize * self.shards as usize;
        let rows = boxes * self.queue_capacity as usize;
        [
            rows * 32,
            boxes * 4,
            (rows * self.queue_payload_bytes()).max(1),
        ]
    }
    pub(crate) fn shared_path_queue_bytes(&self) -> Result<u64> {
        if self.history_layers != 1
            || self.history_algorithm != "HASH"
            || self.parent_transport
            || self.sort_candidates
            || self.packed_candidates()
        {
            return Err("NATIVE_PATH_SHARED_QUEUE_LAYOUT".into());
        }
        if self.compact_transport {
            return Ok(self.transport_bytes());
        }
        Ok(self.dense_queue_bank_count() as u64
            * self
                .dense_queue_bank_extents()
                .into_iter()
                .map(|extent| extent as u64)
                .sum::<u64>())
    }
    pub fn parent_cache_bytes(&self) -> u64 {
        if self.parent_transport {
            u64::from(self.world + 2)
                * u64::from(self.batch)
                * u64::from(self.elements)
                * u64::from(self.state_bytes)
                + u64::from(self.world) * 12
        } else {
            0
        }
    }
    pub fn sort_cache_bytes(&self) -> u64 {
        if self.sort_candidates {
            u64::from(self.shards)
                * (u64::from(self.world) * u64::from(self.queue_capacity) * 88 + 65536)
        } else {
            0
        }
    }
    pub fn with_sort_candidates(mut self, enabled: bool) -> Result<Self> {
        if enabled == self.sort_candidates {
            return Ok(self);
        }
        let old = self.transport_bytes();
        self.sort_candidates = enabled;
        let new = self.transport_bytes();
        self.device_bytes = self
            .device_bytes
            .checked_sub(old)
            .and_then(|b| b.checked_add(new))
            .ok_or("GENERIC_SORT_TRANSPORT_BYTES")?;
        Ok(self)
    }
    pub fn with_parent_transport(mut self, enabled: bool) -> Result<Self> {
        let enabled = enabled && !self.packed_candidates();
        if enabled == self.parent_transport {
            return Ok(self);
        }
        let old = self.transport_bytes();
        self.parent_transport = enabled;
        let new = self.transport_bytes();
        self.device_bytes = self
            .device_bytes
            .checked_sub(old)
            .and_then(|b| b.checked_add(new))
            .ok_or("GENERIC_PARENT_TRANSPORT_BYTES")?;
        Ok(self)
    }

    pub fn with_storage_history(
        elements: u32,
        world: u32,
        shards: u32,
        capacity: u32,
        batch: u32,
        generators: u32,
        generator_bytes: u64,
        state_bytes: u32,
        history_layers: u32,
    ) -> Result<Self> {
        Self::with_storage_history_layout(
            elements,
            world,
            shards,
            capacity,
            batch,
            generators,
            generator_bytes,
            state_bytes,
            history_layers,
            compact_transport_requested(world),
        )?
        .with_requested_compact_pipeline()
    }
    fn with_storage_history_layout(
        elements: u32,
        world: u32,
        shards: u32,
        capacity: u32,
        batch: u32,
        generators: u32,
        generator_bytes: u64,
        state_bytes: u32,
        history_layers: u32,
        compact_transport: bool,
    ) -> Result<Self> {
        if history_layers == 3 && capacity > MAX_ROLLING_CAPACITY {
            return Err("GENERIC_ROLLING_ROW_RANGE".into());
        }
        if ![1, 3].contains(&history_layers) {
            return Err("GENERIC_HISTORY_GEOMETRY".into());
        }
        let mut p = Self::with_storage_layout(
            elements,
            world,
            shards,
            capacity,
            batch,
            generators,
            generator_bytes,
            state_bytes,
            compact_transport,
        )?;
        if history_layers == 3 {
            let old = p.table_slots;
            let slots = u64::from(capacity)
                .checked_mul(6)
                .and_then(u64::checked_next_power_of_two)
                .ok_or("GENERIC_ROLLING_TABLE")?;
            p.device_bytes = p
                .device_bytes
                .checked_add(
                    u64::from(elements)
                        * u64::from(capacity + capacity.min(1000))
                        * u64::from(state_bytes)
                        + u64::from(capacity) * 24
                        + u64::from(slots - old) * 8,
                )
                .ok_or("GENERIC_ROLLING_BYTES")?;
            p.table_slots = slots;
            p.table_layout = "THREE_BANK_SHARED_TOMBSTONE_HISTORY".into();
        }
        p.history_layers = history_layers;
        Ok(p)
    }
    pub fn automatic_storage_history(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        history_layers: u32,
    ) -> Result<Self> {
        Self::automatic_storage_history_with_row_bytes(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            history_layers,
            0,
        )
    }
    pub fn automatic_storage_history_with_row_bytes(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        history_layers: u32,
        extra_row_bytes: u64,
    ) -> Result<Self> {
        Self::automatic_storage_with_history_memory(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            history_layers,
            HistoryMemory::one_history(extra_row_bytes),
        )
    }
    pub(crate) fn automatic_storage_with_history_memory(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        history_layers: u32,
        memory: HistoryMemory,
    ) -> Result<Self> {
        let reserve = crate::generic_memory::admission_reserve_bytes()?;
        let budget = free_bytes
            .checked_sub(reserve)
            .ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
        let ceiling = upper_bound
            .unwrap_or(u64::from(MAX_ARENA_CAPACITY))
            .min(u64::from(if history_layers == 3 {
                MAX_ROLLING_CAPACITY
            } else {
                MAX_ARENA_CAPACITY
            })) as u32;
        if ceiling == 0 {
            return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());
        }
        // Reserve at most a quarter for predictable transport, then admit the
        // largest history arena with the actual bounded queues included.
        let ceiling_batch = ceiling
            .min((1 << 20) / generators.max(1))
            .min(if crate::generic_gemm::requested()? {
                (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
            } else {
                u32::MAX
            })
            .max(1);
        let minimum = Self::with_storage_history(
            elements,
            world,
            shards,
            1,
            1,
            generators,
            generator_bytes,
            state_bytes,
            history_layers,
        )?;
        if memory.required_bytes(&minimum)? > budget {
            return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());
        }
        // Keep the prior per-direction queue budget; reclaim the removed copy
        // for visited and frontier storage instead of increasing the batch.
        let queue_budget =
            (budget / if memory.shared_path_queues { 8 } else { 4 }).max(minimum.transport_bytes());
        let mut lo = 1;
        let mut hi = ceiling_batch;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            let p = Self::with_storage_history(
                elements,
                world,
                shards,
                mid,
                mid,
                generators,
                generator_bytes,
                state_bytes,
                history_layers,
            )?;
            if p.transport_bytes() <= queue_budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Self::automatic_storage_fixed_batch_with_history_memory(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            lo,
            history_layers,
            memory,
        )
    }
    pub fn automatic_storage_fixed_batch_history(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        batch_target: u32,
        history_layers: u32,
    ) -> Result<Self> {
        Self::automatic_storage_fixed_batch_history_with_row_bytes(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            batch_target,
            history_layers,
            0,
        )
    }
    pub fn automatic_storage_fixed_batch_history_with_row_bytes(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        batch_target: u32,
        history_layers: u32,
        extra_row_bytes: u64,
    ) -> Result<Self> {
        Self::automatic_storage_fixed_batch_with_history_memory(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            free_bytes,
            upper_bound,
            state_bytes,
            batch_target,
            history_layers,
            HistoryMemory::one_history(extra_row_bytes),
        )
    }
    fn automatic_storage_fixed_batch_with_history_memory(
        elements: u32,
        world: u32,
        shards: u32,
        generators: u32,
        generator_bytes: u64,
        free_bytes: u64,
        upper_bound: Option<u64>,
        state_bytes: u32,
        batch_target: u32,
        history_layers: u32,
        memory: HistoryMemory,
    ) -> Result<Self> {
        if batch_target == 0 {
            return Err("GENERIC_BATCH_TARGET".into());
        }
        let reserve = crate::generic_memory::admission_reserve_bytes()?;
        let budget = free_bytes
            .checked_sub(reserve)
            .ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
        let ceiling = upper_bound
            .unwrap_or(u64::from(MAX_ARENA_CAPACITY))
            .min(u64::from(if history_layers == 3 {
                MAX_ROLLING_CAPACITY
            } else {
                MAX_ARENA_CAPACITY
            })) as u32;
        if ceiling == 0 {
            return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());
        }
        if memory.required_bytes(&Self::with_storage_history(
            elements,
            world,
            shards,
            1,
            1,
            generators,
            generator_bytes,
            state_bytes,
            history_layers,
        )?)? > budget
        {
            return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());
        }
        let mut low = 1;
        let mut high = ceiling;
        while low < high {
            let mid = low + (high - low + 1) / 2;
            let p = Self::with_storage_history(
                elements,
                world,
                shards,
                mid,
                mid.min(batch_target)
                    .min(if crate::generic_gemm::requested()? {
                        (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
                    } else {
                        u32::MAX
                    }),
                generators,
                generator_bytes,
                state_bytes,
                history_layers,
            )?;
            if memory.required_bytes(&p)? <= budget {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        Self::with_storage_history(
            elements,
            world,
            shards,
            low,
            low.min(batch_target)
                .min(if crate::generic_gemm::requested()? {
                    (1 << 20) / if state_bytes == 1 { 1 } else { elements.max(1) }
                } else {
                    u32::MAX
                }),
            generators,
            generator_bytes,
            state_bytes,
            history_layers,
        )
    }
    pub fn with_sorted_admission(
        mut self,
        lanes: u32,
        owner_bytes: u64,
        driver_bytes: u64,
    ) -> Result<Self> {
        if self.history_algorithm != "HASH"
            || lanes == 0
            || lanes > self.shards
            || lanes > 8
            || owner_bytes == 0
            || driver_bytes
                < (128u64 << 20)
                    .max(u64::from(self.shards) * u64::from(self.history_layers) * 65536)
        {
            return Err("SORTED_PLAN_ADMISSION".into());
        }
        self = self.with_sort_candidates(false)?;
        let obsolete = u64::from(self.table_slots) * 8
            + if self.history_layers == 3 {
                u64::from(self.capacity) * 24
            } else {
                0
            };
        // Buffer::new(0) has a one-byte allocation for the unused table handle.
        self.device_bytes = self
            .device_bytes
            .checked_sub(obsolete)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_add(owner_bytes))
            .and_then(|v| v.checked_add(driver_bytes))
            .ok_or("SORTED_PLAN_BYTES")?;
        self.table_slots = 0;
        self.table_layout = "SORTED_EPOCH_RUNS_SHARED_CREDITS".into();
        self.history_algorithm = "SORTED_RUNS".into();
        self.owner_lanes = lanes;
        self.sorted_owner_bytes = owner_bytes;
        self.driver_graph_reserve_bytes = driver_bytes;
        Ok(self)
    }
    #[cfg(feature = "cuda")]
    pub fn with_sorted_for_graph(
        self,
        graph: &mgbfs_core::graph_definition::GraphDefinitionV2,
        device: i32,
        lanes: u32,
    ) -> Result<Self> {
        use crate::generic_sorted_native::{Input, OwnerShape};
        use mgbfs_core::graph_definition::GraphAction;
        if graph.start.len() != self.elements as usize {
            return Err("SORTED_GRAPH_SHAPE".into());
        }
        let (kind, rows, cols) = match &graph.action {
            GraphAction::Permutation { degree, .. } => (0, *degree, 1),
            GraphAction::Matrix { rows, cols, .. } => (1, *rows, *cols),
        };
        let input = Input {
            elements: self.elements,
            world: self.world,
            shards: self.shards,
            queue_capacity: self.queue_capacity,
            state_bytes: self.state_bytes,
            transport: if self.packed_candidates() {
                1
            } else if self.parent_transport {
                2
            } else {
                0
            },
            kind,
            rows,
            cols,
            generators: graph.generator_count() as u32,
            parent_stride: self.batch,
            hash_bits: 64,
            ..Default::default()
        };
        let shape = OwnerShape::query(&input, self.capacity, self.history_layers, lanes, device)?;
        let reserve =
            (128u64 << 20).max(u64::from(self.shards) * u64::from(self.history_layers) * 65536);
        self.with_sorted_admission(lanes, shape.allocated_bytes as u64, reserve)
    }
    #[cfg(feature = "cuda")]
    pub fn automatic_sorted_for_graph(
        graph: &mgbfs_core::graph_definition::GraphDefinitionV2,
        device: i32,
        world: u32,
        shards: u32,
        generator_bytes: u64,
        free: u64,
        upper: Option<u64>,
        bytes: u32,
        batch_target: u32,
        banks: u32,
        lanes: u32,
    ) -> Result<Self> {
        if batch_target == 0 {
            return Err("SORTED_BATCH_TARGET".into());
        }
        let budget = free
            .checked_sub(crate::generic_memory::admission_reserve_bytes()?)
            .ok_or("SORTED_ADMISSION_HEADROOM")?;
        let ceiling = upper
            .unwrap_or(u64::from(MAX_ARENA_CAPACITY))
            .min(u64::from(if banks == 3 {
                MAX_ROLLING_CAPACITY
            } else {
                MAX_ARENA_CAPACITY
            })) as u32;
        if ceiling == 0 {
            return Err("SORTED_EMPTY_BOUND".into());
        }
        let candidate = |cap: u32| -> Result<Self> {
            Self::with_storage_history(
                graph.start.len() as u32,
                world,
                shards,
                cap,
                cap.min(batch_target),
                graph.generator_count() as u32,
                generator_bytes,
                bytes,
                banks,
            )?
            .with_sorted_for_graph(graph, device, lanes)
        };
        if candidate(1)?.device_bytes > budget {
            return Err("SORTED_ADMISSION_NO_CAPACITY".into());
        }
        let (mut low, mut high) = (1, ceiling);
        while low < high {
            let mid = low + (high - low + 1) / 2;
            match candidate(mid) {
                Ok(p) if p.device_bytes <= budget => low = mid,
                _ => high = mid - 1,
            }
        }
        candidate(low)
    }
    pub fn validate(&self, generators: u32) -> Result<()> {
        let mut expected = Self::with_storage_history_layout(
            self.elements,
            self.world,
            self.shards,
            self.capacity,
            self.batch,
            generators,
            self.generator_bytes,
            self.state_bytes,
            self.history_layers,
            self.compact_transport,
        )?
        .with_parent_transport(self.parent_transport)?
        .with_sort_candidates(self.sort_candidates)?
        .with_compact_lookahead(self.compact_route_lookahead)?
        .with_compact_packet_slots(self.compact_packet_slots)?
        .with_compact_receive_slots(self.compact_receive_slots)?;
        if self.history_algorithm == "SORTED_RUNS" {
            expected = expected.with_sorted_admission(
                self.owner_lanes,
                self.sorted_owner_bytes,
                self.driver_graph_reserve_bytes,
            )?;
        }
        if *self != expected {
            return Err("GENERIC_DISTRIBUTED_PLAN_MUTATED".into());
        }
        Ok(())
    }
}

/// Independent per-rank admission with matched transport geometry and exact
/// weighted hash intervals. No runtime state crosses the host planner.
pub fn heterogeneous_plans(
    elements: u32,
    shards: u32,
    generators: u32,
    generator_bytes: u64,
    free: &[u64],
    upper: Option<u64>,
    state_bytes: u32,
) -> Result<(Vec<GenericDistributedMemoryPlan>, Vec<u64>)> {
    if free.is_empty() || free.len() > 128 {
        return Err("GENERIC_HETEROGENEOUS_WORLD".into());
    }
    let world = free.len() as u32;
    let mut admitted = Vec::new();
    for &bytes in free {
        admitted.push(GenericDistributedMemoryPlan::automatic_storage(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            bytes,
            upper,
            state_bytes,
        )?);
    }
    let batch = admitted.iter().map(|p| p.batch).min().unwrap();
    let mut plans = Vec::new();
    for &bytes in free {
        plans.push(GenericDistributedMemoryPlan::automatic_storage_fixed_batch(
            elements,
            world,
            shards,
            generators,
            generator_bytes,
            bytes,
            upper,
            state_bytes,
            batch,
        )?);
    }
    let total: u64 = plans.iter().map(|p| u64::from(p.capacity)).sum();
    let mut cuts = vec![0];
    let mut cumulative = 0u64;
    for p in &plans {
        cumulative += u64::from(p.capacity);
        cuts.push(
            ((u128::from(cumulative) * (1u128 << 32) + u128::from(total) - 1) / u128::from(total))
                as u64,
        );
    }
    Ok((plans, cuts))
}

pub fn heterogeneous_plans_history(
    elements: u32,
    shards: u32,
    generators: u32,
    generator_bytes: u64,
    free: &[u64],
    upper: Option<u64>,
    state_bytes: u32,
    history_layers: u32,
) -> Result<(Vec<GenericDistributedMemoryPlan>, Vec<u64>)> {
    heterogeneous_plans_history_with_row_bytes(
        elements,
        shards,
        generators,
        generator_bytes,
        free,
        upper,
        state_bytes,
        history_layers,
        0,
    )
}
pub fn heterogeneous_plans_history_with_row_bytes(
    elements: u32,
    shards: u32,
    generators: u32,
    generator_bytes: u64,
    free: &[u64],
    upper: Option<u64>,
    state_bytes: u32,
    history_layers: u32,
    extra_row_bytes: u64,
) -> Result<(Vec<GenericDistributedMemoryPlan>, Vec<u64>)> {
    heterogeneous_plans_with_history_memory(
        elements,
        shards,
        generators,
        generator_bytes,
        free,
        upper,
        state_bytes,
        history_layers,
        HistoryMemory::one_history(extra_row_bytes),
    )
}
pub(crate) fn heterogeneous_plans_with_history_memory(
    elements: u32,
    shards: u32,
    generators: u32,
    generator_bytes: u64,
    free: &[u64],
    upper: Option<u64>,
    state_bytes: u32,
    history_layers: u32,
    memory: HistoryMemory,
) -> Result<(Vec<GenericDistributedMemoryPlan>, Vec<u64>)> {
    if free.is_empty() || free.len() > 128 {
        return Err("GENERIC_HETEROGENEOUS_WORLD".into());
    }
    let world = free.len() as u32;
    let mut admitted = Vec::new();
    for &bytes in free {
        admitted.push(
            GenericDistributedMemoryPlan::automatic_storage_with_history_memory(
                elements,
                world,
                shards,
                generators,
                generator_bytes,
                bytes,
                upper,
                state_bytes,
                history_layers,
                memory,
            )?,
        );
    }
    let batch = admitted.iter().map(|p| p.batch).min().unwrap();
    let mut plans = Vec::new();
    for &bytes in free {
        plans.push(
            GenericDistributedMemoryPlan::automatic_storage_fixed_batch_with_history_memory(
                elements,
                world,
                shards,
                generators,
                generator_bytes,
                bytes,
                upper,
                state_bytes,
                batch,
                history_layers,
                memory,
            )?,
        );
    }
    let total: u64 = plans.iter().map(|p| u64::from(p.capacity)).sum();
    let mut cuts = vec![0];
    let mut cumulative = 0u64;
    for p in &plans {
        cumulative += u64::from(p.capacity);
        cuts.push(
            ((u128::from(cumulative) * (1u128 << 32) + u128::from(total) - 1) / u128::from(total))
                as u64,
        );
    }
    Ok((plans, cuts))
}

#[cfg(test)]
mod packet_plan_tests {
    use super::{GenericDistributedMemoryPlan as Plan, HistoryMemory};

    #[test]
    fn compact_payload_does_not_require_dense_queue_indices() {
        let legacy = Plan::with_storage_layout(12, 128, 4096, 1, 1, 4096, 0, 1, false);
        assert_eq!(legacy.unwrap_err(), "GENERIC_QUEUE_INDEX_RANGE");
        let compact = Plan::with_storage_layout(12, 128, 4096, 1, 1, 4096, 0, 1, true).unwrap();
        assert_eq!(compact.queue_capacity, 4096);
        assert_eq!(
            compact.transport_bytes(),
            6 * 4096 * 32 + 16 * 4096 + 4 * 128 * 4096 + 8 + 4 * 128 + 3
        );
        compact.validate(4096).unwrap();
    }

    #[test]
    fn lookahead_reserves_an_alternate_producer_at_the_same_batch() {
        let single = Plan::with_storage_layout(12, 8, 16, 65536, 8192, 11, 0, 1, true).unwrap();
        let double = single.clone().with_compact_lookahead(true).unwrap();
        let slot_bytes =
            4 * u64::from(single.queue_capacity) * (32 + single.queue_payload_bytes() as u64)
                + 4 * 8 * 16
                + 8;
        assert_eq!(double.queue_capacity, single.queue_capacity);
        assert_eq!(double.device_bytes - single.device_bytes, slot_bytes);
        let serialized = serde_json::to_string(&double).unwrap();
        serde_json::from_str::<Plan>(&serialized)
            .unwrap()
            .validate(11)
            .unwrap();
    }

    #[test]
    fn packet_ring_reserves_two_additional_tx_windows() {
        let single = Plan::with_storage_layout(40, 8, 16, 65_536, 8_192, 11, 0, 8, true).unwrap();
        let ring = single.clone().with_compact_packet_slots(3).unwrap();
        let window_bytes = u64::from(single.queue_capacity)
            * (32 + single.queue_payload_bytes() as u64)
            + 8 * u64::from(single.shards.min(single.queue_capacity));
        assert_eq!(ring.device_bytes - single.device_bytes, 2 * window_bytes);
        assert_eq!(
            ring.transport_bytes() - single.transport_bytes(),
            2 * window_bytes
        );
        assert_eq!(ring.batch, single.batch);
        assert_eq!(ring.queue_capacity, single.queue_capacity);

        let old_wire = serde_json::to_value(&single).unwrap();
        assert!(!old_wire
            .as_object()
            .unwrap()
            .contains_key("compact_packet_slots"));
        assert_eq!(serde_json::from_value::<Plan>(old_wire).unwrap(), single);
        let new_wire = serde_json::to_value(&ring).unwrap();
        let decoded = serde_json::from_value::<Plan>(new_wire).unwrap();
        assert_eq!(decoded, ring);
        decoded.validate(11).unwrap();

        let mut undercharged = decoded;
        undercharged.device_bytes = single.device_bytes;
        assert!(undercharged.validate(11).is_err());
        assert!(single.clone().with_compact_packet_slots(2).is_err());
        let dense = Plan::with_storage_layout(40, 8, 16, 65_536, 8_192, 11, 0, 8, false).unwrap();
        assert!(dense.with_compact_packet_slots(3).is_err());
    }

    #[test]
    fn packet_row_capacity_uses_recipient_count() {
        let one_shard = Plan::with_storage_layout(12, 8, 1, 65536, 65536, 11, 0, 1, true).unwrap();
        let many_shards =
            Plan::with_storage_layout(12, 8, 64, 65536, 65536, 11, 0, 1, true).unwrap();
        assert_eq!(one_shard.queue_capacity, 90112);
        assert_eq!(many_shards.queue_capacity, one_shard.queue_capacity);
        assert!(many_shards.transport_bytes() - one_shard.transport_bytes() < 4096);
    }

    #[test]
    fn bidirectional_paths_reserve_one_queue_workspace_and_two_histories() {
        let memory = HistoryMemory {
            extra_row_bytes: 8,
            shared_path_queues: true,
        };
        for world in [1, 2, 8] {
            let plan =
                Plan::with_storage_layout(16, world, 8, 4096, 1024, 3, 4096, 8, false).unwrap();
            let [records, counts, states] = plan.dense_queue_bank_extents();
            let banks = if world == 1 { 2 } else { 3 };
            let queues = ((records + counts + states) * banks) as u64;
            assert_eq!(queues, plan.transport_bytes());
            let required = 2 * (plan.device_bytes + 8 * u64::from(plan.capacity)) - queues;
            assert_eq!(memory.required_bytes(&plan).unwrap(), required);

            let reserve = crate::generic_memory::admission_reserve_bytes().unwrap();
            let admitted = Plan::automatic_storage_fixed_batch_with_history_memory(
                plan.elements,
                world,
                plan.shards,
                3,
                plan.generator_bytes,
                reserve + required,
                Some(u64::from(plan.capacity) + 1),
                plan.state_bytes,
                plan.batch,
                1,
                memory,
            )
            .unwrap();
            assert_eq!(admitted, plan);
            let duplicate = Plan::automatic_storage_fixed_batch_history_with_row_bytes(
                plan.elements,
                world,
                plan.shards,
                3,
                plan.generator_bytes,
                reserve + required / 2,
                Some(u64::from(plan.capacity)),
                plan.state_bytes,
                plan.batch,
                1,
                8,
            )
            .unwrap();
            assert!(duplicate.capacity < plan.capacity);
        }
    }
}
