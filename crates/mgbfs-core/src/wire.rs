//! Schema2 frame boundary. Field-wise little endian; never transmute host ABI.
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrameKind {
    Dense = 1,
    HashFirst = 2,
    Request = 3,
    Response = 4,
    Receipt = 5,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub kind: FrameKind,
    pub run_tag: u64,
    pub sequence: u64,
    pub batch: u64,
    pub depth: u32,
    pub source: u32,
    pub destination: u32,
    pub count: u32,
}
#[derive(Debug, Clone, Copy)]
pub struct ExpectedFrame {
    pub run_tag: u64,
    pub sequence: u64,
    pub batch: u64,
    pub depth: u32,
    pub source: u32,
    pub destination: u32,
    pub world: u32,
    pub kind: FrameKind,
    pub max_records: u32,
    pub max_payload: u64,
    pub state_stride: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub offset: u64,
    pub bytes: u64,
    pub reserved: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadLayout {
    pub planes: Vec<Plane>,
    pub bytes: u64,
}
pub fn payload_layout(kind: FrameKind, count: u32, stride: u64) -> Result<PayloadLayout> {
    if stride == 0 || stride % 16 != 0 {
        return Err("WIRE_STATE_STRIDE".into());
    }
    let dense = [16, 4, stride];
    let response = [stride];
    let sizes: &[u64] = match kind {
        FrameKind::Dense => &dense,
        FrameKind::HashFirst => &[16, 4, 16],
        FrameKind::Request => &[16],
        FrameKind::Response => &response,
        FrameKind::Receipt => &[32],
    };
    let mut result = PayloadLayout {
        planes: Vec::with_capacity(sizes.len()),
        bytes: 0,
    };
    for &s in sizes {
        let bytes = s.checked_mul(count as u64).ok_or("WIRE_BYTE_OVERFLOW")?;
        let reserved = bytes.checked_add(255).ok_or("WIRE_BYTE_OVERFLOW")? & !255;
        let end = result
            .bytes
            .checked_add(reserved)
            .ok_or("WIRE_BYTE_OVERFLOW")?;
        result.planes.push(Plane {
            offset: result.bytes,
            bytes,
            reserved,
        });
        result.bytes = end;
    }
    Ok(result)
}
impl FrameHeader {
    pub fn encode(self, stride: u64) -> Result<[u8; 64]> {
        let mut b = [0; 64];
        for (offset, value) in [
            (0, 0x4d474232u32),
            (4, 2),
            (8, self.kind as u32),
            (40, self.depth),
            (44, self.source),
            (48, self.destination),
            (52, self.count),
        ] {
            b[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (offset, value) in [
            (16, self.run_tag),
            (24, self.sequence),
            (32, self.batch),
            (56, payload_layout(self.kind, self.count, stride)?.bytes),
        ] {
            b[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        Ok(b)
    }
    pub fn decode(bytes: &[u8], expected: &ExpectedFrame) -> Result<Self> {
        if bytes.len() != 64 {
            return Err("WIRE_HEADER_SIZE".into());
        }
        let u32_at = |o| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u64_at = |o| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if u32_at(0) != 0x4d474232 || u32_at(4) != 2 || u32_at(12) != 0 {
            return Err("WIRE_SCHEMA".into());
        }
        let kind = match u32_at(8) {
            1 => FrameKind::Dense,
            2 => FrameKind::HashFirst,
            3 => FrameKind::Request,
            4 => FrameKind::Response,
            5 => FrameKind::Receipt,
            _ => return Err("WIRE_KIND".into()),
        };
        let h = Self {
            kind,
            run_tag: u64_at(16),
            sequence: u64_at(24),
            batch: u64_at(32),
            depth: u32_at(40),
            source: u32_at(44),
            destination: u32_at(48),
            count: u32_at(52),
        };
        if h.run_tag != expected.run_tag
            || h.sequence != expected.sequence
            || h.batch != expected.batch
            || h.depth != expected.depth
            || h.source != expected.source
            || h.destination != expected.destination
            || h.kind != expected.kind
            || h.source >= expected.world
            || h.destination >= expected.world
        {
            return Err("WIRE_SESSION_OR_TICKET".into());
        }
        if h.count > expected.max_records || u64_at(56) > expected.max_payload {
            return Err("WIRE_CAPACITY".into());
        }
        if payload_layout(h.kind, h.count, expected.state_stride)?.bytes != u64_at(56) {
            return Err("WIRE_PAYLOAD_SIZE".into());
        }
        Ok(h)
    }
}
pub fn validate_payload(payload: &[u8], layout: &PayloadLayout) -> Result<()> {
    if payload.len() as u64 != layout.bytes || layout.planes.len() > 3 {
        return Err("WIRE_PAYLOAD_SIZE".into());
    }
    let mut offset = 0;
    for p in &layout.planes {
        if p.offset != offset || p.reserved % 256 != 0 || p.bytes > p.reserved {
            return Err("WIRE_PLANE_LAYOUT".into());
        }
        let data_end = p.offset.checked_add(p.bytes).ok_or("WIRE_BYTE_OVERFLOW")?;
        offset = p
            .offset
            .checked_add(p.reserved)
            .ok_or("WIRE_BYTE_OVERFLOW")?;
        if offset > layout.bytes {
            return Err("WIRE_PLANE_LAYOUT".into());
        }
        if payload[data_end as usize..offset as usize]
            .iter()
            .any(|&b| b != 0)
        {
            return Err("WIRE_NONZERO_PADDING".into());
        }
    }
    if offset != layout.bytes {
        return Err("WIRE_PLANE_LAYOUT".into());
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginRef {
    pub source: u32,
    pub movement: u16,
    pub parent: u64,
}
impl OriginRef {
    pub fn encode(self) -> [u8; 16] {
        let mut b = [0; 16];
        b[..4].copy_from_slice(&self.source.to_le_bytes());
        b[4..6].copy_from_slice(&self.movement.to_le_bytes());
        b[8..].copy_from_slice(&self.parent.to_le_bytes());
        b
    }
    pub fn decode(bytes: &[u8], world: u32, moves: u32) -> Result<Self> {
        if bytes.len() != 16 || bytes[6..8] != [0, 0] {
            return Err("WIRE_ORIGIN_SIZE_OR_RESERVED".into());
        }
        let o = Self {
            source: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            movement: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            parent: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        };
        if o.source >= world || o.movement as u32 >= moves {
            return Err("WIRE_ORIGIN_RANGE".into());
        }
        Ok(o)
    }
}
