// Host-side admission checks for resident packed bitmap extents. These checks
// are independent of CUDA context creation; device allocations remain owned by
// the caller until all peers have drained the job.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AllocationRegion {
    pub base:u64,
    pub bytes:u64,
    pub alignment:u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExtentAllocation {
    pub rows:u64,
    pub row_bytes:u64,
    pub counts:u64,
    pub count_bytes:u64,
    pub capacity:u64,
    pub banks:u32,
}

/// Validate actual allocations, not only requested layouts. Each extent's
/// rows and counters must be disjoint from all other mutable allocations.
/// An exclusive shard lease owns all banks of the extent.
pub(crate) fn validate_extent_allocations(
    extents:&[ExtentAllocation], other:&[AllocationRegion]
)->Result<(), &'static str> {
    let mut ranges=Vec::with_capacity(extents.len().saturating_mul(2).saturating_add(other.len()));
    fn add(ranges:&mut Vec<(u64,u64)>,r:AllocationRegion)->Result<(), &'static str> {
        if r.base==0 || r.bytes==0 || !r.alignment.is_power_of_two() || r.base%r.alignment!=0 {
            return Err("RESIDENT_INVALID_ALLOCATION");
        }
        let end=r.base.checked_add(r.bytes).ok_or("RESIDENT_ALLOCATION_OVERFLOW")?;
        ranges.push((r.base,end));
        Ok(())
    }
    for e in extents {
        if e.capacity==0 || e.capacity>(1u64<<32) || e.banks!=3 {
            return Err("RESIDENT_EXTENT_GEOMETRY");
        }
        let rows=e.capacity.checked_mul(u64::from(e.banks))
            .and_then(|v|v.checked_mul(8)).ok_or("RESIDENT_EXTENT_OVERFLOW")?;
        let counts=u64::from(e.banks).checked_mul(8).ok_or("RESIDENT_EXTENT_OVERFLOW")?;
        if e.row_bytes<rows || e.count_bytes<counts {
            return Err("RESIDENT_EXTENT_UNDERSIZED");
        }
        add(&mut ranges,AllocationRegion{base:e.rows,bytes:e.row_bytes,alignment:8})?;
        add(&mut ranges,AllocationRegion{base:e.counts,bytes:e.count_bytes,alignment:8})?;
    }
    for &r in other {add(&mut ranges,r)?}
    ranges.sort_unstable_by_key(|v|v.0);
    if ranges.windows(2).any(|w|w[0].1>w[1].0) {
        return Err("RESIDENT_ALLOCATION_OVERLAP");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(rows:u64,counts:u64,capacity:u64)->ExtentAllocation {
        ExtentAllocation { rows, row_bytes:capacity*24, counts, count_bytes:24, capacity, banks:3 }
    }

    #[test]
    fn disjoint_extents_include_multiple_full_u32_segments() {
        let n=1u64<<32;
        let extents=[e(0x1000,0x4000_0000_0000,n),
                     e(0x1000+n*24,0x4000_0000_0020,n)];
        assert!(validate_extent_allocations(&extents,&[]).is_ok());
        assert_eq!(extents.iter().map(|e|e.capacity).sum::<u64>(),1u64<<33);
    }

    #[test]
    fn rejects_row_count_and_extra_allocation_aliases() {
        assert!(validate_extent_allocations(&[e(0x1000,0x2000,2),e(0x1010,0x3000,2)],&[]).is_err());
        assert!(validate_extent_allocations(&[e(0x1000,0x1010,2)],&[]).is_err());
        assert!(validate_extent_allocations(&[e(0x1000,0x2000,2)],
            &[AllocationRegion { base:0x2010,bytes:8,alignment:8 }]).is_err());
        assert!(validate_extent_allocations(&[e(0x1000,0x2000,2)],
            &[AllocationRegion { base:0x3000,bytes:8,alignment:8 }]).is_ok());
    }

    #[test]
    fn rejects_out_of_bounds_sizes_and_address_overflow() {
        let mut x=e(0x1000,0x2000,2);x.row_bytes=47;
        assert!(validate_extent_allocations(&[x],&[]).is_err());
        let mut x=e(0x1000,0x2000,2);x.count_bytes=23;
        assert!(validate_extent_allocations(&[x],&[]).is_err());
        let mut x=e(0x1000,0x2000,2);x.capacity=(1u64<<32)+1;
        assert!(validate_extent_allocations(&[x],&[]).is_err());
        let mut x=e(0x1001,0x2000,2);
        assert!(validate_extent_allocations(&[x],&[]).is_err());
        x.rows=u64::MAX-7;x.row_bytes=48;
        assert!(validate_extent_allocations(&[x],&[]).is_err());
        assert!(validate_extent_allocations(&[],&[AllocationRegion{base:u64::MAX-7,bytes:8,alignment:8}]).is_err());
    }
}
