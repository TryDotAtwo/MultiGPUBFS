use mgbfs_runtime::generic_distributed_memory::GenericDistributedMemoryPlan as Plan;
#[test] fn actual_route_banks_fit_at_1_2_8_128_rank_geometries(){
 for world in [1,2,8,128] {for width in [4,16,257] {for shards in [1,4,16] {
  let p=Plan::automatic(width,world,shards,4,16384,12u64<<30,None).unwrap();p.validate(4).unwrap();
  assert!(p.device_bytes<=((12u64<<30)*9/10));assert!(p.queue_capacity<=p.batch*4);assert!(p.queue_capacity>=4);assert!(p.slots_per_shard>=p.capacity*2);
  let mut bad=p.clone();bad.device_bytes-=1;assert!(bad.validate(4).is_err());
 }}}
}
#[test] fn small_graph_does_not_allocate_vram_sized_queues(){let p=Plan::automatic(4,2,1,2,32,12u64<<30,Some(24)).unwrap();assert_eq!(p.capacity,24);assert_eq!(p.batch,24);assert!(p.device_bytes<40000);}
#[test] fn invalid_bounds_fail_before_any_allocation(){assert!(Plan::new(4,129,1,4,1,2,0).is_err());assert!(Plan::new(4,2,1,4,5,2,0).is_err());assert!(Plan::new(4,2,1,1,1,u32::MAX,0).is_err());assert!(Plan::automatic(4,2,1,2,0,100,None).is_err());}

#[test] fn balanced_queues_bound_padding_independently_of_world(){
 for world in [2,8,128]{let p=Plan::new(16,world,16,4096,1024,4,0).unwrap();assert!(u64::from(p.queue_capacity)*u64::from(world)*16<=4*4096+u64::from(world)*16*4);}
}

#[test]fn unequal_vram_admission_uses_every_rank_capacity_and_matched_queues(){
 use mgbfs_runtime::generic_distributed_memory::heterogeneous_plans;
 let (plans,cuts)=heterogeneous_plans(14,4,3,4096,&[12u64<<30,24u64<<30],None,1).unwrap();
 assert!(plans[1].capacity>plans[0].capacity);assert_eq!(plans[0].batch,plans[1].batch);assert_eq!(plans[0].queue_capacity,plans[1].queue_capacity);
 assert_eq!(cuts[0],0);assert_eq!(cuts[2],1u64<<32);assert!(cuts[1]<(1u64<<31));
 for (p,free) in plans.iter().zip([12u64<<30,24u64<<30]){p.validate(3).unwrap();assert!(p.device_bytes<=free-(1u64<<30).max(free/10));}
 assert!(heterogeneous_plans(14,4,3,4096,&[],None,1).is_err());
}
