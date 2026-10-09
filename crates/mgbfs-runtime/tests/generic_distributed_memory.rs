use mgbfs_runtime::generic_distributed_memory::GenericDistributedMemoryPlan as Plan;
#[test] fn actual_route_banks_fit_at_1_2_8_128_rank_geometries(){
 for world in [1,2,8,128] {for width in [4,16,257] {for shards in [1,4,16] {
  let p=Plan::automatic(width,world,shards,4,16384,12u64<<30,None).unwrap();p.validate(4).unwrap();
  assert!(p.device_bytes<=((12u64<<30)*9/10));assert!(p.queue_capacity<=p.batch*4);assert!(p.queue_capacity>=4);assert!(p.table_slots>=p.capacity*2);
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

#[test]fn packed_candidate_boundary_and_exact_queue_admission(){
 for width in [1,8,16,17,23,24,25,257]{for bytes in [1,8]{
  let p=Plan::with_storage(width,2,4,1024,32,3,4096,bytes).unwrap();
  assert_eq!(p.packed_candidates(),bytes==1&&width<=24);
  assert_eq!(p.queue_payload_bytes(),if bytes==1&&width<=24{0}else{(width*bytes) as usize});p.validate(3).unwrap();
 }}
}

#[test]fn shard_tables_do_not_multiply_local_history_capacity(){
 for shards in [1,3,16,256]{let p=Plan::with_storage(14,2,shards,4096,128,3,4096,1).unwrap();assert_eq!(p.table_slots,8192);assert_eq!(p.table_layout,"SHARD_HASH_REGIONS_SHARED_OVERFLOW");p.validate(3).unwrap();}
 assert!(Plan::with_storage(14,128,4096,4096,1,4096,4096,1).is_err());
}

#[test]fn automatic_plan_uses_remaining_history_budget_after_bounded_transport(){
 for (width,bytes) in [(14,1),(17,1),(257,8)]{
  let free=12u64<<30;let budget=free-(1u64<<30).max(free/10);
  let p=Plan::automatic_storage(width,2,4,3,4096,free,None,bytes).unwrap();assert!(p.transport_bytes()<=budget/4);assert!(p.device_bytes<=budget);
  if p.capacity<(1<<28){assert!(Plan::with_storage(width,2,4,p.capacity+1,p.batch,3,4096,bytes).unwrap().device_bytes>budget);}
 }
}

#[test]fn rolling_history_accounts_for_three_immutable_banks_and_positions(){
 for bytes in [1,8]{let retained=Plan::with_storage(17,2,4,100,8,3,4096,bytes).unwrap();let rolling=Plan::with_storage_history(17,2,4,100,8,3,4096,bytes,3).unwrap();
 assert_eq!(rolling.table_slots,1024);assert_eq!(rolling.history_layers,3);
 assert_eq!(rolling.device_bytes-retained.device_bytes,17*100*bytes as u64*2+100*12+(1024-256)*8);rolling.validate(3).unwrap();
 let mut bad=rolling.clone();bad.history_layers=1;assert!(bad.validate(3).is_err());}
 assert!(Plan::with_storage_history(4,2,4,100,8,3,4096,1,2).is_err());
}
#[test]fn rolling_automatic_capacity_is_exactly_admitted(){
 let p=Plan::automatic_storage_history(14,2,16,3,4096,12u64<<30,None,1,3).unwrap();assert_eq!(p.history_layers,3);assert!(p.device_bytes<=(12u64<<30)-(12u64<<30)/10);p.validate(3).unwrap();
 let (plans,cuts)=mgbfs_runtime::generic_distributed_memory::heterogeneous_plans_history(14,4,3,4096,&[12u64<<30,24u64<<30],None,1,3).unwrap();assert!(plans[0].capacity<plans[1].capacity);assert_eq!(plans[0].batch,plans[1].batch);assert_eq!(*cuts.last().unwrap(),1u64<<32);
}

#[test]fn rolling_logical_geometry_admission_at_1_2_8_128_ranks(){
 for world in [1,2,8,128]{for shards in [1,4,16]{for (width,bytes) in [(4,1),(16,1),(257,8)]{
  let free=12u64<<30;let p=Plan::automatic_storage_history(width,world,shards,4,16384,free,None,bytes,3).unwrap();p.validate(4).unwrap();assert!(p.device_bytes<=free-(1u64<<30).max(free/10));assert!(u64::from(p.capacity)*3<0x80000000);assert!(p.table_slots>=p.capacity*6);assert!(u64::from(p.world)*u64::from(p.shards)*u64::from(p.queue_capacity)<0x7fffffff);
 }}}
}

#[test]fn parent_origin_buffers_are_exactly_admitted(){
 for world in [1,2,8,128]{for history in [1,3]{for (width,bytes) in [(25,1),(257,8)]{
  let full=Plan::with_storage_history(width,world,4,4096,32,3,1024,bytes,history).unwrap();
  let parent=full.clone().with_parent_transport(true).unwrap();parent.validate(3).unwrap();
  assert!(parent.parent_transport);assert_eq!(parent.queue_payload_bytes(),0);
  assert_eq!(parent.parent_cache_bytes(),u64::from(world+2)*32*u64::from(width)*u64::from(bytes)+u64::from(world)*12);
  assert_eq!(parent.device_bytes,full.device_bytes-full.transport_bytes()+parent.transport_bytes());
  assert_eq!(parent.clone().with_parent_transport(false).unwrap(),full);
  let mut bad=parent;bad.device_bytes-=1;assert!(bad.validate(3).is_err());
 }}}
 assert!(!Plan::with_storage(24,2,4,4096,32,3,1024,1).unwrap().with_parent_transport(true).unwrap().parent_transport);
}

#[test]fn radix_origin_scratch_is_fully_admitted(){
 for world in [1,2,8,128]{for width in [14,25,257]{let full=Plan::with_storage_history(width,world,4,4096,32,3,1024,1,3).unwrap();for parent in [false,true]{let base=full.clone().with_parent_transport(parent).unwrap();let sorted=base.clone().with_sort_candidates(true).unwrap();sorted.validate(3).unwrap();assert_eq!(sorted.sort_cache_bytes(),u64::from(sorted.shards)*(u64::from(sorted.world)*u64::from(sorted.queue_capacity)*88+65536));assert_eq!(sorted.device_bytes,base.device_bytes+sorted.sort_cache_bytes());assert_eq!(sorted.with_sort_candidates(false).unwrap(),base);}}}
}

#[test]fn serialized_sorted_admission_replaces_hash_and_position_storage(){
 for history in [1,3]{
  let legacy=Plan::with_storage_history(25,2,8,1000,32,2,4096,1,history).unwrap();
  let reserve=128u64<<20;let owner=123456u64;
  let sorted=legacy.clone().with_sorted_admission(4,owner,reserve).unwrap();
  assert_eq!(sorted.table_slots,0);assert_eq!(sorted.history_algorithm,"SORTED_RUNS");assert!(!sorted.sort_candidates);
  assert_eq!(sorted.device_bytes,legacy.device_bytes-u64::from(legacy.table_slots)*8-if history==3{12000}else{0}+1+owner+reserve);
  sorted.validate(2).unwrap();let roundtrip:Plan=serde_json::from_str(&serde_json::to_string(&sorted).unwrap()).unwrap();assert_eq!(roundtrip,sorted);
  let mut bad=sorted.clone();bad.device_bytes+=1;assert!(bad.validate(2).is_err());
  let mut bad=sorted.clone();bad.table_slots=1;assert!(bad.validate(2).is_err());
  assert!(legacy.clone().with_sorted_admission(0,owner,reserve).is_err());assert!(legacy.with_sorted_admission(4,owner,0).is_err());
 }
}
