use mgbfs_runtime::generic_distributed_memory::GenericDistributedMemoryPlan as Plan;
#[test] fn actual_route_banks_fit_at_1_2_8_128_rank_geometries(){
 for world in [1,2,8,128] {for width in [4,16,257] {for shards in [1,4,16] {
  let p=Plan::automatic(width,world,shards,4,16384,12u64<<30,None).unwrap();p.validate(4).unwrap();
  assert!(p.device_bytes<=((12u64<<30)*9/10));assert_eq!(p.queue_capacity,p.batch*4);assert!(p.slots_per_shard>=p.capacity*2);
  let mut bad=p.clone();bad.device_bytes-=1;assert!(bad.validate(4).is_err());
 }}}
}
#[test] fn small_graph_does_not_allocate_vram_sized_queues(){let p=Plan::automatic(4,2,1,2,32,12u64<<30,Some(24)).unwrap();assert_eq!(p.capacity,24);assert_eq!(p.batch,24);assert!(p.device_bytes<40000);}
#[test] fn invalid_bounds_fail_before_any_allocation(){assert!(Plan::new(4,129,1,4,1,2,0).is_err());assert!(Plan::new(4,2,1,4,5,2,0).is_err());assert!(Plan::new(4,2,1,1,1,u32::MAX,0).is_err());assert!(Plan::automatic(4,2,1,2,0,100,None).is_err());}
