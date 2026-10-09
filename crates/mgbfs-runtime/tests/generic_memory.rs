use mgbfs_runtime::generic_memory::GenericMemoryPlan;
#[test] fn plan_uses_available_vram_with_headroom_and_exact_bytes(){
 let p=GenericMemoryPlan::automatic(9,12u64<<30,None).unwrap();
 assert!(p.device_bytes<=(12u64<<30)*9/10);assert!(p.capacity>1_000_000);assert!(p.table_slots.is_power_of_two());assert!(p.table_slots>=p.capacity*2);
 let small=GenericMemoryPlan::automatic(9,12u64<<30,Some(64)).unwrap();assert_eq!(small.capacity,64);assert!(small.device_bytes<20000);
 assert!(GenericMemoryPlan::automatic(300,100,None).is_err());
}

#[test]fn proven_graph_bounds_do_not_overallocate_small_orbits(){
 use mgbfs_core::graph_definition::{GraphDefinitionV2,GraphAction,MatrixGeneratorV2};use mgbfs_runtime::generic_memory::state_space_bound;
 let mut graph=GraphDefinitionV2{schema:2,name:"bound".into(),generator_names:vec!["rotate".into()],action:GraphAction::Permutation{degree:4,generators:vec![vec![1,2,3,0]]},start:vec![0,1,2,3],expected_max_unique_states:None};
 assert_eq!(state_space_bound(&graph),Some(24));graph.start=vec![0,0,1,2];assert_eq!(state_space_bound(&graph),Some(12));graph.start=vec![7,7,7,7];assert_eq!(state_space_bound(&graph),Some(1));
 graph.action=GraphAction::Matrix{rows:1,cols:1,generators:vec![MatrixGeneratorV2{matrix:vec![1],modulo:7}]};graph.start=vec![20];assert_eq!(state_space_bound(&graph),Some(8));graph.start=vec![0];assert_eq!(state_space_bound(&graph),Some(7));
 graph.action=GraphAction::Matrix{rows:1,cols:1,generators:vec![MatrixGeneratorV2{matrix:vec![1],modulo:0}]};assert_eq!(state_space_bound(&graph),None);
}

#[test]fn compact_storage_admission_matches_exact_payload_savings(){
 let wide=GenericMemoryPlan::with_storage(14,1024,8).unwrap();let compact=GenericMemoryPlan::with_storage(14,1024,1).unwrap();
 assert_eq!(wide.device_bytes-compact.device_bytes,14*1024*2*7);assert!(GenericMemoryPlan::with_storage(14,1024,2).is_err());
 use mgbfs_runtime::generic_distributed_memory::GenericDistributedMemoryPlan as Plan;
 let wide=Plan::with_storage(14,2,4,1024,32,3,4096,8).unwrap();let compact=Plan::with_storage(14,2,4,1024,32,3,4096,1).unwrap();compact.validate(3).unwrap();
 assert_eq!(wide.device_bytes-compact.device_bytes,14*1024*2*7+2*4*u64::from(compact.queue_capacity)*14*8*3-3);
 let free=12u64<<30;assert!(Plan::automatic_storage(14,2,4,3,4096,free,None,1).unwrap().capacity>Plan::automatic_storage(14,2,4,3,4096,free,None,8).unwrap().capacity);
}
