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
