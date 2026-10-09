use mgbfs_runtime::generic_memory::GenericMemoryPlan;
#[test] fn plan_uses_available_vram_with_headroom_and_exact_bytes(){
 let p=GenericMemoryPlan::automatic(9,12u64<<30,None).unwrap();
 assert!(p.device_bytes<=(12u64<<30)*9/10);assert!(p.capacity>1_000_000);assert!(p.table_slots.is_power_of_two());assert!(p.table_slots>=p.capacity*2);
 let small=GenericMemoryPlan::automatic(9,12u64<<30,Some(64)).unwrap();assert_eq!(small.capacity,64);assert!(small.device_bytes<20000);
 assert!(GenericMemoryPlan::automatic(300,100,None).is_err());
}
