#![cfg(feature="cuda")]
use mgbfs_core::graph_definition::GraphDefinitionV2;
use mgbfs_runtime::{generic_native::{GenericNativeBfs,GenericAdvance},generic_memory::GenericMemoryPlan};
fn graph(text:&str)->GraphDefinitionV2{serde_json::from_str(text).unwrap()}
#[test] fn exact_gpu_layers_batch_one_collisions_and_resource_snapshot(){
 let definitions=[
 r#"{"schema":2,"name":"LX","generator_names":["L","X"],"action":{"kind":"permutation","degree":4,"generators":[[1,2,3,0],[1,0,2,3]]},"start":[0,1,2,3],"expected_max_unique_states":null}"#,
 r#"{"schema":2,"name":"rect","generator_names":["M"],"action":{"kind":"matrix","rows":2,"cols":1,"generators":[{"matrix":[1,1,0,1],"modulo":257}]},"start":[256,2],"expected_max_unique_states":null}"#,
 r#"{"schema":2,"name":"overflow","generator_names":["M"],"action":{"kind":"matrix","rows":1,"cols":1,"generators":[{"matrix":[2],"modulo":0}]},"start":[1],"expected_max_unique_states":null}"#];
 for device in 0..2 {for text in definitions {let g=graph(text);let expected=g.exact_layers(1024).unwrap();
  for hash_bits in [64,0] {let mut p=GenericMemoryPlan::new(g.start.len() as u32,1024).unwrap();p.batch=1;
   let mut bfs=GenericNativeBfs::new(&g,device,p,123,hash_bits).unwrap();let mut actual=vec![];
   loop {let mut layer=bfs.sample(1024).unwrap();layer.sort();actual.push(layer);
    match bfs.advance().unwrap(){GenericAdvance::Layer{..}=>(),GenericAdvance::Complete=>break,other=>panic!("unexpected {other:?}")}
   }
   assert_eq!(actual,expected);let mut previous=bfs.previous_small_sample().unwrap();previous.sort();assert_eq!(previous,expected[expected.len()-2]);assert!(bfs.advance().is_err());
  }
 }
 let g=graph(definitions[0]);let mut bfs=GenericNativeBfs::new(&g,device,GenericMemoryPlan::new(4,3).unwrap(),123,0).unwrap();
 loop {let before=bfs.sample(1000).unwrap();match bfs.advance().unwrap(){GenericAdvance::Layer{..}=>(),GenericAdvance::Resource{fatal}=>{assert_ne!(fatal&2,0);assert_eq!(bfs.sample(1000).unwrap(),before);break;},other=>panic!("unexpected {other:?}")}}
 }
}
