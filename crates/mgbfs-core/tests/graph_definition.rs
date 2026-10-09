use mgbfs_core::graph_definition::{GraphDefinitionV2,GraphAction,MatrixGeneratorV2};
fn matrix(g:Vec<(Vec<i64>,u32)>,start:Vec<i64>,n:u32,m:u32)->GraphDefinitionV2 {GraphDefinitionV2{schema:2,name:"test".into(),generator_names:(0..g.len()).map(|i|i.to_string()).collect(),start,expected_max_unique_states:None,action:GraphAction::Matrix{rows:n,cols:m,generators:g.into_iter().map(|(matrix,modulo)|MatrixGeneratorV2{matrix,modulo}).collect()}}}
#[test]fn rectangular_and_wrapping_semantics(){let g=matrix(vec![(vec![1,1,0,1],257)],vec![256,2],2,1);g.validate().unwrap();assert_eq!(g.successor(&g.start,0).unwrap(),vec![1,2]);let g=matrix(vec![(vec![2],7)],vec![i64::MAX],1,1);assert_eq!(g.successor(&g.start,0).unwrap(),vec![5]);let g=matrix(vec![(vec![2],0)],vec![i64::MAX],1,1);assert_eq!(g.successor(&g.start,0).unwrap(),vec![-2]);}
#[test]fn directed_repeated_labels_have_exact_all_visited_oracle(){let g=GraphDefinitionV2{schema:2,name:"lx".into(),generator_names:vec!["l".into(),"x".into()],start:vec![0,0,1,2],expected_max_unique_states:None,action:GraphAction::Permutation{degree:4,generators:vec![vec![1,2,3,0],vec![1,0,2,3]]}};g.validate().unwrap();assert!(!g.inverse_closed().unwrap());assert_eq!(g.exact_layers(100).unwrap().iter().map(Vec::len).sum::<usize>(),12);assert!(g.exact_layers(2).is_err());}
#[test]fn invalid_shapes_and_noncanonical_modular_inverse(){let bad=matrix(vec![(vec![1],3)],vec![1,2],2,1);assert!(bad.validate().is_err());let g=matrix(vec![(vec![1],7)],vec![8],1,1);assert!(!g.inverse_closed().unwrap());let g=matrix(vec![(vec![1],7)],vec![6],1,1);assert!(g.inverse_closed().unwrap());let g=matrix(vec![(vec![1],7),(vec![1],5)],vec![1],1,1);assert!(!g.inverse_closed().unwrap());}
#[test]fn serde_and_identity_exclude_names(){let mut g=matrix(vec![(vec![1],3)],vec![1],1,1);let first=g.semantic_digest().unwrap();g.name="changed".into();g.generator_names=vec!["changed".into()];assert_eq!(g.semantic_digest().unwrap(),first);let raw=serde_json::to_vec(&g).unwrap();let decoded:GraphDefinitionV2=serde_json::from_slice(&raw).unwrap();assert_eq!(decoded.semantic_digest().unwrap(),first);}

#[test]fn wrapped_non_power_two_is_not_an_inverse_proof(){
 let g=matrix(vec![(vec![i64::MAX],3)],vec![2],1,1);
 // a*a wraps to 1, but actual action maps both 1 and 2 to 1 modulo 3.
 assert_eq!(g.successor(&[2],0).unwrap(),vec![1]);
 assert_eq!(g.successor(&[1],0).unwrap(),vec![1]);
 assert!(!g.inverse_closed().unwrap());
 let g=matrix(vec![(vec![i64::MAX],4)],vec![2],1,1);assert!(g.inverse_closed().unwrap());
 let g=matrix(vec![(vec![100],101)],vec![2],1,1);assert!(g.inverse_closed().unwrap());
 let g=matrix(vec![(vec![i64::MAX],0)],vec![2],1,1);assert!(g.inverse_closed().unwrap());
}
