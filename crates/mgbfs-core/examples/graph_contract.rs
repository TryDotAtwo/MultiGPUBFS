use std::io::{self,Read};
use mgbfs_core::graph_definition::GraphDefinitionV2;
fn main(){let mut text=String::new();io::stdin().read_to_string(&mut text).unwrap();let g:GraphDefinitionV2=serde_json::from_str(&text).unwrap();g.validate().unwrap();let digest=g.semantic_digest().unwrap().iter().map(|x|format!("{x:02x}")).collect::<String>();println!("{}",serde_json::json!({"digest":digest,"inverse_closed":g.inverse_closed().unwrap(),"first_child":g.successor(&g.start,0).unwrap()}));}
