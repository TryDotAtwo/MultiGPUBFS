//! Lossless action contract for CayleyPy permutations and int64 matrix states.
use crate::Result;
use serde::{Serialize,Deserialize};
use sha2::{Digest,Sha256};
use std::collections::BTreeSet;

#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixGeneratorV2 {pub matrix:Vec<i64>,pub modulo:u32}
#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(tag="kind",rename_all="snake_case",deny_unknown_fields)]
pub enum GraphAction {
 Permutation {degree:u32,generators:Vec<Vec<u32>>},
 Matrix {rows:u32,cols:u32,generators:Vec<MatrixGeneratorV2>},
}
#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDefinitionV2 {
 pub schema:u32,pub name:String,pub generator_names:Vec<String>,pub action:GraphAction,
 pub start:Vec<i64>,pub expected_max_unique_states:Option<u64>,
}
impl GraphDefinitionV2 {
 pub fn generator_count(&self)->usize {match &self.action {GraphAction::Permutation{generators,..}=>generators.len(),GraphAction::Matrix{generators,..}=>generators.len()}}
 pub fn validate(&self)->Result<()> {
  if self.schema!=2 || self.generator_count()==0 || self.generator_names.len()!=self.generator_count() || self.expected_max_unique_states==Some(0) {return Err("GRAPH_SCHEMA_OR_GENERATORS".into())}
  match &self.action {
   GraphAction::Permutation{degree,generators}=>{
    let n=*degree as usize;
    if n==0 || self.start.len()!=n {return Err("GRAPH_PERMUTATION_SHAPE".into())}
    for g in generators {
     if g.len()!=n {return Err("GRAPH_PERMUTATION_GENERATOR".into())}
     let mut seen=vec![false;n];for &i in g {if i>=*degree || seen[i as usize] {return Err("GRAPH_PERMUTATION_GENERATOR".into())}seen[i as usize]=true;}
    }
   }
   GraphAction::Matrix{rows,cols,generators}=>{
    let n=*rows as usize;let m=*cols as usize;let size=n.checked_mul(m).ok_or("GRAPH_MATRIX_SHAPE")?;
    if n==0 || m==0 || size>u32::MAX as usize || self.start.len()!=size {return Err("GRAPH_MATRIX_SHAPE".into())}
    let square=n.checked_mul(n).ok_or("GRAPH_MATRIX_SHAPE")?;
    for g in generators {if g.matrix.len()!=square || (g.modulo!=0 && !(2..=1u32<<31).contains(&g.modulo)) {return Err("GRAPH_MATRIX_GENERATOR_OR_MODULUS".into())}}
   }
  }
  Ok(())
 }
 pub fn semantic_digest(&self)->Result<[u8;32]> {
  self.validate()?;
  // Value maps serialize in sorted key order, matching Python canonical JSON.
  let v=serde_json::json!({"schema":self.schema,"action":self.action,"start":self.start});
  Ok(Sha256::digest(serde_json::to_vec(&v).map_err(|e|e.to_string())?).into())
 }
 fn multiply(mx:&[i64],state:&[i64],n:usize,m:usize,modulo:u32)->Vec<i64> {
  let mut out=vec![0;n*m];for i in 0..n {for j in 0..m {
   let mut sum=0i64;for k in 0..n {sum=sum.wrapping_add(mx[i*n+k].wrapping_mul(state[k*m+j]));}
   out[i*m+j]=if modulo==0 {sum} else {sum.rem_euclid(i64::from(modulo))};
  }}out
 }
 pub fn successor(&self,state:&[i64],generator:usize)->Result<Vec<i64>> {
  self.validate()?;
  if state.len()!=self.start.len() || generator>=self.generator_count() {return Err("GRAPH_SUCCESSOR_ARGUMENT".into())}
  Ok(match &self.action {
   GraphAction::Permutation{generators,..}=>generators[generator].iter().map(|&i|state[i as usize]).collect(),
   GraphAction::Matrix{rows,cols,generators}=>{let g=&generators[generator];Self::multiply(&g.matrix,state,*rows as usize,*cols as usize,g.modulo)}
  })
 }
 pub fn inverse_closed(&self)->Result<bool> {
  self.validate()?;
  Ok(match &self.action {
   GraphAction::Permutation{degree,generators}=>{
    let available:BTreeSet<Vec<u32>>=generators.iter().cloned().collect();
    generators.iter().all(|g|{let mut inverse=vec![0;*degree as usize];for (i,&x) in g.iter().enumerate(){inverse[x as usize]=i as u32;}available.contains(&inverse)})
   }
   GraphAction::Matrix{rows,generators,..}=>{
    let n=*rows as usize;let modulus=generators[0].modulo;
    if generators.iter().any(|g|g.modulo!=modulus) || (modulus!=0 && self.start.iter().any(|&x|x<0 || x>=i64::from(modulus))) {return Ok(false)}
    // CUDA uses wrapping int64 arithmetic before modular reduction. For
    // non-power-of-two moduli, wrapping is not a homomorphism modulo m.
    // Admit compact history only when every canonical-state dot product
    // is overflow-free; otherwise conservatively retain all history.
    if modulus!=0 && !modulus.is_power_of_two(){
     let bound=i128::from(modulus-1);
     if generators.iter().any(|g|g.matrix.chunks(n).any(|row|row.iter().map(|&v|i128::from(v).abs()*bound).sum::<i128>()>i128::from(i64::MAX))){return Ok(false)}
    }
    let eye:Vec<i64>=(0..n).flat_map(|i|(0..n).map(move |j|i64::from(i==j))).collect();
    let compose=|a:&[i64],b:&[i64]|->Vec<i64>{
     if modulus==0{return Self::multiply(a,b,n,n,0)}
     // Normalize the right operand, as an actual canonical modular state.
     let canonical=b.iter().map(|&x|x.rem_euclid(i64::from(modulus))).collect::<Vec<_>>();
     Self::multiply(a,&canonical,n,n,modulus)
    };
    generators.iter().all(|g|generators.iter().any(|h|compose(&g.matrix,&h.matrix)==eye && compose(&h.matrix,&g.matrix)==eye))
   }
  })
 }
 /// Bounded full-state CPU oracle, never a production compute fallback.
 pub fn exact_layers(&self,maximum_states:usize)->Result<Vec<Vec<Vec<i64>>>> {
  self.validate()?;if maximum_states==0 {return Err("ORACLE_CAPACITY".into())}
  let mut seen=BTreeSet::from([self.start.clone()]);let mut frontier=seen.clone();let mut layers=Vec::new();
  while !frontier.is_empty() {
   layers.push(frontier.iter().cloned().collect());let mut future=BTreeSet::new();
   for x in &frontier {for g in 0..self.generator_count() {let y=self.successor(x,g)?;if !seen.contains(&y){future.insert(y);}if future.len()>maximum_states-seen.len(){return Err("ORACLE_CAPACITY".into())}}}
   seen.extend(future.iter().cloned());frontier=future;
  }Ok(layers)
 }
}
