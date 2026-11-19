//! Checked exact multiset-permutation indexing and distributed bitmap layout.
//! This is a host-side contract and oracle. It is not a GPU BFS backend.
use std::collections::BTreeMap;

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct MultisetDomain { symbols:Vec<i64>, counts:Vec<u32>, elements:usize, size:u64 }

impl MultisetDomain {
 pub fn new(start:&[i64])->Result<Self,&'static str>{
  if start.is_empty()||start.len()>64{return Err("BITMAP_UNSUPPORTED_DEGREE")}
  let mut map=BTreeMap::<i64,u32>::new();
  for &v in start{*map.entry(v).or_default()+=1;}
  let (symbols,counts):(Vec<_>,Vec<_>)=map.into_iter().unzip();
  // Successively insert identical symbols. Every intermediate multinomial is
  // monotone and <= final size; u128 handles the bounded multiply before /k.
  let mut ways=1u128;let mut previous=0u128;
  for &count in &counts{
   for k in 1..=u128::from(count){
    ways=ways.checked_mul(previous+k).ok_or("BITMAP_DOMAIN_OVERFLOW")?/k;
    if ways>u128::from(u64::MAX){return Err("BITMAP_DOMAIN_OVERFLOW")}
   }
   previous+=u128::from(count);
  }
  Ok(Self{symbols,counts,elements:start.len(),size:ways as u64})
 }
 pub fn size(&self)->u64{self.size}
 pub fn symbols(&self)->&[i64]{&self.symbols}
 pub fn counts(&self)->&[u32]{&self.counts}
 pub fn rank(&self,state:&[i64])->Result<u64,&'static str>{
  if state.len()!=self.elements{return Err("BITMAP_STATE_LENGTH")}
  let mut counts=self.counts.clone();let mut ways=u128::from(self.size);let mut rank=0u128;
  for (index,&symbol) in state.iter().enumerate(){
   let i=self.symbols.binary_search(&symbol).map_err(|_|"BITMAP_STATE_SYMBOL")?;
   if counts[i]==0{return Err("BITMAP_STATE_MULTIPLICITY")}
   let n=(self.elements-index) as u128;
   let less:u128=counts[..i].iter().map(|&c|u128::from(c)).sum();
   rank+=ways*less/n;
   ways=ways*u128::from(counts[i])/n;counts[i]-=1;
  }
  if rank>=u128::from(self.size){return Err("BITMAP_RANK_RANGE")}
  Ok(rank as u64)
 }
 pub fn unrank(&self,rank:u64)->Result<Vec<i64>,&'static str>{
  if rank>=self.size{return Err("BITMAP_RANK_RANGE")}
  let mut counts=self.counts.clone();let mut ways=u128::from(self.size);let mut rem=u128::from(rank);
  let mut state=Vec::with_capacity(self.elements);
  for index in 0..self.elements{
   let n=(self.elements-index) as u128;let mut found=None;
   for i in 0..counts.len(){
    if counts[i]==0{continue}
    let block=ways*u128::from(counts[i])/n;
    if rem>=block{rem-=block}else{found=Some((i,block));break}
   }
   let (i,block)=found.ok_or("BITMAP_UNRANK_INVARIANT")?;
   state.push(self.symbols[i]);counts[i]-=1;ways=block;
  }
  if rem!=0{return Err("BITMAP_UNRANK_INVARIANT")}Ok(state)
 }
}

/// Ownership uses rank % world, with local row rank / world. Mapping from a
/// logical owner to a CUDA rank is separate and must be an actual bijection.
#[derive(Clone,Copy,Debug)]
pub struct BitmapPartition{total:u64,world:u32}
impl BitmapPartition{
 pub fn new(total:u64,world:u32)->Result<Self,&'static str>{
  if total==0||world==0||world>128{return Err("BITMAP_PARTITION_RANGE")}
  Ok(Self{total,world})
 }
 pub fn owner_len(self,owner:u32)->Result<u64,&'static str>{
  if owner>=self.world{return Err("BITMAP_OWNER_RANGE")}
  let w=u64::from(self.world);Ok(self.total/w+u64::from(u64::from(owner)<self.total%w))
 }
 pub fn owner_bytes(self,owner:u32)->Result<u64,&'static str>{
  let n=self.owner_len(owner)?;(n/64+u64::from(n%64!=0)).checked_mul(8).ok_or("BITMAP_SIZE_OVERFLOW")
 }
 pub fn locate(self,rank:u64)->Result<(u32,u64),&'static str>{
  if rank>=self.total{return Err("BITMAP_RANK_RANGE")}
  let w=u64::from(self.world);Ok(((rank%w) as u32,rank/w))
 }
 pub fn reconstruct(self,owner:u32,local:u64)->Result<u64,&'static str>{
  if local>=self.owner_len(owner)?{return Err("BITMAP_LOCAL_RANGE")}
  local.checked_mul(u64::from(self.world)).and_then(|v|v.checked_add(u64::from(owner))).filter(|&r|r<self.total).ok_or("BITMAP_RANK_OVERFLOW")
 }
}
