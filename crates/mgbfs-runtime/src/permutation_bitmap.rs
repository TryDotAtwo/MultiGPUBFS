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

/// Exact closed product of multiset domains on position components. The
/// generator action may couple components; this is a closed superdomain, not
/// a claim that every combination is reachable from the starting state.
#[derive(Clone,Debug)]
pub struct PositionOrbitDomain {
 elements:usize,
 components:Vec<(Vec<u32>,MultisetDomain)>,
 size:u64,
}
impl PositionOrbitDomain {
 pub fn new(start:&[i64],generators:&[Vec<u32>])->Result<Self,&'static str>{
  let n=start.len();if n==0||n>64||generators.is_empty(){return Err("BITMAP_ORBIT_SHAPE")}
  fn root(parent:&mut [usize],mut p:usize)->usize{
   while parent[p]!=p {parent[p]=parent[parent[p]];p=parent[p];}p
  }
  let mut parent=(0..n).collect::<Vec<_>>();
  for generator in generators{
   if generator.len()!=n{return Err("BITMAP_ORBIT_GENERATOR")}
   let mut used=0u64;
   for (i,&value) in generator.iter().enumerate(){
    let j=value as usize;if j>=n{return Err("BITMAP_ORBIT_GENERATOR")}
    let bit=1u64<<j;if used&bit!=0{return Err("BITMAP_ORBIT_GENERATOR")}used|=bit;
    let a=root(&mut parent,i);let b=root(&mut parent,j);parent[a]=b;
   }
  }
  let mut groups=BTreeMap::<usize,Vec<u32>>::new();
  for i in 0..n {let r=root(&mut parent,i);groups.entry(r).or_default().push(i as u32);}
  let mut positions:Vec<_>=groups.into_values().collect();positions.sort_by_key(|p|p[0]);
  let mut components=Vec::with_capacity(positions.len());let mut size=1u64;
  for pos in positions{
   let values:Vec<_>=pos.iter().map(|&i|start[i as usize]).collect();
   let domain=MultisetDomain::new(&values)?;
   size=size.checked_mul(domain.size()).ok_or("BITMAP_DOMAIN_OVERFLOW")?;
   components.push((pos,domain));
  }
  Ok(Self{elements:n,components,size})
 }
 pub fn full(start:&[i64])->Result<Self,&'static str>{
  let domain=MultisetDomain::new(start)?;
  Ok(Self{elements:start.len(),components:vec![((0..start.len() as u32).collect(),domain.clone())],size:domain.size()})
 }
 pub fn size(&self)->u64{self.size}
 pub fn rank(&self,state:&[i64])->Result<u64,&'static str>{
  if state.len()!=self.elements{return Err("BITMAP_STATE_LENGTH")}
  let mut rank=0u64;
  for (positions,domain) in &self.components{
   let values:Vec<_>=positions.iter().map(|&i|state[i as usize]).collect();
   let digit=domain.rank(&values)?;
   rank=rank.checked_mul(domain.size()).and_then(|v|v.checked_add(digit)).ok_or("BITMAP_ORBIT_RANK")?;
  }
  if rank>=self.size{return Err("BITMAP_RANK_RANGE")}Ok(rank)
 }
 pub fn unrank(&self,rank:u64)->Result<Vec<i64>,&'static str>{
  if rank>=self.size{return Err("BITMAP_RANK_RANGE")}
  let mut remaining=rank;let mut state=vec![0i64;self.elements];
  for (positions,domain) in self.components.iter().rev(){
   let digit=remaining%domain.size();remaining/=domain.size();
   let values=domain.unrank(digit)?;
   for (&position,value) in positions.iter().zip(values){state[position as usize]=value;}
  }
  if remaining!=0{return Err("BITMAP_ORBIT_RANK")}Ok(state)
 }
 /// Returns exact generator masks when the variable components are pairs.
 /// The least significant bit belongs to the last nonconstant component.
 pub(crate) fn xor_generator_masks(&self, generators: &[Vec<u32>]) -> Option<Vec<u64>> {
  if !self.components.iter().all(|(positions, domain)|
   positions.len() == 1 || (positions.len() == 2 && domain.size() == 2))
  {
   return None;
  }
  generators.iter().map(|generator| {
   if generator.len() != self.elements { return None; }
   let mut mask = 0;
   for (positions, _) in &self.components {
    let first = positions[0];
    if positions.len() == 1 {
     if generator[first as usize] != first { return None; }
     continue;
    }
    let second = positions[1];
    let swapped = match (generator[first as usize], generator[second as usize]) {
     pair if pair == (first, second) => false,
     pair if pair == (second, first) => true,
     _ => return None,
    };
    mask = (mask << 1) | u64::from(swapped);
   }
   Some(mask)
  }).collect()
 }
 /// Layout consumed by a capability-checked CUDA bitmap rank implementation.
 /// The single-component layout is exactly the historical ABI. For multiple
 /// components, the high bit denotes descriptors of seven u32 words each:
 /// length, distinct, positions offset, symbols offset, counts offset,
 /// radix low, radix high. Offsets address flat, immutable per-job buffers.
 pub fn device_layout(&self)->(Vec<i64>,Vec<u32>,u32){
  if self.components.len()==1{
   let domain=&self.components[0].1;
   return (domain.symbols().to_vec(),domain.counts().to_vec(),domain.symbols().len() as u32)
  }
  let mut symbols=Vec::new();let mut metadata=vec![0u32;self.components.len()*7];
  for (i,(positions,domain)) in self.components.iter().enumerate(){
   let base=i*7;metadata[base]=positions.len() as u32;metadata[base+1]=domain.symbols().len() as u32;
   metadata[base+3]=symbols.len() as u32;symbols.extend_from_slice(domain.symbols());
   metadata[base+4]=metadata.len() as u32;metadata.extend_from_slice(domain.counts());
   metadata[base+2]=metadata.len() as u32;metadata.extend_from_slice(positions);
   metadata[base+5]=domain.size() as u32;metadata[base+6]=(domain.size()>>32) as u32;
  }
  (symbols,metadata,0x8000_0000u32|self.components.len() as u32)
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


/// A range-preserving permutation of exact IDs for optional bitmap ownership.
/// Full world*page stripes are rotated; the final incomplete stripe keeps the
/// cyclic layout. Thus the bitmap's size and rank-local lengths do not change.
#[derive(Clone,Copy,Debug)]
pub struct BitmapPageMap{total:u64,world:u32,page:u32,boundary:u64}
impl BitmapPageMap{
 pub fn new(total:u64,world:u32,page:u32)->Result<Self,&'static str>{
  if total==0||world==0||world>128||page>(1<<20)||(page!=0&&!page.is_power_of_two()){return Err("BITMAP_PAGE_MAP_RANGE")}
  let boundary=if page==0{0}else{let stripe=u64::from(page)*u64::from(world);total-total%stripe};
  Ok(Self{total,world,page,boundary})
 }
 pub fn boundary(self)->u64{self.boundary}
 fn rotation(stripe:u64,world:u32)->u64{
  let mut x=stripe;x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);(x^(x>>31))%u64::from(world)
 }
 pub fn map(self,rank:u64)->Result<u64,&'static str>{
  if rank>=self.total{return Err("BITMAP_RANK_RANGE")}
  if rank>=self.boundary{return Ok(rank)}
  let page=u64::from(self.page);let world=u64::from(self.world);let p=rank/page;let stripe=p/world;let slot=p%world;
  let owner=(slot+Self::rotation(stripe,self.world))%world;
  let local=stripe.checked_mul(page).and_then(|v|v.checked_add(rank%page)).ok_or("BITMAP_PAGE_MAP_OVERFLOW")?;
  local.checked_mul(world).and_then(|v|v.checked_add(owner)).filter(|&v|v<self.boundary).ok_or("BITMAP_PAGE_MAP_OVERFLOW")
 }
 pub fn unmap(self,mapped:u64)->Result<u64,&'static str>{
  if mapped>=self.total{return Err("BITMAP_RANK_RANGE")}
  if mapped>=self.boundary{return Ok(mapped)}
  let page=u64::from(self.page);let world=u64::from(self.world);let owner=mapped%world;let local=mapped/world;
  let stripe=local/page;let slot=(owner+world-Self::rotation(stripe,self.world))%world;
  stripe.checked_mul(world).and_then(|v|v.checked_add(slot)).and_then(|v|v.checked_mul(page)).and_then(|v|v.checked_add(local%page)).filter(|&v|v<self.boundary).ok_or("BITMAP_PAGE_MAP_OVERFLOW")
 }
}

#[cfg(test)]
mod page_map_tests {
 use super::{BitmapPartition,BitmapPageMap};
 #[test]
 fn page_map_is_a_bijection_with_a_cyclic_short_tail(){
  for world in [1u32,3,5,7,8] { for page in [0u32,1,2,8,32,64] { for total in 1..=129u64 {
   let map=BitmapPageMap::new(total,world,page).unwrap();
   let partition=BitmapPartition::new(total,world).unwrap();
   let mut seen=vec![false;total as usize];
   for rank in 0..total {
    let mapped=map.map(rank).unwrap();assert!(mapped<total);assert!(!seen[mapped as usize]);seen[mapped as usize]=true;
    assert_eq!(map.unmap(mapped).unwrap(),rank);
    let (owner,local)=partition.locate(mapped).unwrap();
    assert_eq!(partition.reconstruct(owner,local).unwrap(),mapped);
   }
   assert!(seen.into_iter().all(|v|v));
  }}}
 }
 #[test]
 fn page_map_handles_large_arithmetic_without_wrapping(){
  for world in [1u32,3,5,7,8,127,128] { for page in [0u32,1,8,32,64,256,1048576] {
   for total in [1u64,1000,1u64<<32,(1u64<<63)+137,u64::MAX] {
    let map=BitmapPageMap::new(total,world,page).unwrap();
    for rank in [0,total/2,total-1] {let mapped=map.map(rank).unwrap();assert!(mapped<total);assert_eq!(map.unmap(mapped).unwrap(),rank);}
   }
  }}
  assert!(BitmapPageMap::new(4,8,3).is_err());
  assert!(BitmapPageMap::new(4,8,1<<21).is_err());
  assert!(BitmapPageMap::new(4,0,1).is_err());
 }
}

#[cfg(test)]
mod position_orbit_tests {
 use super::{BitmapPartition,PositionOrbitDomain};
 fn check_closed(start:Vec<i64>,gens:Vec<Vec<u32>>,expected:u64){
  let d=PositionOrbitDomain::new(&start,&gens).unwrap();assert_eq!(d.size(),expected);
  let mut seen=std::collections::BTreeSet::new();
  for rank in 0..d.size(){let state=d.unrank(rank).unwrap();assert!(seen.insert(state.clone()));assert_eq!(d.rank(&state).unwrap(),rank);
   for g in &gens {let child:Vec<_>=g.iter().map(|&i|state[i as usize]).collect();let child_rank=d.rank(&child).unwrap();assert_eq!(d.unrank(child_rank).unwrap(),child);}
  }
  assert_eq!(seen.len() as u64,expected);assert!(d.unrank(expected).is_err());
 }
 // Decode the flat descriptor independently of the domain's rank method;
 // this checks the exact metadata read by the CUDA product-domain path.
 fn metadata_rank(d:&PositionOrbitDomain,state:&[i64])->u64{
  let (symbols,meta,kind)=d.device_layout();assert!(kind&0x8000_0000!=0);let components=(kind&0x7fff_ffff) as usize;
  let mut value=0u128;
  for i in 0..components{
   let m=&meta[i*7..i*7+7];let n=m[0] as usize;let count=m[1] as usize;
   let positions=&meta[m[2] as usize..m[2] as usize+n];let alphabet=&symbols[m[3] as usize..m[3] as usize+count];
   let mut counts=meta[m[4] as usize..m[4] as usize+count].to_vec();
   let radix=u128::from(m[5])|(u128::from(m[6])<<32);let mut ways=radix;let mut digit=0u128;
   for (j,&p) in positions.iter().enumerate(){let value=state[p as usize];let k=alphabet.binary_search(&value).unwrap();let smaller:u128=counts[..k].iter().map(|&c|u128::from(c)).sum();
    digit+=ways*smaller/(n-j) as u128;ways=ways*u128::from(counts[k])/(n-j) as u128;counts[k]-=1;}
   value=value*radix+digit;
  }
  value as u64
 }
 #[test] fn metadata_matches_exact_host_rank_and_closed_children(){
  for (start,gens) in [
   (vec![-7,8,10,12,90,91],vec![vec![1,0,3,2,4,5]]),
   (vec![1,1,2,3,3],vec![vec![2,1,0,4,3],vec![1,0,2,3,4]]),
   (vec![0,1,2,3,4,5,6],vec![vec![1,2,0,4,3,6,5]])
  ]{
   let d=PositionOrbitDomain::new(&start,&gens).unwrap();
   for rank in 0..d.size(){let state=d.unrank(rank).unwrap();assert_eq!(metadata_rank(&d,&state),rank);}
  }
 }
 #[test]
 fn pair_orbit_masks_match_original_generator_ids_and_exact_ranks() {
  let start = [7, 99, -3, -15, 2];
  let generators = vec![vec![2, 1, 0, 4, 3], vec![0, 1, 2, 4, 3], vec![0, 1, 2, 3, 4]];
  let domain = PositionOrbitDomain::new(&start, &generators).unwrap();
  let masks = domain.xor_generator_masks(&generators).unwrap();
  assert_eq!(masks, [3, 1, 0]);
  for rank in 0..domain.size() {
   let state = domain.unrank(rank).unwrap();
   for (generator, mask) in generators.iter().zip(&masks) {
    let child: Vec<_> = generator.iter().map(|&position| state[position as usize]).collect();
    assert_eq!(domain.rank(&child).unwrap(), rank ^ mask);
   }
  }
  assert!(domain.xor_generator_masks(&[vec![1, 0, 2, 3, 4]]).is_none());
  let cycle = vec![vec![1, 2, 0]];
  let ternary = PositionOrbitDomain::new(&[0, 1, 2], &cycle).unwrap();
  assert!(ternary.xor_generator_masks(&cycle).is_none());
 }
 #[test] fn component_product_is_closed_even_when_generators_are_coupled(){
  check_closed(vec![-7,8,10,12,90,91],vec![vec![1,0,3,2,4,5]],4);
  check_closed(vec![1,1,2,3,3],vec![vec![2,1,0,4,3],vec![1,0,2,3,4]],3);
  check_closed(vec![2,0,1],vec![vec![1,0,2],vec![0,2,1]],6);
 }
 #[test] fn fixed_positions_shrink_s10_plus_four_domain(){
  let start:Vec<i64>=(0..14).collect();let mut gens=Vec::new();
  for i in 0..9 {let mut g:Vec<u32>=(0..14).collect();g.swap(i,i+1);gens.push(g);}
  let d=PositionOrbitDomain::new(&start,&gens).unwrap();assert_eq!(d.size(),3628800);
  assert_eq!(BitmapPartition::new(d.size(),8).unwrap().owner_bytes(0).unwrap(),56704);
  for rank in [0,1,1024,d.size()-1]{let s=d.unrank(rank).unwrap();assert_eq!(&s[10..],&start[10..]);assert_eq!(d.rank(&s).unwrap(),rank);}
  let mut invalid=start.clone();invalid.swap(0,13);assert!(d.rank(&invalid).is_err());
 }
 #[test] fn arithmetic_checks_true_domain_and_validates_generators(){
  let start:Vec<i64>=(0..21).collect();let identity:Vec<u32>=(0..21).collect();
  assert_eq!(PositionOrbitDomain::new(&start,&[identity.clone()]).unwrap().size(),1);
  let cycle:Vec<u32>=(0..21).map(|x|(x+1)%21).collect();
  assert!(PositionOrbitDomain::new(&start,&[cycle]).is_err());
  let mut bad=identity.clone();bad[0]=bad[1];assert!(PositionOrbitDomain::new(&start,&[bad]).is_err());
  assert!(PositionOrbitDomain::new(&start,&[vec![99;21]]).is_err());
 }
}
