//! Native two-rank NCCL DENSE BFS reference. Torchrun supplies only rank env.
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup, Result};
use mgbfs_cuda::{ffi::*, native_owner::*};
use std::ffi::{c_void, CStr};

#[derive(Clone, Copy)]
pub struct DistributedConfig {
    pub rank: u32,
    pub world: u32,
    pub logical_owner_to_rank: [u32; 2],
    pub batch: u32,
    pub layer_capacity: u32,
    pub future_capacity: u32,
    pub prededup: bool,
    pub generation_variant: u32,
}
fn check(status: i32) -> Result<()> { if status==0 { Ok(()) } else { Err(format!("CUDA_STATUS_{status}")) } }
struct Buffer { ptr:*mut c_void, bytes:usize, stream:*mut c_void }
impl Buffer {
    fn new(bytes:usize,stream:*mut c_void)->Result<Self>{let mut ptr=std::ptr::null_mut();check(unsafe{cudaMalloc(&mut ptr,bytes.max(1))})?;let x=Self{ptr,bytes,stream};check(unsafe{cudaMemsetAsync(ptr,0,bytes.max(1),stream)})?;Ok(x)}
    fn put<T:Copy>(&self,x:&[T])->Result<()>{if std::mem::size_of_val(x)>self.bytes{return Err("UPLOAD_CAPACITY".into())}check(unsafe{cudaMemcpyAsync(self.ptr,x.as_ptr().cast(),std::mem::size_of_val(x),1,self.stream)})?;check(unsafe{cudaStreamSynchronize(self.stream)})}
    fn read<T:Copy>(&self,x:&mut[T])->Result<()>{if std::mem::size_of_val(x)>self.bytes{return Err("READ_CAPACITY".into())}check(unsafe{cudaMemcpy(x.as_mut_ptr().cast(),self.ptr,std::mem::size_of_val(x),2)})}
    fn one<T:Copy+Default>(&self)->Result<T>{let mut x=[T::default()];self.read(&mut x)?;Ok(x[0])}
    unsafe fn at(&self,n:usize)->*mut c_void{self.ptr.cast::<u8>().add(n).cast()}
}
impl Drop for Buffer{fn drop(&mut self){unsafe{cudaFree(self.ptr);}}}
struct Plan(*mut c_void,unsafe extern "C" fn(*mut c_void));
impl Plan{fn new(drop:unsafe extern "C" fn(*mut c_void),create:impl FnOnce(*mut *mut c_void,*mut i8)->i32)->Result<Self>{let mut p=std::ptr::null_mut();let mut e=[0i8;512];if create(&mut p,e.as_mut_ptr())!=0{return Err(unsafe{CStr::from_ptr(e.as_ptr())}.to_string_lossy().into_owned())}Ok(Self(p,drop))}}
impl Drop for Plan{fn drop(&mut self){unsafe{self.1(self.0)}}}
struct Stream(*mut c_void);impl Drop for Stream{fn drop(&mut self){unsafe{cudaStreamSynchronize(self.0);cudaStreamDestroy(self.0);}}}
struct Event(*mut c_void);impl Event{fn new()->Result<Self>{let mut p=std::ptr::null_mut();check(unsafe{cudaEventCreateWithFlags(&mut p,2)})?;Ok(Self(p))}}impl Drop for Event{fn drop(&mut self){unsafe{cudaEventDestroy(self.0);}}}
struct Comm(*mut c_void);impl Drop for Comm{fn drop(&mut self){unsafe{mgbfs_nccl_destroy(self.0)}}}

pub struct DistributedNativeBfs {
    cfg:DistributedConfig,width:usize,stride:usize,moves:u32,candidates:u32,depth:u32,current_count:u32,failed:bool,
    stream:Stream,archive_stream:Stream,archive_done:[Event;2],current_bank:usize,archived_depth:Option<u32>,comm:Comm,generate:Plan,hash:Plan,route:Plan,merge:Plan,settle:Plan,materialize:Plan,
    current_states:Buffer,next_states:Buffer,next_hashes:Buffer,next_state:Buffer,children:Buffer,child_hashes:Buffer,
    sorted_hashes:Buffer,sorted_refs:Buffer,route_count:Buffer,packed_states:Buffer,owner_counts:Buffer,
    recv_states:Buffer,recv_hashes:Buffer,recv_count:Buffer,identity_refs:Buffer,
    future_states:Buffer,future_hashes:Buffer,future_state:Buffer,survivor_hashes:Buffer,survivor_refs:Buffer,survivor_count:Buffer,settle_state:Buffer,
    history:Buffer,history_counts_gpu:Buffer,history_counts:[u32;2],collective_send:Buffer,collective_recv:Buffer,
}
impl DistributedNativeBfs {
    pub fn new(graph:&MatrixGroup,seed:[u8;16],id:[u8;128],cfg:DistributedConfig)->Result<Self>{
        graph.validate()?;if cfg.world!=2||cfg.rank>=2||cfg.logical_owner_to_rank[0]==cfg.logical_owner_to_rank[1]||cfg.logical_owner_to_rank.iter().any(|&x|x>=2)||cfg.batch==0||cfg.layer_capacity==0||cfg.future_capacity==0{return Err("DISTRIBUTED_CONFIG".into())}
        check(unsafe{cudaSetDevice(cfg.rank as i32)})?;let width=graph.start.len();let stride=(width+15)&!15;let moves=graph.generators.len() as u32;let candidates=cfg.batch.checked_mul(moves).ok_or("CANDIDATE_OVERFLOW")?;if candidates>i32::MAX as u32{return Err("CANDIDATE_CAPACITY".into())}
        let mut raw=std::ptr::null_mut();check(unsafe{cudaStreamCreateWithFlags(&mut raw,1)})?;let stream=Stream(raw);
        let mut raw_archive=std::ptr::null_mut();check(unsafe{cudaStreamCreateWithFlags(&mut raw_archive,1)})?;let archive_stream=Stream(raw_archive);
        let mut comm=std::ptr::null_mut();let mut error=[0i8;512];if unsafe{mgbfs_nccl_create(cfg.rank,cfg.world,cfg.rank,id.as_ptr().cast(),&mut comm,error.as_mut_ptr(),512)}!=0{return Err(unsafe{CStr::from_ptr(error.as_ptr())}.to_string_lossy().into_owned())}let comm=Comm(comm);
        let contract=GemmHash::from_seed(width,seed)?;let limbs=contract.limbs();let matrices:Vec<u8>=graph.generators.iter().flatten().copied().collect();let weights=vec![1u32;moves as usize];
        let generate=Plan::new(mgbfs_generate_destroy,|out,e|unsafe{mgbfs_generate_create_macro_variant(graph.rows as u32,moves,graph.modulus as u32,cfg.batch,matrices.as_ptr(),weights.as_ptr(),cfg.generation_variant,out,e,512)})?;
        let hash=Plan::new(mgbfs_hash_destroy,|out,e|unsafe{mgbfs_hash_create(width as u32,candidates,limbs.as_ptr(),contract.offsets.as_ptr(),out,e,512)})?;
        let max_records=candidates.max(cfg.future_capacity);let route=Plan::new(mgbfs_route_destroy,|out,e|unsafe{mgbfs_route_create(max_records,out,e,512)})?;
        let merge=Plan::new(mgbfs_future_merge_destroy,|out,e|unsafe{mgbfs_future_merge_create(stride as u32,cfg.future_capacity,candidates,out,e,512)})?;
        let settle=Plan::new(mgbfs_macro_settle_destroy,|out,e|unsafe{mgbfs_macro_settle_create(cfg.future_capacity,2,cfg.layer_capacity,out,e,512)})?;
        let materialize=Plan::new(mgbfs_materialize_destroy,|out,e|unsafe{mgbfs_materialize_create(stride as u32,cfg.future_capacity,cfg.layer_capacity,out,e,512)})?;
        let b=|n|Buffer::new(n,raw);let state_bytes=cfg.layer_capacity as usize*stride;let current_states=b(state_bytes)?;let history=b(2*cfg.layer_capacity as usize*16)?;let history_counts_gpu=b(8)?;
        let start_hash=contract.hash(&graph.start)?;let start_owner=(start_hash.0[3]>>31) as usize;let start_rank=cfg.logical_owner_to_rank[start_owner];let current_count=(start_rank==cfg.rank) as u32;
        if current_count==1{let mut start=vec![0u8;stride];start[..width].copy_from_slice(&graph.start);current_states.put(&start)?;check(unsafe{cudaMemcpyAsync(history.ptr,start_hash.to_le_bytes().as_ptr().cast(),16,1,raw)})?;}
        let history_counts=[current_count,0];history_counts_gpu.put(&history_counts)?;let identity_refs=b(max_records as usize*8)?;identity_refs.put(&(0..u64::from(max_records)).collect::<Vec<_>>())?;
        let archive_done=[Event::new()?,Event::new()?];check(unsafe{cudaEventRecord(archive_done[0].0,raw)})?;check(unsafe{cudaEventRecord(archive_done[1].0,raw)})?;check(unsafe{cudaStreamSynchronize(raw)})?;
        let result=Self{cfg,width,stride,moves,candidates,depth:0,current_count,failed:false,stream,archive_stream,archive_done,current_bank:0,archived_depth:None,comm,generate,hash,route,merge,settle,materialize,
            current_states,next_states:b(state_bytes)?,next_hashes:b(cfg.layer_capacity as usize*16)?,next_state:b(8)?,children:b(candidates as usize*stride)?,child_hashes:b(candidates as usize*16)?,sorted_hashes:b(max_records as usize*16)?,sorted_refs:b(max_records as usize*8)?,route_count:b(4)?,packed_states:b(candidates as usize*stride)?,owner_counts:b(8)?,recv_states:b(candidates as usize*stride)?,recv_hashes:b(candidates as usize*16)?,recv_count:b(4)?,identity_refs,
            future_states:b(cfg.future_capacity as usize*stride)?,future_hashes:b(cfg.future_capacity as usize*16)?,future_state:b(8)?,survivor_hashes:b(cfg.future_capacity as usize*16)?,survivor_refs:b(cfg.future_capacity as usize*8)?,survivor_count:b(4)?,settle_state:b(std::mem::size_of::<MacroSettleState>())?,history,history_counts_gpu,history_counts,collective_send:b(4)?,collective_recv:b(8)?};
        result.all_max(0)?;Ok(result)
    }
    pub fn depth(&self)->u32{self.depth} pub fn frontier_len(&self)->u32{self.current_count}
    fn all_max(&self,value:u32)->Result<u32>{self.collective_send.put(&[value])?;check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.0,self.collective_send.ptr.cast(),self.collective_recv.ptr.cast(),self.stream.0)})?;check(unsafe{cudaStreamSynchronize(self.stream.0)})?;self.collective_recv.one()}
    fn merge_run(&self,old_bound:u32,states:*const u8,source_count:u32,hashes:*const c_void,refs:*const u64,count:*const u32,incoming_bound:u32)->Result<()>{check(unsafe{mgbfs_future_merge_run_bounded(self.merge.0,self.future_states.ptr.cast(),self.future_hashes.ptr,self.future_state.ptr.cast(),old_bound,states,source_count,hashes,refs,count,incoming_bound,self.stream.0)})}
    fn produce(&mut self)->Result<()> {
        self.future_state.put(&[FrontierState::default()])?;let mut offset=0u32;let mut future_bound=0u32;let trace=std::env::var_os("MGBFS_TRACE_DEPTHS").is_some();
        loop {
            let parents=if offset<self.current_count{self.cfg.batch.min(self.current_count-offset)}else{0};let count=parents*self.moves;
            if trace{eprintln!("MGBFS_BATCH_BEGIN rank={} depth={} offset={} parents={}",self.cfg.rank,self.depth,offset,parents);}
            if parents>0{unsafe{check(mgbfs_generate_run(self.generate.0,self.current_states.at(offset as usize*self.stride).cast(),self.children.ptr.cast(),parents,self.stream.0))?;check(mgbfs_hash_run(self.hash.0,self.children.ptr.cast(),self.child_hashes.ptr.cast(),count,self.stream.0))?;}}
            unsafe{check(mgbfs_route_run(self.route.0,self.child_hashes.ptr,self.identity_refs.ptr.cast(),self.sorted_hashes.ptr,self.sorted_refs.ptr.cast(),self.route_count.ptr.cast(),count,self.cfg.prededup as i32,self.stream.0))?;check(cudaStreamSynchronize(self.stream.0))?;}
            let routed=self.route_count.one::<u32>()?;check(unsafe{mgbfs_exchange_pack(self.stride as u32,self.candidates,self.children.ptr.cast(),count,self.sorted_hashes.ptr,self.sorted_refs.ptr.cast(),routed,self.packed_states.ptr.cast(),self.owner_counts.ptr.cast(),self.stream.0)})?;check(unsafe{cudaStreamSynchronize(self.stream.0)})?;
            let mut owner_counts=[0u32;2];self.owner_counts.read(&mut owner_counts)?;if owner_counts[0]==u32::MAX{return Err("EXCHANGE_SOURCE_REF".into())}
            let local_owner=self.cfg.logical_owner_to_rank.iter().position(|&x|x==self.cfg.rank).unwrap();let send_owner=local_owner^1;let local_offset=if local_owner==0{0}else{owner_counts[0]};let send_offset=if send_owner==0{0}else{owner_counts[0]};
            self.collective_send.put(&[owner_counts[send_owner]])?;check(unsafe{mgbfs_nccl_send_recv(self.comm.0,self.collective_send.ptr,4,self.cfg.rank^1,self.recv_count.ptr,4,self.stream.0)})?;check(unsafe{cudaStreamSynchronize(self.stream.0)})?;let received=self.recv_count.one::<u32>()?;if received>self.candidates{return Err("EXCHANGE_CAPACITY".into())}
            check(unsafe{mgbfs_nccl_send_recv(self.comm.0,self.sorted_hashes.at(send_offset as usize*16),u64::from(owner_counts[send_owner])*16,self.cfg.rank^1,self.recv_hashes.ptr,u64::from(received)*16,self.stream.0)})?;
            check(unsafe{mgbfs_nccl_send_recv(self.comm.0,self.packed_states.at(send_offset as usize*self.stride),u64::from(owner_counts[send_owner])*self.stride as u64,self.cfg.rank^1,self.recv_states.ptr,u64::from(received)*self.stride as u64,self.stream.0)})?;
            self.route_count.put(&[owner_counts[local_owner]])?;self.merge_run(future_bound,unsafe{self.packed_states.at(local_offset as usize*self.stride).cast()},owner_counts[local_owner],unsafe{self.sorted_hashes.at(local_offset as usize*16)},self.identity_refs.ptr.cast(),self.route_count.ptr.cast(),owner_counts[local_owner])?;future_bound=future_bound.checked_add(owner_counts[local_owner]).ok_or("FUTURE_BOUND")?.min(self.cfg.future_capacity);
            self.merge_run(future_bound,self.recv_states.ptr.cast(),received,self.recv_hashes.ptr,self.identity_refs.ptr.cast(),self.recv_count.ptr.cast(),received)?;future_bound=future_bound.checked_add(received).ok_or("FUTURE_BOUND")?.min(self.cfg.future_capacity);check(unsafe{cudaStreamSynchronize(self.stream.0)})?;let future=self.future_state.one::<FrontierState>()?;if future.fatal!=0{return Err(format!("FUTURE_FATAL_{}",future.fatal))}
            offset=offset.saturating_add(parents);let more=(offset<self.current_count) as u32;let any=self.all_max(more)?;if trace{eprintln!("MGBFS_BATCH_END rank={} depth={} offset={} future={} more={} any={}",self.cfg.rank,self.depth,offset,future.count,more,any);}if any==0{break}
        } Ok(())
    }
    fn settle(&mut self,target:u32)->Result<u32>{check(unsafe{cudaStreamWaitEvent(self.stream.0,self.archive_done[self.current_bank^1].0,0)})?;self.next_state.put(&[FrontierState::default()])?;let future=self.future_state.one::<FrontierState>()?;self.route_count.put(&[future.count])?;unsafe{check(mgbfs_macro_settle_run(self.settle.0,self.future_hashes.ptr,self.identity_refs.ptr.cast(),self.route_count.ptr.cast(),self.history.ptr,self.history_counts_gpu.ptr.cast(),self.survivor_hashes.ptr,self.survivor_refs.ptr.cast(),self.survivor_count.ptr.cast(),self.settle_state.ptr.cast(),u64::from(target)+1,self.stream.0))?;check(mgbfs_materialize_run(self.materialize.0,self.future_states.ptr.cast(),future.count,self.survivor_hashes.ptr,self.survivor_refs.ptr.cast(),self.survivor_count.ptr.cast(),self.next_states.ptr.cast(),self.next_hashes.ptr,self.next_state.ptr.cast(),self.stream.0))?;check(cudaStreamSynchronize(self.stream.0))?;}let state=self.next_state.one::<FrontierState>()?;let settled=self.settle_state.one::<MacroSettleState>()?;if state.fatal!=0||settled.fatal!=0||state.count!=settled.count{return Err(format!("SETTLE_FATAL_{}_{}",state.fatal,settled.fatal))}let slot=(target%2) as usize;check(unsafe{cudaMemcpyAsync(self.history.at(slot*self.cfg.layer_capacity as usize*16),self.survivor_hashes.ptr,state.count as usize*16,3,self.stream.0)})?;self.history_counts[slot]=state.count;self.history_counts_gpu.put(&self.history_counts)?;Ok(state.count)}
    pub fn advance(&mut self)->Result<bool>{if self.failed{return Err("DISTRIBUTED_FAILED".into())}let result=self.advance_inner();if result.is_err(){self.failed=true}result}
    fn advance_inner(&mut self)->Result<bool>{self.produce()?;let target=self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;let count=self.settle(target)?;self.depth=target;std::mem::swap(&mut self.current_states,&mut self.next_states);self.current_bank^=1;self.current_count=count;Ok(self.all_max((count>0) as u32)?!=0)}
    pub fn archive_current(&mut self,archive:&mut crate::pinned_archive::PinnedArchive)->Result<()>{if self.failed{return Err("DISTRIBUTED_FAILED".into())}if archive.width!=self.width{return Err("ARCHIVE_STATE_WIDTH".into())}if self.archived_depth==Some(self.depth){return Err("ARCHIVE_DEPTH_ALREADY_SUBMITTED".into())}let s=self.archive_stream.0;let hashes=unsafe{self.history.at((self.depth%2)as usize*self.cfg.layer_capacity as usize*16)};let mut offset=0u32;while offset<self.current_count{let n=archive.rows.min(self.cfg.batch).min(self.current_count-offset);let slot=archive.acquire()?;let copied=(||unsafe{let states=self.current_states.at(offset as usize*self.stride);check(cudaMemcpy2DAsync(slot.ptr,self.width,states,self.stride,self.width,n as usize,2,s))?;check(cudaMemcpyAsync(slot.ptr.cast::<u8>().add(n as usize*self.width).cast(),hashes.cast::<u8>().add(offset as usize*16).cast(),n as usize*16,2,s))?;check(cudaEventRecord(slot.ready,s))})();if let Err(e)=copied{unsafe{cudaStreamSynchronize(s);}self.failed=true;return Err(e)}archive.submit(slot,u64::from(self.depth),n)?;offset+=n}check(unsafe{cudaEventRecord(self.archive_done[self.current_bank].0,s)})?;archive.layer(u64::from(self.depth),u64::from(self.current_count))?;self.archived_depth=Some(self.depth);Ok(())}
    pub fn snapshot(&self)->Result<Vec<Vec<u8>>>{check(unsafe{cudaStreamSynchronize(self.stream.0)})?;let mut bytes=vec![0u8;self.current_count as usize*self.stride];check(unsafe{cudaMemcpy(bytes.as_mut_ptr().cast(),self.current_states.ptr,bytes.len(),2)})?;Ok(bytes.chunks_exact(self.stride).map(|x|x[..self.width].to_vec()).collect())}
}
