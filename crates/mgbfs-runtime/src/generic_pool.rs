//! Bounded rank-local CUDA pool for changing graph geometry.
use std::{cell::{Cell,RefCell},collections::HashSet,ffi::c_void,ptr};
use mgbfs_core::Result;
use crate::generic_native::check;
extern "C" {
 fn cudaDeviceGetDefaultMemPool(pool:*mut *mut c_void,device:i32)->i32;
 fn cudaMemPoolSetAttribute(pool:*mut c_void,attribute:i32,value:*const c_void)->i32;
 fn cudaMemPoolGetAttribute(pool:*mut c_void,attribute:i32,value:*mut c_void)->i32;
 fn cudaMemPoolTrimTo(pool:*mut c_void,bytes:usize)->i32;
 fn cudaMallocAsync(pointer:*mut *mut c_void,bytes:usize,stream:*mut c_void)->i32;
 fn cudaDeviceSynchronize()->i32;
 fn cudaFreeAsync(pointer:*mut c_void,stream:*mut c_void)->i32;
}
thread_local!{static POOL:Cell<*mut c_void>=const{Cell::new(ptr::null_mut())};static OWNED:RefCell<HashSet<usize>>=RefCell::new(HashSet::new());static FAILED:Cell<i32>=const{Cell::new(0)};}
pub fn activate(device:i32)->Result<()>{
 check(unsafe{mgbfs_cuda::native_owner::cudaSetDevice(device)})?;
 let mut pool=ptr::null_mut();let status=unsafe{cudaDeviceGetDefaultMemPool(&mut pool,device)};
 if status==801{return Ok(());}check(status)?;
 let(mut free,mut total)=(0usize,0usize);check(unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
 let threshold=(free as u64).saturating_sub((1u64<<30).max(free as u64/5));
 check(unsafe{cudaMemPoolSetAttribute(pool,4,(&threshold as *const u64).cast())})?;
 POOL.with(|p|p.set(pool));Ok(())
}
pub fn available()->Result<u64>{check(FAILED.with(|v|v.get()))?;POOL.with(|p|{let pool=p.get();if pool.is_null(){return Ok(0);}let(mut reserved,mut used)=(0u64,0u64);check(unsafe{cudaMemPoolGetAttribute(pool,5,(&mut reserved as *mut u64).cast())})?;check(unsafe{cudaMemPoolGetAttribute(pool,7,(&mut used as *mut u64).cast())})?;Ok(reserved.saturating_sub(used))})}
pub fn allocate(bytes:usize)->Result<Option<*mut c_void>>{check(FAILED.with(|v|v.get()))?;POOL.with(|p|{if p.get().is_null(){return Ok(None);}let mut pointer=ptr::null_mut();check(unsafe{cudaMallocAsync(&mut pointer,bytes,ptr::null_mut())})?;check(unsafe{mgbfs_cuda::ffi::cudaStreamSynchronize(ptr::null_mut())})?;OWNED.with(|p|{p.borrow_mut().insert(pointer as usize);});Ok(Some(pointer))})}
pub fn before_free()->Result<()>{if POOL.with(|p|p.get().is_null()){return Ok(());}check(unsafe{cudaDeviceSynchronize()})}
pub fn external_headroom(required:u64)->Result<()>{POOL.with(|p|{let pool=p.get();if pool.is_null(){return Ok(());}let(mut free,mut total)=(0usize,0usize);check(unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;let need=required.saturating_add(256<<20);if free as u64>=need{return Ok(());}let mut reserved=0u64;check(unsafe{cudaMemPoolGetAttribute(pool,5,(&mut reserved as *mut u64).cast())})?;check(unsafe{cudaMemPoolTrimTo(pool,reserved.saturating_sub(need-free as u64) as usize)})})}

pub fn trim_idle()->Result<()>{check(FAILED.with(|v|v.get()))?;if !POOL.with(|p|p.get().is_null()){check(unsafe{mgbfs_cuda::ffi::cudaStreamSynchronize(ptr::null_mut())})?;}POOL.with(|p|if p.get().is_null(){Ok(())}else{check(unsafe{cudaMemPoolTrimTo(p.get(),0)})})}

pub fn release(pointer:*mut c_void)->bool{let owned=OWNED.with(|p|p.borrow_mut().remove(&(pointer as usize)));if !owned{return false;}let status=unsafe{cudaFreeAsync(pointer,ptr::null_mut())};if status!=0{FAILED.with(|v|v.set(status));}true}
