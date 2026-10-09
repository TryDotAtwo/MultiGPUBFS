//! Injective fixed-width state encoding followed by a reversible 128-bit permutation.
use crate::{hash::Hash128, Result};
fn diffuse(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
pub fn pack(state:&[u8],bits:u32)->Result<Hash128>{
 if state.is_empty()||!(1..=8).contains(&bits)||state.len().checked_mul(bits as usize).ok_or("LOSSLESS_KEY_DOMAIN")?>128||state.iter().any(|&x|u32::from(x)>=(1u32<<bits)){return Err("LOSSLESS_KEY_DOMAIN".into());}
 let mut encoded=0u128;for (j,&x) in state.iter().enumerate(){encoded|=u128::from(x)<<(j as u32*bits);}
 let mut lo=encoded as u64;let mut hi=(encoded>>64) as u64;lo^=diffuse(hi^0x9e3779b97f4a7c15);hi^=diffuse(lo^0xd1b54a32d192ed03);Ok(Hash128([lo as u32,(lo>>32)as u32,hi as u32,(hi>>32)as u32]))
}
#[cfg(test)]mod tests{use super::*;#[test]fn bounds(){assert!(pack(&[31;25],5).is_ok());assert!(pack(&[31;26],5).is_err());assert!(pack(&[32],5).is_err());assert!(pack(&[],1).is_err());}#[test]fn fixture(){let k=pack(&[0,1,2,3,4,5,6,7],3).unwrap();let mut hi=u64::from(k.0[2])|(u64::from(k.0[3])<<32);let mut lo=u64::from(k.0[0])|(u64::from(k.0[1])<<32);hi^=diffuse(lo^0xd1b54a32d192ed03);lo^=diffuse(hi^0x9e3779b97f4a7c15);let word=u128::from(lo)|(u128::from(hi)<<64);for j in 0..8{assert_eq!((word>>(j*3))&7,j);}}}
