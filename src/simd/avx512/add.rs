use super::f32x16;
use super::f32x16::F32x16;
use crate::simd::vec::SimdVec;

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;

const CACHE_BLOCK_SIZE: usize = 24_000;

#[cfg(all(avx512, rustc_channel = "nightly"))]
pub fn add_slices(a: &[f32], b: &[f32]) -> Vec<f32> {
    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len(), "{}", msg);

    // Allocate result vector inside the function, as it's part of the work
    let size = a.len();
    let mut c = vec![0.0; size];

    c.par_chunks_mut(CACHE_BLOCK_SIZE) // Use the parameter here
        .zip(a.par_chunks(CACHE_BLOCK_SIZE))
        .zip(b.par_chunks(CACHE_BLOCK_SIZE))
        .for_each(|((c_block, a_block), b_block)| {
            add_block(a_block, b_block, c_block);
        });
    c
}

#[cfg(all(avx512, rustc_channel = "nightly"))]
pub fn add_vecs(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len(), "{}", msg);

    // Allocate result vector inside the function, as it's part of the work
    let size = a.len();
    let mut c = vec![0.0; size];

    c.par_chunks_mut(CACHE_BLOCK_SIZE) // Use the parameter here
        .zip(a.par_chunks(CACHE_BLOCK_SIZE))
        .zip(b.par_chunks(CACHE_BLOCK_SIZE))
        .for_each(|((c_block, a_block), b_block)| {
            add_block(a_block, b_block, c_block);
        });
    c
}

#[cfg(all(avx512, rustc_channel = "nightly"))]
#[inline(always)]
pub fn add_block(a: &[f32], b: &[f32], c: &mut [f32]) {
    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len() && c.len() == a.len(), "{}", msg);

    let chunk_size = f32x16::LANE_COUNT;

    c.chunks_mut(chunk_size)
        .zip(a.chunks(chunk_size))
        .zip(b.chunks(chunk_size))
        .for_each(|((c, a), b)| {
            let a_chunk = F32x16::new(a);
            let b_chunk = F32x16::new(b);

            match c.len().cmp(&chunk_size) {
                std::cmp::Ordering::Less => unsafe {
                    (a_chunk + b_chunk).store_at_partial(c.as_mut_ptr())
                },
                std::cmp::Ordering::Equal => unsafe {
                    (a_chunk + b_chunk).store_at(c.as_mut_ptr())
                },
                std::cmp::Ordering::Greater => unreachable!(),
            }
        });
}
