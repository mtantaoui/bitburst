use super::f32x8;
use super::f32x8::F32x8;
use crate::simd::avx2::f32x8::add;
use crate::simd::vec::SimdVec;

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;

/// .
///
/// # Safety
///
/// .
#[target_feature(enable = "avx")]
pub fn add_slices(a: &[f32], b: &[f32], l2_cache_block_size: usize) -> Vec<f32> {
    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len(), "{}", msg);

    // Allocate result vector inside the function, as it's part of the work
    let size = a.len();
    let mut c = vec![0.0; size];

    c.chunks_mut(l2_cache_block_size) // Use the parameter here
        .zip(a.chunks(l2_cache_block_size))
        .zip(b.chunks(l2_cache_block_size))
        .for_each(|((c_block, a_block), b_block)| {
            add_block(a_block, b_block, c_block);
        });

    // c.par_chunks_mut(l2_cache_block_size) // Use the parameter here
    // .zip(a.par_chunks(l2_cache_block_size))
    // .zip(b.par_chunks(l2_cache_block_size))
    // .for_each(|((c_block, a_block), b_block)| {
    //     add_block(a_block, b_block, c_block);
    // });

    // let chunk_size = f32x8::LANE_COUNT;

    // c.par_chunks_mut(chunk_size)
    //     .zip(a.par_chunks(chunk_size))
    //     .zip(b.par_chunks(chunk_size))
    //     .for_each(|((c, a), b)| {
    //         let a_chunk = F32x8::new(a);
    //         let b_chunk = F32x8::new(b);

    //         match c.len().cmp(&chunk_size) {
    //             std::cmp::Ordering::Less => unsafe {
    //                 // (a_chunk + b_chunk).store_at_partial(c.as_mut_ptr())
    //                 add(a_chunk, b_chunk).store_at_partial(c.as_mut_ptr())
    //             },
    //             std::cmp::Ordering::Equal => unsafe {
    //                 // (a_chunk + b_chunk).store_at(c.as_mut_ptr())
    //                 add(a_chunk, b_chunk).store_at(c.as_mut_ptr())
    //             },
    //             std::cmp::Ordering::Greater => unreachable!("Add_block function"),
    //         }
    //     });
    c
}

/// .
///
/// # Safety
///
/// .
#[target_feature(enable = "avx")]
pub fn add_block(a: &[f32], b: &[f32], c: &mut [f32]) {
    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len() && c.len() == a.len(), "{}", msg);

    let chunk_size = f32x8::LANE_COUNT;

    c.chunks_mut(chunk_size)
        .zip(a.chunks(chunk_size))
        .zip(b.chunks(chunk_size))
        .for_each(|((c, a), b)| {
            let a_chunk = F32x8::new(a);
            let b_chunk = F32x8::new(b);

            match c.len().cmp(&chunk_size) {
                std::cmp::Ordering::Less => unsafe {
                    // (a_chunk + b_chunk).store_at_partial(c.as_mut_ptr())
                    add(a_chunk, b_chunk).store_at_partial(c.as_mut_ptr())
                },
                std::cmp::Ordering::Equal => unsafe {
                    // (a_chunk + b_chunk).store_at(c.as_mut_ptr())
                    add(a_chunk, b_chunk).store_at(c.as_mut_ptr())
                },
                std::cmp::Ordering::Greater => unreachable!("Something went wrong !!"),
            }
        });
}

#[cfg(test)]
mod f32x8_tests {
    use super::add_slices;
    const VECTOR_LENGTH: usize = 1000;

    fn generate_data(len: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = vec![1.0; len];
        let b: Vec<f32> = vec![1.0; len];
        (a, b)
    }

    #[test]
    fn test_new() {
        let (a_vec, b_vec) = generate_data(VECTOR_LENGTH);

        unsafe {
            add_slices(&a_vec, &b_vec, 1024);
        }
    }
}
