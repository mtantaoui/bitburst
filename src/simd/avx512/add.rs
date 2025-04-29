use super::f32x16;
use super::f32x16::F32x16;
use crate::simd::vec::SimdVec;

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

#[cfg(all(avx512, rustc_channel = "nightly"))]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    use rayon::slice::ParallelSlice;

    let msg = format!(
        "Operands must have the same size lhs size:{}, rhs:{}",
        a.len(),
        b.len()
    );
    assert!(a.len() == b.len(), "{}", msg);

    let size = a.len();

    let chunk_size = f32x16::LANE_COUNT;

    let mut c = vec![0.0; size];

    a.par_chunks(chunk_size)
        .zip_eq(b.par_chunks(chunk_size))
        .zip_eq(c.par_chunks_mut(chunk_size))
        .for_each(|((a, b), c)| {
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

    c
}

#[cfg(test)]
mod f32_add_tests {

    use ndarray::Array1;

    #[test]
    fn ndarray_test() {
        // Define two 1D arrays (vectors)
        let vec1 = Array1::from(vec![1.0f32; 1000]);
        let vec2 = Array1::from(vec![4.0f32; 1000]);

        // Add them element-wise
        let result = &vec1 + &vec2;

        // Print the result
        // println!("Result: {:?}", result); // Output: [5.0, 7.0, 9.0]
    }
}
