use std::alloc::{self, Layout};
use std::arch::x86_64::{
    __m512, _mm256_add_ps, _mm256_load_ps, _mm256_store_ps, _mm_add_round_sd,
    _MM_FROUND_TO_NEAREST_INT,
};
use std::slice;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::f32x8;
use super::f32x8::F32x8;
use crate::simd::avx2::f32x8::{add, LANE_COUNT};
use crate::simd::vec::SimdVec;

fn align_slice(s: &[f32]) -> Vec<f32> {
    let len = s.len();
    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), 32).expect("Invalid layout");

    let ptr = unsafe { alloc::alloc(layout) as *mut f32 };
    if ptr.is_null() {
        panic!("Allocation failed");
    }

    unsafe {
        ptr.copy_from_nonoverlapping(s.as_ptr(), len);
        Vec::from_raw_parts(ptr, len, len)
    }
}

/// .
///
/// # Safety
///
/// .
#[target_feature(enable = "avx")]
pub fn add_slices(a: &[f32], b: &[f32], _l2_cache_block_size: usize)
// -> Vec<f32>
{
    if !is_x86_feature_detected!("avx") {
        println!("caca")
    } else {
        let msg = format!(
            "Operands must have the same size lhs size:{}, rhs:{}",
            a.len(),
            b.len()
        );
        assert!(a.len() == b.len(), "{}", msg);

        let size = a.len();
        // let a = align_slice(a);
        // let b = align_slice(b);

        (0..size).into_par_iter().step_by(LANE_COUNT).for_each(|i| {
            // println!("{i} --> {}", i + LANE_COUNT);

            let start = i;
            let end = i + LANE_COUNT;

            let a_chunk = unsafe { F32x8::load_aligned(a[start..end].as_ptr(), 8) };
            let b_chunk = unsafe { F32x8::load_aligned(b[start..end].as_ptr(), 8) };

            _mm256_add_ps(
                a_chunk.elements,
                b_chunk.elements,
                // _MM_FROUND_TO_NEAREST_INT,
            );
        });

        // c
    }
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

#[target_feature(enable = "avx")]
pub unsafe fn add_slices_sequential(a: &[f32], b: &[f32], c: &mut [f32]) {
    let size = a.len();
    let num_chunks = size / LANE_COUNT;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..num_chunks {
        // Sequential loop
        let idx = i * LANE_COUNT;
        let a_vec = _mm256_load_ps(a_ptr.add(idx));
        let b_vec = _mm256_load_ps(b_ptr.add(idx));
        let res = _mm256_add_ps(a_vec, b_vec);
        _mm256_store_ps(c_ptr.add(idx), res);
    }
}
#[derive(Debug)]
pub struct AlignedVecF32 {
    ptr: *mut f32,
    len: usize,
}

impl AlignedVecF32 {
    pub fn zeros(len: usize) -> Self {
        assert!(len % 8 == 0, "Length must be a multiple of 8");

        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), 32).unwrap();

        let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut f32 };

        if ptr.is_null() {
            panic!("Memory allocation failed");
        }

        AlignedVecF32 { ptr, len }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        let len = slice.len();
        assert!(len % 8 == 0, "Slice length must be a multiple of 8");

        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), 32).unwrap();

        let ptr = unsafe { alloc::alloc(layout) as *mut f32 };

        if ptr.is_null() {
            panic!("Memory allocation failed");
        }
        unsafe {
            ptr.copy_from_nonoverlapping(slice.as_ptr(), len);
        }
        AlignedVecF32 { ptr, len }
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedVecF32 {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len * std::mem::size_of::<f32>(), 32).unwrap();
        unsafe {
            alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
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
