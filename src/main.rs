use std::arch::x86_64::*;
use std::iter;

use bitburst::simd::avx2::add;

pub fn saxpy_avx(a: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;

    unsafe {
        let alpha_vec = _mm256_set1_ps(a);

        while i + 8 <= n {
            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_ps(y.as_ptr().add(i));
            let result = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }

        // Fallback to scalar ops for the tail
        while i < n {
            y[i] = a * x[i] + y[i];
            i += 1;
        }
    }
}

fn main() {
    let n = 800;

    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut y: Vec<f32> = iter::repeat(1.0_f32).take(n).collect();

    unsafe { add::add_slices(x.as_slice(), y.as_mut_slice(), 1) };
}
