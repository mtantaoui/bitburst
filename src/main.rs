#[cfg(target_arch = "x86_64")]
use std::arch::asm;
use std::arch::x86_64::{
    __m256d, _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd,
};

/// Performs DAXPY (y = alpha * x + y) on the first 4 elements of the slices
/// using inline assembly (AVX/FMA), requiring 32-byte alignment for x and y.
///
/// This function directly implements the logic of the provided C assembly snippet
/// using Rust's `asm!` macro.
///
/// # Safety
///
/// This function is unsafe because:
/// 1. It executes raw assembly instructions.
/// 2. It requires the `target-feature=+avx,+fma` (or compatible like +avx2)
///    to be enabled for the compilation target.
/// 3. The memory locations pointed to by the start of the `x` and `y` slices **MUST**
///    be aligned to 32 bytes due to `vmovapd`/`vmovaps`. Failure will likely crash.
/// 4. The `x` and `y` slices MUST contain at least 4 elements.
/// 5. It assumes `alpha`, `x`, and `y` point to valid memory locations.
#[cfg(avx2)]
#[target_feature(enable = "avx,fma")]
#[inline] // Enable features just for this function
pub unsafe fn bl_daxpy_asm_4x1(alpha: f64, x: *const f64, y: *mut f64) {
    asm!(
        "vbroadcastsd ymm2, xmm2",         // ymm2 = [alpha, alpha, alpha, alpha]
        "vmovupd      ymm0, [{x_ptr}]",    // ymm0 = *x
        "vmovupd      ymm1, [{y_ptr}]",    // ymm1 = *y
        "vfmadd231pd  ymm1, ymm2, ymm0",   // ymm1 = ymm2 * ymm0 + ymm1
        "vmovupd      [{y_ptr}], ymm1",    // store result to y

        x_ptr = in(reg) x,
        y_ptr = in(reg) y,
        in("xmm2") alpha,

        out("ymm0") _,
        out("ymm1") _,
        lateout("ymm2") _,

        options(nostack, preserves_flags)
    );
}
// --- Fallback for platforms without AVX/FMA (or if not enabled) ---
// (Identical fallback as the intrinsic version)
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
#[inline]
pub unsafe fn bl_daxpy_asm_4x1_rust(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert!(x.len() >= 4, "x slice must have at least 4 elements");
    assert!(y.len() >= 4, "y slice must have at least 4 elements");
    // eprintln!("Warning: Fallback used for bl_daxpy_asm_4x1_rust (inline asm version)");
    for i in 0..4 {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

pub unsafe fn bl_daxpy_4x1(alpha: f64, x: *const f64, y: *mut f64) {
    let x_vec: __m256d = _mm256_loadu_pd(x); // Load x[0..4]
    let y_vec: __m256d = _mm256_loadu_pd(y); // Load y[0..4]
    let a_vec: __m256d = _mm256_broadcast_sd(&alpha); // Broadcast alpha
    let result: __m256d = _mm256_fmadd_pd(a_vec, x_vec, y_vec); // FMA: alpha * x + y
    _mm256_storeu_pd(y, result); // Store result back to y
}
// --- Example Usage (requires careful alignment) ---
fn main() {
    // IMPORTANT: Standard Vec allocation does NOT guarantee 32-byte alignment.
    let mut x_storage = vec![0.0f64; 100]; // Use standard vec, alignment not guaranteed
    let mut y_storage = vec![0.0f64; 100];

    let x_ptr = x_storage.as_mut_ptr();
    let y_ptr = y_storage.as_mut_ptr();
    let x_addr = x_ptr as usize;
    let y_addr = y_ptr as usize;

    let is_aligned = (x_addr % 32 == 0) && (y_addr % 32 == 0);

    println!("Alignment Check (Required 32 bytes):");
    println!(
        "  x address: 0x{:x} -> Aligned: {}",
        x_addr,
        x_addr % 32 == 0
    );
    println!(
        "  y address: 0x{:x} -> Aligned: {}",
        y_addr,
        y_addr % 32 == 0
    );
    if !is_aligned {
        println!(
            "!!! WARNING: Data is likely NOT aligned. Calling the asm! function may crash !!!"
        );
    }

    let x = &mut x_storage[0..4];
    let y = &mut y_storage[0..4];
    x.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    y.copy_from_slice(&[10.0, 11.0, 12.0, 13.0]);

    println!("\nBefore: y = {:?}", y);

    // Call the function - UNSAFE if not aligned
    // Correct version (asm! or fallback) is chosen by cfg attributes.
    unsafe {
        bl_daxpy_asm_4x1(1.0, x.as_ptr(), y.as_mut_ptr());
    }

    println!("After:  y = {:?}", y);

    let expected = [12.0, 15.0, 18.0, 21.0];
    let mut mismatch = false;
    for i in 0..4 {
        if (y[i] - expected[i]).abs() > 1e-9 {
            mismatch = true;
            break;
        }
    }
    if mismatch {
        println!("Result MISMATCH! Expected: {:?}", expected);
    } else {
        println!("Result matches expected: {:?}", expected);
    }
}
// Compile with AVX/FMA enabled:
// RUSTFLAGS="-C target-cpu=native" cargo run --release
// or specifically:
// RUSTFLAGS="-C target-feature=+avx,+fma" cargo run --release
