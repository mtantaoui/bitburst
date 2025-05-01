#[cfg(all(avx512, rustc_channel = "nightly"))]
use bitburst::simd::avx512::add::add_slices;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;

// --- Configuration ---
const VECTOR_LENGTH: usize = 10_000_000; // Choose a representative large size
const BLOCK_SIZES_TO_TEST: &[usize] = &[
    1024, // L1d-ish
    2048, 4096, // Theoretical L1d limit based on 48k cache
    8192, // Lower L2
    16384, 32768, // Mid L2
    65536, 131072, // Upper L2
    262144, // Might be too large, but test if curious
];
// --- End Configuration ---

fn generate_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = vec![1.0; len];
    let b: Vec<f32> = vec![1.0; len];
    (a, b)
}

#[cfg(all(avx512, rustc_channel = "nightly"))]
fn bench_vector_addition(c: &mut Criterion) {
    let (a_vec, b_vec) = generate_data(VECTOR_LENGTH);

    let a_arr = Array1::from_vec(a_vec.clone());
    let b_arr = Array1::from_vec(b_vec.clone());

    let mut group = c.benchmark_group("VectorAdditionComparison");

    group.sample_size(300);
    group.throughput(Throughput::Elements(VECTOR_LENGTH as u64)); // Report ops/element

    // --- Benchmark with different block sizes ---
    for &block_size in BLOCK_SIZES_TO_TEST {
        group.bench_with_input(
            BenchmarkId::new("BitBurst", block_size), // Unique ID for the report
            &block_size,                              // Input parameter passed to the closure
            |bencher, &bs| {
                bencher.iter(|| unsafe {
                    add_slices(black_box(&a_vec), black_box(&b_vec), bs);
                });
            },
        );
    }

    // --- Benchmark ndarray ---
    group.bench_function("ndarray", |bencher| {
        bencher.iter(|| {
            let _ = black_box(&a_arr) + black_box(&b_arr);
        });
    });

    group.finish();
}

// Register the benchmark function
#[cfg(all(avx512, rustc_channel = "nightly"))]
criterion_group!(benches, bench_vector_addition);
// Run the benchmarks defined in the 'benches' group

#[cfg(all(avx512, rustc_channel = "nightly"))]
criterion_main!(benches);
