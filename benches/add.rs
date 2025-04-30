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

fn bench_vector_addition(c: &mut Criterion) {
    // Generate data once
    let (a_vec, b_vec) = generate_data(VECTOR_LENGTH);

    // ndarray setup
    let a_arr = Array1::from_vec(a_vec.clone());
    let b_arr = Array1::from_vec(b_vec.clone());
    // let mut c_arr = Array1::<f32>::zeros(VECTOR_LENGTH); // For ndarray result

    let mut group = c.benchmark_group("VectorAdditionComparison");

    group.sample_size(300);
    group.throughput(Throughput::Elements(VECTOR_LENGTH as u64)); // Report ops/element

    // --- Benchmark Your Implementation with different block sizes ---
    for &block_size in BLOCK_SIZES_TO_TEST {
        // Ensure the destination vector is reset for each run if needed by criterion logic
        // (Criterion runs multiple samples, but usually handles setup correctly.
        // However, modifying c_vec requires care. Cloning template is safest)
        // let mut c_vec = c_vec_template.clone();

        group.bench_with_input(
            BenchmarkId::new("BitBurst", block_size), // Unique ID for the report
            &block_size,                              // Input parameter passed to the closure
            |bencher, &bs| {
                bencher.iter(|| {
                    // Pass black_box to prevent optimization based on constant inputs
                    // Pass the specific block size `bs`
                    unsafe {
                        add_slices(black_box(&a_vec), black_box(&b_vec), bs);
                    }
                    // We modify c_vec in place, so the work isn't optimized away.
                    // black_box(c_vec.as_mut_slice()); // Alternative if needed
                });
            },
        );
    }

    // --- Benchmark ndarray ---
    group.bench_function("ndarray", |bencher| {
        bencher.iter(|| {
            // Perform ndarray addition. Use references.
            // Assigning to c_arr ensures the computation happens.
            // Use black_box on inputs for safety.
            // let mut c_arr =
            let _ = black_box(&a_arr) + black_box(&b_arr);
            // Alternatively, black_box the result directly if not assigning:
            // black_box(black_box(&a_arr) + black_box(&b_arr));
            // black_box(&mut c_arr); // Black box the result container too
        });
    });

    group.finish();
}

// Register the benchmark function
criterion_group!(benches, bench_vector_addition);
// Run the benchmarks defined in the 'benches' group
criterion_main!(benches);
