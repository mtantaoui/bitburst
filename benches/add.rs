use bitburst::simd::avx512::add::add;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

const SIZE: usize = 100_000;

// Benchmark: ndarray vector addition
fn add_ndarray_f32_vectors() {
    let vec1 = Array1::from(vec![1.0f32; SIZE]);
    let vec2 = Array1::from(vec![2.0f32; SIZE]);

    let result = &vec1 + &vec2;
    black_box(result);
}

// Benchmark: custom Vec<f32> vector addition
fn add_bitburst_f32_vectors() {
    let vec1 = vec![1.0f32; SIZE];
    let vec2 = vec![2.0f32; SIZE];

    let result: Vec<f32> = add(vec1.as_slice(), vec2.as_slice());
    black_box(result);
}

fn add_f32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Add Comparison");

    group.bench_function("ndarray vector add", |b| b.iter(add_ndarray_f32_vectors));
    group.bench_function("bitburst vector add", |b| b.iter(add_bitburst_f32_vectors));

    group.finish();
}

criterion_group!(benches, add_f32_benchmark);
criterion_main!(benches);
