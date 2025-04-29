use bitburst::simd::avx512::add::add;

fn main() {
    let n: usize = 10;

    let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=n).map(|i| i as f32).collect();

    let c = add(a.as_slice(), b.as_slice());



    
}
