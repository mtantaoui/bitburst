#[cfg(sse)]
use bitburst::simd::sse::f32x4;

fn main() {
    #[cfg(sse)]
    say_hello()
}
