#[cfg(all(avx512, rustc_channel = "nightly"))]
mod avx512;

#[cfg(avx2)]
mod avx2;

#[cfg(sse)]
mod sse;

#[cfg(neon)]
mod neon;

mod vec;
