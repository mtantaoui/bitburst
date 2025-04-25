#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 16;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct I8x16 {
    size: usize,
    elements: int8x16_t,
}

impl SimdVec<i8> for I8x16 {
    fn new(slice: &[i8]) -> Self {
        assert!(!slice.is_empty(), "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: i8) -> Self {
        Self {
            elements: unsafe { vdupq_n_s8(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const i8, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { vld1q_s8(ptr) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const i8, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);
        // Start with a zero vector
        let mut elements = vdupq_n_s8(0);

        // Load elements individually using vsetq_lane
        match size {
            1 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
            }
            2 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
            }
            3 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
            }
            4 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
            }
            5 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
            }
            6 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
            }
            7 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
            }
            8 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
            }
            9 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
            }
            10 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
            }
            11 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
                elements = vsetq_lane_s8(*ptr.add(10), elements, 10);
            }
            12 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
                elements = vsetq_lane_s8(*ptr.add(10), elements, 10);
                elements = vsetq_lane_s8(*ptr.add(11), elements, 11);
            }
            13 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
                elements = vsetq_lane_s8(*ptr.add(10), elements, 10);
                elements = vsetq_lane_s8(*ptr.add(11), elements, 11);
                elements = vsetq_lane_s8(*ptr.add(12), elements, 12);
            }
            14 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
                elements = vsetq_lane_s8(*ptr.add(10), elements, 10);
                elements = vsetq_lane_s8(*ptr.add(11), elements, 11);
                elements = vsetq_lane_s8(*ptr.add(12), elements, 12);
                elements = vsetq_lane_s8(*ptr.add(13), elements, 13);
            }
            15 => {
                elements = vsetq_lane_s8(*ptr.add(0), elements, 0);
                elements = vsetq_lane_s8(*ptr.add(1), elements, 1);
                elements = vsetq_lane_s8(*ptr.add(2), elements, 2);
                elements = vsetq_lane_s8(*ptr.add(3), elements, 3);
                elements = vsetq_lane_s8(*ptr.add(4), elements, 4);
                elements = vsetq_lane_s8(*ptr.add(5), elements, 5);
                elements = vsetq_lane_s8(*ptr.add(6), elements, 6);
                elements = vsetq_lane_s8(*ptr.add(7), elements, 7);
                elements = vsetq_lane_s8(*ptr.add(8), elements, 8);
                elements = vsetq_lane_s8(*ptr.add(9), elements, 9);
                elements = vsetq_lane_s8(*ptr.add(10), elements, 10);
                elements = vsetq_lane_s8(*ptr.add(11), elements, 11);
                elements = vsetq_lane_s8(*ptr.add(12), elements, 12);
                elements = vsetq_lane_s8(*ptr.add(13), elements, 13);
                elements = vsetq_lane_s8(*ptr.add(14), elements, 14);
            }

            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
        Self { elements, size }
    }

    fn store(&self) -> Vec<i8> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0i8; LANE_COUNT];

        unsafe {
            vst1q_s8(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<i8> {
        match self.size {
            1..LANE_COUNT => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    unsafe fn store_at(&self, ptr: *mut i8) {
        let msg = format!("Size must be == {}", LANE_COUNT);

        assert!(self.size == LANE_COUNT, "{}", msg);

        unsafe {
            vst1q_s8(ptr, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut i8) {
        let msg = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size < LANE_COUNT, "{}", msg);

        match self.size {
            1 => *ptr.add(0) = vgetq_lane_s8(self.elements, 0),
            2 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
            }
            3 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
            }
            4 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
            }
            5 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
            }
            6 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
            }
            7 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
            }
            8 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
            }
            9 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
            }
            10 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
            }
            11 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
                *ptr.add(10) = vgetq_lane_s8(self.elements, 10);
            }
            12 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
                *ptr.add(10) = vgetq_lane_s8(self.elements, 10);
                *ptr.add(11) = vgetq_lane_s8(self.elements, 11);
            }
            13 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
                *ptr.add(10) = vgetq_lane_s8(self.elements, 10);
                *ptr.add(11) = vgetq_lane_s8(self.elements, 11);
                *ptr.add(12) = vgetq_lane_s8(self.elements, 12);
            }
            14 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
                *ptr.add(10) = vgetq_lane_s8(self.elements, 10);
                *ptr.add(11) = vgetq_lane_s8(self.elements, 11);
                *ptr.add(12) = vgetq_lane_s8(self.elements, 12);
                *ptr.add(13) = vgetq_lane_s8(self.elements, 13);
            }
            15 => {
                *ptr.add(0) = vgetq_lane_s8(self.elements, 0);
                *ptr.add(1) = vgetq_lane_s8(self.elements, 1);
                *ptr.add(2) = vgetq_lane_s8(self.elements, 2);
                *ptr.add(3) = vgetq_lane_s8(self.elements, 3);
                *ptr.add(4) = vgetq_lane_s8(self.elements, 4);
                *ptr.add(5) = vgetq_lane_s8(self.elements, 5);
                *ptr.add(6) = vgetq_lane_s8(self.elements, 6);
                *ptr.add(7) = vgetq_lane_s8(self.elements, 7);
                *ptr.add(8) = vgetq_lane_s8(self.elements, 8);
                *ptr.add(9) = vgetq_lane_s8(self.elements, 9);
                *ptr.add(10) = vgetq_lane_s8(self.elements, 10);
                *ptr.add(11) = vgetq_lane_s8(self.elements, 11);
                *ptr.add(12) = vgetq_lane_s8(self.elements, 12);
                *ptr.add(13) = vgetq_lane_s8(self.elements, 13);
                *ptr.add(14) = vgetq_lane_s8(self.elements, 14);
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    fn to_vec(self) -> Vec<i8> {
        let msg = format!("Size must be <= {}", LANE_COUNT);
        assert!(self.size <= LANE_COUNT, "{}", msg);

        if self.size == LANE_COUNT {
            self.store()
        } else {
            self.store_partial()
        }
    }

    fn eq_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a == b elementwise
        let mask = unsafe { vceqq_s8(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s8_u8(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    fn lt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<b elementwise
        let mask = unsafe { vcltq_s8(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s8_u8(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    fn le_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<=b elementwise
        let mask = unsafe { vcleq_s8(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s8_u8(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    fn gt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>b elementwise
        let mask = unsafe { vcgtq_s8(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s8_u8(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    fn ge_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>=b elementwise
        let mask = unsafe { vcgeq_s8(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s8_u8(mask) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for I8x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            I8x16 {
                size: self.size,
                elements: vaddq_s8(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for I8x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self + rhs;
    }
}

impl Sub for I8x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            I8x16 {
                size: self.size,
                elements: vsubq_s8(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for I8x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self - rhs;
    }
}

impl Mul for I8x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            I8x16 {
                size: self.size,
                elements: vmulq_s8(self.elements, rhs.elements),
            }
        }
    }
}

impl MulAssign for I8x16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );
        *self = *self * rhs;
    }
}

impl Eq for I8x16 {}

impl PartialEq for I8x16 {
    fn eq(&self, other: &Self) -> bool {
        assert!(
            self.size == other.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            other.size
        );

        unsafe {
            // Compare lane-by-lane
            let cmp = vceqq_s8(self.elements, other.elements);

            // Reinterpret result as float for mask extraction
            // let mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_s8(vreinterpretq_s8_u8(cmp))), 0);
            let mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_u8(cmp)), 0);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for I8x16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        assert!(
            self.size == other.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            other.size
        );

        unsafe {
            let lt = self.lt_elements(*other).elements;
            let gt = self.gt_elements(*other).elements;
            let eq = self.eq_elements(*other).elements;

            let lt_mask = vandq_u32(vreinterpretq_u32_s8(lt), vdupq_n_u32(0x1));
            let gt_mask = vandq_u32(vreinterpretq_u32_s8(gt), vdupq_n_u32(0x1));
            let eq_mask = vandq_u32(vreinterpretq_u32_s8(eq), vdupq_n_u32(0x1));

            // Compare element-wise using NEON intrinsics
            let lt_mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_u32(lt_mask)), 0);
            let gt_mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_u32(gt_mask)), 0);
            let eq_mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_u32(eq_mask)), 0);

            match (lt_mask, gt_mask, eq_mask) {
                (0xF, 0x0, _) => Some(std::cmp::Ordering::Less), // all lanes less
                (0x0, 0xF, _) => Some(std::cmp::Ordering::Greater), // all lanes greater
                (0x0, 0x0, 0xF) => Some(std::cmp::Ordering::Equal), // all lanes equal
                _ => None,                                       // mixed
            }
        }
    }

    fn lt(&self, other: &Self) -> bool {
        self
            // comparing elementwise
            .lt_elements(*other)
            .to_vec()
            .iter()
            // converting i8 to bool
            .all(|&f| f != 0)
    }

    fn le(&self, other: &Self) -> bool {
        self.le_elements(*other)
            .to_vec()
            .iter()
            // converting i8 to bool
            .all(|&f| f != 0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.gt_elements(*other)
            .to_vec()
            .iter()
            // converting i8 to bool
            .all(|&f| f != 0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.ge_elements(*other)
            .to_vec()
            .iter()
            // converting i8 to bool
            .all(|&f| f != 0)
    }
}

impl BitAnd for I8x16 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Perform bitwise AND between the two uint32x4_t vectors
        let elements = unsafe { vandq_s8(self.elements, rhs.elements) };

        I8x16 {
            size: self.size,
            elements,
        }
    }
}

impl BitAndAssign for I8x16 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self & rhs;
    }
}

impl BitOr for I8x16 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        let elements = unsafe { vorrq_s8(self.elements, rhs.elements) };

        I8x16 {
            size: self.size,
            elements,
        }
    }
}

impl BitOrAssign for I8x16 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self | rhs;
    }
}

#[cfg(test)]
mod i8x16_tests {
    use std::{cmp::min, vec};

    use super::*;

    #[test]
    /// __m128i fields are private and cannot be compared directly
    /// test consist on loading elements to __m128i then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let n = 20;

        (1..=n).for_each(|i| {
            let a1: Vec<i8> = (1..=i).collect();

            let v1 = I8x16::new(&a1);

            let truncated_a1 = a1
                .as_slice()
                .iter()
                .take(v1.size)
                .copied()
                .collect::<Vec<i8>>();

            assert_eq!(truncated_a1, v1.to_vec());
            assert_eq!(min(truncated_a1.len(), LANE_COUNT), v1.size);
        });
    }

    /// Splat method should duplicate one value for all elements of __m128
    #[test]
    fn test_splat() {
        let a = vec![1; 16];

        let v = I8x16::splat(1);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a1: Vec<i8> = vec![100; 20];

        let s1: Vec<i8> = (1..=16).collect();
        let v1 = I8x16::new(&s1);

        unsafe { v1.store_at(a1[0..].as_mut_ptr()) };

        assert_eq!(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 100, 100, 100],
            a1.as_slice()
        );

        let mut a2: Vec<i8> = vec![-1; 20];

        let s2: Vec<i8> = (1..=16).collect();
        let v2 = I8x16::new(&s2);

        unsafe { v2.store_at(a2[4..].as_mut_ptr()) };

        assert_eq!(
            &[-1, -1, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            a2.as_slice()
        );
    }

    #[test]
    fn test_store_at_partial() {
        let n = 15;

        (1..=n).for_each(|i| {
            let mut vector: Vec<i8> = vec![100; 20];

            let a: Vec<i8> = (1..=i).collect();

            let v = I8x16::new(a.as_slice());

            unsafe {
                v.store_at_partial(vector[4..].as_mut_ptr());
            }

            let test = match i {
                1 => &[
                    100, 100, 100, 100, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100, 100,
                ],
                2 => &[
                    100, 100, 100, 100, 1, 2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100, 100,
                ],
                3 => &[
                    100, 100, 100, 100, 1, 2, 3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100,
                ],
                4 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100,
                ],
                5 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100,
                ],
                6 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100,
                ],
                7 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 100, 100, 100, 100, 100, 100, 100,
                    100, 100,
                ],
                8 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 100, 100, 100, 100, 100, 100, 100,
                    100,
                ],
                9 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 100, 100, 100, 100, 100,
                    100,
                ],
                10 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 100, 100, 100, 100, 100,
                ],
                11 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100, 100, 100, 100, 100,
                ],
                12 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 100, 100, 100,
                ],
                13 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 100, 100,
                ],
                14 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 100, 100,
                ],
                15 => &[
                    100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100,
                ],
                _ => panic!("Not a test case"),
            };

            assert_eq!(test, vector.as_slice());
        });
    }

    #[test]
    fn test_add() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(vec![6], (u1 + v1).to_vec());

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![6, 21], (u2 + v2).to_vec());

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(vec![6, 21, 16], (u3 + v3).to_vec());

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(vec![6, 21, 16, 7], (u4 + v4).to_vec());

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(vec![6, 21, 16, 7, 2], (u5 + v5).to_vec());

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12], (u6 + v6).to_vec());

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12, -10], (u7 + v7).to_vec());

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12, -10, -8], (u8 + v8).to_vec());

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12, -10, -8, -12], (u9 + v9).to_vec());

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2],
            (u10 + v10).to_vec()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1],
            (u11 + v11).to_vec()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127],
            (u12 + v12).to_vec()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127, 76],
            (u13 + v13).to_vec()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127, 76, 72],
            (u14 + v14).to_vec()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127, 76, 72, 2],
            (u15 + v15).to_vec()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127, 76, 72, 2, 7],
            (u16 + v16).to_vec()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7, 0]);

        assert_eq!(
            vec![6, 21, 16, 7, 2, 12, -10, -8, -12, -2, -1, 127, 76, 72, 2, 7],
            (u17 + v17).to_vec()
        );
    }

    #[test]
    fn test_add_assign() {
        let mut a = I8x16::new(&[1, 2, 3, 4]);
        let b = I8x16::new(&[4, 3, 2, 1]);

        a += b;

        assert_eq!(vec![5; 4], a.to_vec());
    }
    #[allow(clippy::identity_op)]
    #[test]
    fn test_sub() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(vec![5 - 1], (u1 - v1).to_vec());

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![5 - 1, 11 - 10], (u2 - v2).to_vec());

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(vec![5 - 1, 11 - 10, 9 - 7], (u3 - v3).to_vec());

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(vec![5 - 1, 11 - 10, 9 - 7, 5 - 2], (u4 - v4).to_vec());

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1],
            (u5 - v5).to_vec()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1, 3 - 9],
            (u6 - v6).to_vec()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1, 3 - 9, -9 - (-1)],
            (u7 - v7).to_vec()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3)
            ],
            (u8 - v8).to_vec()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6)
            ],
            (u9 - v9).to_vec()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12)
            ],
            (u10 - v10).to_vec()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8
            ],
            (u11 - v11).to_vec()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27
            ],
            (u12 - v12).to_vec()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27,
                50 - 26
            ],
            (u13 - v13).to_vec()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27,
                50 - 26,
                37 - 35
            ],
            (u14 - v14).to_vec()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27,
                50 - 26,
                37 - 35,
                1 - 1
            ],
            (u15 - v15).to_vec()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27,
                50 - 26,
                37 - 35,
                1 - 1,
                7 - 0
            ],
            (u16 - v16).to_vec()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7, 0]);

        assert_eq!(
            vec![
                5 - 1,
                11 - 10,
                9 - 7,
                5 - 2,
                1 - 1,
                3 - 9,
                -9 - (-1),
                -5 - (-3),
                -6 - (-6),
                10 - (-12),
                -9 - 8,
                100 - 27,
                50 - 26,
                37 - 35,
                1 - 1,
                7 - 0
            ],
            (u17 - v17).to_vec()
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut a = I8x16::new(&[1, 2, 3, 4]);
        let b = I8x16::new(&[4, 3, 2, 1]);

        a -= b;

        assert_eq!(vec![-3, -1, 1, 3], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[test]
    fn test_mul() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(vec![5 * 1], (u1 * v1).to_vec());

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![5 * 1, 11 * 10], (u2 * v2).to_vec());

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(vec![5 * 1, 11 * 10, 9 * 7], (u3 * v3).to_vec());

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(vec![5 * 1, 11 * 10, 9 * 7, 5 * 2], (u4 * v4).to_vec());

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1],
            (u5 * v5).to_vec()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1, 3 * 9],
            (u6 * v6).to_vec()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1, 3 * 9, -9 * (-1)],
            (u7 * v7).to_vec()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3)
            ],
            (u8 * v8).to_vec()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6)
            ],
            (u9 * v9).to_vec()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12)
            ],
            (u10 * v10).to_vec()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8
            ],
            (u11 * v11).to_vec()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                2 * 27
            ],
            (u12 * v12).to_vec()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                2 * 27,
                4 * 26
            ],
            (u13 * v13).to_vec()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                1 * 27,
                5 * 2,
                3 * 35
            ],
            (u14 * v14).to_vec()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                1 * 27,
                5 * 2,
                3 * 35,
                1 * 1
            ],
            (u15 * v15).to_vec()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                1 * 27,
                5 * 2,
                3 * 35,
                1 * 1,
                7 * 0
            ],
            (u16 * v16).to_vec()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 * 1,
                11 * 10,
                9 * 7,
                5 * 2,
                1 * 1,
                3 * 9,
                -9 * (-1),
                -5 * (-3),
                -6 * (-6),
                10 * (-12),
                -9 * 8,
                1 * 27,
                5 * 2,
                3 * 35,
                1 * 1,
                7 * 0
            ],
            (u17 * v17).to_vec()
        );
    }

    #[test]
    fn test_mul_assign() {
        let mut a = I8x16::new(&[1, 2, 3, 4]);
        let b = I8x16::new(&[4, 3, 2, 1]);

        a *= b;

        assert_eq!(vec![4, 6, 6, 4], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5 < 1],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5 < 1, 11 < 10],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2],
            (u4.lt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1],
            (u5.lt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9],
            (u6.lt_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9, -9 < (-1)],
            (u7.lt_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3)
            ],
            (u8.lt_elements(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6)
            ],
            (u9.lt_elements(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12)
            ],
            (u10.lt_elements(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8
            ],
            (u11.lt_elements(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                2 < 27
            ],
            (u12.lt_elements(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                2 < 27,
                4 < 26
            ],
            (u13.lt_elements(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35
            ],
            (u14.lt_elements(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1
            ],
            (u15.lt_elements(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1,
                7 < 0
            ],
            (u16.lt_elements(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1,
                7 < 0
            ],
            (u17.lt_elements(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5 <= 1],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2],
            (u4.le_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1],
            (u5.le_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9],
            (u6.le_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9, -9 <= (-1)],
            (u7.le_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3)
            ],
            (u8.le_elements(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6)
            ],
            (u9.le_elements(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12)
            ],
            (u10.le_elements(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8
            ],
            (u11.le_elements(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                2 <= 27
            ],
            (u12.le_elements(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                2 <= 27,
                4 <= 26
            ],
            (u13.le_elements(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35
            ],
            (u14.le_elements(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1
            ],
            (u15.le_elements(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1,
                7 <= 0
            ],
            (u16.le_elements(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1,
                7 <= 0
            ],
            (u17.le_elements(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5 > 1],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5 > 1, 11 > 10],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2],
            (u4.gt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1],
            (u5.gt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9],
            (u6.gt_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9, -9 > (-1)],
            (u7.gt_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3)
            ],
            (u8.gt_elements(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6)
            ],
            (u9.gt_elements(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12)
            ],
            (u10.gt_elements(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8
            ],
            (u11.gt_elements(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                2 > 27
            ],
            (u12.gt_elements(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                2 > 27,
                4 > 26
            ],
            (u13.gt_elements(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35
            ],
            (u14.gt_elements(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1
            ],
            (u15.gt_elements(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1,
                7 > 0
            ],
            (u16.gt_elements(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1,
                7 > 0
            ],
            (u17.gt_elements(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5 >= 1],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2],
            (u4.ge_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1],
            (u5.ge_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9],
            (u6.ge_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9, -9 >= (-1)],
            (u7.ge_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3)
            ],
            (u8.ge_elements(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6)
            ],
            (u9.ge_elements(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12)
            ],
            (u10.ge_elements(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8
            ],
            (u11.ge_elements(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                2 >= 27
            ],
            (u12.ge_elements(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                2 >= 27,
                4 >= 26
            ],
            (u13.ge_elements(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35
            ],
            (u14.ge_elements(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1
            ],
            (u15.ge_elements(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1,
                7 >= 0
            ],
            (u16.ge_elements(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1,
                7 >= 0
            ],
            (u17.ge_elements(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5 == 1],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5 == 1, 11 == 10],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2],
            (u4.eq_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1],
            (u5.eq_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9],
            (u6.eq_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9, -9 == (-1)],
            (u7.eq_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3)
            ],
            (u8.eq_elements(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6)
            ],
            (u9.eq_elements(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12)
            ],
            (u10.eq_elements(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8
            ],
            (u11.eq_elements(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                2 == 27
            ],
            (u12.eq_elements(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                2 == 27,
                4 == 26
            ],
            (u13.eq_elements(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35
            ],
            (u14.eq_elements(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1
            ],
            (u15.eq_elements(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1,
                7 == 0
            ],
            (u16.eq_elements(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1,
                7 == 0
            ],
            (u17.eq_elements(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!([5 == 1].iter().all(|f| *f), u1 == v1);

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!([5 == 1, 11 == 10].iter().all(|f| *f), u2 == v2);

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!([5 == 1, 11 == 10, 9 == 7].iter().all(|f| *f), u3 == v3);

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2].iter().all(|f| *f),
            u4 == v4
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1]
                .iter()
                .all(|f| *f),
            u5 == v5
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9]
                .iter()
                .all(|f| *f),
            u6 == v6
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9, -9 == (-1)]
                .iter()
                .all(|f| *f),
            u7 == v7
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3)
            ]
            .iter()
            .all(|f| *f),
            u8 == v8
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6)
            ]
            .iter()
            .all(|f| *f),
            u9 == v9
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12)
            ]
            .iter()
            .all(|f| *f),
            u10 == v10
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8
            ]
            .iter()
            .all(|f| *f),
            u11 == v11
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                2 == 27
            ]
            .iter()
            .all(|f| *f),
            u12 == v12
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                2 == 27,
                4 == 26
            ]
            .iter()
            .all(|f| *f),
            u13 == v13
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35
            ]
            .iter()
            .all(|f| *f),
            u14 == v14
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1
            ]
            .iter()
            .all(|f| *f),
            u15 == v15
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1,
                7 == 0
            ]
            .iter()
            .all(|f| *f),
            u16 == v16
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            [
                5 == 1,
                11 == 10,
                9 == 7,
                5 == 2,
                1 == 1,
                3 == 9,
                -9 == (-1),
                -5 == (-3),
                -6 == (-6),
                10 == (-12),
                -9 == 8,
                1 == 27,
                5 == 2,
                3 == 35,
                1 == 1,
                7 == 0
            ]
            .iter()
            .all(|f| *f),
            u17 == v17
        );
    }

    #[test]
    fn test_lt() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!([5 < 1].iter().all(|f| *f), u1 < v1);

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!([5 < 1, 11 < 10].iter().all(|f| *f), u2 < v2);

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!([5 < 1, 11 < 10, 9 < 7].iter().all(|f| *f), u3 < v3);

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!([5 < 1, 11 < 10, 9 < 7, 5 < 2].iter().all(|f| *f), u4 < v4);

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1].iter().all(|f| *f),
            u5 < v5
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9]
                .iter()
                .all(|f| *f),
            u6 < v6
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9, -9 < (-1)]
                .iter()
                .all(|f| *f),
            u7 < v7
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3)
            ]
            .iter()
            .all(|f| *f),
            u8 < v8
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6)
            ]
            .iter()
            .all(|f| *f),
            u9 < v9
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12)
            ]
            .iter()
            .all(|f| *f),
            u10 < v10
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8
            ]
            .iter()
            .all(|f| *f),
            u11 < v11
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                2 < 27
            ]
            .iter()
            .all(|f| *f),
            u12 < v12
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                2 < 27,
                4 < 26
            ]
            .iter()
            .all(|f| *f),
            u13 < v13
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35
            ]
            .iter()
            .all(|f| *f),
            u14 < v14
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1
            ]
            .iter()
            .all(|f| *f),
            u15 < v15
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1,
                7 < 0
            ]
            .iter()
            .all(|f| *f),
            u16 < v16
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            [
                5 < 1,
                11 < 10,
                9 < 7,
                5 < 2,
                1 < 1,
                3 < 9,
                -9 < (-1),
                -5 < (-3),
                -6 < (-6),
                10 < (-12),
                -9 < 8,
                1 < 27,
                5 < 2,
                3 < 35,
                1 < 1,
                7 < 0
            ]
            .iter()
            .all(|f| *f),
            u17 < v17
        );
    }

    #[test]
    fn test_le() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!([5 <= 1].iter().all(|f| *f), u1 <= v1);

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!([5 <= 1, 11 <= 10].iter().all(|f| *f), u2 <= v2);

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!([5 <= 1, 11 <= 10, 9 <= 7].iter().all(|f| *f), u3 <= v3);

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2].iter().all(|f| *f),
            u4 <= v4
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1]
                .iter()
                .all(|f| *f),
            u5 <= v5
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9]
                .iter()
                .all(|f| *f),
            u6 <= v6
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9, -9 <= (-1)]
                .iter()
                .all(|f| *f),
            u7 <= v7
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3)
            ]
            .iter()
            .all(|f| *f),
            u8 <= v8
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6)
            ]
            .iter()
            .all(|f| *f),
            u9 <= v9
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12)
            ]
            .iter()
            .all(|f| *f),
            u10 <= v10
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8
            ]
            .iter()
            .all(|f| *f),
            u11 <= v11
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                2 <= 27
            ]
            .iter()
            .all(|f| *f),
            u12 <= v12
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                2 <= 27,
                4 <= 26
            ]
            .iter()
            .all(|f| *f),
            u13 <= v13
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35
            ]
            .iter()
            .all(|f| *f),
            u14 <= v14
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1
            ]
            .iter()
            .all(|f| *f),
            u15 <= v15
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1,
                7 <= 0
            ]
            .iter()
            .all(|f| *f),
            u16 <= v16
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            [
                5 <= 1,
                11 <= 10,
                9 <= 7,
                5 <= 2,
                1 <= 1,
                3 <= 9,
                -9 <= (-1),
                -5 <= (-3),
                -6 <= (-6),
                10 <= (-12),
                -9 <= 8,
                1 <= 27,
                5 <= 2,
                3 <= 35,
                1 <= 1,
                7 <= 0
            ]
            .iter()
            .all(|f| *f),
            u17 <= v17
        );
    }

    #[test]
    fn test_gt() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!([5 > 1].iter().all(|f| *f), u1 > v1);

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!([5 > 1, 11 > 10].iter().all(|f| *f), u2 > v2);

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!([5 > 1, 11 > 10, 9 > 7].iter().all(|f| *f), u3 > v3);

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!([5 > 1, 11 > 10, 9 > 7, 5 > 2].iter().all(|f| *f), u4 > v4);

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1].iter().all(|f| *f),
            u5 > v5
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9]
                .iter()
                .all(|f| *f),
            u6 > v6
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9, -9 > (-1)]
                .iter()
                .all(|f| *f),
            u7 > v7
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3)
            ]
            .iter()
            .all(|f| *f),
            u8 > v8
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6)
            ]
            .iter()
            .all(|f| *f),
            u9 > v9
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12)
            ]
            .iter()
            .all(|f| *f),
            u10 > v10
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8
            ]
            .iter()
            .all(|f| *f),
            u11 > v11
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                2 > 27
            ]
            .iter()
            .all(|f| *f),
            u12 > v12
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                2 > 27,
                4 > 26
            ]
            .iter()
            .all(|f| *f),
            u13 > v13
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35
            ]
            .iter()
            .all(|f| *f),
            u14 > v14
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1
            ]
            .iter()
            .all(|f| *f),
            u15 > v15
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1,
                7 > 0
            ]
            .iter()
            .all(|f| *f),
            u16 > v16
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            [
                5 > 1,
                11 > 10,
                9 > 7,
                5 > 2,
                1 > 1,
                3 > 9,
                -9 > (-1),
                -5 > (-3),
                -6 > (-6),
                10 > (-12),
                -9 > 8,
                1 > 27,
                5 > 2,
                3 > 35,
                1 > 1,
                7 > 0
            ]
            .iter()
            .all(|f| *f),
            u17 > v17
        );
    }

    #[test]
    fn test_ge() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!([5 >= 1].iter().all(|f| *f), u1 >= v1);

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!([5 >= 1, 11 >= 10].iter().all(|f| *f), u2 >= v2);

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!([5 >= 1, 11 >= 10, 9 >= 7].iter().all(|f| *f), u3 >= v3);

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2].iter().all(|f| *f),
            u4 >= v4
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1]
                .iter()
                .all(|f| *f),
            u5 >= v5
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9]
                .iter()
                .all(|f| *f),
            u6 >= v6
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9, -9 >= (-1)]
                .iter()
                .all(|f| *f),
            u7 >= v7
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3)
            ]
            .iter()
            .all(|f| *f),
            u8 >= v8
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6)
            ]
            .iter()
            .all(|f| *f),
            u9 >= v9
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12)
            ]
            .iter()
            .all(|f| *f),
            u10 >= v10
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8
            ]
            .iter()
            .all(|f| *f),
            u11 >= v11
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                2 >= 27
            ]
            .iter()
            .all(|f| *f),
            u12 >= v12
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 2, 4]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                2 >= 27,
                4 >= 26
            ]
            .iter()
            .all(|f| *f),
            u13 >= v13
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35
            ]
            .iter()
            .all(|f| *f),
            u14 >= v14
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1
            ]
            .iter()
            .all(|f| *f),
            u15 >= v15
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1,
                7 >= 0
            ]
            .iter()
            .all(|f| *f),
            u16 >= v16
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            [
                5 >= 1,
                11 >= 10,
                9 >= 7,
                5 >= 2,
                1 >= 1,
                3 >= 9,
                -9 >= (-1),
                -5 >= (-3),
                -6 >= (-6),
                10 >= (-12),
                -9 >= 8,
                1 >= 27,
                5 >= 2,
                3 >= 35,
                1 >= 1,
                7 >= 0
            ]
            .iter()
            .all(|f| *f),
            u17 >= v17
        );
    }

    #[allow(clippy::erasing_op)]
    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_and() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5u8 & 1u8 != 0],
            (u1.bitand(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0],
            (u2.bitand(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0, 9u8 & 7u8 != 0],
            (u3.bitand(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0
            ],
            (u4.bitand(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0
            ],
            (u5.bitand(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
            ],
            (u6.bitand(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0
            ],
            (u7.bitand(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0
            ],
            (u8.bitand(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0
            ],
            (u9.bitand(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0
            ],
            (u10.bitand(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0
            ],
            (u11.bitand(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 2]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                2u8 & 27u8 != 0
            ],
            (u12.bitand(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 2, 4]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                2u8 & 27u8 != 0,
                4u8 & 26u8 != 0
            ],
            (u13.bitand(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                1u8 & 27u8 != 0,
                5u8 & 2u8 != 0,
                3u8 & 35u8 != 0
            ],
            (u14.bitand(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                1u8 & 27u8 != 0,
                5u8 & 2u8 != 0,
                3u8 & 35u8 != 0,
                1u8 & 1u8 != 0
            ],
            (u15.bitand(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                1u8 & 27u8 != 0,
                5u8 & 2u8 != 0,
                3u8 & 35u8 != 0,
                1u8 & 1u8 != 0,
                7u8 & 0u8 != 0
            ],
            (u16.bitand(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5u8 & 1u8 != 0,
                11u8 & 10u8 != 0,
                9u8 & 7u8 != 0,
                5u8 & 2u8 != 0,
                1u8 & 1u8 != 0,
                3u8 & 9u8 != 0,
                9u8 & 1u8 != 0,
                5u8 & 3u8 != 0,
                6u8 & 6u8 != 0,
                10u8 & 12u8 != 0,
                9u8 & 8u8 != 0,
                1u8 & 27u8 != 0,
                5u8 & 2u8 != 0,
                3u8 & 35u8 != 0,
                1u8 & 1u8 != 0,
                7u8 & 0u8 != 0
            ],
            (u17.bitand(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_or() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(
            vec![5u8 | 1u8 != 0],
            (u1.bitor(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0],
            (u2.bitor(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0, 9u8 | 7u8 != 0],
            (u3.bitor(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0
            ],
            (u4.bitor(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0
            ],
            (u5.bitor(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
            ],
            (u6.bitor(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0
            ],
            (u7.bitor(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0
            ],
            (u8.bitor(v8))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0
            ],
            (u9.bitor(v9))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0
            ],
            (u10.bitor(v10))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0
            ],
            (u11.bitor(v11))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 2]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                2u8 | 27u8 != 0
            ],
            (u12.bitor(v12))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 2, 4]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                2u8 | 27u8 != 0,
                4u8 | 26u8 != 0
            ],
            (u13.bitor(v13))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                1u8 | 27u8 != 0,
                5u8 | 2u8 != 0,
                3u8 | 35u8 != 0
            ],
            (u14.bitor(v14))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                1u8 | 27u8 != 0,
                5u8 | 2u8 != 0,
                3u8 | 35u8 != 0,
                1u8 | 1u8 != 0
            ],
            (u15.bitor(v15))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1, 7]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                1u8 | 27u8 != 0,
                5u8 | 2u8 != 0,
                3u8 | 35u8 != 0,
                1u8 | 1u8 != 0,
                7u8 | 0u8 != 0
            ],
            (u16.bitor(v16))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, 1, 3, 6, 12, 8, 27, 2, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, 9, 5, 6, 10, 9, 1, 5, 3, 1, 7, 0]);

        assert_eq!(
            vec![
                5u8 | 1u8 != 0,
                11u8 | 10u8 != 0,
                9u8 | 7u8 != 0,
                5u8 | 2u8 != 0,
                1u8 | 1u8 != 0,
                3u8 | 9u8 != 0,
                9u8 | 1u8 != 0,
                5u8 | 3u8 != 0,
                6u8 | 6u8 != 0,
                10u8 | 12u8 != 0,
                9u8 | 8u8 != 0,
                1u8 | 27u8 != 0,
                5u8 | 2u8 != 0,
                3u8 | 35u8 != 0,
                1u8 | 1u8 != 0,
                7u8 | 0u8 != 0
            ],
            (u17.bitor(v17))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }
}
