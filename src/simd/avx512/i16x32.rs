#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 32;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct I16x32 {
    size: usize,
    elements: __m512i,
}

impl SimdVec<i16> for I16x32 {
    fn new(slice: &[i16]) -> Self {
        assert!(!slice.is_empty(), "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: i16) -> Self {
        Self {
            elements: unsafe { _mm512_set1_epi16(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const i16, size: usize) -> Self {
        let msg = format!("Size must be == {LANE_COUNT}");
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { _mm512_loadu_si512(ptr as *const __m512i) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const i16, size: usize) -> Self {
        let msg = format!("Size must be < {LANE_COUNT}");
        assert!(size < LANE_COUNT, "{}", msg);

        let mask = (1 << size) - 1;

        Self {
            elements: unsafe { _mm512_maskz_loadu_epi16(mask, ptr) },
            size,
        }
    }

    fn store(&self) -> Vec<i16> {
        let msg = format!("Size must be <= {LANE_COUNT}");

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0i16; LANE_COUNT];

        unsafe {
            _mm512_storeu_si512(vec.as_mut_ptr() as *mut __m512i, self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<i16> {
        match self.size {
            1..LANE_COUNT => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    unsafe fn store_at(&self, ptr: *mut i16) {
        let msg = format!("Size must be == {LANE_COUNT}");

        assert!(self.size == LANE_COUNT, "{}", msg);

        unsafe {
            _mm512_storeu_si512(ptr as *mut __m512i, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut i16) {
        let msg = format!("Size must be < {LANE_COUNT}");

        assert!(self.size < LANE_COUNT, "{}", msg);

        let mask = (1 << self.size) - 1;

        _mm512_mask_storeu_epi16(ptr, mask, self.elements);
    }

    fn to_vec(self) -> Vec<i16> {
        let msg = format!("Size must be <= {LANE_COUNT}");
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
        let mask = unsafe { _mm512_cmpeq_epi16_mask(self.elements, rhs.elements) };
        let elements = unsafe { _mm512_movm_epi16(mask) };

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
        let mask = unsafe { _mm512_cmpgt_epi16_mask(rhs.elements, self.elements) };

        let elements = unsafe { _mm512_movm_epi16(mask) };

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
        let mask = unsafe { _mm512_cmple_epi16_mask(self.elements, rhs.elements) };

        let elements = unsafe { _mm512_movm_epi16(mask) };

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
        let mask = unsafe { _mm512_cmpgt_epi16_mask(self.elements, rhs.elements) };

        let elements = unsafe { _mm512_movm_epi16(mask) };
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
        let mask = unsafe { _mm512_cmpge_epi16_mask(self.elements, rhs.elements) };

        let elements = unsafe { _mm512_movm_epi16(mask) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for I16x32 {
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
            I16x32 {
                size: self.size,
                elements: _mm512_add_epi16(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for I16x32 {
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

impl Sub for I16x32 {
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
            I16x32 {
                size: self.size,
                elements: _mm512_sub_epi16(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for I16x32 {
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

impl Mul for I16x32 {
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

        // Pack 16-bit products into 8-bit integers with saturation
        let elements = unsafe { _mm512_mullo_epi16(self.elements, rhs.elements) };

        I16x32 {
            size: self.size,
            elements,
        }
    }
}

impl MulAssign for I16x32 {
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

impl Eq for I16x32 {}

impl PartialEq for I16x32 {
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
            let mask = _mm512_cmpeq_epi16_mask(self.elements, other.elements);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for I16x32 {
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

            let lt_mask = _mm512_reduce_and_epi64(lt);
            let gt_mask = _mm512_reduce_and_epi64(gt);
            let eq_mask = _mm512_reduce_and_epi64(eq);

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
            // converting i16 to bool
            .all(|&f| f != 0)
    }

    fn le(&self, other: &Self) -> bool {
        self.le_elements(*other)
            .to_vec()
            .iter()
            // converting i16 to bool
            .all(|&f| f != 0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.gt_elements(*other)
            .to_vec()
            .iter()
            // converting i16 to bool
            .all(|&f| f != 0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.ge_elements(*other)
            .to_vec()
            .iter()
            // converting i16 to bool
            .all(|&f| f != 0)
    }
}

impl BitAnd for I16x32 {
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

        unsafe {
            I16x32 {
                size: self.size,
                elements: _mm512_and_si512(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for I16x32 {
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

impl BitOr for I16x32 {
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

        unsafe {
            I16x32 {
                size: self.size,
                elements: _mm512_or_si512(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for I16x32 {
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
mod i16x32_tests {
    use std::{cmp::min, vec};

    use super::*;

    #[test]
    /// __m512i fields are private and cannot be compared directly
    /// test consist on loading elements to __m512i then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let n = 20;

        (1..=n).for_each(|i| {
            let a1: Vec<i16> = (1..=i).collect();

            let v1 = I16x32::new(&a1);

            let truncated_a1 = a1
                .as_slice()
                .iter()
                .take(v1.size)
                .copied()
                .collect::<Vec<i16>>();

            assert_eq!(truncated_a1, v1.to_vec());
            assert_eq!(min(truncated_a1.len(), LANE_COUNT), v1.size);
        });
    }

    /// Splat method should duplicate one value for all elements of __m128
    #[test]
    fn test_splat() {
        let a = vec![1; 32];

        let v = I16x32::splat(1);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a1: Vec<i16> = vec![100; 40];

        let s1: Vec<i16> = (1..=32).collect();
        let v1 = I16x32::new(&s1);

        unsafe { v1.store_at(a1[0..].as_mut_ptr()) };

        assert_eq!(
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 100, 100, 100, 100, 100, 100, 100, 100
            ],
            a1.as_slice()
        );

        let mut a2: Vec<i16> = vec![-1; 40];

        let s2: Vec<i16> = (1..=32).collect();
        let v2 = I16x32::new(&s2);

        unsafe { v2.store_at(a2[4..].as_mut_ptr()) };

        assert_eq!(
            &[
                -1, -1, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, -1, -1, -1, -1
            ],
            a2.as_slice()
        );
    }

    #[test]
    fn test_store_at_partial() {
        let n = 7;

        (1..=n).for_each(|i| {
            let mut vector: Vec<i16> = vec![100; 11];

            let a: Vec<i16> = (1..=i).collect();

            let v = I16x32::new(a.as_slice());

            unsafe {
                v.store_at_partial(vector[4..].as_mut_ptr());
            }

            let test = match i {
                1 => &[100, 100, 100, 100, 1, 100, 100, 100, 100, 100, 100],
                2 => &[100, 100, 100, 100, 1, 2, 100, 100, 100, 100, 100],
                3 => &[100, 100, 100, 100, 1, 2, 3, 100, 100, 100, 100],
                4 => &[100, 100, 100, 100, 1, 2, 3, 4, 100, 100, 100],
                5 => &[100, 100, 100, 100, 1, 2, 3, 4, 5, 100, 100],
                6 => &[100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 100],
                7 => &[100, 100, 100, 100, 1, 2, 3, 4, 5, 6, 7],
                _ => panic!("Not a test case"),
            };

            assert_eq!(test, vector.as_slice());
        });

        let mut vector: Vec<i16> = vec![100; 3];

        let a: Vec<i16> = (1..=1).collect();

        let v = I16x32::new(a.as_slice());

        unsafe {
            v.store_at_partial(vector[2..].as_mut_ptr());
        }

        assert_eq!(vector, [100, 100, 1])
    }

    #[test]
    fn test_add() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(vec![6], (u1 + v1).to_vec());

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(vec![6, 21], (u2 + v2).to_vec());

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(vec![6, 21, 16], (u3 + v3).to_vec());

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(vec![6, 21, 16, 7], (u4 + v4).to_vec());

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(vec![6, 21, 16, 7, 2], (u5 + v5).to_vec());

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12], (u6 + v6).to_vec());

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12, -10], (u7 + v7).to_vec());

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(vec![6, 21, 16, 7, 2, 12, -10, -8], (u8 + v8).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = I16x32::new(&[1, 2, 3, 4]);
        let b = I16x32::new(&[4, 3, 2, 1]);

        a += b;

        assert_eq!(vec![5; 4], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[test]
    fn test_sub() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(vec![5 - 1], (u1 - v1).to_vec());

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(vec![5 - 1, 11 - 10], (u2 - v2).to_vec());

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(vec![5 - 1, 11 - 10, 9 - 7], (u3 - v3).to_vec());

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(vec![5 - 1, 11 - 10, 9 - 7, 5 - 2], (u4 - v4).to_vec());

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1],
            (u5 - v5).to_vec()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1, 3 - 9],
            (u6 - v6).to_vec()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 - 1, 11 - 10, 9 - 7, 5 - 2, 1 - 1, 3 - 9, -9 - (-1)],
            (u7 - v7).to_vec()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_sub_assign() {
        let mut a = I16x32::new(&[1, 2, 3, 4]);
        let b = I16x32::new(&[4, 3, 2, 1]);

        a -= b;

        assert_eq!(vec![-3, -1, 1, 3], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[test]
    fn test_mul() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(vec![5 * 1], (u1 * v1).to_vec());

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(vec![5 * 1, 11 * 10], (u2 * v2).to_vec());

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(vec![5 * 1, 11 * 10, 9 * 7], (u3 * v3).to_vec());

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(vec![5 * 1, 11 * 10, 9 * 7, 5 * 2], (u4 * v4).to_vec());

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1],
            (u5 * v5).to_vec()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1, 3 * 9],
            (u6 * v6).to_vec()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 * 1, 11 * 10, 9 * 7, 5 * 2, 1 * 1, 3 * 9, -9 * (-1)],
            (u7 * v7).to_vec()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_mul_assign() {
        let mut a = I16x32::new(&[1, 2, 3, 4]);
        let b = I16x32::new(&[4, 3, 2, 1]);

        a *= b;

        assert_eq!(vec![4, 6, 6, 4], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5 < 1],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5 < 1, 11 < 10],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2],
            (u4.lt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1],
            (u5.lt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9],
            (u6.lt_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9, -9 < (-1)],
            (u7.lt_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_le_elementwise() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5 <= 1],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2],
            (u4.le_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1],
            (u5.le_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9],
            (u6.le_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9, -9 <= (-1)],
            (u7.le_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_gt_elementwise() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5 > 1],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5 > 1, 11 > 10],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2],
            (u4.gt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1],
            (u5.gt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9],
            (u6.gt_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9, -9 > (-1)],
            (u7.gt_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_ge_elementwise() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5 >= 1],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2],
            (u4.ge_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1],
            (u5.ge_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9],
            (u6.ge_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9, -9 >= (-1)],
            (u7.ge_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_eq_elementwise() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5 == 1],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5 == 1, 11 == 10],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2],
            (u4.eq_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1],
            (u5.eq_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9],
            (u6.eq_elements(v6))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            vec![5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9, -9 == (-1)],
            (u7.eq_elements(v7))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_eq() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!([5 == 1].iter().all(|f| *f), u1 == v1);

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!([5 == 1, 11 == 10].iter().all(|f| *f), u2 == v2);

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!([5 == 1, 11 == 10, 9 == 7].iter().all(|f| *f), u3 == v3);

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2].iter().all(|f| *f),
            u4 == v4
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1]
                .iter()
                .all(|f| *f),
            u5 == v5
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9]
                .iter()
                .all(|f| *f),
            u6 == v6
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 == 1, 11 == 10, 9 == 7, 5 == 2, 1 == 1, 3 == 9, -9 == (-1)]
                .iter()
                .all(|f| *f),
            u7 == v7
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_lt() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!([5 < 1].iter().all(|f| *f), u1 < v1);

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!([5 < 1, 11 < 10].iter().all(|f| *f), u2 < v2);

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!([5 < 1, 11 < 10, 9 < 7].iter().all(|f| *f), u3 < v3);

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!([5 < 1, 11 < 10, 9 < 7, 5 < 2].iter().all(|f| *f), u4 < v4);

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1].iter().all(|f| *f),
            u5 < v5
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9]
                .iter()
                .all(|f| *f),
            u6 < v6
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 < 1, 11 < 10, 9 < 7, 5 < 2, 1 < 1, 3 < 9, -9 < (-1)]
                .iter()
                .all(|f| *f),
            u7 < v7
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_le() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!([5 <= 1].iter().all(|f| *f), u1 <= v1);

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!([5 <= 1, 11 <= 10].iter().all(|f| *f), u2 <= v2);

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!([5 <= 1, 11 <= 10, 9 <= 7].iter().all(|f| *f), u3 <= v3);

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2].iter().all(|f| *f),
            u4 <= v4
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1]
                .iter()
                .all(|f| *f),
            u5 <= v5
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9]
                .iter()
                .all(|f| *f),
            u6 <= v6
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 <= 1, 11 <= 10, 9 <= 7, 5 <= 2, 1 <= 1, 3 <= 9, -9 <= (-1)]
                .iter()
                .all(|f| *f),
            u7 <= v7
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_gt() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!([5 > 1].iter().all(|f| *f), u1 > v1);

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!([5 > 1, 11 > 10].iter().all(|f| *f), u2 > v2);

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!([5 > 1, 11 > 10, 9 > 7].iter().all(|f| *f), u3 > v3);

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!([5 > 1, 11 > 10, 9 > 7, 5 > 2].iter().all(|f| *f), u4 > v4);

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1].iter().all(|f| *f),
            u5 > v5
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9]
                .iter()
                .all(|f| *f),
            u6 > v6
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 > 1, 11 > 10, 9 > 7, 5 > 2, 1 > 1, 3 > 9, -9 > (-1)]
                .iter()
                .all(|f| *f),
            u7 > v7
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[test]
    fn test_ge() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!([5 >= 1].iter().all(|f| *f), u1 >= v1);

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!([5 >= 1, 11 >= 10].iter().all(|f| *f), u2 >= v2);

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!([5 >= 1, 11 >= 10, 9 >= 7].iter().all(|f| *f), u3 >= v3);

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2].iter().all(|f| *f),
            u4 >= v4
        );

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1]
                .iter()
                .all(|f| *f),
            u5 >= v5
        );

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9]
                .iter()
                .all(|f| *f),
            u6 >= v6
        );

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(
            [5 >= 1, 11 >= 10, 9 >= 7, 5 >= 2, 1 >= 1, 3 >= 9, -9 >= (-1)]
                .iter()
                .all(|f| *f),
            u7 >= v7
        );

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

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
    }

    #[allow(clippy::erasing_op)]
    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_and() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5u8 & 1u8 != 0],
            (u1.bitand(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0],
            (u2.bitand(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0, 9u8 & 7u8 != 0],
            (u3.bitand(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

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

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

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

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

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

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, 1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, 9]);

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

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, 1, 3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, 9, 5]);

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
    }

    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_or() {
        let v1 = I16x32::new(&[1]);
        let u1 = I16x32::new(&[5]);

        assert_eq!(
            vec![5u8 | 1u8 != 0],
            (u1.bitor(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I16x32::new(&[1, 10]);
        let u2 = I16x32::new(&[5, 11]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0],
            (u2.bitor(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I16x32::new(&[1, 10, 7]);
        let u3 = I16x32::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0, 9u8 | 7u8 != 0],
            (u3.bitor(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v4 = I16x32::new(&[1, 10, 7, 2]);
        let u4 = I16x32::new(&[5, 11, 9, 5]);

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

        let v5 = I16x32::new(&[1, 10, 7, 2, 1]);
        let u5 = I16x32::new(&[5, 11, 9, 5, 1]);

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

        let v6 = I16x32::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I16x32::new(&[5, 11, 9, 5, 1, 3]);

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

        let v7 = I16x32::new(&[1, 10, 7, 2, 1, 9, 1]);
        let u7 = I16x32::new(&[5, 11, 9, 5, 1, 3, 9]);

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

        let v8 = I16x32::new(&[1, 10, 7, 2, 1, 9, 1, 3]);
        let u8 = I16x32::new(&[5, 11, 9, 5, 1, 3, 9, 5]);

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
    }
}
