#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 16;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct I8x16 {
    size: usize,
    elements: __m128i,
}

impl SimdVec<i8> for I8x16 {
    fn new(slice: &[i8]) -> Self {
        assert!(slice.len() != 0, "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: i8) -> Self {
        Self {
            elements: unsafe { _mm_set1_epi8(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const i8, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { _mm_loadu_si128(ptr as *const __m128i) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const i8, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);

        let elements = match size {
            1 => unsafe { _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *ptr.add(0)) },
            2 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            3 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            4 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            5 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            6 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            7 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            8 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            9 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            10 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            11 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(10),
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            12 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    0,
                    *ptr.add(11),
                    *ptr.add(10),
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            13 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    0,
                    *ptr.add(12),
                    *ptr.add(11),
                    *ptr.add(10),
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            14 => unsafe {
                _mm_set_epi8(
                    0,
                    0,
                    *ptr.add(13),
                    *ptr.add(12),
                    *ptr.add(11),
                    *ptr.add(10),
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            15 => unsafe {
                _mm_set_epi8(
                    0,
                    *ptr.add(14),
                    *ptr.add(13),
                    *ptr.add(12),
                    *ptr.add(11),
                    *ptr.add(10),
                    *ptr.add(9),
                    *ptr.add(8),
                    *ptr.add(7),
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        Self { elements, size }
    }

    fn store(&self) -> Vec<i8> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0i8; LANE_COUNT];

        unsafe {
            _mm_storeu_si128(vec.as_mut_ptr() as *mut __m128i, self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<i8> {
        match self.size {
            1..=15 => self.store().into_iter().take(self.size).collect(),
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
            _mm_storeu_si128(ptr as *mut __m128i, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut i8) {
        let msg = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size < LANE_COUNT, "{}", msg);

        // Create a mask where the first number of elements (size) will be stored at ptr
        let mask = match self.size {
            1 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1),
            2 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1),
            3 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1),
            4 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1),
            5 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1),
            6 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1),
            7 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1),
            8 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1),
            9 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            10 => _mm_set_epi8(0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            11 => _mm_set_epi8(0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            12 => _mm_set_epi8(0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            13 => _mm_set_epi8(0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            14 => _mm_set_epi8(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            15 => _mm_set_epi8(
                0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ),
            _ => panic!("Invalid size"),
        };

        // Use maskmoveu to selectively store first elements
        _mm_maskmoveu_si128(self.elements, mask, ptr);
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
        let elements = unsafe { _mm_cmpeq_epi8(self.elements, rhs.elements) };
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
        let elements = unsafe { _mm_cmplt_epi8(self.elements, rhs.elements) };

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
        let less_than = unsafe { _mm_cmplt_epi8(self.elements, rhs.elements) };
        let equal = unsafe { _mm_cmpeq_epi8(self.elements, rhs.elements) };
        let elements = unsafe { _mm_or_si128(less_than, equal) };

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
        let elements = unsafe { _mm_cmpgt_epi8(self.elements, rhs.elements) }; // Result as float mask

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
        let greater_than = unsafe { _mm_cmpgt_epi8(self.elements, rhs.elements) };
        let equal = unsafe { _mm_cmpeq_epi8(self.elements, rhs.elements) };
        let elements = unsafe { _mm_or_si128(greater_than, equal) };

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
                elements: _mm_add_epi8(self.elements, rhs.elements),
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
                elements: _mm_sub_epi8(self.elements, rhs.elements),
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

        // Sign-extend 8-bit to 16-bit by unpacking with sign bits
        let zero = unsafe { _mm_setzero_si128() };

        let self_elements_lo =
            unsafe { _mm_unpacklo_epi8(self.elements, _mm_cmpgt_epi8(zero, self.elements)) }; // sign-extend low half
        let self_elements_hi =
            unsafe { _mm_unpackhi_epi8(self.elements, _mm_cmpgt_epi8(zero, self.elements)) };
        let rhs_elements_lo =
            unsafe { _mm_unpacklo_epi8(rhs.elements, _mm_cmpgt_epi8(zero, rhs.elements)) };
        let rhs_elements_hi =
            unsafe { _mm_unpackhi_epi8(rhs.elements, _mm_cmpgt_epi8(zero, rhs.elements)) };

        // Multiply 16-bit integers
        let prod_lo = unsafe { _mm_mullo_epi16(self_elements_lo, rhs_elements_lo) };
        let prod_hi = unsafe { _mm_mullo_epi16(self_elements_hi, rhs_elements_hi) };

        // Pack 16-bit products into 8-bit integers with saturation
        let elements = unsafe { _mm_packs_epi16(prod_lo, prod_hi) };

        I8x16 {
            size: self.size,
            elements,
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
            let cmp = _mm_cmpeq_epi8(self.elements, other.elements);

            // Move the mask to integer form
            let mask = _mm_movemask_epi8(cmp);

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

            let lt_mask = _mm_movemask_epi8(lt);
            let gt_mask = _mm_movemask_epi8(gt);
            let eq_mask = _mm_movemask_epi8(eq);

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

        unsafe {
            I8x16 {
                size: self.size,
                elements: _mm_and_si128(self.elements, rhs.elements),
            }
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

        unsafe {
            I8x16 {
                size: self.size,
                elements: _mm_or_si128(self.elements, rhs.elements),
            }
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

        (1..=n).into_iter().for_each(|i| {
            let a1: Vec<i8> = (1..=i).collect();

            let v1 = I8x16::new(&a1);

            let truncated_a1 = a1
                .as_slice()
                .into_iter()
                .take(v1.size)
                .map(|&x| x)
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

        (1..=n).into_iter().for_each(|i| {
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

    #[test]
    fn test_sub() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(vec![4], (u1 - v1).to_vec());

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![4, 1], (u2 - v2).to_vec());

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(vec![4, 1, 2], (u3 - v3).to_vec());

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(vec![4, 1, 2, 3], (u4 - v4).to_vec());

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(vec![4, 1, 2, 3, 0], (u5 - v5).to_vec());

        let v6 = I8x16::new(&[1, 10, 7, 2, 1, 9]);
        let u6 = I8x16::new(&[5, 11, 9, 5, 1, 3]);

        assert_eq!(vec![4, 1, 2, 3, 0, -6], (u6 - v6).to_vec());

        let v7 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1]);
        let u7 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9]);

        assert_eq!(vec![4, 1, 2, 3, 0, -6, -8], (u7 - v7).to_vec());

        let v8 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3]);
        let u8 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5]);

        assert_eq!(vec![4, 1, 2, 3, 0, -6, -8, -2], (u8 - v8).to_vec());

        let v9 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6]);
        let u9 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6]);

        assert_eq!(vec![4, 1, 2, 3, 0, -6, -8, -2, 0], (u9 - v9).to_vec());

        let v10 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12]);
        let u10 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10]);

        assert_eq!(vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22], (u10 - v10).to_vec());

        let v11 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8]);
        let u11 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17],
            (u11 - v11).to_vec()
        );

        let v12 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27]);
        let u12 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73],
            (u12 - v12).to_vec()
        );

        let v13 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26]);
        let u13 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73, 24],
            (u13 - v13).to_vec()
        );

        let v14 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35]);
        let u14 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73, 24, 2],
            (u14 - v14).to_vec()
        );

        let v15 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1]);
        let u15 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73, 24, 2, 0],
            (u15 - v15).to_vec()
        );

        let v16 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0]);
        let u16 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73, 24, 2, 0, 7],
            (u16 - v16).to_vec()
        );

        let v17 = I8x16::new(&[1, 10, 7, 2, 1, 9, -1, -3, -6, -12, 8, 27, 26, 35, 1, 0, 2]);
        let u17 = I8x16::new(&[5, 11, 9, 5, 1, 3, -9, -5, -6, 10, -9, 100, 50, 37, 1, 7, 0]);

        assert_eq!(
            vec![4, 1, 2, 3, 0, -6, -8, -2, 0, 22, -17, 73, 24, 2, 0, 7],
            (u17 - v17).to_vec()
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut a = I8x16::new(&[1, 2, 3, 4]);
        let b = I8x16::new(&[4, 3, 2, 1]);

        a += b;

        assert_eq!(vec![5; 4], a.to_vec());
    }

    #[test]
    fn test_mul() {
        let v1 = I8x16::new(&[1]);
        let u1 = I8x16::new(&[5]);

        assert_eq!(vec![5], (u1 * v1).to_vec());

        let v2 = I8x16::new(&[1, 10]);
        let u2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![5, 110], (u2 * v2).to_vec());

        let v3 = I8x16::new(&[1, 10, 7]);
        let u3 = I8x16::new(&[5, 11, 9]);

        assert_eq!(vec![5, 110, 63], (u3 * v3).to_vec());

        let v4 = I8x16::new(&[1, 10, 7, 2]);
        let u4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(vec![5, 110, 63, 10], (u4 * v4).to_vec());

        let v5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let u5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(vec![5, 110, 63, 10, 1], (u5 * v5).to_vec());
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
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        println!("{:?}", u1.lt_elements(v1).to_vec());

        assert_eq!(
            vec![1 < 5],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![1 < 5, 10 < 11],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![1 < 5, 10 < 11, 9 < 7],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 < 5, 10 < 11, 7 < 9, 2 < 5],
            (u4.lt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 < 5, 10 < 11, 7 < 9, 2 < 5],
            (u5.lt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(
            vec![1 <= 5],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11, 9 <= 7],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11, 7 <= 9, 2 <= 5],
            (u4.le_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11, 7 <= 9, 2 <= 5],
            (u5.le_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(
            vec![1 > 5],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![1 > 5, 10 > 11],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![1 > 5, 10 > 11, 9 > 7],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 > 5, 10 > 11, 7 > 9, 2 > 5],
            (u4.gt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 > 5, 10 > 11, 7 > 9, 2 > 5],
            (u5.gt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(
            vec![1 >= 5],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11, 9 >= 7],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11, 7 >= 9, 2 >= 5],
            (u4.ge_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11, 7 >= 9, 2 >= 5],
            (u5.ge_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        println!("{:?}", (u1.eq_elements(v1)).to_vec());

        assert_eq!(
            vec![1 == 5],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 10]);

        println!("{:?}", (u2.eq_elements(v2)).to_vec());
        assert_eq!(
            vec![1 == 5, 10 == 10],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![1 == 5, 10 == 11, 9 == 7],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 == 5, 10 == 11, 7 == 9, 2 == 5],
            (u4.eq_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        println!("{:?}", (u5.eq_elements(v5)).to_vec());

        assert_eq!(
            vec![1 == 5, 10 == 11, 7 == 9, 2 == 5, 1 == 1],
            (u5.eq_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(vec![1 == 5].iter().all(|f| *f), u1 == v1);

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![1 == 5, 10 == 11].iter().all(|f| *f), u2 == v2);

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(vec![1 == 5, 10 == 11, 9 == 7].iter().all(|f| *f), u3 == v3);

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 == 5, 10 == 11, 7 == 9, 2 == 5].iter().all(|f| *f),
            u4 == v4
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 == 5, 10 == 11, 7 == 9, 2 == 5].iter().all(|f| *f),
            u5 == v5
        );
    }

    #[test]
    fn test_lt() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(vec![1 < 5].iter().all(|f| *f), u1 < v1);

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![1 < 5, 10 < 11].iter().all(|f| *f), u2 < v2);

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(vec![1 < 5, 10 < 11, 9 < 7].iter().all(|f| *f), u3 < v3);

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 < 5, 10 < 11, 7 < 9, 2 < 5].iter().all(|f| *f),
            u4 < v4
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 < 5, 10 < 11, 7 < 9, 2 < 5].iter().all(|f| *f),
            u5 < v5
        );
    }

    #[test]
    fn test_le() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(vec![1 <= 5].iter().all(|f| *f), u1 <= v1);

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![1 <= 5, 10 <= 11].iter().all(|f| *f), u2 <= v2);

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(vec![1 <= 5, 10 <= 11, 9 <= 7].iter().all(|f| *f), u3 <= v3);

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11, 7 <= 9, 2 <= 5].iter().all(|f| *f),
            u4 <= v4
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 <= 5, 10 <= 11, 7 <= 9, 2 <= 5].iter().all(|f| *f),
            u5 <= v5
        );
    }

    #[test]
    fn test_gt() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(vec![1 > 5].iter().all(|f| *f), u1 > v1);

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![1 > 5, 10 > 11].iter().all(|f| *f), u2 > v2);

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(vec![1 > 5, 10 > 11, 9 > 7].iter().all(|f| *f), u3 > v3);

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 > 5, 10 > 11, 7 > 9, 2 > 5].iter().all(|f| *f),
            u4 > v4
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 > 5, 10 > 11, 7 > 9, 2 > 5].iter().all(|f| *f),
            u5 > v5
        );
    }

    #[test]
    fn test_ge() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[5]);

        assert_eq!(vec![1 >= 5].iter().all(|f| *f), u1 >= v1);

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(vec![1 >= 5, 10 >= 11].iter().all(|f| *f), u2 >= v2);

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(vec![1 >= 5, 10 >= 11, 9 >= 7].iter().all(|f| *f), u3 >= v3);

        let u4 = I8x16::new(&[1, 10, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11, 7 >= 9, 2 >= 5].iter().all(|f| *f),
            u4 >= v4
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![1 >= 5, 10 >= 11, 7 >= 9, 2 >= 5].iter().all(|f| *f),
            u5 >= v5
        );
    }

    #[test]
    fn test_and() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[0]);

        assert_eq!(
            vec![false],
            (u1 & v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![true, true],
            (u2 & v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![true, true, true],
            (u3 & v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 0, 7, 2]);
        let v4 = I8x16::new(&[5, 11, 9, 5]);

        assert_eq!(
            vec![true, false, true, true],
            (u4 & v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![true, true, true, true],
            (u5 & v5)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_or() {
        let u1 = I8x16::new(&[1]);
        let v1 = I8x16::new(&[0]);

        assert_eq!(
            vec![true],
            (u1 | v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u2 = I8x16::new(&[1, 10]);
        let v2 = I8x16::new(&[5, 11]);

        assert_eq!(
            vec![true, true],
            (u2 | v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u3 = I8x16::new(&[1, 10, 9]);
        let v3 = I8x16::new(&[5, 11, 7]);

        assert_eq!(
            vec![true, true, true],
            (u3 | v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u4 = I8x16::new(&[1, 0, 7, 0]);
        let v4 = I8x16::new(&[5, 11, 9, 0]);

        assert_eq!(
            vec![true, true, true, false],
            (u4 | v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let u5 = I8x16::new(&[1, 10, 7, 2, 1]);
        let v5 = I8x16::new(&[5, 11, 9, 5, 1]);

        assert_eq!(
            vec![true, true, true, true],
            (u5 | v5)
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }
}
