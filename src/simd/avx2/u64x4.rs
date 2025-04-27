#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct U64x4 {
    size: usize,
    elements: __m256i,
}

impl SimdVec<u64> for U64x4 {
    fn new(slice: &[u64]) -> Self {
        assert!(!slice.is_empty(), "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: u64) -> Self {
        Self {
            elements: unsafe { _mm256_set1_epi64x(value as i64) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const u64, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { _mm256_loadu_si256(ptr as *const __m256i) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const u64, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);

        let elements = match size {
            1 => unsafe { _mm256_set_epi64x(0, 0, 0, *ptr.add(0) as i64) },
            2 => unsafe { _mm256_set_epi64x(0, 0, *ptr.add(1) as i64, *ptr.add(0) as i64) },
            3 => unsafe {
                _mm256_set_epi64x(
                    0,
                    *ptr.add(2) as i64,
                    *ptr.add(1) as i64,
                    *ptr.add(0) as i64,
                )
            },
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        Self { elements, size }
    }

    fn store(&self) -> Vec<u64> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0u64; LANE_COUNT];

        unsafe {
            _mm256_storeu_si256(vec.as_mut_ptr() as *mut __m256i, self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<u64> {
        match self.size {
            1..LANE_COUNT => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    unsafe fn store_at(&self, ptr: *mut u64) {
        let msg = format!("Size must be == {}", LANE_COUNT);

        assert!(self.size == LANE_COUNT, "{}", msg);

        unsafe {
            _mm256_storeu_si256(ptr as *mut __m256i, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut u64) {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(self.size < LANE_COUNT, "{}", msg);

        let mask = match self.size {
            1 => _mm256_setr_epi64x(-1, 0, 0, 0),
            2 => _mm256_setr_epi64x(-1, -1, 0, 0),
            3 => _mm256_setr_epi64x(-1, -1, -1, 0),
            _ => panic!("Invalid size"),
        };

        unsafe {
            _mm256_maskstore_epi64(ptr as *mut i64, mask, self.elements);
        }
    }

    fn to_vec(self) -> Vec<u64> {
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
        let elements = unsafe { _mm256_cmpeq_epi64(self.elements, rhs.elements) };
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
        let elements = unsafe { _mm256_cmpgt_epi64(rhs.elements, self.elements) };

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
        let less_than = unsafe { _mm256_cmpgt_epi64(rhs.elements, self.elements) };
        let equal = unsafe { _mm256_cmpeq_epi64(self.elements, rhs.elements) };
        let elements = unsafe { _mm256_or_si256(less_than, equal) };

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
        let elements = unsafe { _mm256_cmpgt_epi64(self.elements, rhs.elements) }; // Result as float mask

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
        let greater_than = unsafe { _mm256_cmpgt_epi64(self.elements, rhs.elements) };
        let equal = unsafe { _mm256_cmpeq_epi64(self.elements, rhs.elements) };
        let elements = unsafe { _mm256_or_si256(greater_than, equal) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for U64x4 {
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
            U64x4 {
                size: self.size,
                elements: _mm256_add_epi64(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for U64x4 {
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

impl Sub for U64x4 {
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
            U64x4 {
                size: self.size,
                elements: _mm256_sub_epi64(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for U64x4 {
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

impl Mul for U64x4 {
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

        let lo_lo = unsafe { _mm256_mul_epu32(self.elements, rhs.elements) }; // Calculates al*bl

        let self_swap = unsafe { _mm256_shuffle_epi32(self.elements, 0xB1) };
        let rhs_swap = unsafe { _mm256_shuffle_epi32(rhs.elements, 0xB1) };

        let selfh_rhsl = unsafe { _mm256_mul_epu32(self_swap, rhs.elements) };

        let selfl_rhsh = unsafe { _mm256_mul_epu32(self.elements, rhs_swap) };

        let cross_sum = unsafe { _mm256_add_epi64(selfh_rhsl, selfl_rhsh) };

        let cross_sum_shifted = unsafe { _mm256_slli_epi64(cross_sum, 32) };

        let elements = unsafe { _mm256_add_epi64(lo_lo, cross_sum_shifted) };

        U64x4 {
            size: self.size,
            elements,
        }
    }
}

impl MulAssign for U64x4 {
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

impl Eq for U64x4 {}

impl PartialEq for U64x4 {
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
            let cmp = _mm256_cmpeq_epi64(self.elements, other.elements);

            // Move the mask to integer form
            let mask = _mm256_movemask_epi8(cmp);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for U64x4 {
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

            let lt_mask = _mm256_movemask_epi8(lt);
            let gt_mask = _mm256_movemask_epi8(gt);
            let eq_mask = _mm256_movemask_epi8(eq);

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
            // converting u64 to bool
            .all(|&f| f != 0)
    }

    fn le(&self, other: &Self) -> bool {
        self.le_elements(*other)
            .to_vec()
            .iter()
            // converting u64 to bool
            .all(|&f| f != 0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.gt_elements(*other)
            .to_vec()
            .iter()
            // converting u64 to bool
            .all(|&f| f != 0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.ge_elements(*other)
            .to_vec()
            .iter()
            // converting u64 to bool
            .all(|&f| f != 0)
    }
}

impl BitAnd for U64x4 {
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
            U64x4 {
                size: self.size,
                elements: _mm256_and_si256(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for U64x4 {
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

impl BitOr for U64x4 {
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
            U64x4 {
                size: self.size,
                elements: _mm256_or_si256(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for U64x4 {
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
mod u64x4_tests {
    use std::{cmp::min, vec};

    use super::*;

    #[test]
    /// __m256i fields are private and cannot be compared directly
    /// test consist on loading elements to __m256i then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let n = 20;

        (1..=n).for_each(|i| {
            let a1: Vec<u64> = (1..=i).collect();

            let v1 = U64x4::new(&a1);

            let truncated_a1 = a1
                .as_slice()
                .iter()
                .take(v1.size)
                .copied()
                .collect::<Vec<u64>>();

            assert_eq!(truncated_a1, v1.to_vec());
            assert_eq!(min(truncated_a1.len(), LANE_COUNT), v1.size);
        });
    }

    /// Splat method should duplicate one value for all elements of __m256
    #[test]
    fn test_splat() {
        let a = vec![1; 4];

        let v = U64x4::splat(1);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a1: Vec<u64> = vec![100; 20];

        let s1: Vec<u64> = (1..=8).collect();
        let v1 = U64x4::new(&s1);

        unsafe { v1.store_at(a1[0..].as_mut_ptr()) };

        assert_eq!(
            &[
                1, 2, 3, 4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100
            ],
            a1.as_slice()
        );

        let mut a2: Vec<u64> = vec![1; 20];

        let s2: Vec<u64> = (1..=16).collect();
        let v2 = U64x4::new(&s2);

        unsafe { v2.store_at(a2[4..].as_mut_ptr()) };

        assert_eq!(
            &[1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            a2.as_slice()
        );
    }

    #[test]
    fn test_store_at_partial() {
        let n = 1;

        (1..=n).for_each(|i| {
            let mut vector: Vec<u64> = vec![100; 11];

            let a: Vec<u64> = (1..=i).collect();

            let v = U64x4::new(a.as_slice());

            unsafe {
                v.store_at_partial(vector[4..].as_mut_ptr());
            }

            let test = match i {
                1 => &[100, 100, 100, 100, 1, 100, 100, 100, 100, 100, 100],

                _ => panic!("Not a test case"),
            };

            assert_eq!(test, vector.as_slice());
        });

        let mut vector: Vec<u64> = vec![100; 3];

        let a: Vec<u64> = (1..=1).collect();

        let v = U64x4::new(a.as_slice());

        unsafe {
            v.store_at_partial(vector[2..].as_mut_ptr());
        }

        assert_eq!(vector, [100, 100, 1])
    }

    #[test]
    fn test_add() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(vec![6], (u1 + v1).to_vec());

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(vec![6, 21], (u2 + v2).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = U64x4::new(&[1, 2, 3, 4]);
        let b = U64x4::new(&[4, 3, 2, 1]);

        a += b;

        assert_eq!(vec![5; 4], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[test]
    fn test_sub() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(vec![5 - 1], (u1 - v1).to_vec());

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(vec![5 - 1, 11 - 10], (u2 - v2).to_vec());
    }

    #[test]
    fn test_sub_assign() {
        let mut a = U64x4::new(&[7, 4, 3, 4]);
        let b = U64x4::new(&[4, 3, 2, 1]);

        a -= b;

        assert_eq!(vec![3, 1, 1, 3], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[test]
    fn test_mul() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(vec![5 * 1], (u1 * v1).to_vec());

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(vec![5 * 1, 11 * 10], (u2 * v2).to_vec());
    }

    #[test]
    fn test_mul_assign() {
        let mut a = U64x4::new(&[1, 2, 3, 4]);
        let b = U64x4::new(&[4, 3, 2, 1]);

        a *= b;

        assert_eq!(vec![4, 6, 6, 4], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5 < 1],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5 < 1, 11 < 10],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5 <= 1],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5 > 1],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5 > 1, 11 > 10],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5 >= 1],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5 == 1],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5 == 1, 11 == 10],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!([5 == 1].iter().all(|f| *f), u1 == v1);

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!([5 == 1, 11 == 10].iter().all(|f| *f), u2 == v2);
    }

    #[test]
    fn test_lt() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!([5 < 1].iter().all(|f| *f), u1 < v1);

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!([5 < 1, 11 < 10].iter().all(|f| *f), u2 < v2);
    }

    #[test]
    fn test_le() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!([5 <= 1].iter().all(|f| *f), u1 <= v1);

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!([5 <= 1, 11 <= 10].iter().all(|f| *f), u2 <= v2);
    }

    #[test]
    fn test_gt() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!([5 > 1].iter().all(|f| *f), u1 > v1);

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!([5 > 1, 11 > 10].iter().all(|f| *f), u2 > v2);
    }

    #[test]
    fn test_ge() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!([5 >= 1].iter().all(|f| *f), u1 >= v1);

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!([5 >= 1, 11 >= 10].iter().all(|f| *f), u2 >= v2);
    }

    #[allow(clippy::erasing_op)]
    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_and() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5u8 & 1u8 != 0],
            (u1.bitand(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0],
            (u2.bitand(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_or() {
        let v1 = U64x4::new(&[1]);
        let u1 = U64x4::new(&[5]);

        assert_eq!(
            vec![5u8 | 1u8 != 0],
            (u1.bitor(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = U64x4::new(&[1, 10]);
        let u2 = U64x4::new(&[5, 11]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0],
            (u2.bitor(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }
}
