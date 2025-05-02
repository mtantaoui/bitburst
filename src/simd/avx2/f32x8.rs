#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 8;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    size: usize,
    elements: __m256,
}

impl SimdVec<f32> for F32x8 {
    fn new(slice: &[f32]) -> Self {
        assert!(!slice.is_empty(), "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: f32) -> Self {
        Self {
            elements: unsafe { _mm256_set1_ps(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { _mm256_loadu_ps(ptr) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);

        let elements = match size {
            1 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(0)) },
            2 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(1), *ptr.add(0)) },
            3 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            4 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            5 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            6 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            7 => unsafe {
                _mm256_set_ps(
                    0.0,
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            _ => unreachable!(),
        };

        Self { elements, size }
    }

    unsafe fn store(&self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0f32; LANE_COUNT];

        unsafe {
            _mm256_storeu_ps(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    unsafe fn store_partial(&self) -> Vec<f32> {
        match self.size {
            1..LANE_COUNT => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    unsafe fn store_at(&self, ptr: *mut f32) {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        unsafe {
            _mm256_storeu_ps(ptr, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        let msg: String = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mask: __m256i = match self.size {
            1 => _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0),
            2 => _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
            4 => _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            5 => _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0),
            6 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
            7 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0),
            _ => unreachable!("Store at partial"), // Already checked num_elements range
        };

        _mm256_maskstore_ps(ptr, mask, self.elements);
    }

    fn to_vec(self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", LANE_COUNT);
        assert!(self.size <= LANE_COUNT, "{}", msg);

        if self.size == LANE_COUNT {
            unsafe { self.store() }
        } else {
            unsafe { self.store_partial() }
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
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_EQ_OQ) }; // Result as float mask

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
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_LT_OQ) }; // Result as float mask

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
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_LE_OQ) }; // Result as float mask

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
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_GT_OQ) }; // Result as float mask

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
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_GE_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn add(lhs: F32x8, rhs: F32x8) -> F32x8 {
    assert!(
        lhs.size == rhs.size,
        "Operands must have the same size (expected {} lanes, got {} and {})",
        LANE_COUNT,
        lhs.size,
        rhs.size
    );

    F32x8 {
        size: lhs.size,
        elements: unsafe { _mm256_add_ps(lhs.elements, rhs.elements) },
    }
}

impl Add for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_add_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for F32x8 {
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

impl Sub for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_sub_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for F32x8 {
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

impl Mul for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_mul_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl MulAssign for F32x8 {
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

impl Div for F32x8 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            F32x8 {
                size: self.size,
                elements: _mm256_div_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl DivAssign for F32x8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self / rhs;
    }
}

impl Rem for F32x8 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            let div = _mm256_div_ps(self.elements, rhs.elements);
            let floor = _mm256_floor_ps(div);
            let prod = _mm256_mul_ps(floor, rhs.elements);

            let elements = _mm256_sub_ps(self.elements, prod);

            F32x8 {
                size: self.size,
                elements,
            }
        }
    }
}

impl RemAssign for F32x8 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self % rhs;
    }
}

impl Eq for F32x8 {}

impl PartialEq for F32x8 {
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
            let cmp = _mm256_cmp_ps(self.elements, other.elements, _CMP_EQ_OQ);

            // Move the mask to integer form
            let mask = _mm256_movemask_ps(cmp);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for F32x8 {
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

            let lt_mask = _mm256_movemask_ps(lt);
            let gt_mask = _mm256_movemask_ps(gt);
            let eq_mask = _mm256_movemask_ps(eq);

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
            // converting f32 to bool
            .all(|&f| f != 0.0)
    }

    fn le(&self, other: &Self) -> bool {
        self.le_elements(*other)
            .to_vec()
            .iter()
            // converting f32 to bool
            .all(|&f| f != 0.0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.gt_elements(*other)
            .to_vec()
            .iter()
            // converting f32 to bool
            .all(|&f| f != 0.0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.ge_elements(*other)
            .to_vec()
            .iter()
            // converting f32 to bool
            .all(|&f| f != 0.0)
    }
}

impl BitAnd for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_and_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for F32x8 {
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

impl BitOr for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_or_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for F32x8 {
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
mod f32x8_tests {
    use std::vec;

    use super::*;

    #[test]
    /// __m256 fields are private and cannot be compared directly
    /// test consist on loading elements to __m256 then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let a1 = vec![1.0];
        let v1 = F32x8::new(&a1);

        assert_eq!(a1, v1.to_vec());
        assert_eq!(a1.len(), v1.size);

        let a2 = vec![1.0, 2.0];
        let v2 = F32x8::new(&a2);

        assert_eq!(a2, v2.to_vec());
        assert_eq!(a2.len(), v2.size);

        let a3 = vec![1.0, 2.0, 3.0];
        let v3 = F32x8::new(&a3);

        assert_eq!(a3, v3.to_vec());
        assert_eq!(a3.len(), v3.size);

        let a4 = vec![1.0, 2.0, 3.0, 4.0];
        let v4 = F32x8::new(&a4);

        assert_eq!(a4, v4.to_vec());
        assert_eq!(a4.len(), v4.size);
    }

    /// Splat method should duplicate one value for all elements of __m256
    #[test]
    fn test_splat() {
        let a = vec![1.0; LANE_COUNT];

        let v = F32x8::splat(1.0);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];

        let v = F32x8::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        unsafe { v.store_at(a[1..].as_mut_ptr()) };

        assert_eq!(vec![11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 22.0], a);
    }

    #[test]
    fn test_store_at_partial() {
        let mut a3 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v3 = F32x8::new(&[1.0, 2.0, 3.0]);

        unsafe { v3.store_at_partial(a3[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 19.0, 22.0],
            a3
        );

        let mut a2 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v2 = F32x8::new(&[1.0, 2.0]);

        unsafe { v2.store_at_partial(a2[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 18.0, 19.0, 22.0],
            a2
        );

        let mut a1 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v1 = F32x8::new(&[1.0]);

        unsafe { v1.store_at_partial(a1[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 17.0, 18.0, 19.0, 22.0],
            a1
        );
    }

    #[test]
    fn test_add() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());

        let v5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0, 2.0], (u5 + v5).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a += b;

        assert_eq!(vec![5.0; 4], a.to_vec());
    }

    #[test]
    fn test_sub() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());
    }

    #[test]
    fn test_sub_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a -= b;

        assert_eq!(vec![-3.0, -1.0, 1.0, 3.0], a.to_vec());
    }

    #[test]
    fn test_mul() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![5.0], (u1 * v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![5.0, 110.0], (u2 * v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![5.0, 110.0, 63.0], (u3 * v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0], (u4 * v4).to_vec());

        let v5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0, 1.0], (u5 * v5).to_vec());
    }

    #[test]
    fn test_mul_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a *= b;

        assert_eq!(vec![4.0, 6.0, 6.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_div() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(vec![1.0 / 5.0], (u1 / v1).to_vec());

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0], (u2 / v2).to_vec());

        let u3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0], (u3 / v3).to_vec());

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0],
            (u4 / v4).to_vec()
        );

        let u5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0, 1.0 / 1.0],
            (u5 / v5).to_vec()
        );
    }

    #[test]
    fn test_div_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a /= b;

        assert_eq!(vec![1.0 / 4.0, 2.0 / 3.0, 3.0 / 2.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_rem() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(vec![1.0 % 5.0], (u1 % v1).to_vec());

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0], (u2 % v2).to_vec());

        let u3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0], (u3 % v3).to_vec());

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0],
            (u4 % v4).to_vec()
        );

        let u5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0, 1.0 % 1.0],
            (u5 % v5).to_vec()
        );
    }

    #[test]
    fn test_rem_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a %= b;

        assert_eq!(vec![1.0 % 4.0, 2.0 % 3.0, 3.0 % 2.0, 4.0 % 1.0], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(
            vec![1.0 < 5.0],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0],
            (u4.lt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(
            vec![1.0 <= 5.0],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0],
            (u4.le_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(
            vec![1.0 > 5.0],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0],
            (u4.gt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(
            vec![1.0 >= 5.0],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0],
            (u4.ge_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(
            vec![1.0 == 5.0],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 10.0]);

        assert_eq!(
            vec![0x00000000, 0xFFFFFFFF],
            u2.eq_elements(v2)
                .to_vec()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0],
            (u4.eq_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 == 5.0].iter().all(|f| *f), u1 == v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 == 5.0, 10.0 == 11.0].iter().all(|f| *f), u2 == v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0].iter().all(|f| *f),
            u3 == v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0]
                .iter()
                .all(|f| *f),
            u4 == v4
        );
    }

    #[test]
    fn test_lt() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 < 5.0].iter().all(|f| *f), u1 < v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 < 5.0, 10.0 < 11.0].iter().all(|f| *f), u2 < v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0].iter().all(|f| *f),
            u3 < v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0]
                .iter()
                .all(|f| *f),
            u4 < v4
        );
    }

    #[test]
    fn test_le() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 <= 5.0].iter().all(|f| *f), u1 <= v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 <= 5.0, 10.0 <= 11.0].iter().all(|f| *f), u2 <= v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0].iter().all(|f| *f),
            u3 <= v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0]
                .iter()
                .all(|f| *f),
            u4 <= v4
        );
    }

    #[test]
    fn test_gt() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 > 5.0].iter().all(|f| *f), u1 > v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 > 5.0, 10.0 > 11.0].iter().all(|f| *f), u2 > v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0].iter().all(|f| *f),
            u3 > v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0]
                .iter()
                .all(|f| *f),
            u4 > v4
        );
    }

    #[test]
    fn test_ge() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 >= 5.0].iter().all(|f| *f), u1 >= v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 >= 5.0, 10.0 >= 11.0].iter().all(|f| *f), u2 >= v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0].iter().all(|f| *f),
            u3 >= v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0]
                .iter()
                .all(|f| *f),
            u4 >= v4
        );
    }

    #[test]
    fn test_and() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[0.0]);

        assert_eq!(
            vec![false],
            (u1 & v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 & v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 & v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 0.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![true, false, true, true],
            (u4 & v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_or() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[0.0]);

        assert_eq!(
            vec![true],
            (u1 | v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 | v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 | v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 0.0, 7.0, 0.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 0.0]);

        assert_eq!(
            vec![true, true, true, false],
            (u4 | v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }
}
