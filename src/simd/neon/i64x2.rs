#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 2;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct I64x2 {
    size: usize,
    elements: int64x2_t,
}

impl SimdVec<i64> for I64x2 {
    fn new(slice: &[i64]) -> Self {
        assert!(!slice.is_empty(), "Size can't be zero");

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    fn splat(value: i64) -> Self {
        Self {
            elements: unsafe { vdupq_n_s64(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const i64, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { vld1q_s64(ptr) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const i64, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);
        // Start with a zero vector
        let mut elements = vdupq_n_s64(0);

        // Load elements individually using vsetq_lane
        match size {
            1 => {
                elements = vsetq_lane_s64(*ptr.add(0), elements, 0);
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
        Self { elements, size }
    }

    fn store(&self) -> Vec<i64> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0i64; LANE_COUNT];

        unsafe {
            vst1q_s64(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<i64> {
        match self.size {
            1..LANE_COUNT => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    unsafe fn store_at(&self, ptr: *mut i64) {
        let msg = format!("Size must be == {}", LANE_COUNT);

        assert!(self.size == LANE_COUNT, "{}", msg);

        unsafe {
            vst1q_s64(ptr, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut i64) {
        let msg = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size < LANE_COUNT, "{}", msg);

        match self.size {
            1 => *ptr.add(0) = vgetq_lane_s64(self.elements, 0),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    fn to_vec(self) -> Vec<i64> {
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
        let mask = unsafe { vceqq_s64(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s64_u64(mask) };

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
        let mask = unsafe { vcltq_s64(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s64_u64(mask) };

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
        let mask = unsafe { vcleq_s64(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s64_u64(mask) };

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
        let mask = unsafe { vcgtq_s64(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s64_u64(mask) };

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
        let mask = unsafe { vcgeq_s64(self.elements, rhs.elements) };

        let elements = unsafe { vreinterpretq_s64_u64(mask) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for I64x2 {
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
            I64x2 {
                size: self.size,
                elements: vaddq_s64(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for I64x2 {
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

impl Sub for I64x2 {
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
            I64x2 {
                size: self.size,
                elements: vsubq_s64(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for I64x2 {
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

impl Mul for I64x2 {
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

        let self_arr: [i64; 2] = unsafe { core::mem::transmute(self.elements) };
        let rhs_arr: [i64; 2] = unsafe { core::mem::transmute(rhs.elements) };

        let mul = [
            self_arr[0].wrapping_mul(rhs_arr[0]),
            self_arr[1].wrapping_mul(rhs_arr[1]),
        ];

        let elements: int64x2_t = unsafe { core::mem::transmute(mul) };

        I64x2 {
            size: self.size,
            elements,
        }
    }
}

impl MulAssign for I64x2 {
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

impl Eq for I64x2 {}

impl PartialEq for I64x2 {
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
            let cmp = vceqq_s64(self.elements, other.elements);

            // Reinterpret result as float for mask extraction
            // let mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_s64(vreinterpretq_s64_u8(cmp))), 0);
            let mask = vget_lane_u32(vmovn_u64(cmp), 0);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for I64x2 {
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

            let lt_mask = vandq_u32(vreinterpretq_u32_s64(lt), vdupq_n_u32(0x1));
            let gt_mask = vandq_u32(vreinterpretq_u32_s64(gt), vdupq_n_u32(0x1));
            let eq_mask = vandq_u32(vreinterpretq_u32_s64(eq), vdupq_n_u32(0x1));

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
            // converting i64 to bool
            .all(|&f| f != 0)
    }

    fn le(&self, other: &Self) -> bool {
        self.le_elements(*other)
            .to_vec()
            .iter()
            // converting i64 to bool
            .all(|&f| f != 0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.gt_elements(*other)
            .to_vec()
            .iter()
            // converting i64 to bool
            .all(|&f| f != 0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.ge_elements(*other)
            .to_vec()
            .iter()
            // converting i64 to bool
            .all(|&f| f != 0)
    }
}

impl BitAnd for I64x2 {
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
        let elements = unsafe { vandq_s64(self.elements, rhs.elements) };

        I64x2 {
            size: self.size,
            elements,
        }
    }
}

impl BitAndAssign for I64x2 {
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

impl BitOr for I64x2 {
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

        let elements = unsafe { vorrq_s64(self.elements, rhs.elements) };

        I64x2 {
            size: self.size,
            elements,
        }
    }
}

impl BitOrAssign for I64x2 {
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
mod i64x2_tests {
    use std::{cmp::min, vec};

    use super::*;

    #[test]
    /// __m128i fields are private and cannot be compared directly
    /// test consist on loading elements to __m128i then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let n = 20;

        (1..=n).for_each(|i| {
            let a1: Vec<i64> = (1..=i).collect();

            let v1 = I64x2::new(&a1);

            let truncated_a1 = a1
                .as_slice()
                .iter()
                .take(v1.size)
                .copied()
                .collect::<Vec<i64>>();

            assert_eq!(truncated_a1, v1.to_vec());
            assert_eq!(min(truncated_a1.len(), LANE_COUNT), v1.size);
        });
    }

    /// Splat method should duplicate one value for all elements of __m128
    #[test]
    fn test_splat() {
        let a = vec![1; 2];

        let v = I64x2::splat(1);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a1: Vec<i64> = vec![100; 20];

        let s1: Vec<i64> = (1..=8).collect();
        let v1 = I64x2::new(&s1);

        unsafe { v1.store_at(a1[0..].as_mut_ptr()) };

        assert_eq!(
            &[
                1, 2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100
            ],
            a1.as_slice()
        );

        let mut a2: Vec<i64> = vec![-1; 20];

        let s2: Vec<i64> = (1..=16).collect();
        let v2 = I64x2::new(&s2);

        unsafe { v2.store_at(a2[4..].as_mut_ptr()) };

        assert_eq!(
            &[-1, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            a2.as_slice()
        );
    }

    #[test]
    fn test_store_at_partial() {
        let n = 1;

        (1..=n).for_each(|i| {
            let mut vector: Vec<i64> = vec![100; 11];

            let a: Vec<i64> = (1..=i).collect();

            let v = I64x2::new(a.as_slice());

            unsafe {
                v.store_at_partial(vector[4..].as_mut_ptr());
            }

            let test = match i {
                1 => &[100, 100, 100, 100, 1, 100, 100, 100, 100, 100, 100],

                _ => panic!("Not a test case"),
            };

            assert_eq!(test, vector.as_slice());
        });

        let mut vector: Vec<i64> = vec![100; 3];

        let a: Vec<i64> = (1..=1).collect();

        let v = I64x2::new(a.as_slice());

        unsafe {
            v.store_at_partial(vector[2..].as_mut_ptr());
        }

        assert_eq!(vector, [100, 100, 1])
    }

    #[test]
    fn test_add() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(vec![6], (u1 + v1).to_vec());

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(vec![6, 21], (u2 + v2).to_vec());

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(vec![6, 21], (u3 + v3).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = I64x2::new(&[1, 2, 3, 4]);
        let b = I64x2::new(&[4, 3, 2, 1]);

        a += b;

        assert_eq!(vec![5; 2], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[test]
    fn test_sub() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(vec![5 - 1], (u1 - v1).to_vec());

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(vec![5 - 1, 11 - 10], (u2 - v2).to_vec());

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(vec![5 - 1, 11 - 10], (u3 - v3).to_vec());
    }

    #[test]
    fn test_sub_assign() {
        let mut a = I64x2::new(&[1, 2, 3, 4]);
        let b = I64x2::new(&[4, 3, 2, 1]);

        a -= b;

        assert_eq!(vec![-3, -1], a.to_vec());
    }

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[test]
    fn test_mul() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(vec![5 * 1], (u1 * v1).to_vec());

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(vec![5 * 1, 11 * 10], (u2 * v2).to_vec());

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(vec![5 * 1, 11 * 10], (u3 * v3).to_vec());
    }

    #[test]
    fn test_mul_assign() {
        let mut a = I64x2::new(&[1, 2, 3, 4]);
        let b = I64x2::new(&[4, 3, 2, 1]);

        a *= b;

        assert_eq!(vec![4, 6], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5 < 1],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5 < 1, 11 < 10],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 < 1, 11 < 10],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5 <= 1],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 <= 1, 11 <= 10],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5 > 1],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5 > 1, 11 > 10],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 > 1, 11 > 10],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5 >= 1],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 >= 1, 11 >= 10],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5 == 1],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5 == 1, 11 == 10],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5 == 1, 11 == 10],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!([5 == 1].iter().all(|f| *f), u1 == v1);

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!([5 == 1, 11 == 10].iter().all(|f| *f), u2 == v2);

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!([5 == 1, 11 == 10].iter().all(|f| *f), u3 == v3);
    }

    #[test]
    fn test_lt() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!([5 < 1].iter().all(|f| *f), u1 < v1);

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!([5 < 1, 11 < 10].iter().all(|f| *f), u2 < v2);

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!([5 < 1, 11 < 10].iter().all(|f| *f), u3 < v3);
    }

    #[test]
    fn test_le() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!([5 <= 1].iter().all(|f| *f), u1 <= v1);

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!([5 <= 1, 11 <= 10].iter().all(|f| *f), u2 <= v2);

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!([5 <= 1, 11 <= 10].iter().all(|f| *f), u3 <= v3);
    }

    #[test]
    fn test_gt() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!([5 > 1].iter().all(|f| *f), u1 > v1);

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!([5 > 1, 11 > 10].iter().all(|f| *f), u2 > v2);

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!([5 > 1, 11 > 10].iter().all(|f| *f), u3 > v3);
    }

    #[test]
    fn test_ge() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!([5 >= 1].iter().all(|f| *f), u1 >= v1);

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!([5 >= 1, 11 >= 10].iter().all(|f| *f), u2 >= v2);

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!([5 >= 1, 11 >= 10].iter().all(|f| *f), u3 >= v3);
    }

    #[allow(clippy::erasing_op)]
    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_and() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5u8 & 1u8 != 0],
            (u1.bitand(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0],
            (u2.bitand(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 & 1u8 != 0, 11u8 & 10u8 != 0],
            (u3.bitand(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }

    #[allow(clippy::bad_bit_mask)]
    #[test]
    fn test_or() {
        let v1 = I64x2::new(&[1]);
        let u1 = I64x2::new(&[5]);

        assert_eq!(
            vec![5u8 | 1u8 != 0],
            (u1.bitor(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v2 = I64x2::new(&[1, 10]);
        let u2 = I64x2::new(&[5, 11]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0],
            (u2.bitor(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );

        let v3 = I64x2::new(&[1, 10, 7]);
        let u3 = I64x2::new(&[5, 11, 9]);

        assert_eq!(
            vec![5u8 | 1u8 != 0, 11u8 | 10u8 != 0],
            (u3.bitor(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0)
                .collect::<Vec<bool>>()
        );
    }
}
