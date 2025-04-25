#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    size: usize,
    elements: float32x4_t,
}

impl SimdVec<f32> for F32x4 {
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
            elements: unsafe { vdupq_n_f32(value) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { vld1q_f32(ptr) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);

        let elements = match size {
            1 => {
                let v = vdupq_n_f32(0.0);
                vsetq_lane_f32(*ptr.add(0), v, 0)
            }
            2 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                vsetq_lane_f32(*ptr.add(1), v, 1)
            }
            3 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                v = vsetq_lane_f32(*ptr.add(1), v, 1);
                vsetq_lane_f32(*ptr.add(2), v, 2)
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        Self { elements, size }
    }

    fn store(&self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", LANE_COUNT);

        assert!(self.size <= LANE_COUNT, "{}", msg);

        let mut vec = vec![0f32; LANE_COUNT];

        unsafe {
            vst1q_f32(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<f32> {
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
            vst1q_f32(ptr, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        let msg = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size < LANE_COUNT, "{}", msg);

        match self.size {
            3 => {
                let low = vget_low_f32(self.elements); // extract [0, 1]
                vst1_f32(ptr, low); // store [0, 1]

                let third = vgetq_lane_f32(self.elements, 2); // extract element 2
                *ptr.add(2) = third; // store element 2
            }
            2 => {
                let low = vget_low_f32(self.elements);
                vst1_f32(ptr, low);
            }
            1 => {
                let first = vgetq_lane_f32(self.elements, 0);
                *ptr = first;
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    fn to_vec(self) -> Vec<f32> {
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
        let mask = unsafe { vceqq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

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
        let mask = unsafe { vcltq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

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
        let mask = unsafe { vcleq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

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
        let mask = unsafe { vcgtq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

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
        let mask = unsafe { vcgeq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for F32x4 {
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
            F32x4 {
                size: self.size,
                elements: vaddq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for F32x4 {
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

impl Sub for F32x4 {
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
            F32x4 {
                size: self.size,
                elements: vsubq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for F32x4 {
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

impl Mul for F32x4 {
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
            F32x4 {
                size: self.size,
                elements: vmulq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl MulAssign for F32x4 {
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

impl Div for F32x4 {
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
            F32x4 {
                size: self.size,
                elements: vdivq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl DivAssign for F32x4 {
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

impl Rem for F32x4 {
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
            let div = vdivq_f32(self.elements, rhs.elements);
            let floor = vrndq_f32(div);
            let prod = vmulq_f32(floor, rhs.elements);

            let elements = vsubq_f32(self.elements, prod);

            F32x4 {
                size: self.size,
                elements,
            }
        }
    }
}

impl RemAssign for F32x4 {
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

impl Eq for F32x4 {}

impl PartialEq for F32x4 {
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
            let cmp = vceqq_f32(self.elements, other.elements);

            // Reinterpret result as float for mask extraction
            let mask = vget_lane_u32(vmovn_u64(vreinterpretq_u64_u32(cmp)), 0);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for F32x4 {
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

            // Convert float32x4_t to uint32x4_t where we can use bitwise operations
            let lt_mask = vandq_u32(vreinterpretq_u32_f32(lt), vdupq_n_u32(0x1));
            let gt_mask = vandq_u32(vreinterpretq_u32_f32(gt), vdupq_n_u32(0x1));
            let eq_mask = vandq_u32(vreinterpretq_u32_f32(eq), vdupq_n_u32(0x1));

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

impl BitAnd for F32x4 {
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

        // Convert float32x4_t to uint32x4_t
        let self_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(self.elements) }; // Reinterpret as uint32x4_t
        let rhs_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(rhs.elements) }; // Reinterpret as uint32x4_t

        // Perform bitwise AND between the two uint32x4_t vectors
        let result: uint32x4_t = unsafe { vandq_u32(self_u32, rhs_u32) };

        let elements: float32x4_t = unsafe { vreinterpretq_f32_u32(result) };

        F32x4 {
            size: self.size,
            elements,
        }
    }
}

impl BitAndAssign for F32x4 {
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

impl BitOr for F32x4 {
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
        // Convert float32x4_t to uint32x4_t
        let self_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(self.elements) }; // Reinterpret as uint32x4_t
        let rhs_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(rhs.elements) }; // Reinterpret as uint32x4_t

        // Perform bitwise AND between the two uint32x4_t vectors
        let result: uint32x4_t = unsafe { vorrq_u32(self_u32, rhs_u32) };

        let elements: float32x4_t = unsafe { vreinterpretq_f32_u32(result) };

        F32x4 {
            size: self.size,
            elements,
        }
    }
}

impl BitOrAssign for F32x4 {
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
mod f32x4_tests {
    use std::vec;

    use super::*;

    #[test]
    /// __m128 fields are private and cannot be compared directly
    /// test consist on loading elements to __m128 then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let a1 = vec![1.0];
        let v1 = F32x4::new(&a1);

        assert_eq!(a1, v1.to_vec());
        assert_eq!(a1.len(), v1.size);

        let a2 = vec![1.0, 2.0];
        let v2 = F32x4::new(&a2);

        assert_eq!(a2, v2.to_vec());
        assert_eq!(a2.len(), v2.size);

        let a3 = vec![1.0, 2.0, 3.0];
        let v3 = F32x4::new(&a3);

        assert_eq!(a3, v3.to_vec());
        assert_eq!(a3.len(), v3.size);

        let a4 = vec![1.0, 2.0, 3.0, 4.0];
        let v4 = F32x4::new(&a4);

        assert_eq!(a4, v4.to_vec());
        assert_eq!(a4.len(), v4.size);

        // Should take only first 4 f32 elements
        let a5 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v5 = F32x4::new(&a5);

        assert_eq!(vec![1.0, 2.0, 3.0, 4.0], v5.to_vec());
        assert_eq!(4, v5.size);
    }

    /// Splat method should duplicate one value for all elements of __m128
    #[test]
    fn test_splat() {
        let a = vec![1.0, 1.0, 1.0, 1.0];

        let v = F32x4::splat(1.0);

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];

        let v = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);

        unsafe { v.store_at(a[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 22.0],
            a
        );
    }

    #[test]
    fn test_store_at_partial() {
        let mut a3 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v3 = F32x4::new(&[1.0, 2.0, 3.0]);

        unsafe { v3.store_at_partial(a3[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 19.0, 22.0],
            a3
        );

        let mut a2 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v2 = F32x4::new(&[1.0, 2.0]);

        unsafe { v2.store_at_partial(a2[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 18.0, 19.0, 22.0],
            a2
        );

        let mut a1 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v1 = F32x4::new(&[1.0]);

        unsafe { v1.store_at_partial(a1[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 17.0, 18.0, 19.0, 22.0],
            a1
        );
    }

    #[test]
    fn test_add() {
        let v1 = F32x4::new(&[1.0]);
        let u1 = F32x4::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x4::new(&[1.0, 10.0]);
        let u2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x4::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x4::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());

        let v5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u5 + v5).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::new(&[4.0, 3.0, 2.0, 1.0]);

        a += b;

        assert_eq!(vec![5.0; 4], a.to_vec());
    }

    #[test]
    fn test_sub() {
        let v1 = F32x4::new(&[1.0]);
        let u1 = F32x4::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x4::new(&[1.0, 10.0]);
        let u2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x4::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x4::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());
    }

    #[test]
    fn test_sub_assign() {
        let mut a = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::new(&[4.0, 3.0, 2.0, 1.0]);

        a -= b;

        assert_eq!(vec![-3.0, -1.0, 1.0, 3.0], a.to_vec());
    }

    #[test]
    fn test_mul() {
        let v1 = F32x4::new(&[1.0]);
        let u1 = F32x4::new(&[5.0]);

        assert_eq!(vec![5.0], (u1 * v1).to_vec());

        let v2 = F32x4::new(&[1.0, 10.0]);
        let u2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(vec![5.0, 110.0], (u2 * v2).to_vec());

        let v3 = F32x4::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x4::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![5.0, 110.0, 63.0], (u3 * v3).to_vec());

        let v4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0], (u4 * v4).to_vec());

        let v5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0], (u5 * v5).to_vec());
    }

    #[test]
    fn test_mul_assign() {
        let mut a = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::new(&[4.0, 3.0, 2.0, 1.0]);

        a *= b;

        assert_eq!(vec![4.0, 6.0, 6.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_div() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(vec![1.0 / 5.0], (u1 / v1).to_vec());

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0], (u2 / v2).to_vec());

        let u3 = F32x4::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0], (u3 / v3).to_vec());

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0],
            (u4 / v4).to_vec()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0],
            (u5 / v5).to_vec()
        );
    }

    #[test]
    fn test_div_assign() {
        let mut a = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::new(&[4.0, 3.0, 2.0, 1.0]);

        a /= b;

        assert_eq!(vec![1.0 / 4.0, 2.0 / 3.0, 3.0 / 2.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_rem() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(vec![1.0 % 5.0], (u1 % v1).to_vec());

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0], (u2 % v2).to_vec());

        let u3 = F32x4::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0], (u3 % v3).to_vec());

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0],
            (u4 % v4).to_vec()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0],
            (u5 % v5).to_vec()
        );
    }

    #[test]
    fn test_rem_assign() {
        let mut a = F32x4::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::new(&[4.0, 3.0, 2.0, 1.0]);

        a %= b;

        assert_eq!(vec![1.0 % 4.0, 2.0 % 3.0, 3.0 % 2.0, 4.0 % 1.0], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(
            vec![1.0 < 5.0],
            (u1.lt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0],
            (u2.lt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0],
            (u3.lt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0],
            (u4.lt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0],
            (u5.lt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(
            vec![1.0 <= 5.0],
            (u1.le_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0],
            (u2.le_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0],
            (u3.le_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0],
            (u4.le_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0],
            (u5.le_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(
            vec![1.0 > 5.0],
            (u1.gt_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0],
            (u2.gt_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0],
            (u3.gt_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0],
            (u4.gt_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0],
            (u5.gt_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(
            vec![1.0 >= 5.0],
            (u1.ge_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0],
            (u2.ge_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0],
            (u3.ge_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0],
            (u4.ge_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0],
            (u5.ge_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!(
            vec![1.0 == 5.0],
            (u1.eq_elements(v1))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 10.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 10.0],
            (u2.eq_elements(v2))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0],
            (u3.eq_elements(v3))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0],
            (u4.eq_elements(v4))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0],
            (u5.eq_elements(v5))
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!([1.0 == 5.0].iter().all(|f| *f), u1 == v1);

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!([1.0 == 5.0, 10.0 == 11.0].iter().all(|f| *f), u2 == v2);

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0].iter().all(|f| *f),
            u3 == v3
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0]
                .iter()
                .all(|f| *f),
            u4 == v4
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0]
                .iter()
                .all(|f| *f),
            u5 == v5
        );
    }

    #[test]
    fn test_lt() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!([1.0 < 5.0].iter().all(|f| *f), u1 < v1);

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!([1.0 < 5.0, 10.0 < 11.0].iter().all(|f| *f), u2 < v2);

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0].iter().all(|f| *f),
            u3 < v3
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0]
                .iter()
                .all(|f| *f),
            u4 < v4
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0]
                .iter()
                .all(|f| *f),
            u5 < v5
        );
    }

    #[test]
    fn test_le() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!([1.0 <= 5.0].iter().all(|f| *f), u1 <= v1);

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!([1.0 <= 5.0, 10.0 <= 11.0].iter().all(|f| *f), u2 <= v2);

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0].iter().all(|f| *f),
            u3 <= v3
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0]
                .iter()
                .all(|f| *f),
            u4 <= v4
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0]
                .iter()
                .all(|f| *f),
            u5 <= v5
        );
    }

    #[test]
    fn test_gt() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!([1.0 > 5.0].iter().all(|f| *f), u1 > v1);

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!([1.0 > 5.0, 10.0 > 11.0].iter().all(|f| *f), u2 > v2);

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0].iter().all(|f| *f),
            u3 > v3
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0]
                .iter()
                .all(|f| *f),
            u4 > v4
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0]
                .iter()
                .all(|f| *f),
            u5 > v5
        );
    }

    #[test]
    fn test_ge() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[5.0]);

        assert_eq!([1.0 >= 5.0].iter().all(|f| *f), u1 >= v1);

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!([1.0 >= 5.0, 10.0 >= 11.0].iter().all(|f| *f), u2 >= v2);

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0].iter().all(|f| *f),
            u3 >= v3
        );

        let u4 = F32x4::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0]
                .iter()
                .all(|f| *f),
            u4 >= v4
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0]
                .iter()
                .all(|f| *f),
            u5 >= v5
        );
    }

    #[test]
    fn test_and() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[0.0]);

        assert_eq!(
            vec![false],
            (u1 & v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 & v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 & v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 0.0, 7.0, 2.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![true, false, true, true],
            (u4 & v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![true, true, true, true],
            (u5 & v5)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_or() {
        let u1 = F32x4::new(&[1.0]);
        let v1 = F32x4::new(&[0.0]);

        assert_eq!(
            vec![true],
            (u1 | v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x4::new(&[1.0, 10.0]);
        let v2 = F32x4::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 | v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x4::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x4::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 | v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x4::new(&[1.0, 0.0, 7.0, 0.0]);
        let v4 = F32x4::new(&[5.0, 11.0, 9.0, 0.0]);

        assert_eq!(
            vec![true, true, true, false],
            (u4 | v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u5 = F32x4::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x4::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![true, true, true, true],
            (u5 | v5)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }
}
