#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::vec::SimdVec;

pub const LANE_COUNT: usize = 2;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct U64x2 {
    size: usize,
    elements: __m128i,
}

impl SimdVec<u64> for U64x2 {
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
            elements: unsafe { _mm_set1_epi64x(value as i64) },
            size: LANE_COUNT,
        }
    }

    unsafe fn load(ptr: *const u64, size: usize) -> Self {
        let msg = format!("Size must be == {}", LANE_COUNT);
        assert!(size == LANE_COUNT, "{}", msg);

        Self {
            elements: unsafe { _mm_loadu_si128(ptr as *const __m128i) },
            size,
        }
    }

    unsafe fn load_partial(ptr: *const u64, size: usize) -> Self {
        let msg = format!("Size must be < {}", LANE_COUNT);
        assert!(size < LANE_COUNT, "{}", msg);

        let elements = match size {
            1 => unsafe { _mm_set_epi64x(0, *ptr.add(0) as i64) },

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
            _mm_storeu_si128(vec.as_mut_ptr() as *mut __m128i, self.elements);
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
            _mm_storeu_si128(ptr as *mut __m128i, self.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut u64) {
        let msg = format!("Size must be < {}", LANE_COUNT);

        assert!(self.size < LANE_COUNT, "{}", msg);

        match self.size {
            1 => *ptr.add(0) = _mm_cvtsi128_si64(self.elements) as u64,
            _ => panic!("Invalid size"),
        };
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
        let elements = unsafe { _mm_cmpeq_epi64(self.elements, rhs.elements) };
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
        let elements = unsafe { _mm_cmpgt_epi64(rhs.elements, self.elements) };

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
        let less_than = unsafe { _mm_cmpgt_epi64(rhs.elements, self.elements) };
        let equal = unsafe { _mm_cmpeq_epi64(self.elements, rhs.elements) };
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
        let elements = unsafe { _mm_cmpgt_epi64(self.elements, rhs.elements) }; // Result as float mask

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
        let greater_than = unsafe { _mm_cmpgt_epi64(self.elements, rhs.elements) };
        let equal = unsafe { _mm_cmpeq_epi64(self.elements, rhs.elements) };
        let elements = unsafe { _mm_or_si128(greater_than, equal) };

        Self {
            elements,
            size: self.size,
        }
    }
}

impl Add for U64x2 {
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
            U64x2 {
                size: self.size,
                elements: _mm_add_epi64(self.elements, rhs.elements),
            }
        }
    }
}

impl AddAssign for U64x2 {
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

impl Sub for U64x2 {
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
            U64x2 {
                size: self.size,
                elements: _mm_sub_epi64(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for U64x2 {
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

impl Mul for U64x2 {
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

        // Extract low 64 bits
        let self_lo = unsafe { _mm_cvtsi128_si64(self.elements) };
        let rhs_lo = unsafe { _mm_cvtsi128_si64(rhs.elements) };

        // Extract high 64 bits
        let self_hi =
            unsafe { _mm_cvtsi128_si64(_mm_unpackhi_epi64(self.elements, self.elements)) };
        let rhs_hi = unsafe { _mm_cvtsi128_si64(_mm_unpackhi_epi64(rhs.elements, rhs.elements)) };

        // Multiply using scalar ops
        let mul_lo = self_lo.wrapping_mul(rhs_lo);
        let mul_hi = self_hi.wrapping_mul(rhs_hi);

        // Pack back into __m128i
        let elements = unsafe { _mm_set_epi64x(mul_hi, mul_lo) };

        U64x2 {
            size: self.size,
            elements,
        }
    }
}

impl MulAssign for U64x2 {
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

impl Eq for U64x2 {}

impl PartialEq for U64x2 {
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
            let cmp = _mm_cmpeq_epi64(self.elements, other.elements);

            // Move the mask to integer form
            let mask = _mm_movemask_epi8(cmp);

            // All 4 lanes equal => mask == 0b1111 == 0xF
            mask == 0xF
        }
    }
}

impl PartialOrd for U64x2 {
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

impl BitAnd for U64x2 {
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
            U64x2 {
                size: self.size,
                elements: _mm_and_si128(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for U64x2 {
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

impl BitOr for U64x2 {
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
            U64x2 {
                size: self.size,
                elements: _mm_or_si128(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for U64x2 {
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
