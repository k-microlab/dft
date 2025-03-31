#![no_std]
#![feature(generic_const_exprs)]
//! [Discrete Fourier transform][1].
//!
//! The `Transform` trait is responsible for performing the transform. The trait
//! is implemented for both real and complex data. There are three transform
//! operations available: forward, backward, and inverse. The desired operation
//! is specified by the `Operation` enumeration passed to the `Plan::new`
//! function, which precomputes auxiliary information needed for
//! `Transform::transform`. All the operations are preformed in place.
//!
//! When applied to real data, the transform works as follows. If the operation
//! is forward, the data are replaced by the positive frequency half of their
//! complex transform. The first and last components of the complex transform,
//! which are real, are stored in `self[0]` and `self[1]`, respectively. If the
//! operation is backward or inverse, the data are assumed to be stored
//! according to the above convention. See the reference below for further
//! details.
//!
//! ## Example
//!
//! ```
//! use std::mem::MaybeUninit;
//! use dft::{Operation, Plan, c64};
//!
//! const N: usize = 512;
//! let mut factors = MaybeUninit::uninit();
//! let plan = Plan::new(Operation::Forward, &mut factors);
//! let mut data = [c64::new(42.0, 69.0); N];
//! dft::transform(&mut data, &plan);
//! ```
//!
//! ## References
//!
//! 1. W. Press, S. Teukolsky, W. Vetterling, and B. Flannery, “Numerical
//! Recipes 3rd Edition: The Art of Scientific Computing,” Cambridge University
//! Press, 2007.
//!
//! [1]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform

extern crate num_complex;
extern crate num_traits;

use core::mem::MaybeUninit;
use num_complex::Complex;
use num_traits::{Float, FloatConst, One};

/// A complex number with 32-bit parts.
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

/// A complex number with 64-bit parts.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

mod complex;
mod real;

pub use real::unpack;

/// A transform operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    /// The forward transform.
    Forward,
    /// The backward transform.
    Backward,
    /// The inverse transform.
    Inverse,
}

/// A transform plan.
#[derive(Clone, Debug)]
pub struct Plan<'a, T, const N: usize> {
    factors: &'a [Complex<T>; N],
    operation: Operation,
}

/// The transform.
pub trait Transform<T, const N: usize> {
    /// Perform the transform.
    fn transform(&mut self, plan: &Plan<T, N>);
}

impl<'a, T, const N: usize> Plan<'a, T, N>
where
    T: Float + FloatConst,
{
    /// Create a plan for a specific operation and specific number of points.
    ///
    /// The number of points should be a power of two.
    pub fn new(operation: Operation, factors: &'a mut MaybeUninit<[Complex<T>; N]>) -> Self {
        assert!(N.is_power_of_two());
        let one = T::one();
        let two = one + one;
        let sign = if let Operation::Forward = operation {
            -one
        } else {
            one
        };
        let mut i = 0;
        let mut step = 1;
        while step < N {
            let (multiplier, mut factor) = {
                let theta = T::PI() / T::from(step).unwrap();
                let sine = (theta / two).sin();
                (
                    Complex::new(-two * sine * sine, sign * theta.sin()),
                    Complex::one(),
                )
            };
            for _ in 0..step {
                unsafe { factors.as_mut_ptr().cast::<Complex<T>>().add(i).write(factor) }
                i += 1;
                factor = multiplier * factor + factor;
            }
            step <<= 1;
        }
        Plan {
            factors: unsafe { factors.assume_init_ref() },
            operation,
        }
    }
}

/// Perform the transform.
///
/// The function is a shortcut for `Transform::transform`.
#[inline(always)]
pub fn transform<D: ?Sized, T, const N: usize>(data: &mut D, plan: &Plan<T, N>)
where
    D: Transform<T, N>,
{
    Transform::transform(data, plan);
}

#[inline]
pub const unsafe fn as_array<T, const N: usize>(slice: &[T]) -> &[T; N] {
    &*(slice.as_ptr() as *const [_; N])
}

#[inline]
pub const unsafe fn as_mut_array<T, const N: usize>(slice: &mut [T]) -> &mut [T; N] {
    &mut *(slice.as_mut_ptr() as *mut [_; N])
}