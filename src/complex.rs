// The implementation is based on:
// http://www.librow.com/articles/article-10

use num_complex::Complex;
use num_traits::Float;

use {Operation, Plan, Transform};

impl<T, const N: usize> Transform<T, N> for [Complex<T>; N]
where
    T: Float,
{
    fn transform(&mut self, plan: &Plan<T, N>) {
        rearrange(self);
        calculate(self, &plan.factors);
        if let Operation::Inverse = plan.operation {
            scale(self, N);
        }
    }
}

#[inline(always)]
fn calculate<T, const N: usize>(data: &mut [Complex<T>; N], factors: &[Complex<T>; N])
where
    T: Float,
{
    let mut k = 0;
    let mut step = 1;
    while step < N {
        let jump = step << 1;
        for mut i in 0..step {
            while i < N {
                let j = i + step;
                unsafe {
                    let product = *factors.get_unchecked(k) * *data.get_unchecked(j);
                    *data.get_unchecked_mut(j) = *data.get_unchecked(i) - product;
                    *data.get_unchecked_mut(i) = *data.get_unchecked(i) + product;
                }
                i += jump;
            }
            k += 1;
        }
        step <<= 1;
    }
}

#[inline(always)]
fn rearrange<T, const N: usize>(data: &mut [Complex<T>; N]) {
    let mut j = 0;
    for i in 0..N {
        if j > i {
            data.swap(i, j);
        }
        let mut mask = N >> 1;
        while j & mask != 0 {
            j &= !mask;
            mask >>= 1;
        }
        j |= mask;
    }
}

#[inline(always)]
fn scale<T>(data: &mut [Complex<T>], n: usize)
where
    T: Float,
{
    let factor = T::from(n).unwrap().recip();
    for value in data {
        *value = value.scale(factor);
    }
}
