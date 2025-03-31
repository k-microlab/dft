use core::mem::MaybeUninit;
use num_complex::Complex;
use num_traits::Float;
use core::slice::from_raw_parts_mut;

use {Operation, Plan, Transform};

pub(crate) enum Assert<const CHECK: bool> {}
pub(crate) trait IsTrue {}
impl IsTrue for Assert<true> {}

impl<T, const N: usize> Transform<T, N> for [T; N]
where
    T: Float,
    Assert<{N / 2 > 0}>: IsTrue
{
    fn transform(&mut self, plan: &Plan<T, N>) {
        if N / 2 == 0 {
            return;
        }
        let plan: &Plan<T, {N / 2}> = unsafe { core::mem::transmute(plan) };
        let data = unsafe {
            crate::as_mut_array::<_, {N / 2}>(from_raw_parts_mut::<Complex<T>>(self.as_mut_ptr() as *mut _, N / 2))
        };
        match plan.operation {
            Operation::Forward => {
                data.transform(plan);
                compose(data, plan.factors, false);
            }
            Operation::Backward | Operation::Inverse => {
                compose(data, plan.factors, true);
                data.transform(plan);
            }
        }
    }
}

/// Unpack the result produced by the forward transform applied to real data.
///
/// The function decodes the result of an application of `Transform::transform`
/// with `Operation::Forward` to real data. See the top-level description of the
/// crate for further details.
pub fn unpack<'a, T, const N: usize>(data: &[T; N], result: &'a mut MaybeUninit<[Complex<T>; N]>) -> &'a mut [Complex<T>; N]
where
    T: Float,
{
    assert!(N.is_power_of_two());
    let h = N >> 1;
    let base = result.as_mut_ptr().cast::<Complex<T>>();

    unsafe {
        base.write(data[0].into());
    }
    if h == 0 {
        return unsafe { result.assume_init_mut() };
    }
    for i in 1..h {
        let complex = Complex::new(data[2 * i], data[2 * i + 1]);
        unsafe {
            base.add(i).write(complex);
        }
    }
    unsafe {
        base.add(h).write(data[1].into());
    }
    for i in (h + 1)..N {
        unsafe {
            base.add(i).write(base.add(N - i).read().conj());
        }
    }
    unsafe { result.assume_init_mut() }
}

#[inline(always)]
fn compose<T, const N: usize>(data: &mut [Complex<T>; N], factors: &[Complex<T>; N], inverse: bool)
where
    T: Float,
{
    let one = T::one();
    let half = (one + one).recip();
    let h = N >> 1;
    data[0] = Complex::new(data[0].re + data[0].im, data[0].re - data[0].im);
    if inverse {
        data[0] = data[0].scale(half);
    }
    if h == 0 {
        return;
    }
    let m = factors.len();
    let sign: Complex<T> = if inverse { Complex::i() } else { -Complex::i() };
    for i in 1..h {
        let j = N - i;
        let part1 = data[i] + data[j].conj();
        let part2 = data[i] - data[j].conj();
        let product = sign * factors[m - j] * part2;
        data[i] = (part1 + product).scale(half);
        data[j] = (part1 - product).scale(half).conj();
    }
    data[h] = data[h].conj();
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;
    use num_complex::Complex;
    use c64;

    #[test]
    fn unpack() {
        let data = core::array::from_fn::<f64, 4, _>(|i| (i + 1) as f64);
        let mut result = MaybeUninit::uninit();
        assert_eq!(super::unpack(&data, &mut result), &[
            c64::new(1.0, 0.0),
            c64::new(3.0, 4.0),
            c64::new(2.0, 0.0),
            c64::new(3.0, -4.0),
        ]);

        let data = core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64);
        let mut result = MaybeUninit::uninit();
        assert_eq!(super::unpack(&data, &mut result), &[
            c64::new(1.0, 0.0),
            c64::new(3.0, 4.0),
            c64::new(5.0, 6.0),
            c64::new(7.0, 8.0),
            c64::new(2.0, 0.0),
            c64::new(7.0, -8.0),
            c64::new(5.0, -6.0),
            c64::new(3.0, -4.0),
        ]);
    }
}
