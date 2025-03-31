#![no_std]
#![feature(generic_const_exprs)]

extern crate assert;
extern crate dft;
extern crate num_complex;

use core::mem::MaybeUninit;
use num_complex::Complex;
use dft::{transform, unpack, Operation, Plan, c64};

mod fixtures;

#[test]
fn complex_forward_1() {
    let mut data = [c64::new(1.0, -2.0)];
    let mut factors = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert_eq!(data, [c64::new(1.0, -2.0)]);
}

#[test]
fn complex_forward_2() {
    let mut data = [c64::new(1.0, -2.0), c64::new(3.0, -4.0)];
    let mut factors = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert_eq!(data, [c64::new(4.0, -6.0), c64::new(-2.0, 2.0)]);
}

#[test]
fn complex_forward_128() {
    let mut data = fixtures::TIME_DATA_256.clone();
    let mut factors = MaybeUninit::uninit();
    transform(as_c64_mut(&mut data), &Plan::new(Operation::Forward, &mut factors));
    assert::close(data.as_ref(), &fixtures::FREQUENCY_DATA_128_COMPLEX[..], 1e-14);
}

#[test]
fn complex_forward_real_256() {
    let mut data = to_c64(&fixtures::TIME_DATA_256);
    let mut factors = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert::close(
        as_f64(&data).as_ref(),
        &fixtures::FREQUENCY_DATA_256_REAL_UNPACKED[..],
        1e-13,
    );
}

#[test]
fn complex_inverse_128() {
    let mut data = fixtures::FREQUENCY_DATA_128_COMPLEX.clone();
    let mut factors = MaybeUninit::uninit();
    transform(as_c64_mut(&mut data), &Plan::new(Operation::Inverse, &mut factors));
    assert::close(data.as_ref(), &fixtures::TIME_DATA_256[..], 1e-14);
}

/*#[test]
fn real_forward_1() {
    let mut data = [1.0];
    let mut factors = MaybeUninit::uninit();
    let mut output = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert_eq!(unpack(&data, &mut output), &[c64::new(1.0, 0.0)]);
}*/

#[test]
fn real_forward_2() {
    let mut data = [1.0, -2.0];
    let mut factors = MaybeUninit::uninit();
    let mut output = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert_eq!(unpack(&data, &mut output), &[c64::new(-1.0, 0.0), c64::new(3.0, 0.0)]);
}

#[test]
fn real_forward_4() {
    let mut data = [1.0, -2.0, 3.0, -4.0];
    let mut factors = MaybeUninit::uninit();
    let mut output = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert_eq!(unpack(&data, &mut output), &[
        c64::new(-2.0, 0.0),
        c64::new(-2.0, -2.0),
        c64::new(10.0, 0.0),
        c64::new(-2.0, 2.0),
    ]);
}

#[test]
fn real_forward_256() {
    let mut data = fixtures::TIME_DATA_256.clone();
    let mut factors = MaybeUninit::uninit();
    let mut output = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    assert::close(data.as_ref(), &fixtures::FREQUENCY_DATA_256_REAL_PACKED[..], 1e-13);
    let data = unpack(&data, &mut output);
    assert::close(
        as_f64(data).as_ref(),
        &fixtures::FREQUENCY_DATA_256_REAL_UNPACKED[..],
        1e-13,
    );
}

#[test]
fn real_forward_512() {
    let mut data = fixtures::TIME_DATA_512.clone();
    let mut factors = MaybeUninit::uninit();
    let mut result = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    let data = unpack(&data, &mut result);
    assert::close(
        as_f64(data).as_ref(),
        &fixtures::FREQUENCY_DATA_512_REAL_UNPACKED[..],
        1e-12,
    );
}

#[test]
fn real_inverse_256() {
    let mut data = fixtures::FREQUENCY_DATA_256_REAL_PACKED.clone();
    let mut factors = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Inverse, &mut factors));
    assert::close(data.as_ref(), &fixtures::TIME_DATA_256[..], 1e-14);
}

#[test]
fn real_inverse_512() {
    let mut data = fixtures::TIME_DATA_512.clone();
    let mut factors = MaybeUninit::uninit();
    transform(&mut data, &Plan::new(Operation::Forward, &mut factors));
    transform(&mut data, &Plan::new(Operation::Inverse, &mut factors));
    assert::close(data.as_ref(), &fixtures::TIME_DATA_512[..], 1e-14);
}

fn as_f64<const N: usize>(slice: &[c64; N]) -> &[f64; {N * 2}] where [(); {N * 2}]: Sized {
    unsafe {
        dft::as_array(core::slice::from_raw_parts(slice.as_ptr() as *const _, N * 2))
    }
}

fn as_c64_mut<const N: usize>(slice: &mut [f64; N]) -> &mut [c64; {N / 2}] {
    unsafe { 
        dft::as_mut_array(core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut _, N / 2))
    }
}

fn to_c64<const N: usize>(slice: &[f64; N]) -> [c64; N] {
    core::array::from_fn(|i| c64::new(slice[i], 0.0))
}
