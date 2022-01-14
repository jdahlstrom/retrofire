use std::fmt::Debug;
use crate::color::Color;

pub trait PixelFmt {
    type Elem: Clone + Debug;
    const STRIDE: usize;

    fn write(to: &mut [Self::Elem], c: Color);
    fn read(from: &[Self::Elem]) -> Color;
}

#[derive(Debug)]
pub struct Argb32;

impl PixelFmt for Argb32 {
    type Elem = u32;
    const STRIDE: usize = 1;

    fn write(to: &mut [Self::Elem], c: Color) {
        to[0] = c.0
    }

    fn read(from: &[Self::Elem]) -> Color {
        Color(from[0])
    }
}

#[derive(Debug)]
pub struct Identity;


impl PixelFmt for Identity {
    type Elem = Color;
    const STRIDE: usize = 1;

    fn write(to: &mut [Self::Elem], c: Color) {
        to[0] = c;
    }

    fn read(from: &[Self::Elem]) -> Color {
        from[0]
    }
}

#[derive(Debug)]
pub struct Argb4x8;

impl PixelFmt for Argb4x8 {
    type Elem = u8;
    const STRIDE: usize = 4;

    fn write(to: &mut [Self::Elem], c: Color) {
        to.copy_from_slice(&c.to_argb())
    }
    fn read(from: &[Self::Elem]) -> Color {
        Color(u32::from_be_bytes(from.try_into().unwrap()))
    }
}

#[derive(Debug)]
pub struct Bgra4x8;

impl PixelFmt for Bgra4x8 {
    type Elem = u8;
    const STRIDE: usize = 4;

    fn write(to: &mut [Self::Elem], c: Color) {
        let mut elems = c.to_argb();
        elems.reverse();
        to.copy_from_slice(&elems)
    }
    fn read(from: &[Self::Elem]) -> Color {
        Color(u32::from_le_bytes(from.try_into().unwrap()))
    }
}