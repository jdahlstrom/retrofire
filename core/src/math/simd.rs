use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::math::vec::{Affine, Linear};

pub type Vector<E, const DIM: usize, Space = ()> =
    super::vec::Vector<Simd<E, DIM>, Space>;

pub type Vec2 = Vector<f32, 2>;
pub type Vec3 = Vector<f32, 3>;
pub type Vec4 = Vector<f32, 4>;

pub type Vec2i = Vector<i32, 2>;
pub type Vec3i = Vector<i32, 3>;
pub type Vec4i = Vector<i32, 4>;

impl<const N: usize> Affine for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Space = ();
    type Scalar = f32;
    type Diff = Self;

    const DIM: usize = N;

    fn add(&self, other: &Self::Diff) -> Self {
        *self + *other
    }

    fn mul(&self, scalar: Self::Scalar) -> Self {
        *self * Self::splat(scalar)
    }

    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }
}

impl<const DIM: usize> Linear for Simd<f32, DIM>
where
    LaneCount<DIM>: SupportedLaneCount,
{
    fn zero() -> Self {
        Self::splat(0.0)
    }

    fn neg(&self) -> Self {
        -*self
    }
}

impl<const DIM: usize, Space> Affine for Vector<f32, DIM, Space>
where
    LaneCount<DIM>: SupportedLaneCount,
{
    type Space = Space;
    type Scalar = f32;
    type Diff = Self;

    const DIM: usize = DIM;

    fn add(&self, other: &Self) -> Self {
        self.0.add(&other.0).into()
    }
    fn mul(&self, scalar: Self::Scalar) -> Self {
        self.0.mul(scalar).into()
    }
    fn sub(&self, other: &Self) -> Self::Diff {
        self.0.sub(&other.0).into()
    }
}
impl<const DIM: usize, Space> Linear for Vector<f32, DIM, Space>
where
    Simd<f32, DIM>: Linear,
    LaneCount<DIM>: SupportedLaneCount,
{
    fn zero() -> Self {
        Simd::splat(0.0).into()
    }

    fn neg(&self) -> Self {
        self.0.neg().into()
    }
}
