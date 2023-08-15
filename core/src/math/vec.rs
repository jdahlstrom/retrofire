//! Vectors and vector spaces.

use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::{Add, Index, Mul, Neg, Sub};

use crate::math::approx::ApproxEq;
use crate::math::vary::{Iter, Vary};

/// A trait for "vector-like" types. This includes floating-point
/// and integer vectors as well as colors.
///
/// TODO More documentation
pub trait VectorLike: Sized {
    /// The space that `Self` is an element of.
    type Space;
    /// The type of the scalar components of `Self`.
    type Scalar: Sized;

    /// The number of components of this vector.
    const DIM: usize;

    /// Returns a vector with all components zeroed.
    fn zero() -> Self;

    /// Adds `other` to `self` component-wise.
    ///
    /// `add` is commutative and associative.
    fn add(&self, other: &Self) -> Self;
    /// Multiplies all components of `self` by `scalar`.
    ///
    /// `mul` is commutative and associative, and distributes over
    /// `add` and `sub`.
    fn mul(&self, scalar: Self::Scalar) -> Self;
    /// Returns the additive inverse of `self`.
    fn neg(&self) -> Self;
    /// Subtracts `other` from `self`. Equivalent to `self.add(&other.neg())`.
    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }
}

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Real<const DIM: usize, Basis = ()>(PhantomData<Basis>);

#[repr(transparent)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Vector<Repr, Space>(pub Repr, PhantomData<Space>);

impl<Scalar, Space, const N: usize> VectorLike for Vector<[Scalar; N], Space>
where
    Scalar: Copy
        + Default
        + Add<Output = Scalar>
        + Mul<Output = Scalar>
        + Neg<Output = Scalar>,
    [Scalar; N]: Default,
{
    type Space = Space;
    type Scalar = Scalar;

    const DIM: usize = N;

    #[inline]
    fn zero() -> Self {
        Self(Default::default(), PhantomData)
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        let mut res = Self::zero();
        for i in 0..N {
            res.0[i] = self.0[i] + other.0[i];
        }
        res
    }

    #[inline]
    fn mul(&self, scalar: Self::Scalar) -> Self {
        let mut res = Self::zero();
        for i in 0..N {
            res.0[i] = self.0[i] * scalar;
        }
        res
    }

    #[inline]
    fn neg(&self) -> Self {
        let mut res = Self::zero();
        for i in 0..N {
            res.0[i] = -self.0[i];
        }
        res
    }
}

impl<V> Vary for V
where
    V: Copy + VectorLike<Scalar = f32>,
{
    type Iter = Iter<Self>;

    #[inline]
    fn vary(self, step: Self, max: Option<u32>) -> Self::Iter {
        Iter::new(self, step, max)
    }

    #[inline]
    fn lerp(self, other: Self, t: f32) -> Self {
        let delta = other.sub(&self);
        self.add(&delta.mul(t))
    }

    /// Adds `delta` to `self`.
    #[inline]
    fn step(self, delta: Self) -> Self {
        self.add(&delta)
    }
}

impl<Scalar: ApproxEq, Space, const N: usize> ApproxEq<Self, Scalar>
    for Vector<[Scalar; N], Space>
{
    fn approx_eq_eps(&self, other: &Self, eps: &Scalar) -> bool {
        self.0.approx_eq_eps(&other.0, eps)
    }
    fn relative_epsilon() -> Scalar {
        Scalar::relative_epsilon()
    }
}

impl<const DIM: usize, Basis> Debug for Real<DIM, Basis>
where
    Basis: Debug + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "R{}<{:?}>", DIM, Basis::default())
    }
}

impl<Scalar: Debug, Space: Debug + Default, const N: usize> Debug
    for Vector<[Scalar; N], Space>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec<{:?}>{:?}", Space::default(), self.0)
    }
}

impl<Repr, Space> From<Repr> for Vector<Repr, Space> {
    #[inline]
    fn from(els: Repr) -> Self {
        Self(els, PhantomData)
    }
}

impl<Scalar, Space, const N: usize> Index<usize>
    for Vector<[Scalar; N], Space>
{
    type Output = Scalar;
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl<Space, const N: usize> Vector<[f32; N], Space>
where
    [f32; N]: Default,
{
    #[cfg(feature = "std")]
    #[inline]
    pub fn len(&self) -> f32 {
        self.dot(self).sqrt()
    }
    #[cfg(feature = "std")]
    #[inline]
    pub fn normalize(&self) -> Self {
        self.mul(self.len().recip())
    }
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut res = 0.0;
        for i in 0..N {
            res += self.0[i] * other.0[i];
        }
        res
    }
    pub fn scalar_project(&self, other: &Self) -> f32 {
        self.dot(other).mul(&other.dot(other).recip())
    }
    pub fn vector_project(&self, other: &Self) -> Self {
        other.mul(self.scalar_project(other))
    }
}

impl<Sc: Copy, Sp> Vec3<Sc, Sp>
where
    Self: VectorLike<Scalar = Sc, Space = Sp>,
{
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
    #[inline]
    pub fn z(&self) -> Sc {
        self.0[2]
    }

    pub fn cross(&self, other: &Self) -> Self
    where
        Sc: Mul<Output = Sc> + Sub<Output = Sc>,
    {
        let x = self.0[1] * other.0[2] - self.0[2] * other.0[1];
        let y = self.0[2] * other.0[0] - self.0[0] * other.0[2];
        let z = self.0[0] * other.0[1] - self.0[1] * other.0[0];
        [x, y, z].into()
    }
}

pub type Vec2<Scalar = f32, Space = Real<2>> = Vector<[Scalar; 2], Space>;
pub type Vec3<Scalar = f32, Space = Real<3>> = Vector<[Scalar; 3], Space>;
pub type Vec4<Scalar = f32, Space = Real<4>> = Vector<[Scalar; 4], Space>;

#[inline]
pub fn vec2<Sc>(x: Sc, y: Sc) -> Vec2<Sc> {
    [x, y].into()
}

#[inline]
pub fn vec3<Sc>(x: Sc, y: Sc, z: Sc) -> Vec3<Sc> {
    [x, y, z].into()
}

#[inline]
pub fn vec4<Sc>(x: Sc, y: Sc, z: Sc, w: Sc) -> Vec4<Sc> {
    [x, y, z, w].into()
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;

    use super::*;

    mod f32 {
        use super::*;

        #[cfg(feature = "std")]
        #[test]
        fn length() {
            assert_eq!(vec2(3.0, 4.0).len(), 5.0);
            assert_eq!(vec2(3.0, -4.0).len(), 5.0);
        }

        #[test]
        fn vector_addition() {
            assert_eq!(vec2(1.0, 2.0).add(&vec2(-2.0, 1.0)), vec2(-1.0, 3.0));
            assert_eq!(
                vec3(1.0, 2.0, 0.0).add(&vec3(-2.0, 1.0, -1.0)),
                vec3(-1.0, 3.0, -1.0)
            );
        }

        #[test]
        fn scalar_multiplication() {
            assert_eq!(vec2(1.0, -2.0).mul(0.0), vec2(0.0, 0.0));
            assert_eq!(vec3(1.0, -2.0, 3.0).mul(3.0), vec3(3.0, -6.0, 9.0));
            assert_eq!(
                vec4(1.0, -2.0, 0.0, -3.0).mul(3.0),
                vec4(3.0, -6.0, 0.0, -9.0)
            );
        }

        #[test]
        fn from_array() {
            assert_eq!(Vec2::from([1.0, -2.0]), vec2(1.0, -2.0));
            assert_eq!(Vec3::from([1.0, -2.0, 4.0]), vec3(1.0, -2.0, 4.0));
            assert_eq!(
                Vec4::from([1.0, -2.0, 4.0, -3.0]),
                vec4(1.0, -2.0, 4.0, -3.0)
            );
        }
    }

    mod i32 {
        use super::*;

        #[test]
        fn scalar_multiplication() {
            assert_eq!(vec2(1, -2).mul(0), vec2(0, 0));
            assert_eq!(vec3(1, -2, 3).mul(3), vec3(3, -6, 9));
            assert_eq!(vec4(1, -2, 0, -3).mul(3), vec4(3, -6, 0, -9));
        }

        #[test]
        fn from_array() {
            assert_eq!(Vec2::from([1, -2]), vec2(1, -2));
            assert_eq!(Vec3::from([1, -2, 3]), vec3(1, -2, 3));
            assert_eq!(Vec4::from([1, -2, 3, -4]), vec4(1, -2, 3, -4));
        }
    }

    #[test]
    fn dot_product() {
        assert_eq!(vec2(0.5, 0.5).dot(&vec2(-2.0, 2.0)), 0.0);
        assert_eq!(vec2(3.0, 1.0).dot(&vec2(3.0, 1.0)), 10.0);
        assert_eq!(vec2(0.5, 0.5).dot(&vec2(-4.0, -4.0)), -4.0);
    }

    #[test]
    fn cross_product() {
        assert_eq!(
            vec3(1.0, 0.0, 0.0).cross(&vec3(0.0, 1.0, 0.0)),
            vec3(0.0, 0.0, 1.0)
        );
        assert_eq!(
            vec3(0.0, 0.0, 1.0).cross(&vec3(0.0, 1.0, 0.0)),
            vec3(-1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn approx_equal_pass() {
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.01, -9.9), eps = 0.011)
    }
    #[test]
    #[should_panic]
    fn approx_equal_fail() {
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.0 + 1e-5, -10.0 - 1e-5))
    }

    #[test]
    fn debug() {
        assert_eq!(
            alloc::format!("{:?}", vec2(1.0, -2.0)),
            "Vec<R2<()>>[1.0, -2.0]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec3(1.0, -2.0, 3.0)),
            "Vec<R3<()>>[1.0, -2.0, 3.0]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec4(1.0, -2.0, 3.0, -4.0)),
            "Vec<R4<()>>[1.0, -2.0, 3.0, -4.0]"
        );
    }
}