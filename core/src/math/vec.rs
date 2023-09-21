//! Vectors and vector spaces.

use core::array;
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::{Add, Index, Mul, Neg, Sub};

use crate::math::approx::ApproxEq;
use crate::math::vary::{Iter, Vary};

//
// Traits
//

/// Trait for types representing elements of an affine space.
///
/// An `Affine` type must satisfy the following conditions:
///
/// * Addition is commutative and associative
/// * Multiplication is commutative and associative
///
/// # TODO
///
/// * More documentation
/// * Move to own module
pub trait Affine: Sized {
    /// The type of the space that `Self` is the element of.
    type Space;
    /// The scalar type associated with `Self`
    type Scalar: Sized;
    /// The (signed) difference of two values of `Self`.
    /// `Diff` must have the same dimension as `Self`.
    type Diff: Linear;

    /// The dimension of `Self`.
    const DIM: usize;

    /// Adds `diff` to `self` component-wise.
    ///
    /// `add` is commutative and associative.
    fn add(&self, diff: &Self::Diff) -> Self;

    /// Multiplies all components of `self` by `scalar`.
    ///
    /// `mul` is commutative and associative, and distributes over
    /// `add` and `sub` (up to rounding errors):
    /// ```
    /// # use retrofire_core::math::vec::Affine;
    /// # let (v, w, x, a) = (1.0, 2.0, 3.0, 4.0);
    /// v.mul(w) == w.mul(v);
    /// v.mul(w).mul(x) == v.mul(w.mul(x));
    /// v.mul(a).add(&w.mul(a)) == v.add(&w).mul(a);
    /// v.mul(a).sub(&w.mul(a)) == v.add(&w).sub(&a);
    /// ```
    fn mul(&self, scalar: Self::Scalar) -> Self;

    /// Subtracts `other` from `self`, returning the (signed) difference.
    ///
    /// `sub` is anti-commutative: `v.sub(w) == w.sub(v).neg()`.
    fn sub(&self, other: &Self) -> Self::Diff;

    /// Linearly interpolates between `self` and `other`.
    ///
    /// This method does not panic if `t < 0.0` or `t > 1.0`,
    /// or if `t` is `NaN`, but the return value is unspecified.
    /// Individual implementations may offer stronger guarantees.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vec::Affine;
    /// assert_eq!(2.0.lerp(&5.0, 0.0), 2.0);
    /// assert_eq!(2.0.lerp(&5.0, 0.5), 3.5);
    /// assert_eq!(2.0.lerp(&5.0, 1.0), 5.0);
    ///
    /// ```
    #[inline]
    fn lerp(&self, other: &Self, t: <Self::Diff as Affine>::Scalar) -> Self {
        let diff = other.sub(self);
        self.add(&diff.mul(t))
    }
}

/// Trait for types representing elements of a linear space (vector space).
///
/// A `Linear` type is a type that is `Affine` and
/// additionally satisfies the following conditions:
///
/// * The difference type [`Diff`][Affine::Diff] is equal to `Self`
/// * The type has an additive identity, returned by the [`zero`][Self::zero] method
/// * Every value has an additive inverse, returned by the [`neg`][Self::neg] method
///
/// TODO More documentation
/// TODO move to own module
pub trait Linear: Affine<Diff = Self> {
    /// Returns the additive identity of this type.
    fn zero() -> Self;

    /// Returns the additive inverse of `self`.
    fn neg(&self) -> Self;

    /// Subtracts `other` from `self`. Equivalent to `self.add(&other.neg())`.
    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }
}

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Real<const DIM: usize, Basis = ()>(PhantomData<Basis>);

/// A generic vector type. Represents an element of a vector space or a module.
///
/// # Type parameters
/// * `Repr`: Representation of the scalar components of the vector,
/// for example an array or a SIMD vector.
/// * `Space`: The space that the vector is an element of. A tag type used to
/// prevent mixing up vectors of different spaces and bases.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Vector<Repr, Space>(pub Repr, PhantomData<Space>);

impl<Scalar, Space, const DIM: usize> Affine for Vector<[Scalar; DIM], Space>
where
    Scalar: Copy
        + Default
        + Add<Output = Scalar>
        + Sub<Output = Scalar>
        + Mul<Output = Scalar>
        + Neg<Output = Scalar>,
    [Scalar; DIM]: Default,
{
    type Space = Space;
    type Scalar = Scalar;
    type Diff = Self;

    /// The dimension (number of components) of `Self`.
    const DIM: usize = DIM;

    #[inline]
    fn add(&self, other: &Self) -> Self {
        array::from_fn(|i| self.0[i] + other.0[i]).into()
    }

    #[inline]
    fn mul(&self, scalar: Self::Scalar) -> Self {
        array::from_fn(|i| self.0[i] * scalar).into()
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self {
        array::from_fn(|i| self.0[i] - other.0[i]).into()
    }
}

impl<Scalar, Space, const DIM: usize> Linear for Vector<[Scalar; DIM], Space>
where
    Self: Affine<Diff = Self>,
    Scalar: Copy + Default + Neg<Output = Scalar>,
{
    /// Returns the zero vector.
    #[inline]
    fn zero() -> Self {
        array::from_fn(|_| Scalar::default()).into()
    }

    /// Returns the (additive) inverse of `self`.
    #[inline]
    fn neg(&self) -> Self {
        array::from_fn(|i| -self.0[i]).into()
    }
}

/// TODO move to own module
impl Affine for f32 {
    type Space = ();
    type Scalar = f32;
    type Diff = f32;
    const DIM: usize = 1;

    fn add(&self, other: &f32) -> f32 {
        self + other
    }
    fn mul(&self, scalar: f32) -> f32 {
        self * scalar
    }
    fn sub(&self, other: &f32) -> f32 {
        self - other
    }
}
/// TODO move to own module
impl Linear for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn neg(&self) -> f32 {
        -*self
    }
}
/// TODO move to own module
impl<V> Vary for V
where
    Self: Affine<Scalar = f32> + Clone,
    <Self as Affine>::Diff: Clone,
{
    type Iter = Iter<Self>;
    type Diff = <Self as Affine>::Diff;

    #[inline]
    fn vary(&self, step: &Self::Diff, max: Option<u32>) -> Self::Iter {
        Iter::new(self.clone(), step.clone(), max)
    }

    /// Adds `delta` to `self`.
    #[inline]
    fn step(&self, delta: &Self::Diff) -> Self {
        self.add(delta)
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
        let res: [f32; N] = array::from_fn(|i| self.0[i] * other.0[i]);
        res.iter().sum()
    }
    pub fn scalar_project(&self, other: &Self) -> f32 {
        self.dot(other).mul(&other.dot(other).recip())
    }
    pub fn vector_project(&self, other: &Self) -> Self {
        other.mul(self.scalar_project(other))
    }
}

impl<Sc: Copy, Sp> Vec3<Sc, Sp> {
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
