//! Vectors and vector spaces.

use core::array;
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use crate::math::approx::ApproxEq;
use crate::math::vary::{Iter, Vary};

//
// Traits
//

/// Trait for types representing elements of an affine space.

/// # TODO
/// * More documentation
/// * Move to own module
pub trait Affine: Sized {
    /// The type of the space that `Self` is the element of.
    type Space;
    /// The (signed) difference of two values of `Self`.
    /// `Diff` must have the same dimension as `Self`.
    type Diff: Linear;

    /// The dimension of `Self`.
    const DIM: usize;

    /// Adds `diff` to `self` component-wise.
    ///
    /// `add` is commutative and associative.
    fn add(&self, diff: &Self::Diff) -> Self;

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
    fn lerp(&self, other: &Self, t: <Self::Diff as Linear>::Scalar) -> Self {
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
/// # TODO
/// * More documentation
/// * Move to own module?
pub trait Linear: Affine<Diff = Self> {
    /// The scalar type associated with `Self`
    type Scalar: Sized;

    /// Returns the additive identity of `Self`.
    fn zero() -> Self;

    /// Returns the additive inverse of `self`.
    fn neg(&self) -> Self;

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
}

//
// Types
//

/// A generic vector type. Represents an element of a vector space or a module,
/// a generalization of a vector space where the scalars can be integers
/// (technically, the scalar type can be any *ring*-like type).
///
/// # Type parameters
/// * `Repr`: Representation of the scalar components of the vector,
/// for example an array or a SIMD vector.
/// * `Space`: The space that the vector is an element of. A tag type used to
/// prevent mixing up vectors of different spaces and bases.
///
/// # Examples
/// TODO
#[repr(transparent)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Vector<Repr, Space = ()>(pub Repr, PhantomData<Space>);

/// Tag type for real vector spaces (Euclidean spaces) of dimension `DIM`.
/// For example, the type `Real<3>` corresponds to ℝ³.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Real<const DIM: usize, Basis = ()>(PhantomData<Basis>);

/// Tag type for the projective 4-space over reals, 𝗣<sub>4</sub>(ℝ).
/// The properties of this space make it useful for implementing perspective
/// projection. Clipping is also done in the projective space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Proj4;

/// A 2D float vector in `Space` (by default ℝ²).
pub type Vec2<Space = Real<2>> = Vector<[f32; 2], Space>;
/// A 3D float vector in `Space` (by default ℝ³).
pub type Vec3<Space = Real<3>> = Vector<[f32; 3], Space>;
/// A 4D float vector in `Space` (by default ℝ⁴).
pub type Vec4<Space = Real<4>> = Vector<[f32; 4], Space>;

/// A 2D integer vector in `Space` (by default ℤ²).
pub type Vec2i<Space = Real<2>> = Vector<[i32; 2], Space>;
/// A 3D integer vector in `Space` (by default ℤ³).
pub type Vec3i<Space = Real<3>> = Vector<[i32; 3], Space>;
/// A 4D integer vector in `Space` (by default ℤ⁴).
pub type Vec4i<Space = Real<4>> = Vector<[i32; 4], Space>;

impl<R, Sp> Vector<R, Sp> {
    /// Returns a new vector with representation `repr`.
    pub const fn new(repr: R) -> Self {
        Self(repr, PhantomData)
    }

    /// Returns a vector with value equal to `self` but in space `Sp`.
    ///
    /// This method can be used to coerce a vector from one space
    /// to another in order to make types match. One use case is
    /// to cast a "generic" vector returned by one of the constructor
    /// functions to a more specific space.
    // Cannot be const due to E0493 :(
    pub fn to<S>(self) -> Vector<R, S> {
        Vector::new(self.0)
    }
}

impl<Sp, const N: usize> Vector<[f32; N], Sp> {
    /// Returns the length (magnitude) of `self`.
    #[cfg(feature = "std")]
    #[inline]
    pub fn len(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns `self` normalized to unit length.
    #[cfg(feature = "std")]
    #[inline]
    pub fn normalize(&self) -> Self {
        self.mul(self.len().recip())
    }

    /// Returns the dot product of `self` and `other`.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        let res: [f32; N] = array::from_fn(|i| self.0[i] * other.0[i]);
        res.iter().sum()
    }

    /// Returns the scalar projection of `self` onto `other`
    /// (the length of the component of `self` parallel to `other`).
    pub fn scalar_project(&self, other: &Self) -> f32 {
        self.dot(other).mul(&other.dot(other).recip())
    }
    /// Returns the vector projection of `self` onto `other`
    /// (the vector component of `self` parallel to `other`).
    /// ```text
    ///            self
    ///            ^
    ///           /.
    ///         /  .
    ///       /    .
    ///     /      .
    ///   /       _.
    ///  +-------'->-----> other
    ///         result
    /// ```
    pub fn vector_project(&self, other: &Self) -> Self {
        other.mul(self.scalar_project(other))
    }
}

impl<R, Sc, B> Vector<R, Real<2, B>>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the x component of `self`.
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
}

impl<R, Sc, B> Vector<R, Real<3, B>>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the x component of `self`.
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
    /// Returns the z component of `self`.
    #[inline]
    pub fn z(&self) -> Sc {
        self.0[2]
    }
}

impl<Sc, B> Vector<[Sc; 3], Real<3, B>>
where
    Sc: Copy + Mul<Output = Sc> + Sub<Output = Sc>,
{
    /// Returns the cross product of `self` with `other`.
    ///
    /// The result is a vector perpendicular to both input
    /// vectors, with length proportional to the area of
    /// the parallelogram having the vectors as its sides.
    /// Specifically, the length is given by the identity
    ///
    /// |𝗮 × 𝗯| = |𝗮| |𝗯| sin 𝜽
    ///
    /// where |·| denotes the length of a vector and
    /// 𝜽 equals the angle between 𝗮 and 𝗯.
    ///
    /// ```text
    ///        ^
    ///     r  |
    ///     e  |
    ///     s  |    other
    ///     u  |     ^-----------+
    ///     l  |   /           /
    ///     t  | /           /
    ///        +-----------> self
    /// ```
    pub fn cross(&self, other: &Self) -> Self {
        let x = self.0[1] * other.0[2] - self.0[2] * other.0[1];
        let y = self.0[2] * other.0[0] - self.0[0] * other.0[2];
        let z = self.0[0] * other.0[1] - self.0[1] * other.0[0];
        [x, y, z].into()
    }
}

impl<R, Sc> Vector<R, Proj4>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the x component of `self`.
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
    /// Returns the z component of `self`.
    #[inline]
    pub fn z(&self) -> Sc {
        self.0[2]
    }
    /// Returns the w component of `self`.
    #[inline]
    pub fn w(&self) -> Sc {
        self.0[3]
    }

    #[inline]
    pub fn project_to_real(&self) -> Vector<[Sc; 3], Real<3>>
    where
        Sc: Div<Output = Sc>,
    {
        array::from_fn(|i| self.0[i] / self.0[3]).into()
    }
}

//
// Local trait impls
//

impl<Sc, Sp, const DIM: usize> Affine for Vector<[Sc; DIM], Sp>
where
    Sc: Copy
        + Default
        + Add<Output = Sc>
        + Sub<Output = Sc>
        + Neg<Output = Sc>
        + Mul<Output = Sc>,
{
    type Space = Sp;
    type Diff = Self;

    /// The dimension (number of components) of `Self`.
    const DIM: usize = DIM;

    #[inline]
    fn add(&self, other: &Self) -> Self {
        array::from_fn(|i| self.0[i] + other.0[i]).into()
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self {
        array::from_fn(|i| self.0[i] - other.0[i]).into()
    }
}

impl<Sc, Sp, const DIM: usize> Linear for Vector<[Sc; DIM], Sp>
where
    Self: Affine<Diff = Self>,
    Sc: Copy + Default + Neg<Output = Sc> + Mul<Output = Sc>,
{
    type Scalar = Sc;

    /// Returns the zero vector.
    #[inline]
    fn zero() -> Self {
        array::from_fn(|_| Sc::default()).into()
    }
    #[inline]
    fn neg(&self) -> Self {
        array::from_fn(|i| -self.0[i]).into()
    }
    #[inline]
    fn mul(&self, scalar: Self::Scalar) -> Self {
        array::from_fn(|i| self.0[i] * scalar).into()
    }
}

/// TODO move to own module
impl Affine for f32 {
    type Space = ();
    type Diff = f32;
    const DIM: usize = 1;

    fn add(&self, other: &f32) -> f32 {
        self + other
    }
    fn sub(&self, other: &f32) -> f32 {
        self - other
    }
}
/// TODO move to own module
impl Linear for f32 {
    type Scalar = f32;

    fn zero() -> f32 {
        0.0
    }
    fn neg(&self) -> f32 {
        -*self
    }
    fn mul(&self, scalar: f32) -> f32 {
        self * scalar
    }
}
/// TODO move to own module
impl<V> Vary for V
where
    Self: Affine,
    <Self as Affine>::Diff: Linear<Scalar = f32>,
{
    type Iter = Iter<Self>;
    type Diff = <Self as Affine>::Diff;

    #[inline]
    fn vary(self, step: Self::Diff, max: Option<u32>) -> Self::Iter {
        Iter::new(self, step, max)
    }

    /// Adds `delta` to `self`.
    #[inline]
    fn step(&self, delta: &Self::Diff) -> Self {
        self.add(delta)
    }
}

impl<Sc: ApproxEq, Sp, const N: usize> ApproxEq<Self, Sc>
    for Vector<[Sc; N], Sp>
{
    fn approx_eq_eps(&self, other: &Self, eps: &Sc) -> bool {
        self.0.approx_eq_eps(&other.0, eps)
    }
    fn relative_epsilon() -> Sc {
        Sc::relative_epsilon()
    }
}

//
// Foreign trait impls
//

impl<const DIM: usize, B> Debug for Real<DIM, B>
where
    B: Debug + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        const DIMS: [&str; 5] = ["", "", "²", "³", "⁴"];
        let dim = DIMS.get(DIM).unwrap_or(&"?");
        write!(f, "ℝ{}<{:?}>", dim, B::default())
    }
}

impl<R: Debug, Sp: Debug + Default> Debug for Vector<R, Sp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec<{:?}>{:.3?}", Sp::default(), self.0)
    }
}

impl<R, Sp> From<R> for Vector<R, Sp> {
    #[inline]
    fn from(els: R) -> Self {
        Self(els, PhantomData)
    }
}

impl<Sp, Sc: Clone, const DIM: usize> From<Sc> for Vector<[Sc; DIM], Sp> {
    /// Returns a vector with all components equal to `scalar`.
    /// This operation is sometimes called "splat".
    #[inline]
    fn from(scalar: Sc) -> Self {
        array::from_fn(|_| scalar.clone()).into()
    }
}

impl<R, Sp> Index<usize> for Vector<R, Sp>
where
    Self: Linear,
    R: Index<usize, Output = <Self as Linear>::Scalar>,
{
    type Output = <Self as Linear>::Scalar;

    /// Returns the component of `self` with index `i`.
    ///
    /// # Panics
    /// If `i >= Self::DIM`.
    /// Note that `Self::DIM` may not be equal to the the length of `R`.
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < Self::DIM, "index {i} out of bounds ({})", Self::DIM);
        &self.0[i]
    }
}

impl<R, Sp> IndexMut<usize> for Vector<R, Sp>
where
    Self: Linear,
    R: IndexMut<usize, Output = <Self as Linear>::Scalar>,
{
    /// Returns a mutable reference to the component of `self` with index `i`.
    ///
    /// # Panics
    /// If `i >= Self::DIM`.
    /// Note that `Self::DIM` may not be equal to the length of `R`.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < Self::DIM, "index {i} out of bounds ({})", Self::DIM);
        &mut self.0[i]
    }
}

//
// Free functions
//

pub const fn vec2<Sc>(x: Sc, y: Sc) -> Vector<[Sc; 2], Real<2>> {
    Vector([x, y], PhantomData)
}

/// Returns a 3D Euclidean vector with components `x`, `y`, and `z`.
pub const fn vec3<Sc>(x: Sc, y: Sc, z: Sc) -> Vector<[Sc; 3], Real<3>> {
    Vector([x, y, z], PhantomData)
}
/// Returns a 4D Euclidean vector with components `x`, `y`, `z`, and `w`.
#[inline]
pub const fn vec4<Sc>(x: Sc, y: Sc, z: Sc, w: Sc) -> Vector<[Sc; 4], Real<4>> {
    Vector([x, y, z, w], PhantomData)
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
        fn vector_addition() {
            assert_eq!(vec2(1, 2).add(&vec2(-2, 1)), vec2(-1, 3));
            assert_eq!(vec3(1, 2, 0).add(&vec3(-2, 1, -1)), vec3(-1, 3, -1));
        }

        #[test]
        fn scalar_multiplication() {
            assert_eq!(vec2(1, -2).mul(0), vec2(0, 0));
            assert_eq!(vec3(1, -2, 3).mul(3), vec3(3, -6, 9));
            assert_eq!(vec4(1, -2, 0, -3).mul(3), vec4(3, -6, 0, -9));
        }

        #[test]
        fn from_array() {
            assert_eq!(Vec2i::from([1, -2]), vec2(1, -2));
            assert_eq!(Vec3i::from([1, -2, 3]), vec3(1, -2, 3));
            assert_eq!(Vec4i::from([1, -2, 3, -4]), vec4(1, -2, 3, -4));
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

    // TODO Tests for length, normalize, projections, Affine/Linear impls...

    #[test]
    fn debug() {
        assert_eq!(
            alloc::format!("{:?}", vec2(1.0, -2.0)),
            "Vec<ℝ²<()>>[1.000, -2.000]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec3(1.0, -2.0, 3.0)),
            "Vec<ℝ³<()>>[1.000, -2.000, 3.000]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec4(1.0, -2.0, 3.0, -4.0)),
            "Vec<ℝ⁴<()>>[1.000, -2.000, 3.000, -4.000]"
        );
    }
}
