//!

use core::array;
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};
use core::ops::{AddAssign, MulAssign, SubAssign};

use crate::math::approx::ApproxEq;
use crate::math::space::{Affine, Linear, Proj4, Real};

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

/// A 2D float vector in `Space` (by default ℝ²).
pub type Vec2<Space = Real<2>> = Vector<[f32; 2], Space>;
/// A 3D float vector in `Space` (by default ℝ³).
pub type Vec3<Space = Real<3>> = Vector<[f32; 3], Space>;
/// A 4D float vector in `Space` (by default ℝ⁴).
// TODO Are these 4-dim variants really necessary?
pub type Vec4<Space = Real<4>> = Vector<[f32; 4], Space>;

/// A 2D integer vector in `Space` (by default ℤ²).
pub type Vec2i<Space = Real<2>> = Vector<[i32; 2], Space>;
/// A 3D integer vector in `Space` (by default ℤ³).
pub type Vec3i<Space = Real<3>> = Vector<[i32; 3], Space>;
/// A 4D integer vector in `Space` (by default ℤ⁴).
pub type Vec4i<Space = Real<4>> = Vector<[i32; 4], Space>;

/// A 2D unsigned integer vector in `Space` (by default ℕ²).
pub type Vec2u<Space = Real<2>> = Vector<[u32; 2], Space>;
// Will add Vec3u if needed at some point.

//
// Free functions
//

/// Returns a 2D Euclidean vector with components `x` and `y`.
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

/// Returns a vector with all components equal to the argument.
///
/// This operation is also called "broadcast".
///
/// # Examples
/// ```
/// # use retrofire_core::math::{vec3, Vec3};
/// # use retrofire_core::math::vec::splat;
/// let v: Vec3 = splat(1.23);
/// assert_eq!(v, vec3(1.23, 1.23, 1.23));
#[inline]
pub fn splat<Sp, Sc: Clone, const DIM: usize>(s: Sc) -> Vector<[Sc; DIM], Sp> {
    s.into()
}

//
// Inherent impls
//

impl<R, Sp> Vector<R, Sp> {
    /// Returns a new vector with representation `repr`.
    #[inline]
    pub const fn new(repr: R) -> Self {
        Self(repr, PhantomData)
    }

    /// Returns a vector with value equal to `self` but in space `Sp`.
    ///
    /// This method can be used to coerce a vector from one space
    /// to another in order to make types match. One use case is
    /// to cast a "generic" vector returned by one of the constructor
    /// functions to a more specific space.
    // TODO Cannot be const (yet?) due to E0493 :(
    #[inline]
    pub fn to<S>(self) -> Vector<R, S> {
        Vector::new(self.0)
    }
}

// TODO Many of these functions could be more generic
impl<Sp, const N: usize> Vector<[f32; N], Sp> {
    /// Returns the length (magnitude) of `self`.
    #[cfg(feature = "fp")]
    #[inline]
    pub fn len(&self) -> f32 {
        super::float::f32::sqrt(self.dot(self))
    }

    /// Returns `self` normalized to unit length.
    #[cfg(feature = "fp")]
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Self {
        self.mul(self.len().recip())
    }

    /// Returns the length of `self`, squared.
    ///
    /// This avoids taking the square root in cases it's not needed.
    #[inline]
    pub fn len_sqr(&self) -> f32 {
        self.dot(self)
    }

    /// Returns the dot product of `self` and `other`.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        let res: [f32; N] = array::from_fn(|i| self.0[i] * other.0[i]);
        res.iter().sum()
    }

    /// Returns the scalar projection of `self` onto `other`
    /// (the length of the component of `self` parallel to `other`).
    #[must_use]
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
    #[must_use]
    pub fn vector_project(&self, other: &Self) -> Self {
        other.mul(self.scalar_project(other))
    }

    /// Returns a vector of the same dimension as `self` by applying `f`
    /// component-wise.
    #[inline]
    #[must_use]
    pub fn map<T>(self, mut f: impl FnMut(f32) -> T) -> Vector<[T; N], Sp> {
        array::from_fn(|i| f(self[i])).into()
    }

    /// Returns `self` clamped component-wise to the given range.
    ///
    /// In other words, for each component `self[i]`, the result `r` has
    /// `r[i]` equal to `self[i].clamp(min[i], max[i])`.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vec::{vec3,splat};
    /// let v = vec3(0.5, 1.5, -2.0);
    /// // Clamp to the unit cube
    /// let v = v.clamp(&splat(-1.0), &splat(1.0));
    /// assert_eq!(v, vec3(0.5, 1.0, -1.0));
    #[must_use]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        array::from_fn(|i| self[i].clamp(min[i], max[i])).into()
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

    /// Returns the cross product of `self` with `other`.
    ///
    /// The result is a vector perpendicular to both input
    /// vectors, with length proportional to the area of
    /// the parallelogram having the vectors as its sides.
    /// Specifically, the length is given by the identity:
    ///
    /// ```text
    ///     |𝗮 × 𝗯| = |𝗮| |𝗯| sin 𝜽
    /// ```
    ///
    /// where |·| denotes the length of a vector and
    /// 𝜽 equals the angle between 𝗮 and 𝗯.
    ///
    /// ```text
    ///        ^
    ///     r  |
    ///     e  |
    ///     s  |    other
    ///     u  |     ^ - - - - - +
    ///     l  |   /           /
    ///     t  | /           /
    ///        +-----------> self
    /// ```
    pub fn cross(&self, other: &Self) -> Self
    where
        Sc: Mul<Output = Sc> + Sub<Output = Sc>,
        [Sc; 3]: Into<Self>,
    {
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
    // TODO bundle into Scalar trait?
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
        // TODO Profile performance of array::from_fn
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

    /// Returns a vector with all-zero components, also called a null vector.
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
        if let Some(dim) = DIMS.get(DIM) {
            write!(f, "ℝ{}<{:?}>", dim, B::default())
        } else {
            write!(f, "ℝ^{}<{:?}>", DIM, B::default())
        }
    }
}

impl<R: Debug, Sp: Debug + Default> Debug for Vector<R, Sp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec<{:?}>", Sp::default())?;
        Debug::fmt(&self.0, f)
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
    ///
    /// This operation is also called "splat" or "broadcast".
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
    /// Note that `Self::DIM` can be less than the number of elements in `R`.
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
    /// Note that `Self::DIM` can be less than the number of elements in `R`.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < Self::DIM, "index {i} out of bounds ({})", Self::DIM);
        &mut self.0[i]
    }
}

//
// Arithmetic traits
//

/// Implements an operator trait in terms of an op-assign trait.
macro_rules! impl_op {
    ($trait:ident :: $method:ident, $rhs:ty, $op:tt) => {
        impl<R, Sp> $trait<$rhs> for Vector<R, Sp>
        where
            Self: Linear,
        {
            type Output = Self;
            #[inline]
            fn $method(mut self, rhs: $rhs) -> Self {
                self $op rhs; self
            }
        }
    };
}

/// The vector += vector operator.
impl<R, Sp> AddAssign for Vector<R, Sp>
where
    Self: Linear,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Affine::add(&*self, &rhs);
    }
}
// The vector + vector operator.
impl_op!(Add::add, Self, +=);

/// The vector -= vector operator.
impl<R, Sp> SubAssign for Vector<R, Sp>
where
    Self: Linear,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Affine::sub(&*self, &rhs);
    }
}
// The vector - vector operator.
impl_op!(Sub::sub, Self, -=);

// The vector *= scalar operator.
impl<R, Sp> MulAssign<<Self as Linear>::Scalar> for Vector<R, Sp>
where
    Self: Linear,
{
    #[inline]
    fn mul_assign(&mut self, rhs: <Self as Linear>::Scalar) {
        *self = Linear::mul(&*self, rhs);
    }
}
// The vector * scalar operator.
impl_op!(Mul::mul, <Self as Linear>::Scalar, *=);

/// The vector negation operator.
impl<R, Sp> Neg for Vector<R, Sp>
where
    Self: Linear,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        <Self as Linear>::neg(&self)
    }
}

/// The scalar * vector operator.
impl<R, Sp> Mul<Vector<R, Sp>> for <Vector<R, Sp> as Linear>::Scalar
where
    Vector<R, Sp>: Linear,
{
    type Output = Vector<R, Sp>;

    #[inline]
    fn mul(self, rhs: Vector<R, Sp>) -> Self::Output {
        rhs * self
    }
}

//
// Unit tests
//

#[cfg(test)]
mod tests {
    use core::f32::consts::{E, PI};

    use crate::assert_approx_eq;

    use super::*;

    mod f32 {
        use super::*;

        #[cfg(feature = "fp")]
        #[test]
        fn length() {
            assert_eq!(vec2(3.0, 4.0).len(), 5.0);
            assert_eq!(vec2(3.0, -4.0).len(), 5.0);
        }

        #[test]
        fn vector_addition() {
            assert_eq!(vec2(1.0, 2.0) + vec2(-2.0, 1.0), vec2(-1.0, 3.0));
            assert_eq!(
                vec3(1.0, 2.0, 0.0) + vec3(-2.0, 1.0, -1.0),
                vec3(-1.0, 3.0, -1.0)
            );
        }

        #[test]
        fn scalar_multiplication() {
            assert_eq!(vec2(1.0, -2.0) * 0.0, vec2(0.0, 0.0));
            assert_eq!(vec3(1.0, -2.0, 3.0) * 3.0, vec3(3.0, -6.0, 9.0));
            assert_eq!(
                vec4(1.0, -2.0, 0.0, -3.0) * 3.0,
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
            assert_eq!(vec2(1, 2) + vec2(-2, 1), vec2(-1, 3));
            assert_eq!(vec3(1, 2, 0) + vec3(-2, 1, -1), vec3(-1, 3, -1));
        }

        #[test]
        #[allow(clippy::erasing_op)]
        fn scalar_multiplication() {
            assert_eq!(vec2(1, -2) * 0, vec2(0, 0));
            assert_eq!(vec3(1, -2, 3) * 3, vec3(3, -6, 9));
            assert_eq!(vec4(1, -2, 0, -3) * 3, vec4(3, -6, 0, -9));
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
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.01, -9.9), eps = 0.011);
    }
    #[test]
    #[should_panic]
    fn approx_equal_fail() {
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.0 + 1e-5, -10.0 - 1e-5));
    }

    // TODO Tests for length, normalize, projections, Affine/Linear impls...

    #[test]
    fn debug() {
        assert_eq!(
            alloc::format!("{:?}", vec2(1.0, -E)),
            "Vec<ℝ²<()>>[1.0, -2.7182817]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec3(1.0, -2.0, 3.0)),
            "Vec<ℝ³<()>>[1.0, -2.0, 3.0]"
        );
        assert_eq!(
            alloc::format!("{:?}", vec4(1.0, -2.0, PI, -4.0)),
            "Vec<ℝ⁴<()>>[1.0, -2.0, 3.1415927, -4.0]"
        );
    }
}
