//! Real and projective vectors.
//!
//! TODO

use core::{
    array,
    fmt::{Debug, Formatter},
    iter::Sum,
    marker::PhantomData as Pd,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use crate::math::{
    Affine, ApproxEq, Linear, Point,
    float::f32,
    space::{Proj4, Real},
    vary::ZDiv,
};

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
/// TODO examples
#[repr(transparent)]
pub struct Vector<Repr, Space = ()>(pub Repr, Pd<Space>);

/// A 2-vector with `f32` components.
pub type Vec2<Basis = ()> = Vector<[f32; 2], Real<2, Basis>>;
/// A 3-vector with `f32` components.
pub type Vec3<Basis = ()> = Vector<[f32; 3], Real<3, Basis>>;
/// A `f32` 4-vector in the projective 3-space over ℝ, aka P<sub>3</sub>(ℝ).
pub type ProjVec4 = Vector<[f32; 4], Proj4>;

/// A 2-vector with `i32` components.
pub type Vec2i<Basis = ()> = Vector<[i32; 2], Real<2, Basis>>;
/// A 3-vector with `i32` components.
pub type Vec3i<Basis = ()> = Vector<[i32; 3], Real<3, Basis>>;

//
// Free functions
//

/// Returns a real 2-vector with components `x` and `y`.
pub const fn vec2<Sc, B>(x: Sc, y: Sc) -> Vector<[Sc; 2], Real<2, B>> {
    Vector([x, y], Pd)
}

/// Returns a real 3-vector with components `x`, `y`, and `z`.
pub const fn vec3<Sc, B>(x: Sc, y: Sc, z: Sc) -> Vector<[Sc; 3], Real<3, B>> {
    Vector([x, y, z], Pd)
}

/// Returns a vector with all components equal to a scalar.
///
/// This operation is also called "broadcast".
///
/// # Examples
/// ```
/// use retrofire_core::math::{vec3, Vec3, splat};
/// let v: Vec3 = splat(1.23);
/// assert_eq!(v, vec3(1.23, 1.23, 1.23));
#[inline]
pub fn splat<Sp, Sc: Clone, const DIM: usize>(s: Sc) -> Vector<[Sc; DIM], Sp> {
    array::from_fn(|_| s.clone()).into()
}

//
// Inherent impls
//

impl<R, Sp> Vector<R, Sp> {
    /// Returns a new vector with representation `repr`.
    #[inline]
    pub const fn new(repr: R) -> Self {
        Self(repr, Pd)
    }

    /// Returns a vector with value equal to `self` but in space `S`.
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

    /// Returns the affine point equivalent to `self`.
    // TODO Cannot be const (yet?) due to E0493 :(
    #[inline]
    pub fn to_pt(self) -> Point<R, Sp> {
        Point::new(self.0)
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
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::{vec2, Vec2};
    ///
    /// let normalized: Vec2 = vec2(3.0, 4.0).normalize();
    /// assert_approx_eq!(normalized, vec2(0.6, 0.8), eps=1e-2);
    /// assert_approx_eq!(normalized.len_sqr(), 1.0, eps=1e-2);
    /// ```
    ///
    /// # Panics
    /// Panics in dev mode if `self` is a zero vector.
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Self
    where
        Self: Debug,
    {
        #[cfg(feature = "std")]
        use super::float::RecipSqrt;
        use super::float::f32;
        let len_sqr = self.len_sqr();
        assert!(
            len_sqr.is_finite() && !len_sqr.approx_eq_eps(&0.0, &1e-12),
            "cannot normalize a near-zero or non-finite vector: {self:?}"
        );
        *self * f32::recip_sqrt(len_sqr)
    }

    /// Returns `self` clamped component-wise to the given range.
    ///
    /// In other words, for each component `self[i]`, the result `r` has
    /// `r[i]` equal to `self[i].clamp(min[i], max[i])`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::vec::{vec3, Vec3, splat};
    /// let v: Vec3 = vec3(0.5, 1.5, -2.0);
    ///
    /// // Clamp to the unit cube
    /// let clamped = v.clamp(&splat(-1.0), &splat(1.0));
    /// assert_eq!(clamped, vec3(0.5, 1.0, -1.0));
    // TODO f32 and f64 have inherent clamp methods because they're not Ord.
    //      A generic clamp for Sc: Ord would conflict with this one. There is
    //      currently no clean way to support both floats and impl Ord types.
    //      However, VecXi and VecXu should have their own inherent impls.
    #[must_use]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        array::from_fn(|i| self[i].clamp(min[i], max[i])).into()
    }

    /// Returns `true` if every component of `self` is finite,
    /// `false` otherwise.
    ///
    /// See [`f32::is_finite()`].
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|c| c.is_finite())
    }
}

impl<Sc, Sp, const N: usize> Vector<[Sc; N], Sp>
where
    Self: Linear<Scalar = Sc>,
    Sc: Linear<Scalar = Sc> + Copy,
{
    /// Returns the length of `self`, squared.
    ///
    /// This avoids taking the square root in cases it's not needed and works with scalars for
    /// which a square root is not defined.
    #[inline]
    pub fn len_sqr(&self) -> Sc {
        self.dot(self)
    }

    /// Returns the dot product of `self` and `other`.
    #[inline]
    pub fn dot(&self, other: &Self) -> Sc {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(a, b)| a.mul(*b))
            .fold(Sc::zero(), |acc, x| acc.add(&x))
    }

    /// Returns the scalar projection of `self` onto `other`
    /// (the length of the component of `self` parallel to `other`).
    #[must_use]
    pub fn scalar_project(&self, other: &Self) -> Sc
    where
        Sc: Div<Sc, Output = Sc>,
    {
        self.dot(other) / other.dot(other)
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
    pub fn vector_project(&self, other: &Self) -> Self
    where
        Sc: Div<Sc, Output = Sc>,
    {
        other.mul(self.scalar_project(other))
    }
}

impl<Sc: Copy, Sp, const N: usize> Vector<[Sc; N], Sp> {
    /// Returns a vector of the same dimension as `self` by applying `f`
    /// component-wise.
    #[inline]
    #[must_use]
    pub fn map<T>(self, mut f: impl FnMut(Sc) -> T) -> Vector<[T; N], Sp> {
        array::from_fn(|i| f(self.0[i])).into()
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

// TODO Make more general - requires a "Scalar::one()" method
impl<B> Vec2<B> {
    /// Unit vector codirectional with the x-axis.
    pub const X: Self = vec2(1.0, 0.0);

    /// Unit vector codirectional with the y-axis.
    pub const Y: Self = vec2(0.0, 1.0);

    /// Converts `self` into a `Vec3`, with z equal to 0.
    pub fn to_vec3(self) -> Vec3<B> {
        vec3(self.x(), self.y(), 0.0)
    }

    /// Returns `self` rotated 90° counter-clockwise.
    pub fn perp(self) -> Self {
        vec2(-self.y(), self.x())
    }

    /// Returns the "perpendicular dot product" of `self` and `other`.
    ///
    /// This operation is also called a "2D cross product". Like its 3D analog,
    /// it satisfies the identity
    ///
    /// **a**<sup>⟂</sup> · **b** = |**a**| |**b**| sin *θ*,
    ///
    /// where θ is the (signed) angle between **a** and **b**.
    pub fn perp_dot(self, other: Self) -> f32 {
        self.perp().dot(&other)
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
    /// The result is a vector orthogonal with both input vectors, its length
    /// proportional to the area of the parallelogram formed by the vectors.
    /// Specifically, the length is given by the identity:
    ///
    /// ```text
    ///     |𝗮 × 𝗯| = |𝗮| |𝗯| sin 𝜽
    /// ```
    ///
    /// where |·| denotes the length of a vector and 𝜽 equals the angle
    /// between 𝗮 and 𝗯. Specifically, the result has unit length if 𝗮 and 𝗯
    /// are orthogonal and |𝗮| = |𝗯| = 1. The cross product can be used to
    /// produce an *orthonormal basis* from any two non-parallel non-zero
    /// 3-vectors.
    ///
    /// ```text
    ///        ^
    ///     r  |
    ///     e  |
    ///     s  |    other
    ///     u  |     ^ - - - - - - - +
    ///     l  |   /               /
    ///     t  | /               /
    ///        +--------------- > self
    /// ```
    pub fn cross(&self, other: &Self) -> Self
    where
        Sc: Linear<Scalar = Sc>,
        [Sc; 3]: Into<Self>,
    {
        let (s, o) = (self, other);
        [
            s.y().mul(o.z()).sub(&s.z().mul(o.y())),
            s.z().mul(o.x()).sub(&s.x().mul(o.z())),
            s.x().mul(o.y()).sub(&s.y().mul(o.x())),
        ]
        .into()
    }
}

// TODO Make more general - requires a "Scalar::one()" method
impl<B> Vec3<B> {
    /// Unit vector codirectional with the x-axis.
    pub const X: Self = vec3(1.0, 0.0, 0.0);

    /// Unit vector codirectional with the y-axis.
    pub const Y: Self = vec3(0.0, 1.0, 0.0);

    /// Unit vector codirectional with the z-axis.
    pub const Z: Self = vec3(0.0, 0.0, 1.0);
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
}

//
// Local trait impls
//

impl<Sc, Sp, const DIM: usize> Affine for Vector<[Sc; DIM], Sp>
where
    Sc: Linear<Scalar = Sc> + Copy,
{
    type Space = Sp;
    type Diff = Self;

    /// The dimension (number of components) of `Self`.
    const DIM: usize = DIM;

    #[inline]
    fn add(&self, other: &Self) -> Self {
        // TODO Profile performance of array::from_fn
        Self(array::from_fn(|i| self.0[i].add(&other.0[i])), Pd)
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self {
        Self(array::from_fn(|i| self.0[i].sub(&other.0[i])), Pd)
    }
}

impl<Sc, Sp, const DIM: usize> Linear for Vector<[Sc; DIM], Sp>
where
    Sc: Linear<Scalar = Sc> + Copy,
{
    type Scalar = Sc;

    /// Returns a vector with all-zero components, also called a zero vector.
    #[inline]
    fn zero() -> Self {
        Self([Sc::zero(); DIM], Pd)
    }
    #[inline]
    fn mul(&self, scalar: Sc) -> Self {
        self.map(|c| c.mul(scalar))
    }
}

impl<Sc, Sp, const N: usize> ZDiv for Vector<[Sc; N], Sp>
where
    Sc: ZDiv + Copy,
{
    fn z_div(self, z: f32) -> Self {
        self.map(|c| c.z_div(z))
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

// Manual impls of Copy, Clone, Eq, and PartialEq to avoid
// superfluous where S: Trait bound

impl<R: Copy, S> Copy for Vector<R, S> {}

impl<R: Clone, S> Clone for Vector<R, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), Pd)
    }
}

impl<R: Default, S> Default for Vector<R, S> {
    fn default() -> Self {
        Self(R::default(), Pd)
    }
}

impl<R: Eq, S> Eq for Vector<R, S> {}

impl<R: PartialEq, S> PartialEq for Vector<R, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
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
    fn from(repr: R) -> Self {
        Self(repr, Pd)
    }
}

impl<Sp, Sc: Clone, const DIM: usize> From<Sc> for Vector<[Sc; DIM], Sp> {
    /// Returns a vector with all components equal to `scalar`.
    ///
    /// This operation is also called "splat" or "broadcast".
    #[inline]
    fn from(scalar: Sc) -> Self {
        splat(scalar)
    }
}

impl<R, Sp> Index<usize> for Vector<R, Sp>
where
    Self: Affine,
    R: Index<usize>,
{
    type Output = R::Output;

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
    Self: Affine,
    R: IndexMut<usize>,
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

impl<R, Sp> Sum for Vector<R, Sp>
where
    Self: Linear,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, v| Affine::add(&acc, &v))
    }
}

//
// Arithmetic traits
//

/// Implements an operator trait in terms of an op-assign trait.
macro_rules! impl_op {
    ($trait:ident :: $method:ident, $rhs:ty, $op:tt) => {
        impl_op!($trait::$method, $rhs, $op, bound=Linear);
    };
    ($trait:ident :: $method:ident, $rhs:ty, $op:tt, bound=$bnd:path) => {
        impl<R, Sp> $trait<$rhs> for Vector<R, Sp>
        where
            Self: $bnd,
        {
            type Output = Self;
            /// TODO docs for macro-generated operators
            #[inline]
            fn $method(mut self, rhs: $rhs) -> Self {
                self $op rhs; self
            }
        }
    };
}

/// The vector += vector operator.
impl<R, Sp> AddAssign<<Self as Affine>::Diff> for Vector<R, Sp>
where
    Self: Affine,
{
    #[inline]
    fn add_assign(&mut self, rhs: <Self as Affine>::Diff) {
        *self = Affine::add(&*self, &rhs);
    }
}
// The vector + vector operator.
impl_op!(Add::add, <Self as Affine>::Diff, +=, bound=Affine);

/// The vector -= vector operator.
impl<R, Sp> SubAssign<<Self as Affine>::Diff> for Vector<R, Sp>
where
    Self: Affine,
{
    #[inline]
    fn sub_assign(&mut self, rhs: <Self as Affine>::Diff) {
        *self = Affine::add(&*self, &rhs.neg());
    }
}

// The vector - vector operator.
impl_op!(Sub::sub, <Self as Affine>::Diff, -=, bound=Affine);

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

// The vector /= scalar operator.
impl<R, Sp> DivAssign<f32> for Vector<R, Sp>
where
    Self: Linear<Scalar = f32>,
{
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(f32::abs(rhs) > 1e-7, "divisor {rhs} < epsilon");
        *self = Linear::mul(&*self, rhs.recip());
    }
}

// The vector / scalar operator.
impl_op!(Div::div, f32, /=, bound=Linear<Scalar = f32>);

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

impl<R, Sp> Mul<Vector<R, Sp>> for f32
where
    Vector<R, Sp>: Linear<Scalar = f32>,
{
    type Output = Vector<R, Sp>;

    #[inline]
    fn mul(self, rhs: Vector<R, Sp>) -> Self::Output {
        rhs * self
    }
}
impl<R, Sp> Mul<Vector<R, Sp>> for i32
where
    Vector<R, Sp>: Linear<Scalar = i32>,
{
    type Output = Vector<R, Sp>;

    #[inline]
    fn mul(self, rhs: Vector<R, Sp>) -> Self::Output {
        rhs * self
    }
}
impl<R, Sp> Mul<Vector<R, Sp>> for u32
where
    Vector<R, Sp>: Linear<Scalar = u32>,
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
    use core::f32::consts::*;

    use crate::assert_approx_eq;

    use super::*;

    pub const fn vec2<S>(x: S, y: S) -> Vector<[S; 2], Real<2>> {
        super::vec2(x, y)
    }
    pub const fn vec3<S>(x: S, y: S, z: S) -> Vector<[S; 3], Real<3>> {
        super::vec3(x, y, z)
    }
    pub const fn vec4<S>(x: S, y: S, z: S, w: S) -> Vector<[S; 4], Real<4>> {
        Vector::new([x, y, z, w])
    }

    mod f32 {
        use super::*;

        #[cfg(feature = "fp")]
        #[test]
        fn length() {
            assert_approx_eq!(vec2(1.0, 1.0).len(), SQRT_2);
            assert_approx_eq!(vec2(-3.0, 4.0).len(), 5.0);
            assert_approx_eq!(vec3(1.0, -2.0, 3.0).len(), 14.0f32.sqrt());
        }

        #[test]
        fn length_squared() {
            assert_eq!(vec2(1.0, 1.0).len_sqr(), 2.0);
            assert_eq!(vec2(-4.0, 3.0).len_sqr(), 25.0);
            assert_eq!(vec3(1.0, -2.0, 3.0).len_sqr(), 14.0);
        }

        #[test]
        fn normalize() {
            assert_approx_eq!(vec2(3.0, 4.0).normalize(), vec2(0.6, 0.8));

            let sqrt_14 = 14.0f32.sqrt();
            assert_approx_eq!(
                vec3(1.0, 2.0, 3.0).normalize(),
                vec3(1.0 / sqrt_14, 2.0 / sqrt_14, 3.0 / sqrt_14)
            );
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
            assert_eq!(3.0 * vec3(1.0, -2.0, 3.0), vec3(3.0, -6.0, 9.0));
            assert_eq!(
                vec4(1.0, -2.0, 0.0, -3.0) * 3.0,
                vec4(3.0, -6.0, 0.0, -9.0)
            );
            assert_eq!(
                3.0 * vec4(1.0, -2.0, 0.0, -3.0),
                vec4(3.0, -6.0, 0.0, -9.0)
            );
        }

        #[test]
        fn scalar_division() {
            assert_eq!(vec2(1.0, -2.0) / 1.0, vec2(1.0, -2.0));
            assert_eq!(vec3(3.0, -6.0, 9.0) / 3.0, vec3(1.0, -2.0, 3.0));
            assert_eq!(
                vec4(3.0, -6.0, 0.0, -9.0) / 3.0,
                vec4(1.0, -2.0, 0.0, -3.0)
            );
        }

        #[test]
        fn dot_product() {
            assert_eq!(vec2(1.0, -2.0).dot(&vec2(2.0, 3.0)), -4.0);
            assert_eq!(vec3(1.0, -2.0, 3.0).dot(&vec3(2.0, 3.0, -1.0)), -7.0);
        }

        #[test]
        fn indexing() {
            let mut v = vec2(1.0, 2.0);
            assert_eq!(v[1], 2.0);
            v[0] = 3.0;
            assert_eq!(v.0, [3.0, 2.0]);

            let mut v = vec3(1.0, 2.0, 3.0);
            assert_eq!(v[1], 2.0);
            v[2] = 4.0;
            assert_eq!(v.0, [1.0, 2.0, 4.0]);
        }

        #[test]
        fn from_array() {
            assert_eq!(Vec2::from([1.0, -2.0]), vec2(1.0, -2.0));
            assert_eq!(Vec3::from([1.0, -2.0, 4.0]), vec3(1.0, -2.0, 4.0));
            assert_eq!(
                Vector::from([1.0, -2.0, 4.0, -3.0]),
                vec4(1.0, -2.0, 4.0, -3.0)
            );
        }

        #[test]
        fn perp() {
            assert_eq!(Vec2::<()>::zero().perp(), Vec2::zero());
            assert_eq!(Vec2::<()>::X.perp(), Vec2::Y);
            assert_eq!(vec2(-0.2, -1.5).perp(), vec2(1.5, -0.2));
        }

        #[test]
        fn perp_dot() {
            const X: Vec2 = Vec2::X;
            const Y: Vec2 = Vec2::Y;

            assert_eq!(X.perp_dot(X), 0.0);
            assert_eq!(X.perp_dot(Y), 1.0);
            assert_eq!((2.0 * Y).perp_dot(3.0 * X), -6.0);
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
        fn vector_subtraction() {
            assert_eq!(vec2(1, 2) - vec2(-2, 3), vec2(3, -1));
            assert_eq!(vec3(1, 2, 0) - vec3(-2, 1, 2), vec3(3, 1, -2));
        }

        #[test]
        #[allow(clippy::erasing_op)]
        fn scalar_multiplication() {
            assert_eq!(vec2(1, -2) * 0, vec2(0, 0));

            assert_eq!(vec3(1, -2, 3) * 3, vec3(3, -6, 9));
            assert_eq!(3 * vec3(1, -2, 3), vec3(3, -6, 9));

            assert_eq!(vec4(1, -2, 0, -3) * 3, vec4(3, -6, 0, -9));
            assert_eq!(3 * vec4(1, -2, 0, -3), vec4(3, -6, 0, -9));
        }

        #[test]
        fn dot_product() {
            assert_eq!(vec2(1, -2).dot(&vec2(2, 3)), -4);
            assert_eq!(vec3(1, -2, 3).dot(&vec3(2, 3, -1)), -7);
        }

        #[test]
        fn indexing() {
            let mut v = vec2(1, 2);
            assert_eq!(v[1], 2);
            v[0] = 3;
            assert_eq!(v.0, [3, 2]);

            let mut v = vec3(1, 2, 3);
            assert_eq!(v[1], 2);
            v[2] = 4;
            assert_eq!(v.0, [1, 2, 4]);
        }

        #[test]
        fn from_array() {
            assert_eq!(Vec2i::from([1, -2]), vec2(1, -2));
            assert_eq!(Vec3i::from([1, -2, 3]), vec3(1, -2, 3));
        }
    }

    const X: Vec3 = Vec3::X;
    const Y: Vec3 = Vec3::Y;
    const Z: Vec3 = Vec3::Z;

    #[test]
    fn cross_product_basis_vectors() {
        assert_eq!(X.cross(&Y), Z);
        assert_eq!(Y.cross(&X), -Z);

        assert_eq!(Y.cross(&Z), X);
        assert_eq!(Z.cross(&Y), -X);

        assert_eq!(Z.cross(&X), Y);
        assert_eq!(X.cross(&Z), -Y);
    }

    #[test]
    fn cross_product_parallelogram_area() {
        let a = 3.0 * Y;
        let b = 2.0 * X - Y;
        assert_eq!(a.cross(&b).len_sqr(), 6.0 * 6.0);
    }

    #[test]
    fn iterator_sum() {
        let vs = [vec2(-1.0, 2.0), vec2(0.0, 2.0), vec2(3.0, -1.0)];
        assert_eq!(vs.into_iter().sum::<Vec2>(), vec2(2.0, 3.0));
    }

    #[test]
    fn approx_equal_pass() {
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.01, -9.9), eps = 0.011);
    }
    #[test]
    #[should_panic]
    fn approx_equal_fail() {
        let eps = 2.0 * f32::relative_epsilon();
        assert_approx_eq!(vec2(1.0, -10.0), vec2(1.0 + eps, -10.0 - eps));
    }

    // TODO Tests for projections

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
