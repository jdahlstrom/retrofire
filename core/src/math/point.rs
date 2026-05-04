use core::{
    array,
    fmt::{Debug, Formatter},
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use super::{Affine, ApproxEq, Linear, Vec3, Vector, vary::ZDiv, vec::Vec4};

#[repr(transparent)]
pub struct Point<Repr>(pub Repr);

/// A 2-point with `f32` components.
pub type Point2 = Point<[f32; 2]>;
/// A 3-point with `f32` components.
pub type Point3 = Point<[f32; 3]>;

/// A 2-point with `u32` components.
pub type Point2u = Point<[u32; 2]>;

/// Returns a real 2-point with `x` and `y` components.
#[inline]
pub const fn pt2<Sc>(x: Sc, y: Sc) -> Point<[Sc; 2]> {
    Point([x, y])
}
/// Returns a real 3-point with `x`, `y`, and `z` components.
#[inline]
pub const fn pt3<Sc>(x: Sc, y: Sc, z: Sc) -> Point<[Sc; 3]> {
    Point([x, y, z])
}

impl<R> Point<R> {
    #[inline]
    pub const fn new(repr: R) -> Self {
        Self(repr)
    }

    /// Returns a point with value equal to `self` but in space `S`.
    #[inline]
    pub const fn to(self) -> Point<R>
    where
        R: Copy, // TODO Needed for now due to E0493
    {
        Point::new(self.0)
    }

    /// Returns the vector equivalent to `self`.
    #[inline]
    pub const fn to_vec(self) -> Vector<R>
    where
        R: Copy, // TODO Needed for now due to E0493
    {
        Vector::new(self.0)
    }
}

impl<Sc: Copy, const N: usize> Point<[Sc; N]> {
    /// Returns a vector of the same dimension as `self` by applying `f`
    /// component-wise.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{pt3};
    ///
    /// let p = pt3::<i32, ()>(1, 2, 3);
    /// assert_eq!(p.map(|x| x as f32 + 0.5), pt3(1.5, 2.5, 3.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn map<T>(self, f: impl FnMut(Sc) -> T) -> Point<[T; N]> {
        self.0.map(f).into()
    }

    /// Returns a vector of the same dimension as `self` by applying `f`
    /// component-wise to `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::pt3;
    ///
    /// let a = pt3::<f32, ()>(1.0, 2.0, 3.0);
    /// let b = pt3(4, 3, 2);
    /// assert_eq!(a.zip_map(b, |x, exp| x.powi(exp)), pt3(1.0, 8.0, 9.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn zip_map<T: Copy, U>(
        self,
        other: Point<[T; N]>,
        mut f: impl FnMut(Sc, T) -> U,
    ) -> Point<[U; N]> {
        array::from_fn(|i| f(self.0[i], other.0[i])).into()
    }
}

impl<const N: usize> Point<[f32; N]> {
    /// Returns the canonical origin point (0, …, 0).
    #[inline]
    pub const fn origin() -> Self {
        Self::new([0.0; N])
    }

    /// Returns the Euclidean distance between `self` and another point.
    ///
    /// # Example
    /// ```
    /// use retrofire_core::math::{Point2, pt2};
    ///
    /// let x3: Point2 = pt2(3.0, 0.0);
    /// let y4 = pt2(0.0, 4.0);
    /// assert_eq!(x3.distance(&y4), 5.0);
    /// ```
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        self.sub(other).len()
    }

    /// Returns the square of the Euclidean distance between `self` and another
    /// point.
    ///
    /// The squared distance is faster to compute than the [distance][Self::distance].
    ///
    /// # Example
    /// ```
    /// use retrofire_core::math::{Point2, pt2};
    ///
    /// let x3: Point2 = pt2(3.0, 0.0);
    /// let y4 = pt2(0.0, 4.0);
    /// assert_eq!(x3.distance_sqr(&y4), 5.0 * 5.0);
    /// ```
    #[inline]
    pub fn distance_sqr(&self, other: &Self) -> f32 {
        self.sub(other).len_sqr()
    }

    /// Returns `self` clamped component-wise to the given range.
    ///
    /// The result is a vector `v` such that for each valid index `i`,
    /// `v[i]` is equal to `self[i].clamp(min[i], max[i])`.
    ///
    /// See also [`f32::clamp`].
    ///
    /// # Panics
    /// If `min[i] > max[i]` for any valid index `i`,
    /// or if either `min` or `max` contains a NaN.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{pt3, Point3};
    ///
    /// let pt: Point3 = pt3(0.5, 1.5, -2.0);
    /// // Clamp to the unit cube
    /// let clamped = pt.clamp(&pt3(0.0, 0.0, 0.0), &pt3(1.0, 1.0, 1.0));
    /// assert_eq!(clamped, pt3(0.5, 1.0, 0.0));
    /// ```
    #[must_use]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        array::from_fn(|i| self.0[i].clamp(min.0[i], max.0[i])).into()
    }

    /// Returns `self`, moved towards another point by an offset.
    ///
    /// If the distance to `other` is less than `d`, returns `other`. That is,
    /// this method never "overshoots" the target. Negative values of `d` result
    /// in moving away from `other`. If `self` equals `other`, returns `other`
    /// independent of the value of `d`.
    ///
    /// To translate by a *relative* offset instead, use [`lerp`][super::Lerp::lerp].
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{Point2, pt2};
    ///
    /// let a: Point2 = pt2(0.0, 0.0);
    /// let b: Point2 = pt2(4.0, 3.0);
    /// // Move two units along the line y = 3x/4:
    /// assert_eq!(a.approach(&b, 2.0), pt2(1.6, 1.2));
    /// // Movement is clamped to b:
    /// assert_eq!(a.approach(&b, 10.0), b);
    /// // Negative values of `d` move away from b:
    /// assert_eq!(a.approach(&b, -1.0), pt2(-0.8, -0.6));
    /// // Approaching the point itself does nothing:
    /// assert_eq!(a.approach(&a, 1.0), a);
    /// assert_eq!(a.approach(&a, -1.0), a);
    /// ```
    #[must_use]
    pub fn approach(&self, other: &Self, d: f32) -> Self {
        let v = *other - *self;
        let l = v.len();
        if d < l && l != 0.0 {
            *self + d / l * v
        } else {
            *other
        }
    }
}

impl<Sc: Copy> Point<[Sc; 2]> {
    /// Returns the x component of `self`.
    #[inline]
    pub const fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub const fn y(&self) -> Sc {
        self.0[1]
    }

    /// Converts `self` to a `Point3`, with z equal to 0.
    #[inline]
    pub fn to_pt3(self) -> Point<[Sc; 3]>
    where
        Sc: Linear,
    {
        pt3(self.x(), self.y(), Sc::zero())
    }
}

impl Point2 {
    /// Converts `self` to homogeneous representation.
    #[inline]
    pub const fn to_hom(&self) -> Vec3 {
        Vector::new([self.0[0], self.0[1], 1.0])
    }
}

impl<Sc: Copy> Point<[Sc; 3]> {
    /// Returns the x component of `self`.
    #[inline]
    pub const fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub const fn y(&self) -> Sc {
        self.0[1]
    }
    /// Returns the z component of `self`.
    #[inline]
    pub const fn z(&self) -> Sc {
        self.0[2]
    }
}

impl Point3 {
    // Converts `self´ to homogeneous representation.
    #[inline]
    pub const fn to_hom(&self) -> Vec4 {
        Vector::new([self.0[0], self.0[1], self.0[2], 1.0])
    }
}

//
// Local trait impls
//

impl<ScSelf, ScDiff, const N: usize> Affine for Point<[ScSelf; N]>
where
    ScSelf: Affine<Diff = ScDiff> + Copy,
    ScDiff: Linear<Scalar = ScDiff> + Copy,
{
    type Space = ();
    type Diff = Vector<[ScDiff; N]>;
    const DIM: usize = N;

    #[inline]
    fn add(&self, other: &Self::Diff) -> Self {
        // TODO Profile performance of array::from_fn
        Self(array::from_fn(|i| self.0[i].add(&other.0[i])))
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self::Diff {
        Vector::new(array::from_fn(|i| self.0[i].sub(&other.0[i])))
    }
}

impl<Sc, const N: usize> ZDiv for Point<[Sc; N]>
where
    Sc: ZDiv + Copy,
{
    #[inline]
    fn z_div(mut self, z: f32) -> Self {
        for c in &mut self.0 {
            *c = c.z_div(z);
        }
        self
    }
}

impl<Sc: ApproxEq, const N: usize> ApproxEq<Sc> for Point<[Sc; N]> {
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

impl<R: Copy> Copy for Point<R> {}

impl<R: Clone> Clone for Point<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: Default> Default for Point<R> {
    fn default() -> Self {
        Self(R::default())
    }
}

impl<R: Debug> Debug for Point<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Point")?;
        Debug::fmt(&self.0, f)
    }
}

impl<R: Eq> Eq for Point<R> {}

impl<R: PartialEq> PartialEq for Point<R> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Point <-> repr conversions
impl<R> From<R> for Point<R> {
    #[inline]
    fn from(repr: R) -> Self {
        Self::new(repr)
    }
}
impl<Sc, const N: usize> From<Point<[Sc; N]>> for [Sc; N] {
    #[inline]
    fn from(v: Point<[Sc; N]>) -> Self {
        v.0
    }
}

// Point <-> tuple conversions
impl<Sc> From<(Sc, Sc)> for Point<[Sc; 2]> {
    #[inline]
    fn from(xy: (Sc, Sc)) -> Self {
        Self::new(xy.into())
    }
}
impl<Sc> From<Point<[Sc; 2]>> for (Sc, Sc) {
    #[inline]
    fn from(v: Point<[Sc; 2]>) -> Self {
        v.0.into()
    }
}
impl<Sc> From<(Sc, Sc, Sc)> for Point<[Sc; 3]> {
    #[inline]
    fn from(xyz: (Sc, Sc, Sc)) -> Self {
        Self::new(xyz.into())
    }
}
impl<Sc> From<Point<[Sc; 3]>> for (Sc, Sc, Sc) {
    #[inline]
    fn from(v: Point<[Sc; 3]>) -> Self {
        v.0.into()
    }
}

impl<R: Index<usize>> Index<usize> for Point<R> {
    type Output = R::Output;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        self.0.index(i)
    }
}
impl<R: IndexMut<usize>> IndexMut<usize> for Point<R> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut R::Output {
        self.0.index_mut(i)
    }
}

impl<R> Add<<Self as Affine>::Diff> for Point<R>
where
    Self: Affine,
{
    type Output = Self;

    #[inline]
    fn add(self, other: <Self as Affine>::Diff) -> Self {
        Affine::add(&self, &other)
    }
}
impl<R> AddAssign<<Self as Affine>::Diff> for Point<R>
where
    Self: Affine,
{
    #[inline]
    fn add_assign(&mut self, other: <Self as Affine>::Diff) {
        *self = Affine::add(self, &other);
    }
}

impl<R> Sub<<Self as Affine>::Diff> for Point<R>
where
    Self: Affine,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: <Self as Affine>::Diff) -> Self {
        Affine::add(&self, &other.neg())
    }
}
impl<R> SubAssign<<Self as Affine>::Diff> for Point<R>
where
    Self: Affine,
{
    #[inline]
    fn sub_assign(&mut self, other: <Self as Affine>::Diff) {
        *self = Affine::add(self, &other.neg());
    }
}

impl<R, Diff> Sub for Point<R>
where
    Self: Affine<Diff = Diff>,
{
    type Output = Diff;

    #[inline]
    fn sub(self, other: Self) -> Diff {
        Affine::sub(&self, &other)
    }
}

impl<R> Neg for Point<R>
where
    Self: Mul<f32, Output = Self>,
{
    type Output = Self;

    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<R: Copy, Sc> Mul<Sc> for Point<R>
where
    Self: Affine<Diff = Vector<R>>,
    Vector<R>: Linear<Scalar = Sc>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Sc) -> Self {
        self.to_vec().mul(rhs).to_pt()
    }
}
impl<R: Copy> Mul<Point<R>> for f32
where
    Point<R>: Affine<Diff = Vector<R>>,
    Vector<R>: Linear<Scalar = f32>,
{
    type Output = Point<R>;

    #[inline]
    fn mul(self, rhs: Point<R>) -> Point<R> {
        rhs * self
    }
}
impl<R: Copy, Sc> MulAssign<Sc> for Point<R>
where
    Self: Affine<Diff = Vector<R>>,
    Vector<R>: Linear<Scalar = Sc>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Sc) {
        *self = *self * rhs;
    }
}

impl<R: Copy> Div<f32> for Point<R>
where
    Self: Affine<Diff = Vector<R>>,
    Vector<R>: Linear<Scalar = f32>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self {
        self * rhs.recip()
    }
}
impl<R: Copy> DivAssign<f32> for Point<R>
where
    Self: Affine<Diff = Vector<R>>,
    Vector<R>: Linear<Scalar = f32>,
{
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
#[allow(non_upper_case_globals)]
mod tests {
    use super::*;
    use crate::math::{Lerp, vec2, vec3};

    mod f32 {
        use super::*;
        use crate::assert_approx_eq;

        const pt2: fn(f32, f32) -> Point2 = super::pt2;
        const pt3: fn(f32, f32, f32) -> Point3 = super::pt3;
        #[test]
        fn vector_addition() {
            assert_eq!(pt2(1.0, 2.0) + vec2(-2.0, 3.0), pt2(-1.0, 5.0));
            assert_eq!(
                pt3(1.0, 2.0, 3.0) + vec3(-2.0, 3.0, 1.0),
                pt3(-1.0, 5.0, 4.0)
            )
        }
        #[test]
        fn vector_subtraction() {
            assert_eq!(pt2(1.0, 2.0) - vec2(-2.0, 3.0), pt2(3.0, -1.0));
            assert_eq!(
                pt3(1.0, 2.0, 3.0) - vec3(-2.0, 3.0, 1.0),
                pt3(3.0, -1.0, 2.0)
            )
        }
        #[test]
        fn point_subtraction() {
            assert_eq!(pt2(1.0, 2.0) - pt2(-2.0, 3.0), vec2(3.0, -1.0));
            assert_eq!(
                pt3(1.0, 2.0, 3.0) - pt3(-2.0, 3.0, 1.0),
                vec3(3.0, -1.0, 2.0)
            )
        }
        #[test]
        fn scalar_multiplication() {
            assert_eq!(pt2(1.0, -2.0) * 3.0, pt2(3.0, -6.0));
            assert_eq!(pt3(1.0, -2.0, 3.0) * 0.5, pt3(0.5, -1.0, 1.5));

            assert_eq!(3.0 * pt2(1.0, -2.0), pt2(3.0, -6.0));
            assert_eq!(0.5 * pt3(1.0, -2.0, 3.0), pt3(0.5, -1.0, 1.5));
        }
        #[test]
        fn scalar_division() {
            assert_eq!(pt2(1.0, -2.0) / 0.5, pt2(2.0, -4.0));
            assert_eq!(pt3(1.0, -2.0, 3.0) / 2.0, pt3(0.5, -1.0, 1.5));
        }
        #[test]
        fn point_point_distance_sqr() {
            assert_eq!(pt2(1.0, -1.0).distance_sqr(&pt2(-2.0, 3.0)), 25.0);
            assert_eq!(
                pt3(1.0, -3.0, 2.0).distance_sqr(&pt3(-2.0, 3.0, 4.0)),
                49.0
            );
        }
        #[test]
        fn point_point_distance() {
            assert_approx_eq!(pt2(1.0, -1.0).distance(&pt2(-2.0, 3.0)), 5.0);
            assert_approx_eq!(
                pt3(1.0, -3.0, 2.0).distance(&pt3(-2.0, 3.0, 4.0)),
                7.0
            );
        }
        #[test]
        fn point2_clamp() {
            let (min, max) = (&pt2(-2.0, -1.0), &pt2(3.0, 2.0));
            assert_eq!(pt2(1.0, -1.0).clamp(min, max), pt2(1.0, -1.0));
            assert_eq!(pt2(3.0, -2.0).clamp(min, max), pt2(3.0, -1.0));
            assert_eq!(pt2(-3.0, 4.0).clamp(min, max), pt2(-2.0, 2.0));
        }
        #[test]
        fn point3_clamp() {
            let (min, max) = (&pt3(-2.0, -1.0, 0.0), &pt3(3.0, 2.0, 1.0));
            assert_eq!(
                pt3(1.0, -1.0, 0.0).clamp(min, max),
                pt3(1.0, -1.0, 0.0)
            );
            assert_eq!(
                pt3(3.0, -2.0, -1.0).clamp(min, max),
                pt3(3.0, -1.0, 0.0)
            );
            assert_eq!(
                pt3(-3.0, 4.0, 2.0).clamp(min, max),
                pt3(-2.0, 2.0, 1.0)
            );
        }
        #[test]
        fn point2_indexing() {
            let mut p = pt2(2.0, -1.0);
            assert_eq!(p[0], p.x());
            assert_eq!(p[1], p.y());

            p[1] -= 1.0;
            assert_eq!(p[1], -2.0);
        }
        #[test]
        fn point3_indexing() {
            let mut p = pt3(2.0, -1.0, 3.0);
            assert_eq!(p[0], p.x());
            assert_eq!(p[1], p.y());
            assert_eq!(p[2], p.z());

            p[2] += 1.0;
            assert_eq!(p[2], 4.0);
        }
        #[test]
        #[should_panic]
        fn point2_index_oob() {
            _ = pt2(1.0, 2.0)[2];
        }
        #[test]
        fn point2_lerp() {
            assert_eq!(
                pt2(2.0, -1.0).lerp(&pt2(-2.0, 3.0), 0.25),
                pt2(1.0, 0.0)
            );
        }
    }

    mod u32 {
        use super::*;

        const pt2: fn(u32, u32) -> Point2u = super::pt2;

        #[test]
        fn vector_addition() {
            assert_eq!(pt2(1_u32, 2) + vec2(1_i32, -2), pt2(2_u32, 0));
        }

        #[test]
        fn vector_subtraction() {
            assert_eq!(pt2(3_u32, 2) - vec2(3_i32, -1), pt2(0_u32, 3));
        }

        #[test]
        fn point_subtraction() {
            assert_eq!(pt2(3_u32, 2) - pt2(3_u32, 3), vec2(0, -1));
        }

        #[test]
        fn indexing() {
            let mut p = pt2(1u32, 2);
            assert_eq!(p[1], 2);
            p[0] = 3;
            assert_eq!(p.0, [3, 2]);
        }

        #[test]
        fn from_array() {
            assert_eq!(Point2u::from([1, 2]), pt2(1, 2));
        }
    }
}
