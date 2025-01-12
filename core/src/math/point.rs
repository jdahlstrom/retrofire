use core::{
    array,
    fmt::{Debug, Formatter},
    marker::PhantomData as Pd,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
};

use super::{space::Real, vary::ZDiv, Affine, ApproxEq, Linear, Vector};

#[repr(transparent)]
pub struct Point<Repr, Space = ()>(pub Repr, Pd<Space>);

/// A 2-point with `f32` components.
pub type Point2<Basis = ()> = Point<[f32; 2], Real<2, Basis>>;
/// A 3-point with `f32` components.
pub type Point3<Basis = ()> = Point<[f32; 3], Real<3, Basis>>;

/// A 2-point with `u32` components.
pub type Point2u<Basis = ()> = Point<[u32; 2], Real<2, Basis>>;

/// Returns a real 2-point with `x` and `y` components.
pub const fn pt2<Sc, B>(x: Sc, y: Sc) -> Point<[Sc; 2], Real<2, B>> {
    Point([x, y], Pd)
}
/// Returns a real 3-point with `x`, `y`, and `z` components.
pub const fn pt3<Sc, B>(x: Sc, y: Sc, z: Sc) -> Point<[Sc; 3], Real<3, B>> {
    Point([x, y, z], Pd)
}

impl<R, Sp> Point<R, Sp> {
    #[inline]
    pub fn new(repr: R) -> Self {
        Self(repr, Pd)
    }

    /// Returns a point with value equal to `self` but in space `S`.
    // TODO Cannot be const (yet?) due to E0493 :(
    #[inline]
    pub fn to<S>(self) -> Point<R, S> {
        Point(self.0, Pd)
    }

    /// Returns the vector equivalent to `self`.
    // TODO Cannot be const (yet?) due to E0493 :(
    #[inline]
    pub fn to_vec(self) -> Vector<R, Sp> {
        Vector::new(self.0)
    }
}

impl<Sc, Sp, const N: usize> Point<[Sc; N], Sp> {
    /// Returns a point of the same dimension as `self` by applying `f`
    /// component-wise.
    #[inline]
    #[must_use]
    pub fn map<T>(self, f: impl FnMut(Sc) -> T) -> Point<[T; N], Sp> {
        self.0.map(f).into()
    }
}

impl<const N: usize, B> Point<[f32; N], Real<N, B>> {
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
    #[cfg(feature = "fp")]
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        self.sub(other).len()
    }
    /// Returns the square of the Euclidean distance between `self` and another
    /// point.
    ///
    /// Faster to compute than [distance][Self::distance].
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
    #[must_use]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        array::from_fn(|i| self.0[i].clamp(min.0[i], max.0[i])).into()
    }
}

impl<R, B, Sc> Point<R, Real<2, B>>
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

impl<R, Sc, B> Point<R, Real<3, B>>
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

//
// Local trait impls
//

impl<Sc, DSc, Sp, const N: usize> Affine for Point<[Sc; N], Sp>
where
    Sc: Affine<Diff = DSc> + Copy,
    DSc: Linear<Scalar = DSc> + Copy,
{
    type Space = Sp;
    type Diff = Vector<[DSc; N], Sp>;
    const DIM: usize = N;

    #[inline]
    fn add(&self, other: &Self::Diff) -> Self {
        // TODO Profile performance of array::from_fn
        Self(array::from_fn(|i| self.0[i].add(&other.0[i])), Pd)
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self::Diff {
        Vector::new(array::from_fn(|i| self.0[i].sub(&other.0[i])))
    }
}

impl<Sc, Sp, const N: usize> ZDiv for Point<[Sc; N], Sp>
where
    Sc: ZDiv + Copy,
{
    fn z_div(self, z: f32) -> Self {
        self.map(|c| c.z_div(z))
    }
}

impl<Sc: ApproxEq, Sp, const N: usize> ApproxEq<Self, Sc>
    for Point<[Sc; N], Sp>
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

impl<R: Copy, S> Copy for Point<R, S> {}

impl<R: Clone, S> Clone for Point<R, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), Pd)
    }
}

impl<R: Default, S> Default for Point<R, S> {
    fn default() -> Self {
        Self(R::default(), Pd)
    }
}

impl<R: Debug, Sp: Debug + Default> Debug for Point<R, Sp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Point<{:?}>", Sp::default())?;
        Debug::fmt(&self.0, f)
    }
}

impl<R: Eq, S> Eq for Point<R, S> {}

impl<R: PartialEq, S> PartialEq for Point<R, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<R, Sp> From<R> for Point<R, Sp> {
    #[inline]
    fn from(repr: R) -> Self {
        Self(repr, Pd)
    }
}

impl<R: Index<usize>, Sp> Index<usize> for Point<R, Sp> {
    type Output = R::Output;

    fn index(&self, i: usize) -> &Self::Output {
        self.0.index(i)
    }
}

impl<R: IndexMut<usize>, Sp> IndexMut<usize> for Point<R, Sp> {
    fn index_mut(&mut self, i: usize) -> &mut R::Output {
        self.0.index_mut(i)
    }
}

impl<R, Sp> Add<<Self as Affine>::Diff> for Point<R, Sp>
where
    Self: Affine,
{
    type Output = Self;

    fn add(self, other: <Self as Affine>::Diff) -> Self {
        Affine::add(&self, &other)
    }
}

impl<R, Sp> AddAssign<<Self as Affine>::Diff> for Point<R, Sp>
where
    Self: Affine,
{
    fn add_assign(&mut self, other: <Self as Affine>::Diff) {
        *self = Affine::add(self, &other);
    }
}

impl<R, Sp> Sub<<Self as Affine>::Diff> for Point<R, Sp>
where
    Self: Affine,
{
    type Output = Self;

    fn sub(self, other: <Self as Affine>::Diff) -> Self {
        Affine::add(&self, &other.neg())
    }
}

impl<R, Sp> SubAssign<<Self as Affine>::Diff> for Point<R, Sp>
where
    Self: Affine,
{
    fn sub_assign(&mut self, other: <Self as Affine>::Diff) {
        *self = Affine::add(self, &other.neg());
    }
}

impl<R, Sp> Sub for Point<R, Sp>
where
    Self: Affine,
{
    type Output = <Self as Affine>::Diff;

    fn sub(self, other: Self) -> Self::Output {
        Affine::sub(&self, &other)
    }
}

#[cfg(test)]
#[allow(non_upper_case_globals)]
mod tests {
    use super::*;
    use crate::math::{vec2, vec3, Lerp};

    const pt2: fn(f32, f32) -> Point2 = super::pt2;
    const pt3: fn(f32, f32, f32) -> Point3 = super::pt3;

    #[test]
    fn point_plus_vec_gives_point() {
        assert_eq!(pt2(1.0, 2.0) + vec2(-2.0, 3.0), pt2(-1.0, 5.0));
        assert_eq!(
            pt3(1.0, 2.0, 3.0) + vec3(-2.0, 3.0, 1.0),
            pt3(-1.0, 5.0, 4.0)
        )
    }
    #[test]
    fn point_minus_vec_gives_point() {
        assert_eq!(pt2(1.0, 2.0) - vec2(-2.0, 3.0), pt2(3.0, -1.0));
        assert_eq!(
            pt3(1.0, 2.0, 3.0) - vec3(-2.0, 3.0, 1.0),
            pt3(3.0, -1.0, 2.0)
        )
    }
    #[test]
    fn point_minus_point_gives_vec() {
        assert_eq!(pt2(1.0, 2.0) - pt2(-2.0, 3.0), vec2(3.0, -1.0));
        assert_eq!(
            pt3(1.0, 2.0, 3.0) - pt3(-2.0, 3.0, 1.0),
            vec3(3.0, -1.0, 2.0)
        )
    }
    #[test]
    fn point_point_dist_sqr() {
        assert_eq!(pt2(1.0, -1.0).distance_sqr(&pt2(-2.0, 3.0)), 25.0);
        assert_eq!(
            pt3(1.0, -3.0, 2.0).distance_sqr(&pt3(-2.0, 3.0, 4.0)),
            49.0
        );
    }
    #[test]
    #[cfg(feature = "fp")]
    fn point_point_dist() {
        assert_eq!(pt2(1.0, -1.0).distance(&pt2(-2.0, 3.0)), 5.0);
        assert_eq!(pt3(1.0, -3.0, 2.0).distance(&pt3(-2.0, 3.0, 4.0)), 7.0);
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
        assert_eq!(pt3(1.0, -1.0, 0.0).clamp(min, max), pt3(1.0, -1.0, 0.0));
        assert_eq!(pt3(3.0, -2.0, -1.0).clamp(min, max), pt3(3.0, -1.0, 0.0));
        assert_eq!(pt3(-3.0, 4.0, 2.0).clamp(min, max), pt3(-2.0, 2.0, 1.0));
    }
    #[test]
    fn point2_index() {
        let mut p = pt2(2.0, -1.0);
        assert_eq!(p[0], p.x());
        assert_eq!(p[1], p.y());

        p[1] -= 1.0;
        assert_eq!(p[1], -2.0);
    }
    #[test]
    fn point3_index() {
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
        assert_eq!(pt2(2.0, -1.0).lerp(&pt2(-2.0, 3.0), 0.25), pt2(1.0, 0.0));
    }
}
