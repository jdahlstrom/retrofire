use core::{
    array,
    fmt::{Debug, Formatter},
    marker::PhantomData as Pd,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
};

use super::{Affine, ApproxEq, Linear, Vector, space::Real, vary::ZDiv};

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
    pub const fn new(repr: R) -> Self {
        Self(repr, Pd)
    }

    /// Returns a point with value equal to `self` but in space `S`.
    #[inline]
    pub const fn to<S>(self) -> Point<R, S>
    where
        R: Copy, // TODO Needed for now due to E0493
    {
        Point::new(self.0)
    }

    /// Returns the vector equivalent to `self`.
    #[inline]
    pub const fn to_vec(self) -> Vector<R, Sp>
    where
        R: Copy, // TODO Needed for now due to E0493
    {
        Vector::new(self.0)
    }
}

impl<Sc: Copy, Sp, const N: usize> Point<[Sc; N], Sp> {
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
    pub fn map<T>(self, f: impl FnMut(Sc) -> T) -> Point<[T; N], Sp> {
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
        other: Point<[T; N], Sp>,
        mut f: impl FnMut(Sc, T) -> U,
    ) -> Point<[U; N], Sp> {
        array::from_fn(|i| f(self.0[i], other.0[i])).into()
    }
}

impl<const N: usize, B> Point<[f32; N], Real<N, B>> {
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
    #[cfg(feature = "fp")]
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
}

impl<Sc: Copy, B> Point<[Sc; 2], Real<2, B>> {
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
}

impl Point2 {
    /// Converts `self` into a `Point3`, with z equal to 0.
    #[inline]
    pub const fn to_pt3(self) -> Point3 {
        pt3(self.x(), self.y(), 0.0)
    }
}

impl<Sc: Copy, B> Point<[Sc; 3], Real<3, B>> {
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

//
// Local trait impls
//

impl<ScSelf, ScDiff, Sp, const N: usize> Affine for Point<[ScSelf; N], Sp>
where
    ScSelf: Affine<Diff = ScDiff> + Copy,
    ScDiff: Linear<Scalar = ScDiff> + Copy,
{
    type Space = Sp;
    type Diff = Vector<[ScDiff; N], Sp>;
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
    #[inline]
    fn z_div(mut self, z: f32) -> Self {
        for c in &mut self.0 {
            *c = c.z_div(z);
        }
        self
    }
}

impl<Sc: ApproxEq, Sp, const N: usize> ApproxEq<Sc> for Point<[Sc; N], Sp> {
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
/*
impl<B> From<Point3<B>> for HomVec3<B> {
    fn from(p: Point3<B>) -> Self {
        let [x, y, z] = p.0;
        [x, y, z, 1.0].into()
    }
}

impl<B> From<Point2<B>> for HomVec2<B> {
    fn from(p: Point2<B>) -> Self {
        let [x, y] = p.0;
        [x, y, 1.0].into()
    }
}*/

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
    use crate::math::{Lerp, vec2, vec3};

    mod f32 {
        use super::*;

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
        fn point_point_distance_sqr() {
            assert_eq!(pt2(1.0, -1.0).distance_sqr(&pt2(-2.0, 3.0)), 25.0);
            assert_eq!(
                pt3(1.0, -3.0, 2.0).distance_sqr(&pt3(-2.0, 3.0, 4.0)),
                49.0
            );
        }
        #[test]
        #[cfg(feature = "fp")]
        fn point_point_distance() {
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

        const pt2: fn(u32, u32) -> Point2u = super::super::pt2;

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
