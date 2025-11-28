//! Bézier curves and splines.
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::geom::{Polyline, Ray};
use crate::mat;

use super::{Affine, Lerp, Linear, Mat4, Parametric, Point, Vector};

/// A cubic Bézier curve, defined by four control points.
///
/// TODO More info about Béziers
///
/// ```text
///
///  p1
///   \         ____
///    \     _-´    `--_                      p3
///     \   /           `-_                    \
///      \ |               `-_                 |\
///       \|                  `-__            /  \
///        \                      `---_____--´    \
///        p0                                      \
///                                                 p2
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CubicBezier<T>(pub [T; 4]);

/// Interpolates smoothly from 0.0 to 1.0 as `t` goes from 0.0 to 1.0.
///
/// Returns 0 for all `t` <= 0 and 1 for all `t` >= 1. Has a continuous
/// first derivative.
pub fn smoothstep(t: f32) -> f32 {
    step(t, &0.0, &1.0, |t| t * t * (3.0 - 2.0 * t))
}

/// Even smoother version of [`smoothstep`].
///
/// Has continuous first and second derivatives.
pub fn smootherstep(t: f32) -> f32 {
    step(t, &0.0, &1.0, |t| t * t * t * (10.0 + t * (6.0 * t - 15.0)))
}

/// Helper for defining step functions.
///
/// Returns `min` if t ≤ 0, `max` if t ≥ 1, and `f(t)` if 0 < t < 1.
#[inline]
pub fn step<T: Clone, F>(t: f32, min: &T, max: &T, f: F) -> T
where
    F: FnOnce(f32) -> T,
{
    if t <= 0.0 {
        min.clone()
    } else if t >= 1.0 {
        max.clone()
    } else {
        f(t)
    }
}

/// The characteristic matrix of a cubic Bézier spline.
const _BEZIER_MAT: Mat4 = mat![
     1.0,  0.0,  0.0,  0.0;
    -3.0,  3.0,  0.0,  0.0;
     3.0, -6.0,  3.0,  0.0;
    -1.0,  3.0, -3.0,  1.0;
];

/// The characteristic matrix of a cubic Hermite spline.
const _HERMITE_MAT: Mat4 = mat![
     1.0,  0.0,  0.0,  0.0;
     0.0,  1.0,  0.0,  0.0;
    -3.0, -2.0,  3.0, -1.0;
     2.0,  1.0, -2.0,  1.0;
];

/// The characteristic matrix of a cubic Catmull-Rom spline.
const _CATMULL_ROM_MAT: Mat4 = mat![
     0.0,  1.0,  0.0,  0.0;
    -0.5,  0.0,  0.5,  0.0;
     1.0, -2.5,  2.0, -0.5;
    -0.5,  1.5, -1.5,  0.5;
];

/// The characteristic matrix of a cubic B-spline.
const _B_SPLINE_MAT: Mat4 = {
    const F16: f32 = 1.0 / 6.0;
    const F23: f32 = 2.0 / 3.0;
    mat![
        F16,   F23,  F16,  0.0;
       -0.5,   0.0,  0.5,  0.0;
        0.5,  -1.0,  0.5,  0.0;
       -F16,  0.5, -0.5,  F16;
    ]
};

//
// Inherent impls
//

impl<T: Lerp> CubicBezier<T> {
    /// Evaluates the value of `self` at `t`.
    ///
    /// For t < 0, returns the first control point. For t > 1, returns the last
    /// control point. Uses [De Casteljau's algorithm][1].
    ///
    /// [1]: https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
    pub fn eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = &self.0;
        step(t, p0, p3, |t| {
            let p01 = p0.lerp(p1, t);
            let p12 = p1.lerp(p2, t);
            let p23 = p2.lerp(p3, t);
            p01.lerp(&p12, t).lerp(&p12.lerp(&p23, t), t)
        })
    }
}
impl<T> CubicBezier<T>
where
    T: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
    /// Evaluates the value of `self` at `t`.
    ///
    /// For t < 0, returns the first control point. For t > 1, returns the last
    /// control point.
    ///
    /// Directly evaluates the cubic. Faster but possibly less numerically
    /// stable than [`Self::eval`].
    pub fn fast_eval(&self, t: f32) -> T {
        let [p0, .., p3] = &self.0;
        step(t, p0, p3, |t| {
            // Add a linear combination of the three coefficients
            // to `p0` to get the result
            let [co3, co2, co1] = self.coefficients();
            p0.add(&co3.mul(t).add(&co2).mul(t).add(&co1).mul(t))
        })
    }

    /// Returns the tangent, or direction vector, of `self` at `t`.
    ///
    /// Clamps `t` to the range [0, 1].
    pub fn tangent(&self, t: f32) -> T::Diff {
        let [p0, p1, p2, p3] = &self.0;
        let t = t.clamp(0.0, 1.0);

        //   3 (3 (p1 - p2) + (p3 - p0)) * t^2
        // + 6 ((p0 - p1 + p2 - p1) * t
        // + 3 (p1 - p0)

        let co2: T::Diff = p1.sub(p2).mul(3.0).add(&p3.sub(p0));
        let co1: T::Diff = p0.sub(p1).add(&p2.sub(p1)).mul(2.0);
        let co0: T::Diff = p1.sub(p0);

        co2.mul(t).add(&co1).mul(t).add(&co0).mul(3.0)
    }

    /// Returns the coefficients used to evaluate the spline.
    ///
    /// These are constant as long as the control points do not change,
    /// so they can be precomputed when the spline is evaluated several times,
    /// for example by an iterator.
    ///
    /// The coefficient values are, from the first to the last:
    /// ```text
    /// co3 = (p3 - p0) + 3 * (p1 - p2)
    /// co2 = 3 * (p0 - p1) + 3 * (p2 - p1)
    /// co1 = 3 * (p1 - p0)
    /// ```
    /// The value of the spline at *t* is then computed as:
    /// ```text
    /// co3 * t^3 + co2 * t^2 + co1 * t + p0
    ///
    /// = (((co3 * t) + co2 * t) + co1 * t) + p0.
    /// ```
    fn coefficients(&self) -> [T::Diff; 3] {
        let [p0, p1, p2, p3] = &self.0;

        // Rewrite the parametric equation into a form where three of the
        // coefficients are vectors, their linear combination added to `p0`
        // so the equation can be expressed for affine types:
        //
        //   (p3 - p0) * t^3 + (p1 - p2) * 3t^3
        // + (p0 + p2) * 3t^2 - p1 * 6t^2
        // + (p1 - p0) * 3t
        // + p0
        // = ((p3 - p0 + 3(p1 - p2)) * t^3
        // + 3(p0 - p1 + p2 - p1) * t^2
        // + 3(p1 - p0) * t
        // + p0
        // = ((((p3 - p0 + 3(p1 - p2))) * t
        //      + 3(p0 - p1 + p2 - p1)) * t)
        //          + 3(p1 - p0)) * t)
        //              + p0
        let p3_p0 = p3.sub(p0);
        let p1_p0_3 = p1.sub(p0).mul(3.0);
        let p1_p2_3 = p1.sub(p2).mul(3.0);
        [p3_p0.add(&p1_p2_3), p1_p0_3.add(&p1_p2_3).neg(), p1_p0_3]
    }
}

/// A curve composed of one or more concatenated
/// [cubic Bézier curves][CubicBezier].
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BezierSpline<T>(Vec<T>);

pub struct HermiteSpline<T: Affine>(Vec<Ray<T>>);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CatmullRomSpline<T>(Vec<T>);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BSpline<T>(Vec<T>);

type Pt<const N: usize, Sp> = Point<[f32; N], Sp>;
type V<const N: usize, Sp> = Vector<[f32; N], Sp>;

//
// Inherent impls
//

impl<T> BezierSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone + Debug> + Clone + Debug,
{
    /// Creates a Bézier spline from the given control points. The number of
    /// elements in `pts` must be 3n + 1 for some positive integer n.
    ///
    /// Consecutive points in `pts` make up Bézier curves such that:
    /// * `pts[0..=3]` define the first curve,
    /// * `pts[3..=6]` define the second curve,
    ///
    /// and so on.
    ///
    /// # Panics
    /// If `pts.len() < 4` or if `pts.len() % 3 != 1`.
    pub fn new(pts: &[T]) -> Self {
        assert!(
            pts.len() >= 4 && pts.len() % 3 == 1,
            "length must be 3n+1 for some integer n > 0, was {}",
            pts.len()
        );
        Self(pts.to_vec())
    }

    /// Constructs a Bézier spline from (position, tangent) pairs.
    ///
    /// Specifically, for each pair of consecutive rays (P0, d0) and (P1, d1),
    /// the result contains one cubic Bézier curve segment with control points
    /// (P0, P0 + d0, P1 - d1, P1). The next segment, defined by (P1, d1) and
    /// the next ray (P2, d2), would in turn give (P1, P1 + d1, P2 - d2, P2).
    pub fn from_rays<I>(rays: I) -> Self
    where
        I: IntoIterator<Item = Ray<T>>,
    {
        let mut rays = rays.into_iter().peekable();
        let mut first = true;
        let mut pts = Vec::with_capacity(2 * rays.size_hint().0);
        while let Some(ray) = rays.next() {
            if !first {
                pts.push(ray.eval(-1.0));
            }
            first = false;
            pts.push(ray.0.clone());
            if rays.peek().is_some() {
                pts.push(ray.eval(1.0));
            }
        }
        Self::new(&pts)
    }

    /// Evaluates `self` at position `t`.
    ///
    /// Returns the first point if `t < 0` and the last point if `t > 1`.
    pub fn eval(&self, t: f32) -> T {
        let [first, .., last] = &self.0[..] else {
            panic!("invariant: self.0.len() >= 4")
        };
        step(t, first, last, |t| {
            let (t, seg) = self.segment(t);
            CubicBezier(seg).fast_eval(t)
        })
    }

    /// Returns the tangent of `self` at `t`.
    ///
    /// Clamps `t` to the range [0, 1].
    pub fn tangent(&self, t: f32) -> T::Diff {
        let (t, seg) = self.segment(t);
        CubicBezier(seg).tangent(t)
    }

    fn segment(&self, t: f32) -> (f32, [T; 4]) {
        // The number of Bézier curve segments in this spline
        let n_segs = ((self.0.len() - 1) / 3) as f32;
        let t = t * n_segs;
        // The integral part is the index of the segment we want
        use super::float::f32;
        let seg = f32::floor(t).min(n_segs - 1.0);
        // The fractional part is the local parameter value
        let u = t - seg;
        let idx = 3 * (seg as usize);
        let seg: &[T; 4] = self.0[idx..][..4]
            .try_into()
            .expect("idx is guaranteed to be <= len - 4");

        (u, seg.clone())
    }

    /// Approximates `self` as a chain of line segments.
    ///
    /// Recursively subdivides the curve into two half-curves, stopping once
    /// the approximation error is less than `error`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{BezierSpline, Point2, pt2};
    ///
    /// let curve = BezierSpline::<Point2>::new(
    ///     &[pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(1.0, 0.0)]
    /// );
    /// // Find an approximation with error less than 0.01
    /// let approx = curve.approximate(0.01);
    ///
    /// // Euclidean length of the polyline approximation
    /// assert_eq!(approx.len(), 1.9969313);
    ///
    /// // Number of line segments used by the approximation
    /// assert_eq!(approx.0.len(), 17);
    /// ```
    ///
    /// # Panics
    /// If `err` ≤ 0.
    pub fn approximate<Sp, const DIM: usize>(&self, error: f32) -> Polyline<T>
    where
        T: Affine<Diff = Vector<[f32; DIM], Sp>>,
    {
        assert!(error > 0.0);
        self.approximate_with(&|e: &T::Diff| e.len_sqr() < error * error)
    }

    /// Approximates `self` as a chain of line segments.
    ///
    /// Recursively subdivides the curve into two half-curves, stopping once
    /// the approximation error is small enough, as determined by the `halt`
    /// function.
    ///
    /// Given a curve segment between some points `p` and `r`, the parameter
    /// passed to `halt` is the distance to the real midpoint `q` from its
    /// linear approximation `q'`. If `halt` returns `true`, the line segment
    /// `pr` is returned as the approximation of this curve segment, otherwise
    /// the bisection continues.
    ///
    /// Note that this heuristic does not work well in certain edge cases
    /// (consider, for example, an S-shaped curve where `q'` is very close
    /// to `q`, yet a straight line would be a poor approximation). However,
    /// in practice it tends to give reasonable results.
    ///
    /// ```text
    ///                 ___--- q ---___
    ///             _--´       |       `--_
    ///         _--´           |           `--_
    ///      _-p ------------- q' ------------ r-_
    ///   _-´                                     `-_
    /// ```
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{BezierSpline, Point2, pt2};
    ///
    /// let curve = BezierSpline::<Point2>::new(
    ///     &[pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(1.0, 0.0)]
    /// );
    /// let approx = curve.approximate_with(|err| err.len_sqr() < 0.01 * 0.01);
    ///
    /// // Euclidean length of the polyline approximation
    /// assert_eq!(approx.len(), 1.9969313);
    ///
    /// // Number of line segments used by the approximation
    /// assert_eq!(approx.0.len(), 17);
    /// ```
    pub fn approximate_with<Pred>(&self, halt: Pred) -> Polyline<T>
    where
        Pred: Fn(&T::Diff) -> bool,
    {
        let len = self.0.len();
        let mut res = approx(self, 0.0, 1.0, 10 + len.ilog2(), &halt);
        res.push(self.0[len - 1].clone());
        Polyline(res)
    }
}

impl<Sp, const N: usize> HermiteSpline<Pt<N, Sp>>
where
    Pt<N, Sp>: Affine<Diff = V<N, Sp>> + Lerp,
{
    pub fn new(
        rays: impl IntoIterator<Item = Ray<Point<[f32; N], Sp>>>,
    ) -> Self {
        let rays: Vec<_> = rays.into_iter().collect();
        assert!(rays.len() >= 1);
        Self(rays)
    }

    pub fn eval(&self, t: f32) -> Pt<N, Sp> {
        let len = self.0.len();
        step(t, &self.0[0].0, &self.0[len - 1].0, |t| {
            let t = t * (len as f32 - 1.0);

            let (t, Ray(p0, d0), Ray(p1, d1)) = self.segment(t);
            let ts = [1.0, t, t * t, t * t * t];
            let [_, b1, b2, b3] = Self::bernstein(ts);

            // S(t) = b * P

            //   b0 * p0 + b1 * d0 + b2 * p1 + b3 * d1
            // = b0 * p0 + b2 * p1 // Affine: b0 + b2 = 1: lerp
            // + b1 * d0 + b3 * d1 // Linear

            //   b0 * p0 + b2 * p1
            // = (1 - b2) * p0 + b2 * p1
            // = p0 + b2 * (p1 - p0)

            p0.add(&p1.sub(&p0).mul(b2)) // Affine
                .add(&d0.mul(b1).add(&d1.mul(b3))) // Linear
        })
    }

    pub fn approximate(&self, error: f32) -> Polyline<Pt<N, Sp>>
    where
        Pt<N, Sp>: Lerp,
        Sp: Debug + Default,
    {
        assert!(error > 0.0);
        self.approximate_with(&|e: &V<N, Sp>| e.len_sqr() < error * error)
    }

    pub fn approximate_with<Pred>(&self, halt: Pred) -> Polyline<Pt<N, Sp>>
    where
        Pt<N, Sp>: Lerp,
        Sp: Debug + Default,
        Pred: Fn(&V<N, Sp>) -> bool,
    {
        let len = self.0.len();
        let mut res = approx(self, 0.0, 1.0, 10 + len.ilog2(), &halt);
        res.push(self.0[len - 1].0);
        Polyline(res)
    }

    fn bernstein([_0, t1, t2, t3]: [f32; 4]) -> [f32; 4] {
        // Characteristic matrix M
        //  1.0,  0.0,  0.0,  0.0;
        //  0.0,  1.0,  0.0,  0.0;
        // -3.0, -2.0,  3.0, -1.0;
        //  2.0,  1.0, -2.0,  1.0;

        // b = ts * M
        let _0 = 1.0 - 3.0 * t2 + 2.0 * t3; // = 1 - b2
        let b1 = t1 - 2.0 * t2 + t3;
        let b2 = 3.0 * t2 - 2.0 * t3;
        let b3 = -t2 + t3;
        [1.0 - b2, b1, b2, b3]
    }

    // Returns the curve segment corresponding to the global t value.
    // Precondition: 0 <= t < self.0.len() - 1
    fn segment(&self, t: f32) -> (f32, &Ray<Pt<N, Sp>>, &Ray<Pt<N, Sp>>) {
        debug_assert!(0.0 <= t && t < self.0.len() as f32 - 1.0);
        let i = t as usize;
        let u = t - i as f32;
        (u, &self.0[i], &self.0[i + 1])
    }
}

impl<const N: usize, Sp> CatmullRomSpline<Pt<N, Sp>> {
    pub fn new(pts: &[Pt<N, Sp>]) -> Self {
        Self(pts.to_vec())
    }

    pub fn eval(&self, t: f32) -> Pt<N, Sp> {
        // Doesn't pass through first and last
        let [_, second, .., second_last, _] = self.0.as_slice() else {
            unreachable!("invariant: len >= 4")
        };
        step(t, second, second_last, |t| {
            let t = t * (self.0.len() as f32 - 3.0) + 1.0;

            let (t, pts) = self.segment(t);
            let ts = [1.0, t, t * t, t * t * t];

            // S(t) = b * P
            Affine::combine(&Self::bernstein(ts), &pts)
        })
    }

    fn bernstein([t0, t1, t2, t3]: [f32; 4]) -> [f32; 4] {
        // Characteristic matrix M
        //  0.0,  1.0,  0.0,  0.0;
        // -0.5,  0.0,  0.5,  0.0;
        //  1.0, -2.5,  2.0, -0.5;
        // -0.5,  1.5, -1.5,  0.5;

        // b = ts * M
        let b0 = (-t1 + 2.0 * t2 - t3) / 2.0;
        let b1 = (2.0 * t0 - 5.0 * t2 + 3.0 * t3) / 2.0;
        let b2 = (t1 + 4.0 * t2 - 3.0 * t3) / 2.0;
        let b3 = (-t2 + t3) / 2.0;
        [b0, b1, b2, b3]
    }

    // Returns the curve segment corresponding to the global t value.
    // Precondition: 1 <= t < self.0.len() - 2
    fn segment(&self, t: f32) -> (f32, [Pt<N, Sp>; 4]) {
        debug_assert!(1.0 <= t && t < self.0.len() as f32 - 2.0, "t = {t}");
        let i = t as usize;
        let u = t - i as f32;
        let pts = self.0[i - 1..i + 3]
            .try_into()
            .expect("slice has four elements");
        (u, pts)
    }
}

impl<const N: usize, Sp> BSpline<Pt<N, Sp>> {
    pub fn new(pts: &[Pt<N, Sp>]) -> Self {
        Self(pts.to_vec())
    }

    pub fn eval(&self, t: f32) -> Pt<N, Sp> {
        let [_, second, .., second_last, _] = self.0.as_slice() else {
            unreachable!("invariant: len >= 4")
        };
        step(t, second, second_last, |t| {
            let t = t * (self.0.len() as f32 - 3.0) + 1.0;

            let (t, pts) = self.segment(t);
            let ts = [1.0, t, t * t, t * t * t];

            // S(t) = b * P
            Affine::combine(&Self::bernstein(ts), &pts)
        })
    }

    fn bernstein([_0, t1, t2, t3]: [f32; 4]) -> [f32; 4] {
        // Characteristic matrix M
        //  1/6,   2/3,  1/6,  0.0;
        // -0.5,   0.0,  0.5,  0.0;
        //  0.5,  -1.0,  0.5,  0.0;
        // -1/6,  0.5, -0.5,   1/6;

        // b = ts * M
        let b0 = (1.0 - 3.0 * t1 + 3.0 * t2 + t3) / 6.0;
        let b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
        let b2 = (1.0 + 3.0 * t1 + 3.0 * t2 - 3.0 * t3) / 6.0;
        let b3 = t3 / 6.0;
        [b0, b1, b2, b3]
    }

    // Returns the curve segment corresponding to the global t value.
    // Precondition: 1 <= t < self.0.len() - 2
    fn segment(&self, t: f32) -> (f32, [Pt<N, Sp>; 4]) {
        debug_assert!(1.0 <= t && t < self.0.len() as f32 - 2.0);
        let i = t as usize;
        let u = t - i as f32;
        let pts = self.0[i - 1..i + 3]
            .try_into()
            .expect("slice has four elements");
        (u, pts)
    }
}

fn approx<T: Affine<Diff: Lerp> + Lerp>(
    spline: &impl Parametric<T>,
    a: f32,
    b: f32,
    max_dep: u32,
    halt: &impl Fn(&T::Diff) -> bool,
) -> Vec<T> {
    let mut res = Vec::with_capacity(32);
    do_approx(spline, a, b, max_dep, halt, &mut res);
    res
}

fn do_approx<T: Affine<Diff: Lerp> + Lerp>(
    spline: &impl Parametric<T>,
    a: f32,
    b: f32,
    max_dep: u32,
    halt: &impl Fn(&T::Diff) -> bool,
    accum: &mut Vec<T>,
) {
    let mid = a.midpoint(b);

    let ap = spline.eval(a);
    let bp = spline.eval(b);

    let real = spline.eval(mid);
    let approx = ap.midpoint(&bp);

    if max_dep == 0 || (halt(&(real.sub(&approx)))) {
        accum.push(ap);
    } else {
        do_approx(spline, a, mid, max_dep - 1, halt, accum);
        do_approx(spline, mid, b, max_dep - 1, halt, accum);
    }
}

//
// Local trait impls
//

impl<T> Parametric<T> for CubicBezier<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    fn eval(&self, t: f32) -> T {
        self.fast_eval(t)
    }
}

impl<T> Parametric<T> for BezierSpline<T>
where
    T: Affine + Clone + Debug,
    T::Diff: Linear<Scalar = f32> + Clone + Debug,
{
    fn eval(&self, t: f32) -> T {
        self.eval(t)
    }
}

impl<const N: usize, Sp> Parametric<Point<[f32; N], Sp>>
    for HermiteSpline<Point<[f32; N], Sp>>
where
    Pt<N, Sp>: Affine<Diff = Vector<[f32; N], Sp>> + Lerp,
    Sp: Debug + Default,
{
    fn eval(&self, t: f32) -> Pt<N, Sp> {
        self.eval(t)
    }
}

impl<const N: usize, Sp> Parametric<Point<[f32; N], Sp>>
    for CatmullRomSpline<Point<[f32; N], Sp>>
where
    Pt<N, Sp>: Affine<Diff = Vector<[f32; N], Sp>> + Lerp,
    Sp: Debug + Default,
{
    fn eval(&self, t: f32) -> Pt<N, Sp> {
        self.eval(t)
    }
}

impl<const N: usize, Sp> Parametric<Point<[f32; N], Sp>>
    for BSpline<Point<[f32; N], Sp>>
where
    Pt<N, Sp>: Affine<Diff = Vector<[f32; N], Sp>> + Lerp,
    Sp: Debug + Default,
{
    fn eval(&self, t: f32) -> Pt<N, Sp> {
        self.eval(t)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::{Parametric, Point2, Vec2, pt2, vec2};
    use alloc::vec;

    use super::*;

    #[test]
    fn smoothstep_test() {
        assert_eq!(0.0, smoothstep(-10.0));
        assert_eq!(0.0, smoothstep(0.0));

        assert_eq!(0.15625, smoothstep(0.25));
        assert_eq!(0.50000, smoothstep(0.5));
        assert_eq!(0.84375, smoothstep(0.75));

        assert_eq!(1.0, smoothstep(1.0));
        assert_eq!(1.0, smoothstep(10.0));

        assert_eq!(0.15625, smoothstep.eval(0.25));
    }

    #[test]
    fn smootherstep_test() {
        assert_eq!(0.0, smootherstep(-10.0));
        assert_eq!(0.0, smootherstep(0.0));

        assert_eq!(0.103515625, smootherstep(0.25));
        assert_eq!(0.5, smootherstep(0.5));
        assert_eq!(0.8964844, smootherstep(0.75));

        assert_eq!(1.0, smootherstep(1.0));
        assert_eq!(1.0, smootherstep(10.0));

        assert_eq!(0.103515625, smootherstep.eval(0.25));
    }

    #[test]
    fn bezier_spline_eval_eq_fast_eval() {
        let b: CubicBezier<Vec2> = CubicBezier(
            [[0.0, 0.0], [0.0, 2.0], [1.0, -1.0], [1.0, 1.0]].map(Vec2::from),
        );
        for i in 0..11 {
            let t = i as f32 / 10.0;
            let (v, u) = (b.eval(t), b.fast_eval(t));
            assert_approx_eq!(v.x(), u.x(), eps = 1e-5);
            assert_approx_eq!(v.y(), u.y(), eps = 1e-5);
        }
    }

    #[test]
    fn bezier_spline_eval_1d() {
        let b = CubicBezier([0.0, 2.0, -1.0, 1.0]);

        assert_eq!(b.eval(-1.0), 0.0);
        assert_eq!(b.eval(0.00), 0.0);
        assert_eq!(b.eval(0.25), 0.71875);
        assert_eq!(b.eval(0.50), 0.5);
        assert_eq!(b.eval(0.75), 0.28125);
        assert_eq!(b.eval(1.00), 1.0);
        assert_eq!(b.eval(2.00), 1.0);
    }

    #[test]
    fn bezier_spline_eval_2d_vec() {
        let b = CubicBezier(
            [[0.0, 0.0], [0.0, 2.0], [1.0, -1.0], [1.0, 1.0]]
                .map(Vec2::<()>::from),
        );

        assert_eq!(b.eval(-1.0), vec2(0.0, 0.0));
        assert_eq!(b.eval(0.00), vec2(0.0, 0.0));
        assert_eq!(b.eval(0.25), vec2(0.15625, 0.71875));
        assert_eq!(b.eval(0.50), vec2(0.5, 0.5));
        assert_eq!(b.eval(0.75), vec2(0.84375, 0.281250));
        assert_eq!(b.eval(1.00), vec2(1.0, 1.0));
        assert_eq!(b.eval(2.00), vec2(1.0, 1.0));
    }

    #[test]
    fn bezier_spline_eval_2d_point() {
        let b = CubicBezier(
            [[0.0, 0.0], [0.0, 2.0], [1.0, -1.0], [1.0, 1.0]]
                .map(Point2::<()>::from),
        );

        assert_eq!(b.eval(-1.0), pt2(0.0, 0.0));
        assert_eq!(b.eval(0.00), pt2(0.0, 0.0));
        assert_eq!(b.eval(0.25), pt2(0.15625, 0.71875));
        assert_eq!(b.eval(0.50), pt2(0.5, 0.5));
        assert_eq!(b.eval(0.75), pt2(0.84375, 0.281250));
        assert_eq!(b.eval(1.00), pt2(1.0, 1.0));
        assert_eq!(b.eval(2.00), pt2(1.0, 1.0));
    }

    #[test]
    fn bezier_spline_tangent_1d() {
        let b = CubicBezier([0.0, 2.0, -1.0, 1.0]);

        assert_eq!(b.tangent(-1.0), 6.0);
        assert_eq!(b.tangent(0.00), 6.0);
        assert_eq!(b.tangent(0.25), 0.375);
        assert_eq!(b.tangent(0.50), -1.5);
        assert_eq!(b.tangent(0.75), 0.375);
        assert_eq!(b.tangent(1.00), 6.0);
        assert_eq!(b.tangent(2.00), 6.0);
    }

    #[test]
    fn bezier_spline_tangent_2d() {
        let b = CubicBezier(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
                .map(Point2::<()>::from),
        );

        assert_eq!(b.tangent(-1.0), vec2(0.0, 3.0),);
        assert_eq!(b.tangent(0.0), vec2(0.0, 3.0),);
        assert_eq!(b.tangent(0.25), vec2(1.125, 0.75),);
        assert_eq!(b.tangent(0.5), vec2(1.5, 0.0),);
        assert_eq!(b.tangent(0.75), vec2(1.125, 0.75),);
        assert_eq!(b.tangent(1.0), vec2(0.0, 3.0),);
        assert_eq!(b.tangent(2.0), vec2(0.0, 3.0),);
    }

    #[test]
    fn bezier_spline_eval() {
        let c = BezierSpline(vec![0.0, 0.8, 0.9, 1.0, 0.6, 0.5, 0.5]);
        assert_eq!(c.eval(-1.0), 0.0);
        assert_eq!(c.eval(0.0), 0.0);
        assert_approx_eq!(c.eval(0.25), 0.7625);
        assert_eq!(c.eval(0.5), 1.0);
        assert_eq!(c.eval(0.75), 0.6);
        assert_eq!(c.eval(1.0), 0.5);
        assert_eq!(c.eval(2.0), 0.5);
    }

    #[test]
    fn hermite_spline() {
        let h = HermiteSpline::new([
            Ray(pt2::<_, ()>(0.0, 0.0), vec2(1.0, 0.0)),
            Ray(pt2(1.0, 1.0), vec2(1.0, 0.0)),
        ]);

        assert_eq!(h.eval(0.0), pt2(0.0, 0.0));
        assert_approx_eq!(h.eval(0.2), pt2(0.2, 0.104));
        assert_eq!(h.eval(0.5), pt2(0.5, 0.5));
        assert_approx_eq!(h.eval(0.8), pt2(0.8, 0.896));
        assert_eq!(h.eval(1.0), pt2(1.0, 1.0));
    }
}
