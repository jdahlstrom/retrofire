//! Bézier curves and splines.
use alloc::vec::Vec;
use core::{array::from_fn, fmt::Debug, marker::PhantomData};

use crate::geom::{Polyline, Ray};
use crate::mat;

use super::{Affine, Lerp, Linear, Mat4, Parametric, Vector};

/// A cubic Bézier curve, defined by four control points.
///
/// TODO More info about Béziers
///
/// ```text
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
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CubicBezier<T>(pub [T; 4]);

/// A cubic Hermite curve, defined by two control points and the velocity
/// vectors at the control points.
///
/// Hermite curves are closely related to Bézier curves, via the identity
///
/// H(p0, d0, p1, d1) = B(p0, p0 + d0/3, p1 - d1/3, p1),
///
/// or, equivalently,
///
/// B(p0, p1, p2, p3) = H(p0, 3(p1 - p0), p3, 3(p2 - p3)).
///
/// ```text
///  d0
///   ^
///    \
///     \        ____
///      \    _-´    `--_                       p1
///       \  /           `-_                     \
///        \|               `-_                  |\
///         \                  `-__             /  \
///          p0                     `---_____--´    \
///                                                  \
///                                                   \
///                                                    v
///                                                    d1
/// ```
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CubicHermite<P, D>(pub [P; 2], pub [D; 2]);

/// A piecewise curve composed of concatenated [cubic Bézier curves][CubicBezier].
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BezierSpline<T>(Vec<T>);

/// A piecewise curve composed of concatenated [cubic Hermite curves][CubicHermite].
#[derive(Debug, Clone, Eq, PartialEq)]
// HACK: The PhantomData field only exists to force the derive impls
//       to include the correct `T::Diff: Trait` bounds
pub struct HermiteSpline<T: Affine>(Vec<Ray<T>>, PhantomData<T::Diff>);

/// A piecewise curve composed of concatenated cubic curves.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CatmullRomSpline<T>(Vec<T>);

/// A piecewise curve composed of concatenated cubic curves.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BSpline<T>(Vec<T>);

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

/// Approximates a curve as a chain of line segments.
///
/// Recursively subdivides the curve into two half-curves, stopping once
/// the approximation error is less than `error`.
///
/// # Examples
/// ```
/// use retrofire_core::math::{BezierSpline, Point2, pt2};
/// use retrofire_core::math::spline::approximate;
///
/// let curve = BezierSpline::<Point2>::new(
///     [pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(1.0, 0.0)]
/// );
/// // Find an approximation with error less than 0.01
/// let approx = approximate(&curve, 0.01);
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
pub fn approximate<T, Sp, const DIM: usize>(
    curve: &impl Parametric<T>,
    error: f32,
) -> Polyline<T>
where
    T: Affine<Diff = Vector<[f32; DIM], Sp>>,
{
    assert!(error > 0.0);
    approximate_with(curve, &|e: &T::Diff| e.len_sqr() < error * error)
}

/// Approximates a curve as a chain of line segments.
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
/// use retrofire_core::math::spline::approximate_with;
///
/// let curve = BezierSpline::<Point2>::new(
///     [pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(1.0, 0.0)]
/// );
/// // Find an approximation with error less than 0.01
/// let approx = approximate_with(&curve, |err| err.len_sqr() < 0.01 * 0.01);
///
/// // Euclidean length of the polyline approximation
/// assert_eq!(approx.len(), 1.9969313);
///
/// // Number of line segments used by the approximation
/// assert_eq!(approx.0.len(), 17);
/// ```
pub fn approximate_with<T: Affine<Diff: Linear<Scalar = f32>>>(
    curve: &impl Parametric<T>,
    halt: impl Fn(&T::Diff) -> bool,
) -> Polyline<T> {
    let mut res = Vec::new();
    do_approx(curve, 0.0, 1.0, 10, &halt, &mut res);
    res.push(curve.eval(1.0));
    Polyline(res)
}

fn do_approx<T: Affine<Diff: Linear<Scalar = f32>>>(
    c: &impl Parametric<T>,
    a: f32,
    b: f32,
    max_dep: u32,
    halt: &impl Fn(&T::Diff) -> bool,
    accum: &mut Vec<T>,
) {
    let mid = a.midpoint(b);

    let ap = c.eval(a);
    let bp = c.eval(b);

    let real = c.eval(mid);
    let approx = ap.add(&bp.sub(&ap).mul(0.5));

    if max_dep == 0 || halt(&real.sub(&approx)) {
        accum.push(ap);
    } else {
        do_approx(c, a, mid, max_dep - 1, halt, accum);
        do_approx(c, mid, b, max_dep - 1, halt, accum);
    }
}

//
// Inherent impls
//

impl<T: Lerp> CubicBezier<T> {
    /// Returns the point of `self` at the given *t* value. Uses
    /// [De Casteljau's algorithm][1].
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    ///
    /// [1]: https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
    pub fn eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = &self.0;
        let p01 = p0.lerp(p1, t);
        let p12 = p1.lerp(p2, t);
        let p23 = p2.lerp(p3, t);
        p01.lerp(&p12, t).lerp(&p12.lerp(&p23, t), t)
    }
}

impl<T> CubicBezier<T>
where
    T: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
    /// Returns the point of `self` at the given *t* value.
    ///
    /// Directly evaluates the cubic polynomial. Faster but possibly less
    /// numerically stable than [`Self::eval`].  Values of *t* outside the
    /// interval [0, 1] are accepted and extrapolate the curve beyond the
    /// control points.
    pub fn fast_eval(&self, t: f32) -> T {
        // Add a linear combination of the three coefficients
        // to `p0` to get the result
        let p0 = &self.0[0];
        let [co3, co2, co1] = self.coefficients();
        p0.add(&co3.mul(t).add(&co2).mul(t).add(&co1).mul(t))
    }

    /// Returns the velocity vector of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> T::Diff {
        let [p0, p1, p2, p3] = &self.0;

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

impl<P> CubicHermite<P, P::Diff>
where
    P: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
    // Characteristic matrix M
    //  1.0,  0.0,  0.0,  0.0;
    //  0.0,  1.0,  0.0,  0.0;
    // -3.0, -2.0,  3.0, -1.0;
    //  2.0,  1.0, -2.0,  1.0;

    /// Returns the point of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn eval(&self, t: f32) -> P {
        let Self([p0, p1], [d0, d1]) = self;
        let [_0, t1, t2, t3] = [1.0, t, t * t, t * t * t];

        // b = ts * M
        let _0 = 1.0 - 3.0 * t2 + 2.0 * t3; // = 1 - b2
        let b1 = t1 - 2.0 * t2 + t3;
        let b2 = 3.0 * t2 - 2.0 * t3;
        let b3 = -t2 + t3;

        // H(t) = b * P

        //   b0 * p0 + b1 * d0 + b2 * p1 + b3 * d1
        // = b0 * p0 + b2 * p1 // Affine: b0 + b2 = 1: lerp
        // + b1 * d0 + b3 * d1 // Linear

        //   b0 * p0 + b2 * p1
        // = (1 - b2) * p0 + b2 * p1
        // = p0 + b2 * (p1 - p0)

        p0.add(&p1.sub(&p0).mul(b2)) // Affine part
            .add(&d0.mul(b1).add(&d1.mul(b3))) // Linear part
    }

    /// Returns the velocity vector of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> P::Diff {
        let Self([p0, p1], [d0, d1]) = self;
        // Derivatives of the powers of t
        let [_0, _1, t2, t3] = [0.0, 1.0, 2.0 * t, 3.0 * t * t];

        // b = ts * M
        let _0 = 0.0 - 3.0 * t2 + 2.0 * t3; // = -b2
        let b1 = 1.0 - 2.0 * t2 + t3;
        let b2 = 3.0 * t2 - 2.0 * t3;
        let b3 = -t2 + t3;

        // H(t) = b * P

        //   b0 * p0 + b1 * d0 + b2 * p1 + b3 * d1
        // = b0 * p0 + b2 * p1 // b0 = -b2
        // + b1 * d0 + b3 * d1 // Linear

        //   b0 * p0 + b2 * p1
        // = -b2 * p0 + b2 * p1
        // = b2 * p1 - b2 * p0
        // = b2 * (p1 - p0)

        // Only vectors as expected:
        // b2·(p1 - p0) + b1·d0 + b3·d1
        p1.sub(&p0)
            .mul(b2)
            .add(&d0.mul(b1).add(&d1.mul(b3)))
    }
}

impl<T> BezierSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    /// Creates a Bézier spline from the given control points.
    ///
    /// The number of elements in `pts` must be 3*n* + 1 for some positive integer
    /// *n*. Consecutive points in `pts` make up Bézier curves such that:
    /// * (p0, p1, p2, p3) define the first curve,
    /// * (p3, p4, p5, p6) define the second curve,
    ///
    /// and so on.
    ///
    /// # Panics
    /// If the number of points *n* < 4 or if *n* ≠ 1 (mod 3).
    pub fn new(pts: impl IntoIterator<Item = T>) -> Self {
        let pts = pts.into_iter().collect::<Vec<_>>();
        let len = pts.len();
        assert!(
            len >= 4 && len % 3 == 1,
            "length must be 3n+1 for some integer n > 0, was {len}",
        );
        Self(pts)
    }

    /// Constructs a Bézier spline from (position, tangent) pairs.
    ///
    /// For each pair of consecutive rays (P, **v**) and (Q, **u**), the result
    /// contains one cubic Bézier curve segment with control points (P, P +
    /// **v**, Q - **u**, Q).
    ///
    /// # Panics
    /// If the number of rays < 2.
    pub fn from_rays(rays: impl IntoIterator<Item = Ray<T>>) -> Self {
        let pts: Vec<_> = rays
            .into_iter()
            .flat_map(|Ray(p, d)| [p.add(&d.neg()), p.clone(), p.add(&d)])
            .collect();
        Self::new(pts[1..pts.len() - 1].into_iter().cloned())
    }

    /// Returns the point of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn eval(&self, t: f32) -> T {
        let (u, seg) = self.segment(t);
        seg.fast_eval(u)
    }

    /// Returns the velocity vector of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> T::Diff {
        let (u, seg) = self.segment(t);
        seg.velocity(u)
    }

    /// Returns the list of control points of `self`.
    pub fn control_points(&self) -> &[T] {
        &self.0
    }

    /// Returns the spline segment and local *t* value corresponding to
    /// the given global *t* value.
    ///
    fn segment(&self, t: f32) -> (f32, CubicBezier<T>) {
        // Consecutive segments share an endpoint:
        // [B0  B1  B2  B3]
        //             [B3  B4  B5  B6]
        //                         [B6  B7  B8  B9]
        //                                     [B9 ...
        // If the number of segs is n, the number of control points is 3n + 1,
        // thus if the number of points is l, the number of segs is (l - 1) / 3.
        let num_segs = (self.0.len() - 1) / 3;
        // Rescale from [0, 1] to [0, num_segs]
        let t = t * num_segs as f32;
        use super::float::f32;
        // Calculate the segment index.
        let seg_i = (t as usize).min(num_segs - 1);
        // The leftover part is the local t value. This is the fractional part
        // for 0 <= t < segs. t = segs maps to u = 1 of the last subsegment.
        // Values of t < 0 or t > segs result in u < 0 or u > 1 and extrapolate
        // beyond the first or last subsegment, respectively.
        let u = t - seg_i as f32;
        // Index of the first control point of the segment
        let i = 3 * seg_i;
        let seg = from_fn(|j| self.0[i + j].clone());
        (u, CubicBezier(seg))
    }
}

impl<T> HermiteSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    /// Creates a new Hermite spline from a sequence of rays.
    ///
    /// Each ray (P<sub>i</sub>, **v**<sub>i</sub>) makes up a point
    /// P<sub>i</sub> on the curve and the velocity (velocity) vector
    /// **v**<sub>i</sub> of the curve at that point. Thus,
    /// the ray lies tangent to the curve at point P<sub>i</sub>.
    ///
    /// # Panics
    /// If `rays` has fewer than two items.
    pub fn new(rays: impl IntoIterator<Item = Ray<T>>) -> Self {
        let rays: Vec<_> = rays.into_iter().collect();
        assert!(
            rays.len() >= 2,
            "a Hermite spline requires at least two points and two vectors"
        );
        Self(rays, PhantomData)
    }

    /// Returns the subsegment and local *t* value corresponding to the given
    /// global *t* value.
    fn segment(&self, t: f32) -> (f32, CubicHermite<T, T::Diff>) {
        // Scale from [0, 1] to [0, len-1]
        let t = t * (self.0.len() - 1) as f32;
        // Calculate the index of the subsegment. There are len-1 subsegments:
        // (0, 1), (1,2), ..., (len-2, len-1).
        let i = (t as usize).min(self.0.len() - 2);
        // The leftover part is the local t value. This is the fractional part
        // for 0 <= t < len-1. t = len-1 maps to u = 1 of the last subsegment.
        // Values of t < 0 or t > len-1 result in u < 0 or u > 1 and extrapolate
        // beyond the first or last subsegment, respectively.
        let u = t - i as f32;
        // Ok: i <= self.0.len() - 2
        let Ray(p0, d0) = self.0[i].clone();
        let Ray(p1, d1) = self.0[i + 1].clone();
        (u, CubicHermite([p0, p1], [d0, d1]))
    }

    /// Returns the point of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn eval(&self, t: f32) -> T {
        let (u, seg) = self.segment(t);
        seg.eval(u)
    }

    /// Returns the velocity vector of `self`
    /// at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> T::Diff {
        let (u, seg) = self.segment(t);
        seg.velocity(u)
    }
}

impl<T> CatmullRomSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
    const _CHAR_MAT: Mat4 = mat![
         0.0,  1.0,  0.0,  0.0;
        -0.5,  0.0,  0.5,  0.0;
         1.0, -2.5,  2.0, -0.5;
        -0.5,  1.5, -1.5,  0.5;
    ];

    pub fn new(pts: impl IntoIterator<Item = T>) -> Self {
        let pts: Vec<_> = pts.into_iter().collect();
        assert!(
            pts.len() >= 4,
            "a Catmull–Rom spline requires at least four points"
        );
        Self(pts)
    }

    /// Returns the point of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn eval(&self, t: f32) -> T {
        let (t, [p0, p1, p2, p3]) = crb_segment(&self.0, t);
        let [_0, t1, t2, t3] = [1.0, t, t * t, t * t * t];

        let _0 = -t1 + 2.0 * t2 - t3;
        let b1 = 2.0 - 5.0 * t2 + 3.0 * t3;
        let b2 = t1 + 4.0 * t2 - 3.0 * t3;
        let b3 = -t2 + t3;

        // b0 + b1 + b2 + b3 = 1
        // b0 = 1 - b1 - b2 - b3

        //    b0·P0 + b1·P1 + b2·P2 + b3·P3
        // = (1 - b1 - b2 - b3)·P0 + b1·P1 + b2·P2 + b3·P3
        // = P0 - b1·P0 - b2·P0 - b3·P0 + b1·P1 + b2·P2 + b3·P3
        // = P0 + b1·(P1 - P0) + b2·(P2 - P0) + b3·(P3 - P0)

        let v01 = p1.sub(&p0).mul(b1);
        let v02 = &p2.sub(&p0).mul(b2);
        let v03 = p3.sub(&p0).mul(b3);
        p0.add(&v01.add(&v02).add(&v03).mul(1.0 / 2.0))
    }

    /// Returns the velocity vector of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> T::Diff {
        let (t, [p0, p1, p2, p3]) = crb_segment(&self.0, t);
        let [_0, _1, t2, t3] = [0.0, 1.0, 2.0 * t, 3.0 * t * t];

        // ⎛ b0 ⎞        ⎛ -1 + 2·t2 -   t3 ⎞
        // ⎜ b1 ⎟ =  1/2 ⎜    - 5·t2 + 3·t3 ⎟
        // ⎜ b2 ⎟        ⎜  1 + 4·t2 - 3·t3 ⎟
        // ⎝ b3 ⎠        ⎝    -   t2 +   t3 ⎠

        let b1 = -5.0 * t2 + 3.0 * t3;
        let b2 = 1.0 + 4.0 * t2 - 3.0 * t3;
        let b3 = -t2 + t3;

        // b0 + b1 + b2 + b3 = 0 <=> b0 = -(b1 + b2 + b3)
        //
        //    b0·P0 + b1·P1 + b2·P2 + b3·P3
        // = -(b1 + b2 + b3)·P0 + b1·P1 + b2·P2 + b3·P3
        // = b1·P1 + b2·P2 + b3·P3 - b1·P0 - b2·P0 - b3·P0
        // = b1·(P1 - P0) + b2·(P2 - P0) + b3·(P3 - P0)

        let v01 = p1.sub(&p0).mul(b1);
        let v02 = p2.sub(&p0).mul(b2);
        let v03 = p3.sub(&p0).mul(b3);
        v01.add(&v02).add(&v03).mul(1.0 / 2.0)
    }
}

impl<T> BSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
    const _CHAR_MAT: Mat4 = {
        const _1_6: f32 = 1.0 / 6.0;
        const _2_3: f32 = 2.0 / 3.0;
        mat![
            _1_6, _2_3, _1_6,  0.0;
            -0.5,  0.0,  0.5,  0.0;
             0.5, -1.0,  0.5,  0.0;
           -_1_6,  0.5, -0.5, _1_6;
        ]
    };

    pub fn new(pts: impl IntoIterator<Item = T>) -> Self {
        let pts: Vec<_> = pts.into_iter().collect();
        assert!(pts.len() >= 4, "a B-spline requires at least four points");
        Self(pts)
    }

    /// Returns the point of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn eval(&self, t: f32) -> T {
        let (t, [p0, p1, p2, p3]) = crb_segment(&self.0, t);
        let [_0, t1, t2, t3] = [1.0, t, t * t, t * t * t];

        let _0 = 1.0 - 3.0 * t1 + 3.0 * t2 + t3;
        let b1 = 4.0 - 6.0 * t2 + 3.0 * t3;
        let b2 = 1.0 + 3.0 * t1 + 3.0 * t2 - 3.0 * t3;
        let b3 = t3;

        let v01 = p1.sub(&p0).mul(b1);
        let v02 = p2.sub(&p0).mul(b2);
        let v03 = p3.sub(&p0).mul(b3);
        p0.add(&v01.add(&v02).add(&v03).mul(1.0 / 6.0))
    }

    /// Returns the velocity vector of `self` at the given *t* value.
    ///
    /// Values of *t* outside the interval [0, 1] are accepted and extrapolate
    /// the curve beyond the control points.
    pub fn velocity(&self, t: f32) -> T::Diff {
        let (t, [p0, p1, p2, p3]) = crb_segment(&self.0, t);
        let [_0, _1, t2, t3] = [0.0, 1.0, 2.0 * t, 3.0 * t * t];

        // ⎛ b0 ⎞        ⎛ -3 + 3·t2 -   t3 ⎞
        // ⎜ b1 ⎟ = 1/6  ⎜    - 6·t2 + 3·t3 ⎟
        // ⎜ b2 ⎟        ⎜  3 + 3·t2 - 3·t3 ⎟
        // ⎝ b3 ⎠        ⎝               t3 ⎠

        let b1 = -6.0 * t2 + 3.0 * t3;
        let b2 = 3.0 + 3.0 * t2 - 3.0 * t3;
        let b3 = t3;

        // b0 + b1 + b2 + b3 = 0 <=> b0 = -(b1 + b2 + b3)
        //
        //   b0·P0 + b1·P1 + b2·P2 + b3·P3
        // = -(b1 + b2 + b3)·P0 + b1·P1 + b2·P2 + b3·P3
        // = b1·P1 + b2·P2 + b3·P3 - b1·P0 - b2·P0 - b3·P0
        // = b1·(P1 - P0) + b2·(P2 - P0) + b3·(P3 - P0)

        let v01 = p1.sub(&p0).mul(b1);
        let v02 = p2.sub(&p0).mul(b2);
        let v03 = p3.sub(&p0).mul(b3);
        v01.add(&v02).add(&v03).mul(1.0 / 6.0)
    }
}

/// Returns the curve segment and local *t* value corresponding to
/// the global *t* value of a Catmull–Rom or B-spline.
fn crb_segment<T: Clone>(pts: &[T], t: f32) -> (f32, &[T; 4]) {
    let t = 1.0 + t * (pts.len() as f32 - 3.0);

    let i = (t as usize).clamp(1, pts.len() - 3);
    let u = t - i as f32;
    let pts = pts[i - 1..i + 3] // OK: 1 <= i < len - 3
        .try_into()
        .expect("3 - (-1) = 4");
    (u, pts)
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
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    fn eval(&self, t: f32) -> T {
        self.eval(t)
    }
}

impl<T> Parametric<T> for HermiteSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    fn eval(&self, t: f32) -> T {
        self.eval(t)
    }
}

impl<T> Parametric<T> for CatmullRomSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    fn eval(&self, t: f32) -> T {
        self.eval(t)
    }
}

impl<T> Parametric<T> for BSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    fn eval(&self, t: f32) -> T {
        self.eval(t)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::{Parametric, Point2, Vec2, pt2, vec2};

    use super::*;

    const TEST_T_VALS: [f32; 7] = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0];

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
    fn cubic_bezier_eval_eq_fast_eval() {
        let b = CubicBezier(
            [[0.0, 0.0], [0.0, 2.0], [1.0, -1.0], [1.0, 1.0]].map(<Vec2>::from),
        );
        for i in 0..11 {
            let t = i as f32 / 10.0;
            let (v, u) = (b.eval(t), b.fast_eval(t));
            assert_approx_eq!(v.x(), u.x(), eps = 1e-5);
            assert_approx_eq!(v.y(), u.y(), eps = 1e-5);
        }
    }

    #[test]
    fn cubic_bezier_f32_eval() {
        let b = CubicBezier([0.0, 2.0, -1.0, 1.0]);

        let expected = [-31.0, 0.0, 0.71875, 0.5, 0.28125, 1.0, 32.0];
        let actual = TEST_T_VALS.map(|t| b.eval(t));

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_bezier_vec2_eval() {
        let b = CubicBezier::<Vec2>([
            vec2(0.0, 0.0),
            vec2(0.0, 2.0),
            vec2(1.0, -1.0),
            vec2(1.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [5.0, -31.0], [0.0, 0.0], [0.15625, 0.71875], [0.5, 0.5],
            [0.84375, 0.281250], [1.0, 1.0], [-4.0, 32.0],
        ];
        let actual = TEST_T_VALS.map(|t| b.eval(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_bezier_point2_eval() {
        let b = CubicBezier::<Point2>([
            pt2(0.0, 0.0),
            pt2(0.0, 2.0),
            pt2(1.0, -1.0),
            pt2(1.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [5.0, -31.0], [0.0, 0.0], [0.15625, 0.71875], [0.5, 0.5],
            [0.84375, 0.281250], [1.0, 1.0], [-4.0, 32.0],
        ];
        let actual = TEST_T_VALS.map(|t| b.eval(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_bezier_f32_velocity() {
        let b = CubicBezier([0.0, 2.0, -1.0, 1.0]);

        let expected = [66.0, 6.0, 0.375, -1.5, 0.375, 6.0, 66.0];
        let actual = TEST_T_VALS.map(|t| b.velocity(t));

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_bezier_point2_velocity() {
        #[rustfmt::skip]
        let b = CubicBezier::<Point2>([
            pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 0.0), pt2(1.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [-12.0, 27.0],  [0.0, 3.0], [1.125, 0.75], [1.5, 0.0],
            [1.125, 0.75], [0.0, 3.0], [-12.0, 27.0],
        ];
        let actual = TEST_T_VALS.map(|t| b.velocity(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_hermite_f32_eval() {
        let h = CubicHermite([0.0, 0.0], [2.0, -2.0]);

        let expected = [-4.0, 0.0, 0.375, 0.5, 0.375, 0.0, -4.0];
        let actual = TEST_T_VALS.map(|t| h.eval(t));

        assert_eq!(expected, actual);
    }

    #[test]
    fn cubic_hermite_f32_velocity() {
        let h = CubicHermite([0.0, 0.0], [1.0, -1.0]);

        let expected = [3.0, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0];
        let actual = TEST_T_VALS.map(|t| h.velocity(t));

        assert_eq!(expected, actual);
    }

    #[test]
    fn bezier_hermite_equivalence() {
        let [p0, p1, p2, p3] =
            [pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 0.0), pt2(1.0, 1.0)];
        let b = CubicBezier::<Point2>([p0, p1, p2, p3]);
        let h = CubicHermite::<Point2, _>(
            [p0, p3],
            [3.0 * (p1 - p0), 3.0 * (p3 - p2)],
        );
        assert_eq!(
            TEST_T_VALS.map(|t| b.eval(t).0),
            TEST_T_VALS.map(|t| h.eval(t).0)
        );
    }

    #[test]
    fn bezier_spline_segment() {
        let b = BezierSpline::new([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]);

        let first = CubicBezier([0.0, 0.2, 0.4, 0.5]);
        let second = CubicBezier([0.5, 0.6, 0.8, 1.0]);

        assert_eq!(b.segment(-1.0), (-2.0, first));
        assert_eq!(b.segment(0.0), (0.0, first));
        assert_eq!(b.segment(0.5), (0.0, second));
        assert_eq!(b.segment(1.0), (1.0, second));
        assert_eq!(b.segment(2.0), (3.0, second));
    }

    #[test]
    fn bezier_spline_f32_eval() {
        let b = BezierSpline::new([0.0, 0.8, 0.9, 1.0, 0.6, 0.5, 0.5]);

        let expected = [-18.8, 0.0, 0.7625, 1.0, 0.6, 0.5, 0.1];
        let actual = TEST_T_VALS.map(|t| b.eval(t));

        assert_approx_eq!(expected, actual);
    }

    #[test]
    fn bezier_spline_point2_from_rays() {
        #[rustfmt::skip]
        let expected = BezierSpline::<Point2>::new([
            pt2(0.0, 0.0),pt2(0.0, 2.0),pt2(1.0, -1.0),pt2(1.0, 1.0)
        ]);
        let actual = BezierSpline::<Point2>::from_rays([
            Ray(pt2(0.0, 0.0), vec2(0.0, 2.0)),
            Ray(pt2(1.0, 1.0), vec2(0.0, 2.0)),
        ]);

        assert_eq!(
            TEST_T_VALS.map(|t| expected.eval(t)),
            TEST_T_VALS.map(|t| actual.eval(t))
        );
    }

    #[test]
    fn bezier_spline_point2_velocity() {
        #[rustfmt::skip]
        let b = BezierSpline::<Point2>::new([
            pt2(0.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 0.0), pt2(1.0, 1.0),
        ]);
        #[rustfmt::skip]
        let expected = [
            vec2(-12.0, 27.0), vec2(0.0, 3.0), vec2(1.125, 0.75), vec2(1.5, 0.0),
            vec2(1.125, 0.75), vec2(0.0, 3.0), vec2(-12.0, 27.0),
        ];
        let actual = TEST_T_VALS.map(|t| b.velocity(t));

        assert_eq!(expected, actual);
    }

    #[test]
    fn hermite_spline_segment() {
        let b =
            HermiteSpline::new([Ray(0.0, 0.4), Ray(0.5, 0.1), Ray(1.0, 0.6)]);

        let first = CubicHermite([0.0, 0.5], [0.4, 0.1]);
        let second = CubicHermite([0.5, 1.0], [0.1, 0.6]);

        assert_eq!(b.segment(-1.0), (-2.0, first));
        assert_eq!(b.segment(0.0), (0.0, first));
        assert_eq!(b.segment(0.5), (0.0, second));
        assert_eq!(b.segment(1.0), (1.0, second));
        assert_eq!(b.segment(2.0), (3.0, second));
    }

    #[test]
    fn hermite_spline_point2_eval() {
        let h = HermiteSpline::<Point2>::new([
            Ray(pt2(0.0, 0.0), vec2(1.0, 0.0)),
            Ray(pt2(1.0, 1.0), vec2(1.0, 0.0)),
        ]);

        #[rustfmt::skip]
        let expected = [
            [-1.0, 5.0], [0.0, 0.0], [0.25, 0.15625], [0.5, 0.5],
            [0.75, 0.84375], [1.0, 1.0], [2.0, -4.0]
        ];
        let actual = TEST_T_VALS.map(|t| h.eval(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn hermite_spline_point2_velocity() {
        let h = HermiteSpline::<Point2>::new([
            Ray(pt2(0.0, 0.0), vec2(1.0, 0.0)),
            Ray(pt2(1.0, 1.0), vec2(1.0, 0.0)),
        ]);

        #[rustfmt::skip]
        let expected = [
            [1.0, -12.0], [1.0, 0.0], [1.0, 1.125], [1.0, 1.5],
            [1.0, 1.125], [1.0, 0.0], [1.0, -12.0]
        ];
        let actual = TEST_T_VALS.map(|t| h.velocity(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn catmull_rom_spline_point2_eval() {
        #[rustfmt::skip]
        let c = CatmullRomSpline::<Point2>::new([
            pt2(-1.0, 0.0), pt2(0.0, 0.0), pt2(1.0, 0.0),
            pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(2.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [33.0, -18.0], [0.0, 0.0], [0.890625, -0.0703125], [0.5, 0.5],
            [0.109375, 1.0703125], [1.0, 1.0], [-32.0, 19.0]
        ];
        let actual = TEST_T_VALS.map(|t| c.eval(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn catmull_rom_spline_point2_gradient() {
        #[rustfmt::skip]
        let c = CatmullRomSpline::<Point2>::new([
            pt2(-1.0, 0.0), pt2(0.0, 0.0), pt2(1.0, 0.0),
            pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(2.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [-32.0, 16.5], [1.0, 0.0], [0.8125, 0.09375], [-1.5, 1.25],
            [0.8125, 0.09375], [1.0, 0.0], [-32.0, 16.5]
        ];
        let actual = TEST_T_VALS.map(|t| c.velocity(t).0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn b_spline_point2_eval() {
        #[rustfmt::skip]
        let b = BSpline::<Point2>::new([
            pt2(-1.0, 0.0), pt2(0.0, 0.0), pt2(1.0, 0.0),
            pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(2.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [6.0, -4.5], [0.0, 0.0], [0.609375, 0.0703125], [0.5, 0.5],
            [0.39062497, 0.92968756], [1.0, 1.0], [-5.0, 5.5]
        ];
        let actual = TEST_T_VALS.map(|t| b.eval(t).0);

        assert_approx_eq!(expected, actual);
    }

    #[test]
    fn b_spline_point2_gradient() {
        #[rustfmt::skip]
        let b = BSpline::<Point2>::new([
            pt2(-1.0, 0.0), pt2(0.0, 0.0), pt2(1.0, 0.0),
            pt2(0.0, 1.0), pt2(1.0, 1.0), pt2(2.0, 1.0),
        ]);

        #[rustfmt::skip]
        let expected = [
            [-8.0, 4.5], [1.0, 0.0], [0.4375, 0.28125], [-0.5, 0.75],
            [0.4375, 0.28125], [1.0, 0.0], [-8.0, 4.5]
        ];
        let actual = TEST_T_VALS.map(|t| b.velocity(t).0);

        assert_eq!(expected, actual);
    }
}
