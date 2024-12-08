//! Bézier curves and splines.

use alloc::{vec, vec::Vec};
use core::{array, fmt::Debug};

use crate::geom::Ray;
use crate::math::{Affine, Lerp, Linear};

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

impl<T> CubicBezier<T>
where
    T: Affine<Diff: Linear<Scalar = f32>> + Clone,
{
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

impl<T> BezierSpline<T>
where
    T: Affine<Diff: Linear<Scalar = f32> + Clone> + Clone,
{
    /// Creates a Bézier curve from the given control points. The number of
    /// elements in `pts` must be 3n + 1 for some positive integer n.
    ///
    /// Consecutive points in `pts` make up Bézier curves such that:
    /// * `pts[0..=3]` define the first curve,
    /// * `pts[3..=6]` define the second curve,
    ///
    /// and so on.
    ///
    /// # Panics
    /// If `pts.len()` < 4 or if `pts.len()` mod 3 ≠ 1.
    pub fn new(pts: &[T]) -> Self {
        assert!(
            pts.len() >= 4 && pts.len() % 3 == 1,
            "length must be 3n+1 for some integer n > 0, was {}",
            pts.len()
        );
        Self(pts.to_vec())
    }

    pub fn from_rays<I>(rays: I) -> Self
    where
        I: IntoIterator<Item = Ray<T, T::Diff>>,
    {
        let mut rays = rays.into_iter().peekable();
        let mut first = true;
        let mut pts = vec![];
        while let Some(Ray(p, v)) = rays.next() {
            if !first {
                pts.push(p.add(&v.neg()));
            }
            first = false;
            pts.push(p.clone());
            if rays.peek().is_some() {
                pts.push(p.add(&v));
            }
        }
        Self::new(&pts)
    }

    /// Evaluates `self` at position `t`.
    ///
    /// Returns the first point if `t` < 0 and the last point if `t` > 1.
    pub fn eval(&self, t: f32) -> T {
        // invariant self.0.len() != 0 -> last always exists
        step(t, &self.0[0], self.0.last().unwrap(), |t| {
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
        let segs = ((self.0.len() - 1) / 3) as f32;
        // TODO use floor and make the code cleaner
        let seg = ((t * segs) as u32 as f32).min(segs - 1.0);
        let t2 = t * segs - seg;
        let idx = 3 * (seg as usize);
        (t2, array::from_fn(|k| self.0[idx + k].clone()))
    }

    /// Approximates `self` as a sequence of line segments.
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
    /// use retrofire_core::math::{BezierSpline, vec2, Vec2};
    ///
    /// let curve = BezierSpline::<Vec2>::new(
    ///     &[vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0)]
    /// );
    /// let approx = curve.approximate(|err| err.len_sqr() < 0.01*0.01);
    /// assert_eq!(approx.len(), 17);
    /// ```
    pub fn approximate(&self, halt: impl Fn(&T::Diff) -> bool) -> Vec<T> {
        let len = self.0.len();
        let mut res = Vec::with_capacity(3 * len);
        self.do_approx(0.0, 1.0, 10 + len.ilog2(), &halt, &mut res);
        res.push(self.0[len - 1].clone());
        res
    }

    fn do_approx(
        &self,
        a: f32,
        b: f32,
        max_dep: u32,
        halt: &impl Fn(&T::Diff) -> bool,
        accum: &mut Vec<T>,
    ) {
        let mid = a.lerp(&b, 0.5);

        let ap = self.eval(a);
        let bp = self.eval(b);

        let real = self.eval(mid);
        let approx = ap.lerp(&bp, 0.5);

        if max_dep == 0 || halt(&real.sub(&approx)) {
            accum.push(ap);
        } else {
            self.do_approx(a, mid, max_dep - 1, halt, accum);
            self.do_approx(mid, b, max_dep - 1, halt, accum);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::assert_approx_eq;
    use crate::math::{pt2, vec2, Point2, Vec2};

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
}
