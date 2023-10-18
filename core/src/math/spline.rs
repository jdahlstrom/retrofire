//! Bezier curves and splines.

use alloc::vec::Vec;

use crate::math::{Affine, Linear};

/// A cubic Bezier curve, defined by four control points.
///
/// TODO More info about Beziers
///
/// ```text
///                                          p3
///   p1        ____                           \
///    \     _-´    `--_                        \
///     \   /           `-_                     |\
///      \ |               `-_                  | \
///       \|                  `-_               |  \
///        \                     `-__          /    \
///        p0                        `---____-´      p2
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CubicBezier<T>(pub [T; 4]);

/// Interpolates smoothly from 0.0 to 1.0 as `t` goes from 0.0 to 1.0.
///
/// Returns 0 for all `t` <= 0 and 1 for all `t` >= 1. Has a continuous
/// first derivative.
pub fn smoothstep(t: f32) -> f32 {
    step(t, 0.0, 1.0, |t| t * t * (3.0 - 2.0 * t))
}

/// Even smoother version of [`smoothstep`].
///
/// Has continuous first and second derivatives.
pub fn smootherstep(t: f32) -> f32 {
    step(t, 0.0, 1.0, |t| t * t * t * (10.0 + t * (6.0 * t - 15.0)))
}

/// Helper for defining step functions.
///
/// Returns `min` if t ≤ 0, `max` if t ≥ 1, and `f(t)` if 0 < t < 1.
#[inline]
pub fn step<T, F>(t: f32, min: T, max: T, f: F) -> T
where
    F: FnOnce(f32) -> T,
{
    if t <= 0.0 {
        min
    } else if t >= 1.0 {
        max
    } else {
        f(t)
    }
}

impl<T> CubicBezier<T>
where
    T: Affine + Copy,
    T::Diff: Linear<Scalar = f32>,
{
    /// Evaluates the value of `self` at `t`
    ///
    /// Uses De Casteljau's algorithm.
    pub fn eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        step(t, p0, p3, |t| {
            let p01 = p0.lerp(&p1, t);
            let p12 = p1.lerp(&p2, t);
            let p23 = p2.lerp(&p3, t);
            p01.lerp(&p12, t).lerp(&p12.lerp(&p23, t), t)
        })
    }
}
impl<T: Linear<Scalar = f32> + Copy> CubicBezier<T> {
    pub fn fast_eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = &self.0;
        step(t, *p0, *p3, |t| {
            let p2_3 = &p1.mul(3.0);
            let p3_3 = &p2.mul(3.0);

            let d0 = p0;
            let d1 = p0.neg().add(p2_3);
            let d2 = p0.add(p2_3).mul(-2.0).add(p3_3);
            let d3 = p0.add(p2_3).sub(p3_3).add(p3);

            d0.add(&d1.add(&d2.add(&d3.mul(t)).mul(t)).mul(t))
        })
    }

    /// Returns the tangent, or "direction", of `self` at `t`.
    pub fn tangent(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        let p1_3 = p1.mul(3.0);
        let p2_3 = p2.mul(3.0);

        let d0 = p0.neg().add(&p1_3);
        let d1 = p0.add(&p1_3).mul(-2.0).add(&p2_3).mul(2.0);
        let d2 = p0.add(&p1_3).sub(&p2_3).add(&p3).mul(3.0);

        step(t, d0.add(&p1_3), d0.add(&d1).add(&d2), |t| {
            d0.add(&d1.add(&d2.mul(t)).mul(t))
        })
    }
}

/// A curve composed of one or more concatenated [cubic Bezier curves][Bezier].
#[derive(Debug, Clone, PartialEq)]
pub struct BezierSpline<T>(Vec<T>);

impl<T: Linear<Scalar = f32> + Copy> BezierSpline<T> {
    /// Creates a bezier curve from the given control points. The number of
    /// elements in `pts` must be n * 3 + 1 for some integer n > 0.
    ///
    /// Consecutive points in `pts` make up Bezier curves such that:
    /// * `pts[0..=3]` define the first curve,
    /// * `pts[3..=6]` define the second curve,
    ///
    /// and so on.
    ///
    /// # Panics
    /// If `pts.len() % 3 != 1`.
    pub fn new(pts: &[T]) -> Self {
        assert_eq!(
            pts.len() % 3,
            1,
            "length must be 3n+1 for some integer n, was {}",
            pts.len()
        );
        Self(pts.to_vec())
    }

    /// Evaluates `self` at position `t`.
    ///
    /// Returns the first point if `t` < 0 and the last point if `t` > 1.
    pub fn eval(&self, t: f32) -> T {
        step(t, self.0[0], self.0[self.0.len() - 1], |t| {
            let (t, seg) = self.segment(t);
            CubicBezier(seg).eval(t)
        })
    }

    /// Returns the tangent of `self` at `t`.
    pub fn tangent(&self, t: f32) -> T {
        let (t, seg) = self.segment(t);
        CubicBezier(seg).tangent(t)
    }

    fn segment(&self, t: f32) -> (f32, [T; 4]) {
        let segs = ((self.0.len() - 1) / 3) as f32;
        let seg = (t * segs).floor().min(segs - 1.0);
        let t2 = t * segs - seg;
        let idx = 3 * (seg as usize);
        (t2, (self.0[idx..idx + 4]).try_into().unwrap())
    }

    /// Approximates `self` as a sequence of line segments.
    ///
    /// Recursively subdivides the curve into two half-curves, stopping
    /// once `halt` returns `true`.
    pub fn approximate(&self, halt: &impl Fn(T, T) -> bool) -> Vec<T> {
        let len = self.0.len();
        let mut res = Vec::with_capacity(3 * len);
        self.do_approx(0.0, 1.0, 10 + len.ilog2(), halt, &mut res);
        res.push(self.0[len - 1]);
        res
    }

    fn do_approx(
        &self,
        a: f32,
        b: f32,
        max_dep: u32,
        halt: &impl Fn(T, T) -> bool,
        accum: &mut Vec<T>,
    ) {
        let mid = 0.5 * (a + b);

        let midp = self.eval(mid);

        let ap = self.eval(a);
        let bp = self.eval(b);

        let a_to_mid = midp.sub(&ap);
        let mid_to_b = bp.sub(&midp);

        if max_dep == 0 || halt(a_to_mid, mid_to_b) {
            //eprintln!("Halting! d={} a={} b={} ab={:?}", max_dep, a, b, a_to_b);
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
    use crate::math::Vec2;

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
    fn bezier_spline_eval_eq_fasteval() {
        let b: CubicBezier<Vec2> = CubicBezier(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]].map(Vec2::new),
        );
        for i in 0..11 {
            let t = i as f32 / 10.0;
            let (v, u) = (b.eval(t), b.fast_eval(t));
            assert_approx_eq!(v.x(), u.x());
            assert_approx_eq!(v.y(), u.y());
        }
    }

    #[test]
    fn bezier_spline_tangent_1d() {
        let b = CubicBezier([0.0, 0.0, 1.0, 1.0]);

        assert_eq!(0.0, b.tangent(-1.0));
        assert_eq!(0.0, b.tangent(0.0));
        assert_eq!(1.125, b.tangent(0.25));
        assert_eq!(1.500, b.tangent(0.5));
        assert_eq!(1.125, b.tangent(0.75));
        assert_eq!(0.0, b.tangent(1.0));
        assert_eq!(0.0, b.tangent(2.0));
    }

    #[test]
    fn bezier_spline_approx_1d() {
        let b = BezierSpline(vec![2.0, 5.0, -5.0, 10.0]);

        let approx =
            b.approximate(&|d0: f32, d1: f32| (1.0 - d1 / d0).abs() < 0.5);

        for _pt in approx.into_iter().step_by(1) {
            //eprintln!("{:>w$}", '*', w = (pt * 10.0) as usize);
        }

        for i in 0..16 {
            let _pt = b.eval(i as f32 / 15.0);
            //eprintln!("{:>w$}", '*', w = (pt * 10.0) as usize);
        }
    }

    #[test]
    fn bezier_spline_eval() {
        let c = BezierSpline(vec![0.0, 0.8, 0.9, 1.0, 0.6, 0.5, 0.5]);
        assert_eq!(0.0, c.eval(-1.0));
        assert_eq!(0.0, c.eval(0.0));
        assert_eq!(0.7625, c.eval(0.25));
        assert_eq!(1.0, c.eval(0.5));
        assert_eq!(0.6, c.eval(0.75));
        assert_eq!(0.5, c.eval(1.0));
        assert_eq!(0.5, c.eval(2.0));
    }
}
