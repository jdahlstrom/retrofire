use crate::{lerp, Linear};
use crate::transform::Transform;
use crate::mat::Mat4;
use std::fmt::Debug;

pub fn smoothstep(t: f32) -> f32 {
    interpolate(t, 0.0, 1.0, || {
        t * t * (3.0 - 2.0 * t)
    })
}

pub fn smootherstep(t: f32) -> f32 {
    interpolate(t, 0.0, 1.0, || {
        t * t * t * (10.0 + t * (6.0 * t - 15.0))
    })
}

pub fn interpolate<T, F>(t: f32, min: T, max: T, f: F) -> T
where
    F: FnOnce() -> T
{
    match t {
        _ if t <= 0.0 => min,
        _ if t >= 1.0 => max,
        _ => f()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bezier<T>([T; 4]);

impl<T: Linear<f32> + Copy> Bezier<T> {

    pub fn eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        interpolate(t, p0, p3, || {
            let p01 = lerp(t, p0, p1);
            let p12 = lerp(t, p1, p2);
            let p23 = lerp(t, p2, p3);
            lerp(t, lerp(t, p01, p12), lerp(t, p12, p23))
        })
    }

    pub fn fast_eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        interpolate(t, p0, p3, || {
            let p2_3 = p1.mul(3.0);
            let p3_3 = p2.mul(3.0);

            let d0 = p0;
            let d1 = p0.neg().add(p2_3);
            let d2 = p0.add(p2_3).mul(-2.0).add(p3_3);
            let d3 = p0.add(p2_3).sub(p3_3).add(p3);

            d0.add(d1.add(d2.add(d3.mul(t)).mul(t)).mul(t))
        })
    }

    pub fn tangent(&self, t: f32) -> T {
        let t = t.clamp(0.0, 1.0);

        let [p0, p1, p2, p3] = self.0;

        let p2_3 = p1.mul(3.0);
        let p3_3 = p2.mul(3.0);

        let d0 = p0.neg().add(p2_3);
        let d1 = p0.add(p2_3).mul(-2.0).add(p3_3).mul(2.0);
        let d2 = p0.add(p2_3).sub(p3_3).add(p3).mul(3.0);

        d0.add(d1.add(d2.mul(t)).mul(t))
    }
}

impl<T: Transform> Transform for Bezier<T> {
    fn transform_mut(&mut self, m: &Mat4) {
        self.0.transform_mut(m)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BezierCurve<T>(Vec<T>);

impl<T: Linear<f32> + Copy> BezierCurve<T> {

    pub fn new(pts: &[T]) -> Self {
        assert_eq!((pts.len() - 1) % 3, 0,
                   "length must be of form 3n+1, was {}", pts.len());
        Self(pts.to_vec())
    }

    /// Evaluates this Bezier curve at position `t`.
    pub fn eval(&self, t: f32) -> T {
        interpolate(t, self.0[0], self.0[self.0.len() - 1], || {
            let (t, seg) = self.segment(t);
            Bezier([seg[0], seg[1], seg[2], seg[3]]).eval(t)
        })
    }

    pub fn tangent(&self, t: f32) -> T {
        let (t, seg) = self.segment(t);
        Bezier([seg[0], seg[1], seg[2], seg[3]]).tangent(t)
    }

    fn segment(&self, t: f32) -> (f32, &[T]) {
        let segs = ((self.0.len() - 1) / 3) as f32;
        let seg = (t * segs).floor().min(segs - 1.0);
        let t2 = t * segs - seg;
        let idx = 3 * (seg as usize);
        //eprintln!("t={} -> seg={}, t={}", t, seg, t2);
        (t2, &self.0[idx..idx + 4])
    }

    /// Approximates this Bezier curve as a sequence of line segments.
    ///
    /// Recursively subdivides the curve into two half-curves, stopping
    /// once `halt` returns `true`.
    pub fn approximate(&self, halt: &impl Fn(T, T) -> bool) -> Vec<T> {
        let mut res = Vec::with_capacity(3 * self.0.len());
        self.do_approx(0.0, 1.0, 10, halt, &mut res);
        res.push(self.0[self.0.len()-1]);
        res
    }

    fn do_approx(
        &self,
        a: f32, b: f32,
        max_dep: usize,
        halt: &impl Fn(T, T) -> bool,
        accum: &mut Vec<T>
    ) {
        let mid = 0.5 * (a + b);

        let midp = self.eval(mid);

        let ap = self.eval(a);
        let bp = self.eval(b);

        let a_to_b = bp.sub(ap);
        let a_to_mid = midp.sub(ap);

        if max_dep == 0 || halt(a_to_b.mul(0.5), a_to_mid) {
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
    use super::*;
    use crate::tests::util::*;
    use crate::vec::dir;

    #[test]
    fn bezier_spline_eval_eq_fasteval() {
        let b = Bezier([
            (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
        ]);
        for i in 0..11 {
            let t = i as f32 / 10.0;
            let (x, y) = (b.eval(t), b.fast_eval(t));
            assert_approx_eq(x.0, y.0);
            assert_approx_eq(x.1, y.1);
        }
    }

    #[test]
    fn bezier_spline_tangent() {
        let b = Bezier([0.0, 0.0, 1.0, 1.0]);

        assert_eq!(0.0, b.tangent(-1.0));
        assert_eq!(0.0, b.tangent(0.0));
        assert_eq!(1.125, b.tangent(0.25));
        assert_eq!(1.500, b.tangent(0.5));
        assert_eq!(1.125, b.tangent(0.75));
        assert_eq!(0.0, b.tangent(1.0));
        assert_eq!(0.0, b.tangent(2.0));

    }

    #[test]
    fn bezier_curve_approx_1d() {
        let b = BezierCurve::new(&[2.0, 5.0, -5.0, 10.0]);

        let approx = b.approximate(&|d0: f32, d1: f32| (1.0 - d1/d0).abs() < 0.5);

        for pt in approx.into_iter().step_by(1) {
            eprintln!("{:>w$}", '*', w = (pt * 10.0) as usize);
        }

        for i in 0..16 {
            let pt = b.eval(i as f32 / 15.0);
            eprintln!("{:>w$}", '*', w = (pt * 10.0) as usize);
        }
    }

    #[test]
    fn bezier_curve_approx_2d() {
        let b = BezierCurve::new(&[(0.0, 0.0), (0.0, 10.0), (1.0, 10.0), (1.0, 0.0)]);

        let approx = b.approximate(&|(x0, y0), (x1, y1)| {
            let d0 = dir(x0, y0, 0.0);
            let d1 = dir(x1, y1, 0.0);

            (1.0 - d0.scalar_project(d1)).abs() < 0.2
        });

        for pt in approx.into_iter().step_by(1) {
            eprintln!("{:>w$}", '*', w = (pt.1 * 10.0) as usize);
        }

        for i in 0..16 {
            let pt = b.eval(i as f32 / 15.0);
            eprintln!("{:>w$}", '*', w = (pt.1 * 10.0) as usize);
        }
    }

    #[test]
    fn bezier_curve() {
        let c = BezierCurve::new(&[
            0.0, 0.8, 0.9, 1.0, 0.6, 0.5, 0.5
        ]);
        assert_eq!(0.0, c.eval(-1.0));
        assert_eq!(0.0, c.eval(0.0));
        assert_eq!(0.7625, c.eval(0.25));
        assert_eq!(1.0, c.eval(0.5));
        assert_eq!(0.6, c.eval(0.75));
        assert_eq!(0.5, c.eval(1.0));
        assert_eq!(0.5, c.eval(2.0));
    }

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
        assert_eq!(0.500000000, smootherstep(0.5));
        assert_eq!(0.896484400, smootherstep(0.75));

        assert_eq!(1.0, smootherstep(1.0));
        assert_eq!(1.0, smootherstep(10.0));
    }
}
