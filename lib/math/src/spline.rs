use crate::{lerp, Linear};
use crate::transform::Transform;
use crate::mat::Mat4;

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
    if t < 0.0 {
        min
    } else if t < 1.0 {
        f()
    } else {
        max
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bezier<T>([T; 4]);

impl<T: Linear<f32> + Copy> Bezier<T> {

    pub fn eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        interpolate(t, p0, p1, || {
            let p01 = lerp(t, p0, p1);
            let p12 = lerp(t, p1, p2);
            let p23 = lerp(t, p2, p3);
            lerp(t, lerp(t, p01, p12), lerp(t, p12, p23))
        })
    }

    pub fn fast_eval(&self, t: f32) -> T {
        let [p0, p1, p2, p3] = self.0;
        interpolate(t, p0, p1, || {
            let p2_3 = p1.mul(3.0);
            let p3_3 = p2.mul(3.0);

            let d0 = p0;
            let d1 = p0.neg().add(p2_3);
            let d2 = p0.add(p2_3).mul(-2.0).add(p3_3);
            let d3 = p0.add(p2_3).add(p3_3.neg()).add(p3);

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
        let d2 = p0.add(p2_3).add(p3_3.neg()).add(p3).mul(3.0);

        d0.add(d1.add(d2.mul(t)).mul(t))
    }
}

impl<T: Transform> Transform for Bezier<T> {
    fn transform(&mut self, m: &Mat4) {
        self.0.transform(m)
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

    pub fn eval(&self, t: f32) -> T {
        interpolate(t, self.0[0], self.0[self.0.len() - 1], || {
            let segs = ((self.0.len() - 1) / 3) as f32;

            let t = t * segs;
            let seg = t.floor().min(segs - 1.0);
            let t = t - seg;

            let idx = 3 * (seg as usize);

            let cps = &self.0[idx..idx + 4];
            Bezier([cps[0], cps[1], cps[2], cps[3]]).eval(t)
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::util::*;

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
