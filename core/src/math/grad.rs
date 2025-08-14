use alloc::{format, vec::Vec};
use core::fmt::Debug;

use super::{Lerp, Parametric, Point2};

pub struct Gradient<T> {
    shape: Shape,
    map: Vec<(f32, T)>,
}

pub enum Shape {
    Linear(Point2, Point2),
    Radial(Point2, f32),
    Conical(Point2),
}

impl<T: Lerp + Clone + Debug> Gradient<T> {
    pub fn new(shape: Shape, map: impl IntoIterator<Item = (f32, T)>) -> Self {
        let map: Vec<_> = map.into_iter().collect();
        assert!(!map.is_empty());
        Self { shape, map }
    }

    pub fn eval(&self, p: Point2) -> T {
        let t = match self.shape {
            Shape::Linear(p0, p1) => (p - p0).scalar_project(&(p1 - p0)),

            Shape::Radial(p0, r) => (p - p0).len() / r,
            Shape::Conical(p) => {
                todo!();
            }
        };
        self.map.eval(t)
    }
}

impl<T: Lerp + Clone + Debug> Parametric<T> for Vec<(f32, T)> {
    fn eval(&self, t: f32) -> T {
        if let (Some((t0, v0)), Some((tn, vn))) = (self.first(), self.last()) {
            if t < *t0 {
                return v0.clone();
            } else if t >= *tn {
                return vn.clone();
            }
        } else {
            unreachable!("failed invariant !self.is_empty()")
        };

        // t is between some consecutive t0, t1. Find the corresponding values
        // and lerp between them.
        for [(t0, v0), (t1, v1)] in
            self.windows(2).map(|win| [&win[0], &win[1]])
        {
            if *t0 <= t && t < *t1 {
                return v0.lerp(&v1, (t - t0) / (t1 - t0));
            }
        }
        unreachable!("t={t}, map={:?}", self)
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn t_less_than_or_eq_to_min() {
        let g = Gradient::new([(0.5, 1.23f32)]);
        assert_eq!(g.eval(-10.0), 1.23);
        assert_eq!(g.eval(0.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }
    #[test]
    fn t_greater_than_or_eq_to_max() {
        let g = Gradient::new([(0.5, 1.23f32)]);
        assert_eq!(g.eval(10.0), 1.23);
        assert_eq!(g.eval(1.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }

    #[test]
    fn t_between_min_max() {
        let g = Gradient::new([(0.2, 1.23f32), (0.6, 2.23f32)]);

        assert_eq!(g.eval(0.2), 1.23);
        assert_eq!(g.eval(0.4), 1.73);
        assert_eq!(g.eval(0.6), 2.23);
    }
}
