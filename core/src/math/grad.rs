use alloc::vec::Vec;
use core::fmt::Debug;

use super::{Lerp, Parametric, Point2, inv_lerp, smoothstep};

pub struct Gradient2<T> {
    pub shape: Shape,
    pub stops: Vec<(f32, T)>,
    pub frequency: f32,
}

pub enum Shape {
    Linear(Point2, Point2),
    Radial(Point2, f32),
    Conical(Point2),
}

/// Two-dimensional gradient.
impl<T: Lerp + Clone + Debug> Gradient2<T> {
    pub fn new(
        shape: Shape,
        stops: impl IntoIterator<Item = (f32, T)>,
    ) -> Self {
        let stops: Vec<_> = stops.into_iter().collect();
        assert!(!stops.is_empty(), "at least one stop must be supplied");
        Self { shape, stops, frequency: 1.0 }
    }

    /// Returns the value of `self` at the given point.
    pub fn eval(&self, p: Point2) -> T {
        let mut t = match self.shape {
            Shape::Linear(p0, p1) => (p - p0).scalar_project(&(p1 - p0)),
            Shape::Radial(p0, r) => (p - p0).len() / r,
            Shape::Conical(p0) => (p - p0).atan().to_turns(),
        };
        t = (t * self.frequency).rem_euclid(1.0);
        //t = smoothstep(t);

        self.stops.eval(t)
    }
}

impl<T: Lerp + Clone> Parametric<T> for Vec<(f32, T)> {
    fn eval(&self, t: f32) -> T {
        let res = self.binary_search_by(|(u, _)| u.total_cmp(&t));
        match res {
            Ok(i) => self[i].1.clone(),
            Err(i) => {
                match i {
                    0 => self[0].1.clone(),                        // t < t0
                    i if i == self.len() => self[i - 1].1.clone(), // t > tn
                    i => {
                        let (t1, v1) = &self[i - 1]; // ok: i != 0
                        let (t2, v2) = &self[i]; // ok: i != len
                        // Remap t such that t=0 -> v1 and t=1 -> v2
                        v1.lerp(v2, inv_lerp(t, *t1, *t2))
                    }
                }
            }
        }

        /*if t < *t0 {
            return v0.clone();
        }
        for ((t0, v0), (t1, v1)) in
            self.windows(2).map(|win| (&win[0], &win[1]))
        {
            if *t0 <= t && t < *t1 {
                let t = inv_lerp(t, *t0, *t1);
                let t = if t < 0.5 { 0.0 } else { 1.0 };
                return v0.lerp(&v1, t);
            }
        }
        // if not yet returned, t >= tn
        vn.clone()*/
    }
}

mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn t_less_than_or_eq_to_min() {
        let g = vec![(0.5, 1.23f32)];
        assert_eq!(g.eval(-10.0), 1.23);
        assert_eq!(g.eval(0.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }
    #[test]
    fn t_greater_than_or_eq_to_max() {
        let g = vec![(0.5, 1.23f32)];
        assert_eq!(g.eval(10.0), 1.23);
        assert_eq!(g.eval(1.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }

    #[test]
    fn t_between_min_max() {
        let g = vec![(0.2, 1.23f32), (0.6, 2.23f32)];

        assert_eq!(g.eval(0.2), 1.23);
        assert_eq!(g.eval(0.4), 1.73);
        assert_eq!(g.eval(0.6), 2.23);
    }
}
