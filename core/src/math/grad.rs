use alloc::vec::Vec;
use core::fmt::Debug;

use super::{Lerp, Parametric, Point2, inv_lerp};

pub struct Gradient2<T> {
    pub shape: Shape,
    pub stops: Stops<T>,
    pub frequency: f32,
}

pub enum Shape {
    Linear(Point2, Point2),
    Radial(Point2, f32),
    Conical(Point2),
}

pub struct Stops<T>(Vec<(f32, T)>);

/// Two-dimensional gradient.
impl<T: Lerp + Clone + Debug> Gradient2<T> {
    pub fn new(
        shape: Shape,
        stops: impl IntoIterator<Item = (f32, T)>,
    ) -> Self {
        let stops = Stops::new(stops);
        assert!(!stops.0.is_empty(), "at least one stop must be supplied");
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

impl<T> Stops<T> {
    pub fn new(it: impl IntoIterator<Item = (f32, T)>) -> Self {
        let mut t0 = f32::MIN;
        let stops = it
            .into_iter()
            .inspect(|&(t, _)| {
                assert!(t >= t0, "t values must be nondecreasing");
                t0 = t;
            })
            .collect();
        Self(stops)
    }
}

impl<T: Lerp> Parametric<T> for Stops<T> {
    fn eval(&self, t: f32) -> T {
        let v = &self.0[..];
        debug_assert!(!v.is_empty(), "failed invariant");
        let res = v.binary_search_by(|(u, _)| u.total_cmp(&t));
        match res {
            Ok(i) => v[i].1.clone(),                      // t == t_i
            Err(0) => v[0].1.clone(),                     // t < t_0
            Err(i) if i == v.len() => v[i - 1].1.clone(), // t > t_n
            Err(i) => {
                // 0 < i < len
                let (t1, v1) = &v[i - 1];
                let (t2, v2) = &v[i];
                // Remap t such that t=0 -> v1 and t=1 -> v2
                v1.lerp(v2, inv_lerp(t, *t1, *t2))
            }
        }
    }
}

mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn t_less_than_or_eq_to_min() {
        let g = Stops(vec![(0.5, 1.23f32)]);
        assert_eq!(g.eval(-10.0), 1.23);
        assert_eq!(g.eval(0.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }
    #[test]
    fn t_greater_than_or_eq_to_max() {
        let g = Stops(vec![(0.5, 1.23f32)]);
        assert_eq!(g.eval(10.0), 1.23);
        assert_eq!(g.eval(1.0), 1.23);
        assert_eq!(g.eval(0.5), 1.23);
    }

    #[test]
    fn t_between_min_max() {
        let g = Stops(vec![(0.2, 1.23f32), (0.6, 2.23f32)]);

        assert_eq!(g.eval(0.2), 1.23);
        assert_eq!(g.eval(0.4), 1.73);
        assert_eq!(g.eval(0.6), 2.23);
    }
}
