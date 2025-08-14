use alloc::vec::Vec;
use core::fmt::Debug;

use super::{Lerp, Parametric, Point2, inv_lerp};

/// A position-based color progression that can be used to fill a 2D surface.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient2<T> {
    /// The shape of the gradient.
    pub kind: Kind<Point2>,
    /// The sequence of colors to interpolate between.
    pub map: ColorMap<T>,
}

/// The shape of a gradient.
///
/// Maps a point to a *t* value used to look up the respective color
/// in the color sequence of a gradient.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Kind<Pt> {
    /// A linear, or axial, gradient between two points.
    ///
    /// Given two points Q and R and an input point P, perpendicularly projects
    /// P onto the line crossing Q and R and returns the corresponding *t*
    /// value clamped to [0, 1] such that t = 0 at Q and t = 1 at R.
    Linear(Pt, Pt),
    /// A circularly symmetric gradient of some radius around a point.
    ///
    /// Given a center point Q, radius *r*, and input point P, maps the distance
    /// |P - Q| to *t* values such that:
    /// * *t* = 0 when P = Q, and
    /// * *t* = 1 when |P - Q| >= *r*.
    #[cfg(feature = "fp")]
    Radial(Pt, f32),
    /// TODO
    #[cfg(feature = "fp")]
    Conical(Pt),
}

/// A sequence of (number, color) pairs, mapping t values to colors.
/// The numbers must be in a *nondecreasing* order.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorMap<T>(Vec<(f32, T)>);

impl<T: Lerp> Gradient2<T> {
    /// Creates a new gradient.
    ///
    /// # Panics
    /// If there are no stops, or not all the stop values are nondecreasing
    pub fn new(
        kind: Kind<Point2>,
        stops: impl IntoIterator<Item = (f32, T)>,
    ) -> Self {
        Self {
            kind,
            map: ColorMap::new(stops),
        }
    }

    /// Returns the value of `self` at the given point.
    pub fn eval(&self, p: Point2) -> T {
        let t = match self.kind {
            Kind::Linear(p0, p1) => (p - p0).scalar_project(&(p1 - p0)),
            #[cfg(feature = "fp")]
            Kind::Radial(p0, r) => (p - p0).len() / r,
            #[cfg(feature = "fp")]
            Kind::Conical(p0) => {
                let angle = (p - p0).atan();
                // map negative angles to positive
                use super::float::f32;
                f32::rem_euclid(angle.to_turns(), 1.0)
            }
        };
        self.map.eval(t)
    }
}

impl<T> ColorMap<T> {
    /// Creates a new color map.
    ///
    /// # Panics
    /// If there are no stops, or not all the stop values are nondecreasing
    pub fn new(it: impl IntoIterator<Item = (f32, T)>) -> Self {
        let mut t0 = f32::MIN;
        let stops: Vec<_> = it
            .into_iter()
            .inspect(|&(t, _)| {
                assert!(t >= t0, "t values must be nondecreasing");
                t0 = t;
            })
            .collect();
        assert!(!stops.is_empty(), "at least one stop is required");
        Self(stops)
    }
}

impl<T: Lerp> Parametric<T> for ColorMap<T> {
    fn eval(&self, t: f32) -> T {
        let v = &self.0[..];
        debug_assert!(!v.is_empty(), "failed invariant");
        let res = v.binary_search_by(|(u, _)| u.total_cmp(&t));
        match res {
            // t == t_i
            Ok(i) => v[i].1.clone(),
            // t < t_0
            Err(0) => v[0].1.clone(),
            // t > t_n
            Err(i) if i == v.len() => v[i - 1].1.clone(),
            // 0 < i < len
            Err(i) => {
                let (t1, v1) = &v[i - 1]; // ok: 0 < i
                let (t2, v2) = &v[i]; // ok: i < len
                // Remap t such that t=0 -> v1 and t=1 -> v2
                v1.lerp(v2, inv_lerp(t, *t1, *t2))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use crate::math::pt2;
    use alloc::vec;

    #[test]
    fn linear_gradient() {
        let p = pt2(-1.0, 0.0);
        let q = pt2(2.0, 1.0);
        let g = Gradient2::new(Kind::Linear(p, q), [(0.25, 0.9), (0.6, 0.1)]);
        let v = q - p;

        // start point, t=0
        assert_eq!(g.eval(p), 0.9);

        // perpendicular to start point
        assert_eq!(g.eval(p + v.perp()), 0.9);
        assert_eq!(g.eval(p + 100.0 * v.perp()), 0.9);

        // t < 0
        assert_eq!(g.eval(pt2(-2.0, 0.0)), 0.9);
        assert_eq!(g.eval(pt2(-100.0, 0.0)), 0.9);

        // t = 0.25
        assert_eq!(g.eval(p + 0.25 * v), 0.9);

        // t = (0.6 + 0.25)/2
        assert_approx_eq!(g.eval(p + 0.425 * v), 0.5);
        assert_approx_eq!(g.eval(p + 0.425 * v + v.perp()), 0.5);

        // t = 0.6
        assert_eq!(g.eval(p + 0.6 * v), 0.1);

        // end point, t = 1
        assert_eq!(g.eval(q), 0.1);

        // t > 1
        assert_eq!(g.eval(pt2(3.0, 1.0)), 0.1);
        assert_eq!(g.eval(pt2(100.0, 1.0)), 0.1);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn conical_gradient() {
        let g = Gradient2::new(
            Kind::Conical(pt2(2.0, -1.0)),
            [(0.25, 0.9), (0.6, 0.1)],
        );

        // center point and points to its right with same y should be t=0
        assert_eq!(g.eval(pt2(2.0, -1.0)), 0.9);
        assert_eq!(g.eval(pt2(3.0, -1.0)), 0.9);
        assert_eq!(g.eval(pt2(100.0, -1.0)), 0.9);

        // t=0.25 corresponds to directly up, should be 0.9 until that
        assert_eq!(g.eval(pt2(2.0, 0.0)), 0.9);
        assert_eq!(g.eval(pt2(2.0, 100.0)), 0.9);

        // left
        assert_approx_eq!(g.eval(pt2(-1.0, -1.0)), 0.328572);
        assert_approx_eq!(g.eval(pt2(-100.0, -1.0)), 0.328572);

        // down
        assert_eq!(g.eval(pt2(2.0, -2.0)), 0.1);
        assert_eq!(g.eval(pt2(2.0, -100.0)), 0.1);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn radial_gradient() {
        let g = Gradient2::new(
            Kind::Radial(pt2(2.0, -1.0), 2.0),
            [(0.25, 0.9), (0.5, 0.1)],
        );

        assert_eq!(g.eval(pt2(2.0, -1.0)), 0.9); // t=0.0

        assert_eq!(g.eval(pt2(2.0, -1.5)), 0.9); // t=0.25
        assert_eq!(g.eval(pt2(1.5, -1.0)), 0.9); // t=0.25

        assert_eq!(g.eval(pt2(2.0, -1.75)), 0.5); // t=0.375
        assert_eq!(g.eval(pt2(1.25, -1.0)), 0.5); // t=0.375

        assert_eq!(g.eval(pt2(2.0, -3.0)), 0.1); // t=0.5
        assert_eq!(g.eval(pt2(0.0, -1.0)), 0.1); // t=0.5

        assert_eq!(g.eval(pt2(2.0, 2.0)), 0.1); // t=1.0
        assert_eq!(g.eval(pt2(0.0, -1.0)), 0.1); // t=1.0
    }

    #[test]
    fn t_less_than_or_eq_to_min_gives_min() {
        let g = ColorMap(vec![(0.2, -1.23f32), (0.8, 1.23f32)]);
        assert_eq!(g.eval(-10.0), -1.23);
        assert_eq!(g.eval(0.0), -1.23);
        assert_eq!(g.eval(0.2), -1.23);
    }
    #[test]
    fn t_greater_than_or_eq_to_max_gives_min() {
        let g = ColorMap(vec![(0.2, -1.23f32), (0.8, 1.23f32)]);
        assert_eq!(g.eval(0.8), 1.23);
        assert_eq!(g.eval(1.0), 1.23);
        assert_eq!(g.eval(10.0), 1.23);
    }

    #[test]
    fn t_between_min_max_interpolates() {
        let g = ColorMap(vec![(0.2, 1.23f32), (0.6, 2.23f32)]);

        assert_eq!(g.eval(0.2), 1.23);
        assert_eq!(g.eval(0.4), 1.73);
        assert_eq!(g.eval(0.6), 2.23);
    }

    #[test]
    fn nonincreasing_t_gives_sharp_change() {
        let g = ColorMap(vec![(0.4, -1.23f32), (0.4, 1.23f32)]);

        assert_eq!(g.eval(0.2), -1.23);
        assert_eq!(g.eval(0.4f32.next_down()), -1.23);
        assert_eq!(g.eval(0.4), 1.23);
        assert_eq!(g.eval(0.6), 1.23);
    }

    #[test]
    fn single_stop_gradient_has_constant_value() {
        let g = ColorMap(vec![(0.5, 1.23f32)]);
        assert_eq!(g.eval(10.0), 1.23);
        assert_eq!(g.eval(-1.0), 1.23);
        assert_eq!(g.eval(0.0), 1.23);
        assert_eq!(g.eval(0.2), 1.23);
        assert_eq!(g.eval(0.8), 1.23);
        assert_eq!(g.eval(1.0), 1.23);
        assert_eq!(g.eval(10.0), 1.23);
    }

    #[test]
    #[should_panic(expected = "at least one stop is required")]
    fn stops_with_zero_entries_panics() {
        _ = ColorMap::<()>::new([]);
    }

    #[test]
    #[should_panic(expected = "t values must be nondecreasing")]
    fn stops_with_nondecreasing_t_panics() {
        _ = ColorMap::<()>::new([(0.8, ()), (0.2, ())]);
    }
}
