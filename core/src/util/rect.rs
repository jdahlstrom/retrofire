//! Rectangular regions; essentially two-dimensional ranges.

use core::fmt::Debug;
use core::ops::{Bound::*, RangeBounds, RangeFull, Sub};

use crate::math::vec::{Real, Vector};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rect<T = usize> {
    /// The left bound of `self`, if any.
    pub left: Option<T>,
    /// The top bound of `self`, if any.
    pub top: Option<T>,
    /// The right bound of `self`, if any.
    pub right: Option<T>,
    /// The bottom bound of `self`, if any.
    pub bottom: Option<T>,
}

impl<T: Copy> Rect<T> {
    pub fn width(&self) -> Option<T::Output>
    where
        T: Ord + Sub,
    {
        let (r, l) = (self.right?, self.left?);
        Some(r - r.min(l)) // Clamp width to 0
    }
    pub fn height(&self) -> Option<T::Output>
    where
        T: Ord + Sub,
    {
        let (b, t) = (self.bottom?, self.top?);
        Some(b - b.min(t)) // Clamp height to 0
    }

    pub fn is_empty(&self) -> bool
    where
        T: PartialEq,
    {
        self.left == self.right || self.top == self.bottom
    }

    pub fn contains(&self, x: T, y: T) -> bool
    where
        T: PartialOrd,
    {
        let [horiz, vert] = self.bounds();
        horiz.contains(&x) && vert.contains(&y)
    }

    pub fn bounds(&self) -> [impl RangeBounds<T>; 2] {
        let left = self.left.map(Included).unwrap_or(Unbounded);
        let top = self.top.map(Included).unwrap_or(Unbounded);
        let right = self.right.map(Excluded).unwrap_or(Unbounded);
        let bottom = self.bottom.map(Excluded).unwrap_or(Unbounded);

        [(left, right), (top, bottom)]
    }

    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self
    where
        T: Ord,
    {
        pub fn zip_with<T>(
            a: Option<T>,
            b: Option<T>,
            f: fn(T, T) -> T,
        ) -> Option<T> {
            match (a, b) {
                (None, None) => None,
                (some, None) | (None, some) => some,
                (Some(s), Some(o)) => Some(f(s, o)),
            }
        }
        Self {
            left: zip_with(self.left, other.left, T::max),
            top: zip_with(self.top, other.top, T::max),
            right: zip_with(self.right, other.right, T::min),
            bottom: zip_with(self.bottom, other.bottom, T::min),
        }
    }
}

impl<R: RangeBounds<usize>, S: RangeBounds<usize>> From<(R, S)> for Rect {
    fn from((x, y): (R, S)) -> Self {
        let resolve = |b, i, e| match b {
            Included(&x) => Some(x + i),
            Excluded(&x) => Some(x + e),
            Unbounded => None,
        };
        let left = resolve(x.start_bound(), 0, 1);
        let top = resolve(y.start_bound(), 0, 1);
        let right = resolve(x.end_bound(), 1, 0);
        let bottom = resolve(y.end_bound(), 1, 0);

        Self { left, top, right, bottom }
    }
}

type Vec2u = Vector<[usize; 2], Real<2>>;

impl From<(Vec2u, Vec2u)> for Rect {
    /// Creates a `Rect` from two vectors designating the left-top
    /// and right-bottom corners of the `Rect`.
    fn from((l_t, r_b): (Vec2u, Vec2u)) -> Self {
        Self {
            left: Some(l_t.x()),
            top: Some(l_t.y()),
            right: Some(r_b.x()),
            bottom: Some(r_b.y()),
        }
    }
}

impl<T> From<RangeFull> for Rect<T> {
    fn from(_: RangeFull) -> Self {
        Self {
            left: None,
            top: None,
            right: None,
            bottom: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::vec::vec2;

    use super::*;

    fn rect<T>(
        l: impl Into<Option<T>>,
        t: impl Into<Option<T>>,
        r: impl Into<Option<T>>,
        b: impl Into<Option<T>>,
    ) -> Rect<T> {
        Rect {
            left: l.into(),
            top: t.into(),
            right: r.into(),
            bottom: b.into(),
        }
    }

    #[test]
    fn extents_bounded() {
        let r = rect(10, 20, 100, 120);
        assert_eq!(r.width(), Some(90));
        assert_eq!(r.height(), Some(100));
    }
    #[test]
    fn extents_unbounded() {
        let r = rect(None, 20, 100, None);
        assert_eq!(r.width(), None);
        assert_eq!(r.height(), None);
    }
    #[test]
    fn extents_empty() {
        let r = rect(10, 20, -100, 20);
        assert_eq!(r.width(), Some(0));
        assert_eq!(r.height(), Some(0));
    }
    #[test]
    fn contains() {
        let r = rect(None, -10i32, 100, None);
        assert!(r.contains(0, 0));
        assert!(!r.contains(0, -20));
        assert!(r.contains(-9999, 9999));
        assert!(r.contains(99, 0));
        assert!(!r.contains(100, 0));
    }

    #[test]
    fn bounds() {
        let r = rect(None, -10i32, 100, None);
        let [h, v] = r.bounds();

        assert_eq!(h.start_bound(), Unbounded);
        assert_eq!(v.start_bound(), Included(&-10));
        assert_eq!(h.end_bound(), Excluded(&100));
        assert_eq!(v.end_bound(), Unbounded);
    }

    #[test]
    fn intersect() {
        let r = rect(10, 20, 100, 40);
        let s = rect(30, 0, 60, 50);

        assert_eq!(r.intersect(&s), rect(30, 20, 60, 40));
    }
    #[test]
    fn intersect_unbounded() {
        let r = rect(0, 0, 10, None);
        let s = rect(0, 10, 10, None);

        assert_eq!(r.intersect(&s), rect(0, 10, 10, None));
    }
    #[test]
    fn intersect_unbounded_to_bounded() {
        let r = rect(None, 0, 30, 10);
        let s = rect(10, 0, None, 10);

        assert_eq!(r.intersect(&s), rect(10, 0, 30, 10));
    }

    #[test]
    fn intersect_disjoint() {
        let r = rect(None, -10i32, 100, None);
        let s = rect(100, None, None, None);

        let t = r.intersect(&s);
        assert_eq!(t.width(), Some(0));
        assert_eq!(t.height(), None);
    }

    #[test]
    fn from_pair_of_ranges() {
        assert_eq!(Rect::from((2..5, 4..8)), rect(2, 4, 5, 8));
        assert_eq!(Rect::from((2.., 4..8)), rect(2, 4, None, 8));
        assert_eq!(Rect::from((2..5, ..8)), rect(2, None, 5, 8));
        assert_eq!(Rect::from((2..=5, 4..=8)), rect(2, 4, 6, 9));
        assert_eq!(Rect::from((.., ..)), rect(None, None, None, None));
    }

    #[test]
    fn from_range_full() {
        assert_eq!(Rect::<()>::from(..), rect(None, None, None, None));
    }

    #[test]
    fn from_pair_of_vecs() {
        assert_eq!(
            Rect::from((vec2(10, 20), vec2(40, 80))),
            rect(10, 20, 40, 80)
        );
    }
}
