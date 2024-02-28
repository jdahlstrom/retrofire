use core::fmt::Debug;
use core::ops::{Bound::*, Range, RangeBounds, RangeFull, Sub};

use crate::math::vec::Vec2u;

/// An axis-aligned rectangular region.
///
/// Each of the four sides of a `Rect` can be either bounded (`Some(_)`) or
/// unbounded (`None`). If bounded, the start bounds (left and top) are always
/// inclusive, and the end bounds (right and bottom) are always exclusive.
///
/// If `left` and `right` are unbounded and `right` ≤ `left`, the `Rect` is
/// considered empty. The same holds for `top` and `bottom`. This matches the
/// semantics of the standard `Range` type.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rect<T = u32> {
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
    /// Returns the width of `self`, or `None` if horizontally unbounded.
    ///
    /// Returns 0 if `right` <= `left`.
    pub fn width(&self) -> Option<T::Output>
    where
        T: Ord + Sub,
    {
        let (r, l) = (self.right?, self.left?);
        Some(r - r.min(l)) // Clamp width to 0
    }
    /// Returns the height of `self`, or `None` if vertically unbounded.
    ///
    /// Returns 0 if `bottom` <= `top`.
    pub fn height(&self) -> Option<T::Output>
    where
        T: Ord + Sub,
    {
        let (b, t) = (self.bottom?, self.top?);
        Some(b - b.min(t)) // Clamp height to 0
    }

    /// Returns whether `self` contains no points.
    pub fn is_empty(&self) -> bool
    where
        T: PartialOrd,
    {
        let Rect { left, top, right, bottom } = self;
        // Empty if either extent is a bounded, empty range
        let h_empty = left.is_some() && right.is_some() && left >= right;
        let v_empty = top.is_some() && bottom.is_some() && top >= bottom;
        h_empty || v_empty
    }

    /// Returns whether the point (x, y)  is contained within `self`.
    pub fn contains(&self, x: T, y: T) -> bool
    where
        T: PartialOrd,
    {
        let [horiz, vert] = self.bounds();
        horiz.contains(&x) && vert.contains(&y)
    }

    /// Returns the horizontal and vertical extents of `self`.
    pub fn bounds(&self) -> [impl RangeBounds<T>; 2] {
        let left = self.left.map(Included).unwrap_or(Unbounded);
        let top = self.top.map(Included).unwrap_or(Unbounded);
        let right = self.right.map(Excluded).unwrap_or(Unbounded);
        let bottom = self.bottom.map(Excluded).unwrap_or(Unbounded);

        [(left, right), (top, bottom)]
    }

    /// Returns the intersection of `self` and `other`.
    ///
    /// The intersection is a rect that contains exactly the points contained
    /// by both `self` and `other`. If the input rects are disjoint, i.e.
    /// contain no common points, the result is an empty rect with unspecified
    /// top-left coordinates.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self
    where
        T: Ord,
    {
        fn extremum<T>(
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
            left: extremum(self.left, other.left, T::max),
            top: extremum(self.top, other.top, T::max),
            right: extremum(self.right, other.right, T::min),
            bottom: extremum(self.bottom, other.bottom, T::min),
        }
    }
}

impl<H: RangeBounds<u32>, V: RangeBounds<u32>> From<(H, V)> for Rect<u32> {
    /// Creates a `Rect` from two ranges specifying the horizontal and
    /// vertical extents of the `Rect` respectively.
    fn from((horiz, vert): (H, V)) -> Self {
        let resolve = |b, i, e| match b {
            Included(&x) => Some(x + i),
            Excluded(&x) => Some(x + e),
            Unbounded => None,
        };
        let left = resolve(horiz.start_bound(), 0, 1);
        let top = resolve(vert.start_bound(), 0, 1);
        let right = resolve(horiz.end_bound(), 1, 0);
        let bottom = resolve(vert.end_bound(), 1, 0);

        Self { left, top, right, bottom }
    }
}

impl From<Range<Vec2u>> for Rect<u32> {
    /// Creates a `Rect` from two vectors designating the left-top
    /// and right-bottom corners of the `Rect`.
    fn from(r: Range<Vec2u>) -> Self {
        Self {
            left: Some(r.start.x()),
            top: Some(r.start.y()),
            right: Some(r.end.x()),
            bottom: Some(r.end.y()),
        }
    }
}

impl<T> From<RangeFull> for Rect<T> {
    /// Creates a `Rect` with all sides unbounded.
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
    fn is_empty_bounded() {
        assert!(rect(10, 10, 10, 20).is_empty());
        assert!(rect(10, 20, 10, 10).is_empty());
    }
    #[test]
    fn is_empty_negative_extent() {
        assert!(rect(20, 10, 10, 20).is_empty());
        assert!(rect(10, 20, 20, 10).is_empty());
    }

    #[test]
    fn is_empty_unbounded() {
        assert!(rect(10, 10, 10, None).is_empty());
        assert!(rect(10, 10, None, 10).is_empty());
        assert!(rect(10, None, 10, 10).is_empty());
        assert!(rect(None, 10, 10, 10).is_empty());
    }
    #[test]
    fn is_empty_bounded_not_empty() {
        assert!(!rect(10, 10, 20, 20).is_empty());
        assert!(!rect(10, -10, 20, 10).is_empty());
    }

    #[test]
    fn is_empty_unbounded_not_empty() {
        assert!(!rect(10, 10, 20, None).is_empty());
        assert!(!rect(10, 10, None, 20).is_empty());
        assert!(!rect(10, None, 20, 10).is_empty());
        assert!(!rect(None, 10, 10, 20).is_empty());

        assert!(!rect(10, None, 20, None).is_empty());
        assert!(!rect(None, 10, 10, None).is_empty());
        assert!(!rect::<i32>(None, None, None, None).is_empty());
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
    fn intersect_two_bounded() {
        let r = rect(10, 20, 100, 40);
        let s = rect(30, 0, 60, 50);

        assert_eq!(r.intersect(&s), rect(30, 20, 60, 40));
    }
    #[test]
    fn intersect_bounded_and_unbounded() {
        let r = rect(10, 20, 50, 40);
        let s = rect(30, 0, None, 50);

        assert_eq!(r.intersect(&s), rect(30, 20, 50, 40));
    }
    #[test]
    fn intersect_two_unbounded_to_unbounded() {
        let r = rect(0, 0, 10, None);
        let s = rect(0, 10, 10, None);

        assert_eq!(r.intersect(&s), rect(0, 10, 10, None));
    }
    #[test]
    fn intersect_two_unbounded_to_bounded() {
        let r = rect(None, 0, 30, 10);
        let s = rect(10, 0, None, 10);

        assert_eq!(r.intersect(&s), rect(10, 0, 30, 10));
    }
    #[test]
    fn intersect_empty() {
        let r = rect(0, 0, 20, 20);
        let s = rect(10, 10, 10, 10);
        assert!(r.intersect(&s).is_empty());
    }
    #[test]
    fn intersect_full() {
        let r = rect(0, 0, 10, 20);
        let s = (..).into();
        assert_eq!(r.intersect(&s), r);
    }
    #[test]
    fn intersect_disjoint() {
        let r = rect(None, -10i32, 100, None);
        let s = rect(100, None, None, None);

        let t = r.intersect(&s);
        assert!(t.is_empty());
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
    fn from_range_of_vecs() {
        assert_eq!(
            Rect::from(vec2(10, 20)..vec2(40, 80)),
            rect(10, 20, 40, 80)
        );
    }
}
