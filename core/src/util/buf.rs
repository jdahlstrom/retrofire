//! Two-dimensional buffers, with owned and borrowed variants.
//!
//! Useful for storing pixel data of any kind, among other things.

use alloc::{vec, vec::Vec};
use core::fmt::{self, Debug, Formatter};
use core::iter;
use core::ops::{Deref, DerefMut};

use crate::util::Dims;

use inner::Inner;

//
// Traits
//

/// A trait for types that can provide a view of their data as a [`Slice2`].
pub trait AsSlice2<T> {
    /// Returns a borrowed `Slice2` view of `Self`.
    fn as_slice2(&self) -> Slice2<T>;
}

/// A trait for types that can provide a mutable view of their data
/// as a [`MutSlice2`].
pub trait AsMutSlice2<T> {
    /// Returns a mutably borrowed `MutSlice2` view of `Self`.
    fn as_mut_slice2(&mut self) -> MutSlice2<T>;
}

//
// Types
//

/// A rectangular 2D buffer that owns its elements, backed by a `Vec`.
///
/// Unlike `Vec`, however, `Buf2` cannot be resized after construction
/// without explicitly copying the contents to a new, larger buffer.
///
/// `Buf2` stores its elements contiguously, in standard row-major order,
/// such that the coordinate pair (x, y) maps to index `buf.width() * y + x`
/// in the backing vector.
///
/// # Examples
/// ```
/// # use retrofire_core::util::buf::*;
/// # use retrofire_core::math::vec::*;
/// // Elements initialized with `Default::default()`
/// let mut buf = Buf2::new((4, 4));
/// // Indexing with a 2D vector (x, y) yields element at row y, column x:
/// buf[vec2(2, 1)] = 123;
/// // Indexing with an usize i yields row with index i as a slice:
/// assert_eq!(buf[1usize], [0, 0, 123, 0]);
/// // Thus you can also do this, row first, column second:
/// assert_eq!(buf[1usize][2], 123)
/// ```
#[derive(Clone)]
#[repr(transparent)]
pub struct Buf2<T>(Inner<T, Vec<T>>);

/// An immutable rectangular view to a region of a [`Buf2`], another `Slice2`,
/// or in general any `&[T]` slice of memory. A two-dimensional analog to `&[T]`.
///
/// A `Slice2` may be non-contiguous:
/// ```text
/// +------stride-----+
/// |    ____w____    |
/// |   |r0_______|   |
/// |   |r1_______| h |
/// |   |r2_______|   |
/// +-----------------+
/// ```
/// TODO More documentation
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Slice2<'a, T>(Inner<T, &'a [T]>);

/// A mutable rectangular view to a region of a `Buf2`, a `Slice2`,
/// or in general any `&[T]` slice of memory.
#[repr(transparent)]
pub struct MutSlice2<'a, T>(Inner<T, &'a mut [T]>);

//
// Inherent impls
//

impl<T> Buf2<T> {
    /// Returns a buffer with the given dimensions, with elements initialized
    /// in row-major order with values yielded by `init`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::util::buf::Buf2;
    ///
    /// let buf = Buf2::new_from((3, 3), 1..);
    /// assert_eq!(buf.data(), [1, 2, 3,
    ///                         4, 5, 6,
    ///                         7, 8, 9]);
    /// ```
    ///
    /// # Panics
    /// If `w * h > isize::MAX`, or if `init` has fewer than `w * h` elements.
    pub fn new_from<I>((w, h): Dims, init: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let ww = isize::try_from(w).ok();
        let hh = isize::try_from(h).ok();
        let len = ww.and_then(|w| hh.and_then(|h| w.checked_mul(h)));
        let Some(len) = len else {
            panic!(
                "w * h cannot exceed isize::MAX ({w} * {h} > {})",
                isize::MAX
            );
        };
        let data: Vec<_> = init.into_iter().take(len as usize).collect();
        assert_eq!(
            data.len(),
            len as usize,
            "insufficient items in iterator ({} < {len}",
            data.len()
        );
        Self(Inner::new((w, h), w, data))
    }

    /// Returns a buffer of size `w` × `h`, with every element initialized to
    /// `T::default()`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::util::buf::Buf2;
    ///
    /// let buf: Buf2<i32> = Buf2::new((3, 3));
    /// assert_eq!(buf.data(), [0, 0, 0,
    ///                         0, 0, 0,
    ///                         0, 0, 0]);
    /// ```
    ///
    /// # Panics
    /// If `w * h > isize::MAX`.
    pub fn new((w, h): Dims) -> Self
    where
        T: Default + Clone,
    {
        let data = vec![T::default(); (w * h) as usize];
        Self(Inner::new((w, h), w, data))
    }

    /// Returns a buffer of size `w` × `h`, initialized by repeatedly calling
    /// the given function.
    ///
    /// For each element, `init_fn(x, y)` is invoked, where `x` is the column
    /// index and `y` the row index of the element being initialized. The
    /// function invocations occur in row-major order.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::util::buf::Buf2;
    ///
    /// let buf = Buf2::new_with((3, 3), |x, y| 10 * y + x);
    /// assert_eq!(buf.data(), [ 0,  1, 2,
    ///                         10, 11, 12,
    ///                         20, 21, 22]);
    /// ```
    ///
    /// # Panics
    /// If `w * h > isize::MAX`.
    pub fn new_with<F>((w, h): Dims, mut init_fn: F) -> Self
    where
        F: FnMut(u32, u32) -> T,
    {
        let (mut x, mut y) = (0, 0);
        Self::new_from(
            (w, h),
            iter::from_fn(|| {
                let res = init_fn(x, y);
                x += 1;
                if x == w {
                    (x, y) = (0, y + 1);
                }
                Some(res)
            }),
        )
    }

    /// Returns a view of the backing data of `self`.
    pub fn data(&self) -> &[T] {
        self.0.data()
    }

    /// Returns a mutable view of the backing data of `self`.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.0.data_mut()
    }
}

impl<'a, T> Slice2<'a, T> {
    /// Returns a new `Slice2` view to `data` with the given dimensions
    /// and stride.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::util::buf::Slice2;
    /// let data = &[0, 1, 2, 3, 4, 5, 6];
    /// let slice = Slice2::new((2, 2), 3, data);
    /// assert_eq!(&slice[0usize], &[0, 1]);
    /// assert_eq!(&slice[1usize], &[3, 4]);
    /// ```
    /// Above, `slice` represents a 2×2 rectangle with stride 3, such that
    /// the first row maps to `data[0..2]` and the second to `data[3..5]`:
    /// ```text
    ///  slice[0]    slice[1]
    ///     |           |
    /// ,---´---.   ,---´---.
    /// +---+---+---+---+---+---+---+
    /// | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    /// +---+---+---+---+---+---+---+
    /// ```
    /// Internally, this is implemented as the borrow `&data[0..5]`.
    /// Semantically, however, `slice` does not contain `data[2]` as
    /// an element, and attempting to access it eg. with `slice[0][2]`
    /// will panic, as expected.
    ///
    /// # Panics
    /// if `stride < width` or if the slice would overflow `data`.
    ///
    pub fn new(dims: Dims, stride: u32, data: &'a [T]) -> Self {
        Self(Inner::new(dims, stride, data))
    }
}

impl<'a, T> MutSlice2<'a, T> {
    /// Returns a new `MutSlice2` view to `data` with dimensions `w` and `h`
    /// and stride `stride`.
    ///
    /// See [`Slice2::new`] for more information.
    pub fn new(dims: Dims, stride: u32, data: &'a mut [T]) -> Self {
        Self(Inner::new(dims, stride, data))
    }
}

//
// Local trait impls
//

impl<T> AsSlice2<T> for Buf2<T> {
    #[inline]
    fn as_slice2(&self) -> Slice2<T> {
        self.0.as_slice2()
    }
}
impl<T> AsSlice2<T> for &Buf2<T> {
    #[inline]
    fn as_slice2(&self) -> Slice2<T> {
        self.0.as_slice2()
    }
}
impl<T> AsSlice2<T> for Slice2<'_, T> {
    #[inline]
    fn as_slice2(&self) -> Slice2<T> {
        self.0.as_slice2()
    }
}
impl<T> AsSlice2<T> for MutSlice2<'_, T> {
    #[inline]
    fn as_slice2(&self) -> Slice2<T> {
        self.0.as_slice2()
    }
}

impl<T> AsMutSlice2<T> for Buf2<T> {
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<T> {
        self.0.as_mut_slice2()
    }
}
impl<T> AsMutSlice2<T> for &mut Buf2<T> {
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<T> {
        self.0.as_mut_slice2()
    }
}
impl<T> AsMutSlice2<T> for MutSlice2<'_, T> {
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<T> {
        self.0.as_mut_slice2()
    }
}

//
// Foreign trait impls
//

impl<T> Debug for Buf2<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.debug_fmt(f, "Buf2")
    }
}
impl<T> Debug for Slice2<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.debug_fmt(f, "Slice2")
    }
}
impl<T> Debug for MutSlice2<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.debug_fmt(f, "Slice2Mut")
    }
}

impl<T> Deref for Buf2<T> {
    type Target = Inner<T, Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a, T> Deref for Slice2<'a, T> {
    type Target = Inner<T, &'a [T]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a, T> Deref for MutSlice2<'a, T> {
    type Target = Inner<T, &'a mut [T]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Buf2<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<'a, T> DerefMut for MutSlice2<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub mod inner {
    use core::{
        fmt::Formatter,
        iter::zip,
        marker::PhantomData,
        ops::{Deref, DerefMut, Index, IndexMut, Range},
    };

    use crate::{math::vec::Vec2u, util::rect::Rect, util::Dims};

    use super::{AsSlice2, MutSlice2, Slice2};

    /// A helper type that abstracts over owned and borrowed buffers.
    ///
    /// The types `Buf2`, `Slice2`, and `MutSlice2` deref to `Inner`.
    #[derive(Copy, Clone)]
    pub struct Inner<T, D> {
        dims: Dims,
        stride: u32,
        data: D,
        _pd: PhantomData<T>,
    }

    impl<T, D> Inner<T, D> {
        /// Returns the width of `self`.
        #[inline]
        pub fn width(&self) -> u32 {
            self.dims.0
        }
        /// Returns the height of `self`.
        #[inline]
        pub fn height(&self) -> u32 {
            self.dims.1
        }
        /// Returns the stride of `self`.
        #[inline]
        pub fn stride(&self) -> u32 {
            self.stride
        }
        /// Returns whether the rows of `self` are stored as one contiguous
        /// slice, without gaps between rows.
        ///
        /// `Buf2` instances are always contiguous. A `Slice2` or `MutSlice2`
        /// instance is contiguous if its width equals its stride, if its
        /// height is 1, or if it is empty.
        pub fn is_contiguous(&self) -> bool {
            let (w, h) = self.dims;
            self.stride == w || h <= 1 || w == 0
        }
        /// Returns whether `self` contains no elements.
        pub fn is_empty(&self) -> bool {
            self.dims.0 == 0 || self.dims.1 == 0
        }

        /// Returns the linear index corresponding to the coordinates,
        /// even if out of bounds.
        #[inline]
        fn to_index(&self, x: u32, y: u32) -> usize {
            (y * self.stride + x) as usize
        }

        /// Returns the linear index corresponding to the coordinates,
        /// or panics if either x or y is out of bounds.
        #[inline]
        fn to_index_strict(&self, x: u32, y: u32) -> usize {
            self.to_index_checked(x, y).unwrap_or_else(|| {
                let (w, h) = self.dims;
                panic!(
                    "position (x={x}, y={y}) out of bounds (0..{w}, 0..{h})",
                )
            })
        }
        /// Returns the linear index corresponding to the coordinates,
        /// or `None` if x or y is out of bounds.
        #[inline]
        fn to_index_checked(&self, x: u32, y: u32) -> Option<usize> {
            let (w, h) = self.dims;
            (x < w && y < h).then(|| self.to_index(x, y))
        }

        /// Returns the dimensions and linear range corresponding to the rect.
        fn resolve_bounds(&self, rect: &Rect<u32>) -> (Dims, Range<usize>) {
            let (w, h) = self.dims;

            let l = rect.left.unwrap_or(0);
            let t = rect.top.unwrap_or(0);
            let r = rect.right.unwrap_or(w);
            let b = rect.bottom.unwrap_or(h);

            // Assert that left <= right <= width and top <= bottom <= height.
            // Note that this permits left == width or top == height, but only
            // when left == right or top == bottom, that is, when the range is
            // empty. This matches the way slice indexing works.
            assert!(l <= r, "range left ({l}) > right ({r})");
            assert!(t <= b, "range top ({l}) > bottom ({r})");
            assert!(r <= w, "range right ({r}) > width ({w})");
            assert!(b <= h, "range bottom ({b}) > height ({h})");

            // (l, t) is now guaranteed to be in bounds
            let start = self.to_index(l, t);
            // Slice end is the end of the last row
            let end = if b == t {
                self.to_index(r, t)
            } else {
                // b != 0 because b >= t && b != t
                self.to_index(r, b - 1)
            };
            ((r - l, b - t), start..end)
        }

        /// A helper for implementing `Debug`.
        pub(super) fn debug_fmt(
            &self,
            f: &mut Formatter,
            name: &str,
        ) -> core::fmt::Result {
            f.debug_struct(name)
                .field("dims", &self.dims)
                .field("stride", &self.stride)
                .finish()
        }
    }

    impl<T, D: Deref<Target = [T]>> Inner<T, D> {
        /// # Panics
        /// if `stride < w` or if the slice would overflow `data`.
        #[rustfmt::skip]
        pub(super) fn new(dims @ (w, h): Dims, stride: u32, data: D) -> Self {
            assert!(w <= stride, "width ({w}) > stride ({stride})");

            let len = data.len();
            assert!(
                h <= 1 || stride as usize <= len,
                "stride ({stride}) > data length ({len})"
            );
            assert!(h as usize <= len, "height ({h}) > data length ({len})");
            if h > 0 {
                let size = (h - 1) * stride + w;
                assert!(
                    size as usize <= len,
                    "required size ({size}) > data length ({len})"
                );
            }
            Self { dims, stride, data, _pd: PhantomData }
        }

        /// Returns the data of `self` as a linear slice.
        pub(super) fn data(&self) -> &[T] {
            &self.data
        }

        /// Borrows `self` as a `Slice2`.
        pub fn as_slice2(&self) -> Slice2<T> {
            Slice2::new(self.dims, self.stride, &self.data)
        }

        /// Returns a borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice(&self, rect: impl Into<Rect>) -> Slice2<T> {
            let (dims, rg) = self.resolve_bounds(&rect.into());
            Slice2::new(dims, self.stride, &self.data[rg])
        }

        /// Returns a reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get(&self, pos: impl Into<Vec2u>) -> Option<&T> {
            let [x, y] = pos.into().0;
            self.to_index_checked(x, y).map(|i| &self.data[i])
        }

        /// Returns an iterator over the rows of `self` as `&[T]` slices.
        ///
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows(&self) -> impl Iterator<Item = &[T]> {
            self.data
                .chunks(self.stride as usize)
                .map(|row| &row[..self.dims.0 as usize])
        }

        /// Returns an iterator over the elements of `self` in row-major order.
        ///
        /// First returns the elements on row 0 from left to right, followed by
        /// the elements on row 1, and so on.
        pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
            self.rows().flatten()
        }
    }

    impl<T, D: DerefMut<Target = [T]>> Inner<T, D> {
        /// Returns a mutably borrowed rectangular slice of `self`.
        pub fn as_mut_slice2(&mut self) -> MutSlice2<T> {
            MutSlice2::new(self.dims, self.stride, &mut self.data)
        }

        /// Returns the data of `self` as a single mutable slice.
        pub(super) fn data_mut(&mut self) -> &mut [T] {
            &mut self.data
        }

        /// Returns an iterator over the rows of this buffer as `&mut [T]`.
        ///
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
            self.data
                .chunks_mut(self.stride as usize)
                .map(|row| &mut row[..self.dims.0 as usize])
        }

        /// Returns a mutable iterator over all the elements of `self`,
        /// yielded in row-major order.
        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.rows_mut().flatten()
        }

        /// Fills `self` with clones of the value.
        pub fn fill(&mut self, val: T)
        where
            T: Clone,
        {
            if self.is_contiguous() {
                self.data.fill(val);
            } else {
                self.rows_mut()
                    .for_each(|row| row.fill(val.clone()));
            }
        }
        /// Fills `self` by calling a function for each element.
        ///
        /// Calls `f(x, y)` for every element, where  `x` and `y` are the column
        /// and row indices of the element. Proceeds in row-major order.
        pub fn fill_with<F>(&mut self, mut fill_fn: F)
        where
            F: FnMut(u32, u32) -> T,
        {
            for (row, y) in zip(self.rows_mut(), 0..) {
                for (item, x) in zip(row, 0..) {
                    *item = fill_fn(x, y);
                }
            }
        }

        /// Copies each element in `other` to the same position in `self`.
        ///
        /// This operation is often called "blitting".
        ///
        /// # Panics
        /// if the dimensions of `self` and `other` do not match.
        #[doc(alias = "blit")]
        pub fn copy_from(&mut self, other: impl AsSlice2<T>)
        where
            T: Copy,
        {
            let other = other.as_slice2();

            assert_eq!(
                self.dims, other.dims,
                "dimension mismatch (self: {:?}, other: {:?})",
                self.dims, other.dims
            );
            for (dest, src) in self.rows_mut().zip(other.rows()) {
                dest.copy_from_slice(src);
            }
        }

        /// Returns a mutable reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get_mut(&mut self, pos: impl Into<Vec2u>) -> Option<&mut T> {
            let [x, y] = pos.into().0;
            self.to_index_checked(x, y)
                .map(|i| &mut self.data[i])
        }

        /// Returns a mutably borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice_mut(&mut self, rect: impl Into<Rect>) -> MutSlice2<T> {
            let (dims, rg) = self.resolve_bounds(&rect.into());
            MutSlice2(Inner::new(dims, self.stride, &mut self.data[rg]))
        }
    }

    impl<T, D: Deref<Target = [T]>> Index<usize> for Inner<T, D> {
        type Output = [T];

        /// Returns a reference to the row at index `i`.
        ///
        /// The returned slice has length `self.width()`.
        ///
        /// # Panics
        /// If `row >= self.height()`.
        #[inline]
        fn index(&self, i: usize) -> &[T] {
            let idx = self.to_index_strict(0, i as u32);
            let w = self.dims.0 as usize;
            &self.data[idx..][..w]
        }
    }

    impl<T, D> IndexMut<usize> for Inner<T, D>
    where
        Self: Index<usize, Output = [T]>,
        D: DerefMut<Target = [T]>,
    {
        /// Returns a mutable reference to the row at index `i`.
        ///
        /// The returned slice has length `self.width()`.
        ///
        /// # Panics
        /// If `row >= self.height()`.
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut [T] {
            let idx = self.to_index_strict(0, row as u32);
            let w = self.dims.0 as usize;
            &mut self.data[idx..][..w]
        }
    }

    impl<T, D, Pos> Index<Pos> for Inner<T, D>
    where
        D: Deref<Target = [T]>,
        Pos: Into<Vec2u>,
    {
        type Output = T;

        /// Returns a reference to the element at position `pos`.
        ///
        /// # Panics
        /// If `pos` is out of bounds.
        #[inline]
        fn index(&self, pos: Pos) -> &T {
            let [x, y] = pos.into().0;
            &self.data[self.to_index_strict(x, y)]
        }
    }

    impl<T, D, Pos> IndexMut<Pos> for Inner<T, D>
    where
        D: DerefMut<Target = [T]>,
        Pos: Into<Vec2u>,
    {
        /// Returns a mutable reference to the element at position `pos`.
        ///
        /// # Panics
        /// If `pos` is out of bounds.
        #[inline]
        fn index_mut(&mut self, pos: Pos) -> &mut T {
            let [x, y] = pos.into().0;
            let idx = self.to_index_strict(x, y);
            &mut self.data[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::vec::vec2;

    use super::*;

    #[test]
    fn buf_new_from() {
        let buf = Buf2::new_from((3, 2), 1..);
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn buf_new() {
        let buf: Buf2<i32> = Buf2::new((3, 2));
        assert_eq!(buf.data(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn buf_new_with() {
        let buf = Buf2::new_with((3, 2), |x, y| x + y);
        assert_eq!(buf.data(), &[0, 1, 2, 1, 2, 3]);
    }

    #[test]
    fn buf_extents() {
        let buf: Buf2<()> = Buf2::new((4, 5));
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 5);
        assert_eq!(buf.stride(), 4);
    }

    #[test]
    fn buf_index_and_get() {
        let buf = Buf2::new_with((4, 5), |x, y| x * 10 + y);

        assert_eq!(buf[2usize], [2, 12, 22, 32]);

        assert_eq!(buf[[0, 0]], 0);
        assert_eq!(buf[vec2(1, 0)], 10);
        assert_eq!(buf[vec2(3, 4)], 34);

        assert_eq!(buf.get([2, 3]), Some(&23));
        assert_eq!(buf.get([4, 4]), None);
        assert_eq!(buf.get([3, 5]), None);
    }

    #[test]
    fn buf_index_mut_and_get_mut() {
        let mut buf = Buf2::new_with((4, 5), |x, y| x * 10 + y);

        buf[2usize][1] = 123;
        assert_eq!(buf[2usize], [2, 123, 22, 32]);

        buf[vec2(2, 3)] = 234;
        assert_eq!(buf[[2, 3]], 234);

        *buf.get_mut([3, 4]).unwrap() = 345;
        assert_eq!(buf.get_mut([3, 4]), Some(&mut 345));
        assert_eq!(buf.get_mut([4, 4]), None);
        assert_eq!(buf.get_mut([3, 5]), None);
    }

    #[test]
    #[should_panic = "position (x=4, y=0) out of bounds (0..4, 0..5)"]
    fn buf_index_x_out_of_bounds_should_panic() {
        let buf = Buf2::new((4, 5));
        let _: i32 = buf[[4, 0]];
    }

    #[test]
    #[should_panic = "position (x=0, y=4) out of bounds (0..5, 0..4)"]
    fn buf_index_y_out_of_bounds_should_panic() {
        let buf = Buf2::new((5, 4));
        let _: i32 = buf[[0, 4]];
    }

    #[test]
    #[should_panic = "position (x=0, y=5) out of bounds (0..4, 0..5)"]
    fn buf_index_row_out_of_bounds_should_panic() {
        let buf = Buf2::new((4, 5));
        let _: &[i32] = &buf[5usize];
    }

    #[test]
    fn buf_slice_range_full() {
        let buf: Buf2<()> = Buf2::new((4, 5));

        let slice = buf.slice(..);
        assert_eq!(slice.width(), 4);
        assert_eq!(slice.height(), 5);
        assert_eq!(slice.stride(), 4);

        let slice = buf.slice((.., ..));
        assert_eq!(slice.width(), 4);
        assert_eq!(slice.height(), 5);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_range_inclusive() {
        let buf: Buf2<()> = Buf2::new((4, 5));
        let slice = buf.slice((1..=3, 0..=3));
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 4);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_range_to() {
        let buf: Buf2<()> = Buf2::new((4, 5));

        let slice = buf.slice((..2, ..4));
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.height(), 4);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_range_from() {
        let buf: Buf2<()> = Buf2::new((4, 5));

        let slice = buf.slice((3.., 2..));
        assert_eq!(slice.width(), 1);
        assert_eq!(slice.height(), 3);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_empty_range() {
        let buf: Buf2<()> = Buf2::new((4, 5));

        let empty = buf.slice(vec2(1, 1)..vec2(1, 3));
        assert_eq!(empty.width(), 0);
        assert_eq!(empty.height(), 2);
        assert_eq!(empty.stride(), 4);

        let empty = buf.slice(vec2(1, 1)..vec2(3, 1));
        assert_eq!(empty.width(), 2);
        assert_eq!(empty.height(), 0);
        assert_eq!(empty.stride(), 4);
    }

    #[test]
    #[should_panic = "range right (5) > width (4)"]
    fn buf_slice_x_out_of_bounds_should_panic() {
        let buf: Buf2<()> = Buf2::new((4, 5));
        buf.slice((0..5, 1..3));
    }

    #[test]
    #[should_panic = "range bottom (6) > height (5)"]
    fn buf_slice_y_out_of_bounds_should_panic() {
        let buf: Buf2<()> = Buf2::new((4, 5));
        buf.slice((1..3, 0..6));
    }

    #[test]
    #[should_panic = "width (4) > stride (3)"]
    fn slice_stride_less_than_width_should_panic() {
        let _ = Slice2::new((4, 4), 3, &[0; 16]);
    }

    #[test]
    #[should_panic = "required size (19) > data length (16)"]
    fn slice_larger_than_data_should_panic() {
        let _ = Slice2::new((4, 4), 5, &[0; 16]);
    }

    #[test]
    fn slice_extents() {
        let buf: Buf2<()> = Buf2::new((10, 10));

        let slice = buf.slice((1..4, 2..8));
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 6);
        assert_eq!(slice.stride(), 10);
        assert_eq!(slice.data().len(), 5 * 10 + 3);
    }

    #[test]
    fn slice_contiguity() {
        let buf: Buf2<()> = Buf2::new((10, 10));
        // Buf2 is always contiguous
        assert!(buf.is_contiguous());

        // Empty slice is contiguous
        assert!(buf.slice((2..2, 2..8)).is_contiguous());
        assert!(buf.slice((2..8, 2..2)).is_contiguous());
        // One-row slice is contiguous
        assert!(buf.slice((2..8, 2..3)).is_contiguous());
        // Slice spanning whole width of buf is contiguous
        assert!(buf.slice((0..10, 2..8)).is_contiguous());
        assert!(buf.slice((.., 2..8)).is_contiguous());

        // Slice not spanning the width of buf is not contiguous
        assert!(!buf.slice((2..=2, 1..9)).is_contiguous());
        assert!(!buf.slice((2..4, 0..9)).is_contiguous());
        assert!(!buf.slice((2..4, 1..10)).is_contiguous());
    }

    #[test]
    #[rustfmt::skip]
    fn slice_fill() {
        let mut buf = Buf2::new((5, 4));
        let mut slice = buf.slice_mut((2.., 1..3));

        slice.fill(1);

        assert_eq!(
            buf.data(),
            &[0, 0, 0, 0, 0,
              0, 0, 1, 1, 1,
              0, 0, 1, 1, 1,
              0, 0, 0, 0, 0]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn slice_fill_with() {
        let mut buf = Buf2::new((5, 4));
        let mut slice = buf.slice_mut((2.., 1..3));

        slice.fill_with(|x, y| x + y);

        assert_eq!(
            buf.data(),
            &[0, 0, 0, 0, 0,
              0, 0, 0, 1, 2,
              0, 0, 1, 2, 3,
              0, 0, 0, 0, 0]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn slice_copy_from() {
        let mut dest = Buf2::new((5, 4));
        let src = Buf2::new_with((3, 3), |x, y| x + y);

        dest.slice_mut((1..4, 1..)).copy_from(src);

        assert_eq!(
            dest.data(),
            &[0, 0, 0, 0, 0,
              0, 0, 1, 2, 0,
              0, 1, 2, 3, 0,
              0, 2, 3, 4, 0]
        );
    }

    #[test]
    fn slice_index() {
        let buf = Buf2::new_with((5, 4), |x, y| x * 10 + y);
        let slice = buf.slice((2.., 1..3));

        assert_eq!(slice[vec2(0, 0)], 21);
        assert_eq!(slice[vec2(1, 0)], 31);
        assert_eq!(slice[vec2(2, 1)], 42);

        assert_eq!(slice.get(vec2(2, 1)), Some(&42));
        assert_eq!(slice.get(vec2(2, 2)), None);
    }

    #[test]
    fn slice_index_mut() {
        let mut buf = Buf2::new_with((5, 5), |x, y| x * 10 + y);
        let mut slice = buf.slice_mut((2.., 1..3));

        slice[[2, 1]] = 123;
        assert_eq!(slice[vec2(2, 1)], 123);

        assert_eq!(slice.get_mut(vec2(2, 1)), Some(&mut 123));
        assert_eq!(slice.get(vec2(2, 2)), None);

        buf[[2, 2]] = 321;
        let slice = buf.slice((1.., 2..));
        assert_eq!(slice[[1, 0]], 321);
    }

    #[test]
    fn slice_rows() {
        let buf = Buf2::new_with((5, 4), |x, y| x * 10 + y);
        let slice = buf.slice((2..4, 1..));

        let mut rows = slice.rows();
        assert_eq!(rows.next(), Some(&[21, 31][..]));
        assert_eq!(rows.next(), Some(&[22, 32][..]));
        assert_eq!(rows.next(), Some(&[23, 33][..]));
        assert_eq!(rows.next(), None);
    }

    #[test]
    fn slice_rows_mut() {
        let mut buf = Buf2::new_with((5, 4), |x, y| x * 10 + y);
        let mut slice = buf.slice_mut((2..4, 1..));

        let mut rows = slice.rows_mut();
        assert_eq!(rows.next(), Some(&mut [21, 31][..]));
        assert_eq!(rows.next(), Some(&mut [22, 32][..]));
        assert_eq!(rows.next(), Some(&mut [23, 33][..]));
        assert_eq!(rows.next(), None);
    }

    #[test]
    fn buf_ref_as_slice() {
        fn foo<T: AsSlice2<u32>>(buf: T) -> u32 {
            buf.as_slice2().width()
        }
        let buf = Buf2::new((2, 2));
        let w = foo(&buf);
        assert_eq!(w, buf.width());
    }

    #[test]
    fn buf_ref_as_slice_mut() {
        fn foo<T: AsMutSlice2<u32>>(mut buf: T) {
            buf.as_mut_slice2()[[1, 1]] = 42;
        }
        let mut buf = Buf2::new((2, 2));
        foo(&mut buf);
        assert_eq!(buf[[1, 1]], 42);
    }
}
