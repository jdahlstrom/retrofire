//! Two-dimensional buffers, with owned and borrowed variants.
//!
//! Useful for storing pixel data of any kind, among other things.

use alloc::{vec, vec::Vec};
use core::fmt::{Debug, Formatter};
use core::ops::{Deref, DerefMut};

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
/// such that the coordinate pair (x, y) maps to the index
/// ```text
/// buf.width() * y + x
/// ```
/// in the backing vector.
///
/// # Examples
/// ```
/// # use retrofire_core::util::buf::*;
/// # use retrofire_core::math::vec::*;
/// // Elements initialized with `Default::default()`
/// let mut buf = Buf2::new(4, 4);
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
/// A `Slice2` may be discontiguous:
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
    /// Returns a buffer with size `w` × `h`, with elements initialized in
    /// row-major order with values yielded by `init`.
    ///
    /// # Panics
    /// If there are fewer than `w * h` elements in `init`.
    pub fn new_from<I>(w: u32, h: u32, init: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = init.into_iter().take((w * h) as usize).collect();
        assert_eq!(data.len(), (w * h) as usize);
        Self(Inner::new(w, h, w, data))
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized by calling `T::default()`.
    pub fn new(w: u32, h: u32) -> Self
    where
        T: Clone + Default,
    {
        let data = vec![T::default(); (w * h) as usize];
        Self(Inner::new(w, h, w, data))
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized by calling `init_fn(x, y)` where x is the column index
    /// and y the row index of the element being initialized.
    pub fn new_with<F>(w: u32, h: u32, init_fn: F) -> Self
    where
        F: Clone + FnMut(u32, u32) -> T,
    {
        let init = (0..h).flat_map(move |y| {
            let mut init_fn = init_fn.clone();
            (0..w).map(move |x| init_fn(x, y)) //
        });
        Self::new_from(w, h, init)
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
    /// Returns a new `Slice2` view to `data` with dimensions `w` and `h`
    /// and stride `stride`.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::util::buf::Slice2;
    /// let data = &[0, 1, 2, 3, 4, 5, 6];
    /// let slice = Slice2::new(2, 2, 3, data);
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
    pub fn new(width: u32, height: u32, stride: u32, data: &'a [T]) -> Self {
        Self(Inner::new(width, height, stride, data))
    }
}

impl<'a, T> MutSlice2<'a, T> {
    /// Returns a new `MutSlice2` view to `data` with dimensions `w` and `h`
    /// and stride `stride`.
    ///
    /// See [`Slice2::new`] for more information.
    pub fn new(w: u32, h: u32, stride: u32, data: &'a mut [T]) -> Self {
        Self(Inner::new(w, h, stride, data))
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
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.debug_fmt(f, "Buf2")
    }
}
impl<T> Debug for Slice2<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.debug_fmt(f, "Slice2")
    }
}
impl<T> Debug for MutSlice2<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
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

mod inner {
    use core::fmt::Formatter;
    use core::marker::PhantomData;
    use core::ops::{Deref, DerefMut, Index, IndexMut, Range};

    use crate::math::vec::Vec2u;
    use crate::util::rect::Rect;

    use super::{AsSlice2, MutSlice2, Slice2};

    /// A helper type that abstracts over owned and borrowed buffers.
    /// The types `Buf2`, `Slice2`, and `MutSlice2` deref to `Inner`.
    #[derive(Copy, Clone)]
    pub struct Inner<T, D> {
        w: u32,
        h: u32,
        stride: u32,
        data: D,
        _pd: PhantomData<T>,
    }

    impl<T, D> Inner<T, D> {
        /// Returns the width of `self`.
        #[inline]
        pub fn width(&self) -> u32 {
            self.w
        }
        /// Returns the height of `self`.
        #[inline]
        pub fn height(&self) -> u32 {
            self.h
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
            self.stride == self.w || self.h <= 1 || self.w == 0
        }
        /// Returns whether `self` contains no elements.
        pub fn is_empty(&self) -> bool {
            self.w == 0 || self.h == 0
        }

        #[inline]
        fn to_index(&self, x: u32, y: u32) -> usize {
            (y * self.stride + x) as usize
        }
        #[inline]
        fn to_index_strict(&self, x: u32, y: u32) -> usize {
            self.to_index_checked(x, y).unwrap_or_else(|| {
                panic!(
                    "position (x={x}, y={y}) out of bounds (0..{}, 0..{})",
                    self.w, self.h
                )
            })
        }
        #[inline]
        fn to_index_checked(&self, x: u32, y: u32) -> Option<usize> {
            (x < self.w && y < self.h).then(|| self.to_index(x, y))
        }

        fn resolve_bounds(&self, rect: &Rect<u32>) -> (u32, u32, Range<usize>) {
            let l = rect.left.unwrap_or(0);
            let t = rect.top.unwrap_or(0);
            let r = rect.right.unwrap_or(self.w);
            let b = rect.bottom.unwrap_or(self.h);

            assert!(l <= r, "range left ({l}) > right ({r})");
            assert!(t <= b, "range top ({l}) > bottom ({r})");
            assert!(r <= self.w, "range right ({r}) > width ({})", self.w);
            assert!(b <= self.h, "range bottom ({b}) > height ({})", self.h);

            let start = self.to_index(l, t);
            // Slice end is the end of the last row
            let end = if b == t {
                self.to_index(r, t)
            } else {
                // b != 0 because b > t
                self.to_index(r, b - 1)
            };
            (r - l, b - t, start..end)
        }

        /// A helper for implementing `Debug`.
        pub(super) fn debug_fmt(
            &self,
            f: &mut Formatter,
            name: &str,
        ) -> core::fmt::Result {
            f.debug_struct(name)
                .field("w", &self.w)
                .field("h", &self.h)
                .field("stride", &self.stride)
                .finish()
        }
    }

    impl<T, D: Deref<Target = [T]>> Inner<T, D> {
        /// # Panics
        /// if `stride < w` or if the slice would overflow `data`.
        #[rustfmt::skip]
        pub(super) fn new(w: u32, h: u32, stride: u32, data: D) -> Self {
            let len = data.len() as u32;
            assert!(w <= stride, "width ({w}) > stride ({stride})");
            assert!(
                h <= 1 || stride <= len,
                "stride ({stride}) > data length ({len})"
            );
            assert!(h <= len, "height ({h}) > date length ({len})");
            if h > 0 {
                let size = (h - 1) * stride + w;
                assert!(
                    size <= len,
                    "required size ({size}) > data length ({len})"
                );
            }
            Self { w, h, stride, data, _pd: PhantomData }
        }

        /// Returns the data of `self` as a linear slice.
        pub(super) fn data(&self) -> &[T] {
            &self.data
        }

        /// Borrows `self` as a `Slice2`.
        pub fn as_slice2(&self) -> Slice2<T> {
            Slice2::new(self.w, self.h, self.stride, &self.data)
        }

        /// Returns a borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice(&self, rect: impl Into<Rect>) -> Slice2<T> {
            let (w, h, rg) = self.resolve_bounds(&rect.into());
            Slice2::new(w, h, self.stride, &self.data[rg])
        }

        /// Returns a reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get(&self, pos: impl Into<Vec2u>) -> Option<&T> {
            let [x, y] = pos.into().0;
            self.to_index_checked(x, y).map(|i| &self.data[i])
        }

        /// Returns an iterator over the rows of `self` as `&[T]` slices.
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows(&self) -> impl Iterator<Item = &[T]> {
            self.data
                .chunks(self.stride as usize)
                .map(|row| &row[..self.w as usize])
        }

        /// Returns an iterator over all the elements of `self` in row-major
        /// order: first the elements on row 0 from left to right, followed
        /// by the elements on row 1, and so on.
        pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
            self.rows().flatten()
        }
    }

    impl<T, D: DerefMut<Target = [T]>> Inner<T, D> {
        /// Returns a mutably borrowed rectangular slice of `self`.
        pub fn as_mut_slice2(&mut self) -> MutSlice2<T> {
            MutSlice2::new(self.w, self.h, self.stride, &mut self.data)
        }
        /// Returns the data of `self` as a single mutable slice.
        pub(super) fn data_mut(&mut self) -> &mut [T] {
            &mut self.data
        }
        /// Returns an iterator over the rows of this buffer as &mut [T].
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
            self.data
                .chunks_mut(self.stride as usize)
                .map(|row| &mut row[..self.w as usize])
        }

        /// Returns an iterator over all the elements of `self` in row-major
        /// order: first the elements on row 0 from left to right, followed
        /// by the elements on row 1, and so on.
        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.rows_mut().flatten()
        }

        /// Fills `self` with clones of `val`.
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
        /// Fills `self` by invoking `f(x, y)` for every element, where
        /// `x` and `y` are the column and row of the element, respectively.
        pub fn fill_with<F>(&mut self, mut fill_fn: F)
        where
            F: Copy + FnMut(u32, u32) -> T,
        {
            let w = self.w;
            let mut fill = (0..self.h).flat_map(move |y| {
                (0..w).map(move |x| fill_fn(x, y)) //
            });
            if self.is_contiguous() {
                self.data.fill_with(|| fill.next().unwrap());
            } else {
                self.rows_mut().for_each(|row| {
                    row.fill_with(|| fill.next().unwrap()); //
                })
            }
        }

        /// Copies each element in `src` to the same position in `self`.
        ///
        /// This operation is often called "blitting".
        ///
        /// # Panics
        /// if the dimensions of `self` and `src` don't match.
        #[doc(alias = "blit")]
        pub fn copy_from(&mut self, src: impl AsSlice2<T>)
        where
            T: Copy,
        {
            let src = src.as_slice2();
            assert_eq!(
                self.w, src.w,
                "width ({}) != source width ({})",
                self.w, src.w
            );
            assert_eq!(
                self.h, src.h,
                "height ({}) != source height ({})",
                self.h, src.h
            );
            for (dest, src) in self.rows_mut().zip(src.rows()) {
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
            let (w, h, rg) = self.resolve_bounds(&rect.into());
            MutSlice2(Inner::new(w, h, self.stride, &mut self.data[rg]))
        }
    }

    impl<T, D: Deref<Target = [T]>> Index<usize> for Inner<T, D> {
        type Output = [T];

        /// Returns a reference to the row of `self` at index `i`.
        /// The returned slice has length `self.width()`.
        #[inline]
        fn index(&self, i: usize) -> &[T] {
            let idx = self.to_index_strict(0, i as u32);
            &self.data[idx..][..self.w as usize]
        }
    }

    impl<T, D> IndexMut<usize> for Inner<T, D>
    where
        Self: Index<usize, Output = [T]>,
        D: DerefMut<Target = [T]>,
    {
        /// Returns a mutable reference to the row of `self` at index `i`.
        /// The returned slice has length `self.width()`.
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut [T] {
            let idx = self.to_index_strict(0, row as u32);
            let w = self.w as usize;
            &mut self.data[idx..][..w]
        }
    }

    impl<T, D, Pos> Index<Pos> for Inner<T, D>
    where
        D: Deref<Target = [T]>,
        Pos: Into<Vec2u>,
    {
        type Output = T;

        /// Returns a reference to the element of `self` at position `pos`.
        /// # Panics
        /// If `pos` is out of bounds of `self`.
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
        /// Returns a mutable reference to the element of `self`
        /// at position `pos`.
        /// # Panics
        /// If `pos` is out of bounds of `self`.
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
        let buf = Buf2::new_from(3, 2, 1..);
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn buf_new() {
        let buf: Buf2<i32> = Buf2::new(3, 2);
        assert_eq!(buf.data(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn buf_new_with() {
        let buf = Buf2::new_with(3, 2, |x, y| x + y);
        assert_eq!(buf.data(), &[0, 1, 2, 1, 2, 3]);
    }

    #[test]
    fn buf_extents() {
        let buf: Buf2<()> = Buf2::new(8, 10);
        assert_eq!(buf.width(), 8);
        assert_eq!(buf.height(), 10);
        assert_eq!(buf.stride(), 8);
    }

    #[test]
    fn buf_index_and_get() {
        let buf = Buf2::new_with(4, 5, |x, y| x * 10 + y);

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
        let mut buf = Buf2::new_with(4, 5, |x, y| x * 10 + y);

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
        let buf = Buf2::new(4, 5);
        let _: i32 = buf[[4, 0]];
    }

    #[test]
    #[should_panic = "position (x=0, y=4) out of bounds (0..5, 0..4)"]
    fn buf_index_y_out_of_bounds_should_panic() {
        let buf = Buf2::new(5, 4);
        let _: i32 = buf[[0, 4]];
    }

    #[test]
    #[should_panic = "position (x=0, y=5) out of bounds (0..4, 0..5)"]
    fn buf_index_row_out_of_bounds_should_panic() {
        let buf = Buf2::new(4, 5);
        let _: &[i32] = &buf[5usize];
    }

    #[test]
    fn buf_slice_range_full() {
        let buf: Buf2<()> = Buf2::new(4, 5);

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
        let buf: Buf2<()> = Buf2::new(4, 5);

        let slice = buf.slice((1..=3, 0..=3));
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 4);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_range_to() {
        let buf: Buf2<()> = Buf2::new(4, 5);

        let slice = buf.slice((..2, ..4));
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.height(), 4);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_range_from() {
        let buf: Buf2<()> = Buf2::new(4, 5);

        let slice = buf.slice((3.., 2..));
        assert_eq!(slice.width(), 1);
        assert_eq!(slice.height(), 3);
        assert_eq!(slice.stride(), 4);
    }

    #[test]
    fn buf_slice_empty_range() {
        let buf: Buf2<()> = Buf2::new(4, 5);

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
        let buf: Buf2<()> = Buf2::new(4, 5);
        buf.slice((0..5, 1..3));
    }

    #[test]
    #[should_panic = "range bottom (6) > height (5)"]
    fn buf_slice_y_out_of_bounds_should_panic() {
        let buf: Buf2<()> = Buf2::new(4, 5);
        buf.slice((1..3, 0..6));
    }

    #[test]
    #[should_panic = "width (4) > stride (3)"]
    fn slice_stride_less_than_width_should_panic() {
        let _ = Slice2::new(4, 4, 3, &[0; 16]);
    }

    #[test]
    #[should_panic = "required size (19) > data length (16)"]
    fn slice_larger_than_data_should_panic() {
        let _ = Slice2::new(4, 4, 5, &[0; 16]);
    }

    #[test]
    fn slice_extents() {
        let buf: Buf2<()> = Buf2::new(10, 10);

        let slice = buf.slice((1..4, 2..8));
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 6);
        assert_eq!(slice.stride(), 10);
        assert_eq!(slice.data().len(), 5 * 10 + 3);
    }

    #[test]
    fn slice_contiguity() {
        let buf: Buf2<()> = Buf2::new(10, 10);
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
        let mut buf = Buf2::new(5, 4);
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
        let mut buf = Buf2::new(5, 4);
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
        let mut dest = Buf2::new(5, 4);
        let src = Buf2::new_with(3, 3, |x, y| x + y);

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
        let buf = Buf2::new_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice((2.., 1..3));

        assert_eq!(slice[vec2(0, 0)], 21);
        assert_eq!(slice[vec2(1, 0)], 31);
        assert_eq!(slice[vec2(2, 1)], 42);

        assert_eq!(slice.get(vec2(2, 1)), Some(&42));
        assert_eq!(slice.get(vec2(2, 2)), None);
    }

    #[test]
    fn slice_index_mut() {
        let mut buf = Buf2::new_with(5, 5, |x, y| x * 10 + y);
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
        let buf = Buf2::new_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice((2..4, 1..));

        let mut rows = slice.rows();
        assert_eq!(rows.next(), Some(&[21, 31][..]));
        assert_eq!(rows.next(), Some(&[22, 32][..]));
        assert_eq!(rows.next(), Some(&[23, 33][..]));
        assert_eq!(rows.next(), None);
    }

    #[test]
    fn slice_rows_mut() {
        let mut buf = Buf2::new_with(5, 4, |x, y| x * 10 + y);
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
        let buf = Buf2::new(2, 2);
        let w = foo(&buf);
        assert_eq!(w, buf.width());
    }

    #[test]
    fn buf_ref_as_slice_mut() {
        fn foo<T: AsMutSlice2<u32>>(mut buf: T) {
            buf.as_mut_slice2()[[1, 1]] = 42;
        }
        let mut buf = Buf2::new(2, 2);
        foo(&mut buf);
        assert_eq!(buf[[1, 1]], 42);
    }
}
