use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use core::iter::repeat;
use core::ops::{Deref, DerefMut};

use inner::Inner;

//
// Traits
//

/// A trait for types that can provide a view of their data as a `Slice2`
pub trait AsSlice2<T> {
    /// Returns a borrowed `Slice2` view of `Self`.
    fn as_slice2(&self) -> Slice2<T>;
}

/// A trait for types that can provide a mutable view of their data
/// as a `MutSlice2`
pub trait AsMutSlice2<T> {
    /// Returns a mutably borrowed `MutSlice2` view of `Self`.
    fn as_mut_slice2(&mut self) -> MutSlice2<T>;
}

//
// Types
//

/// A rectangular 2D buffer that owns its elements, backed by a `Vec`.
///
/// `Buf2` stores its elements contiguously, in standard row-major order,
/// such that element (x, y) maps to element at index
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
/// let mut buf = Buf2::new_default(4, 4);
/// // Indexing with a 2D vector (x, y) yields element at row y, column x:
/// buf[vec2(2, 1)] = 123;
/// // Indexing with an usize i yields row with index i as a slice:
/// assert_eq!(&buf[1usize], &[0, 0, 123, 0]);
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
    /// Returns a buffer with size `w` × `h`, with elements initialized
    /// with values from `init` in row-major order.
    ///
    /// # Panics
    /// If there are fewer than `w * h` elements in `init`.
    pub fn new<I>(w: usize, h: usize, init: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = init.into_iter().take(w * h).collect();
        assert_eq!(data.len(), w * h);
        Self(Inner::new(w, h, w, data))
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized by calling `T::default()`.
    pub fn new_default(w: usize, h: usize) -> Self
    where
        T: Clone + Default,
    {
        Self::new(w, h, repeat(T::default()))
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized by calling `init_fn(x, y)` where x is the column index
    /// and y the row index of the element being initialized.
    pub fn new_with<F>(w: usize, h: usize, mut init_fn: F) -> Self
    where
        F: Copy + FnMut(usize, usize) -> T,
    {
        let init = (0..h).flat_map(move |y| {
            (0..w).map(move |x| init_fn(x, y)) //
        });
        Self::new(w, h, init)
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
    pub fn new(
        width: usize,
        height: usize,
        stride: usize,
        data: &'a [T],
    ) -> Self {
        Self(Inner::new(width, height, stride, data))
    }
}

impl<'a, T> MutSlice2<'a, T> {
    /// Returns a new `MutSlice2` view to `data` with dimensions `w` and `h`
    /// and stride `stride`.
    pub fn new(w: usize, h: usize, stride: usize, data: &'a mut [T]) -> Self {
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
        self.0.as_mut_slice()
    }
}
impl<T> AsMutSlice2<T> for MutSlice2<'_, T> {
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<T> {
        self.0.as_mut_slice()
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

    use crate::math::vec::Vec2i;
    use crate::util::buf::{MutSlice2, Slice2};
    use crate::util::rect::Rect;

    /// A helper type that abstracts over owned and borrowed buffers.
    /// The types `Buf2`, `Slice2`, and `MutSlice2` deref to `Inner`.
    #[derive(Copy, Clone)]
    pub struct Inner<T, D> {
        w: usize,
        h: usize,
        stride: usize,
        data: D,
        _pd: PhantomData<T>,
    }

    impl<T, D> Inner<T, D> {
        /// Returns the width of `self`.
        #[inline]
        pub fn width(&self) -> usize {
            self.w
        }
        /// Returns the height of `self`.
        #[inline]
        pub fn height(&self) -> usize {
            self.h
        }
        /// Returns the stride of `self`.
        #[inline]
        pub fn stride(&self) -> usize {
            self.stride
        }
        /// Returns whether the rows of `self` are stored contiguously
        /// in memory. `Buf2` instances are always contiguous. `Slice2`
        /// and `MutSlice2` instances are contiguous if their width equals
        /// their stride, if their height is 1, or if they are empty.
        pub fn is_contiguous(&self) -> bool {
            self.stride == self.w || self.h <= 1 || self.w == 0
        }
        /// Returns whether `self` has no elements (if its width or height is 0).
        pub fn is_empty(&self) -> bool {
            self.w == 0 || self.h == 0
        }

        #[inline]
        fn to_index(&self, x: usize, y: usize) -> usize {
            y * self.stride + x
        }
        #[inline]
        fn to_index_checked(&self, x: i32, y: i32) -> Option<usize> {
            if x < 0 || (x as usize) >= self.w {
                return None;
            }
            if y < 0 || (y as usize) >= self.h {
                return None;
            }
            Some(y as usize * self.stride + x as usize)
        }

        fn resolve_bounds(&self, rect: &Rect) -> (Range<usize>, Range<usize>) {
            let l = rect.left.unwrap_or(0);
            let t = rect.top.unwrap_or(0);
            let r = rect.right.unwrap_or(self.w);
            let b = rect.bottom.unwrap_or(self.h);

            if l > self.w || t > self.h {
                self.position_out_of_bounds(l as _, t as _);
            }
            if r > self.w || b > self.h {
                self.position_out_of_bounds(l as _, t as _);
            }
            (l..r, t..b)
        }

        #[cold]
        #[inline(never)]
        #[track_caller]
        fn position_out_of_bounds(&self, x: i32, y: i32) -> ! {
            panic!(
                "position (x={x}, y={y}) out of bounds (0..{}, 0..{})",
                self.w, self.h
            )
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
        pub(super) fn new(w: usize, h: usize, stride: usize, data: D)
            -> Self
        {
            assert!(stride >= w);
            assert!(h == 0 || (h - 1) * stride + w <= data.len());
            Self { w, h, stride, data, _pd: PhantomData, }
        }

        /// Returns the data of `self` as a linear slice.
        pub(super) fn data(&self) -> &[T] {
            &self.data
        }

        /// Borrows `self` as a `Slice2`.
        pub fn as_slice2(&self) -> Slice2<T> {
            Slice2(Inner::new(self.w, self.h, self.stride, self.data()))
        }

        /// Returns a borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice(&self, rect: &Rect) -> Slice2<T> {
            let (x, y) = self.resolve_bounds(rect);
            let start = self.to_index(x.start, y.start);
            let end = self.to_index(x.end, y.end - 1).max(start);
            Slice2::new(
                x.end - x.start,
                y.end - y.start,
                self.stride,
                &self.data()[start..end],
            )
        }

        /// Returns a reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get(&self, pos: Vec2i) -> Option<&T> {
            self.to_index_checked(pos.x(), pos.y())
                .map(|i| &self.data()[i])
        }

        /// Returns an iterator over the rows of `self` as `&[T]` slices.
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows(&self) -> impl Iterator<Item = &[T]> {
            self.data()
                .chunks(self.stride)
                .map(|row| &row[..self.w])
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
        pub fn as_mut_slice(&mut self) -> MutSlice2<T> {
            MutSlice2::new(self.w, self.h, self.stride, self.data_mut())
        }
        /// Returns the data of `self` as a single mutable slice.
        pub(super) fn data_mut(&mut self) -> &mut [T] {
            &mut self.data
        }
        /// Returns an iterator over the rows of this buffer as &mut [T].
        /// The length of each slice equals [`self.width()`](Self::width).
        pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
            self.data
                .chunks_exact_mut(self.stride)
                .map(|row| &mut row[..self.w])
        }

        /// Returns an iterator over all the elements of `self` in row-major
        /// order: first the elements on row 0 from left to right, followed
        /// by the elements on row 1, and so on.
        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.rows_mut().flatten()
        }

        /// Fills the buffer with clones of `val`.
        pub fn fill(&mut self, val: T)
        where
            T: Clone,
        {
            if self.is_contiguous() {
                self.data_mut().fill(val);
            } else {
                self.rows_mut()
                    .for_each(|row| row.fill(val.clone()));
            }
        }
        /// Fills the buffer by invoking `f(x, y)` for every element, where
        /// `x` and `y` are the column and row of the element, respectively.
        pub fn fill_with<F>(&mut self, mut fill_fn: F)
        where
            F: Copy + FnMut(usize, usize) -> T,
        {
            let w = self.w;
            let mut fill = (0..self.h).flat_map(move |y| {
                (0..w).map(move |x| fill_fn(x, y)) //
            });
            if self.is_contiguous() {
                self.data_mut().fill_with(|| fill.next().unwrap())
            } else {
                self.rows_mut().for_each(|row| {
                    row.fill_with(|| fill.next().unwrap()) //
                })
            }
        }

        /// Returns a mutable reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get_mut(&mut self, pos: Vec2i) -> Option<&mut T> {
            self.to_index_checked(pos.x(), pos.y())
                .map(|i| &mut self.data[i])
        }

        /// Returns a mutably borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice_mut(&mut self, rect: &Rect) -> MutSlice2<T> {
            let (x, y) = self.resolve_bounds(rect);
            let range =
                self.to_index(x.start, y.start)..self.to_index(x.end, y.end);
            MutSlice2(Inner::new(
                x.len(),
                y.len(),
                self.stride,
                &mut self.data_mut()[range],
            ))
        }
    }

    impl<T, D: Deref<Target = [T]>> Index<usize> for Inner<T, D> {
        type Output = [T];

        /// Returns a reference to the row of `self` at index `i`.
        /// The returned slice has length `self.width()`.
        #[inline]
        fn index(&self, i: usize) -> &[T] {
            &self.data()[i * self.stride..][..self.w]
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
            let idx = row * self.stride;
            let w = self.w;
            &mut self.data_mut()[idx..idx + w]
        }
    }

    impl<T, D, Pos> Index<Pos> for Inner<T, D>
    where
        D: Deref<Target = [T]>,
        Pos: Into<Vec2i>,
    {
        type Output = T;

        /// Returns a reference to the element of `self` at position `pos`.
        /// # Panics
        /// If `pos` is out of bounds of `self`.
        #[inline]
        fn index(&self, pos: Pos) -> &T {
            let [x, y] = pos.into().0;
            // TODO Better error message in debug
            let idx = self
                .to_index_checked(x, y)
                .unwrap_or_else(|| self.position_out_of_bounds(x, y));
            &self.data[idx]
        }
    }

    impl<T, D, Pos> IndexMut<Pos> for Inner<T, D>
    where
        D: DerefMut<Target = [T]>,
        Pos: Into<Vec2i>,
    {
        /// Returns a mutable reference to the element of `self`
        /// at position `pos`.
        /// # Panics
        /// If `pos` is out of bounds of `self`.
        #[inline]
        fn index_mut(&mut self, pos: Pos) -> &mut T {
            let [x, y] = pos.into().0;
            // TODO Better error message in debug
            let idx = self
                .to_index_checked(x, y)
                .unwrap_or_else(|| self.position_out_of_bounds(x, y));
            &mut self.data[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::vec::vec2;

    use super::*;

    #[test]
    fn buf_new() {
        let buf = Buf2::new(3, 2, 1..);
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn buf_new_default() {
        let buf: Buf2<i32> = Buf2::new_default(3, 2);
        assert_eq!(buf.data(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn buf_new_with() {
        let buf = Buf2::new_with(3, 2, |x, y| x + y);
        assert_eq!(buf.data(), &[0, 1, 2, 1, 2, 3]);
    }

    #[test]
    fn buf_extents() {
        let buf: Buf2<()> = Buf2::new_default(8, 10);
        assert_eq!(buf.width(), 8);
        assert_eq!(buf.height(), 10);
        assert_eq!(buf.stride(), 8);
    }

    #[test]
    fn buf_indexing() {
        let buf = Buf2::new_with(4, 5, |x, y| x * 10 + y);

        assert_eq!(buf[[0, 0]], 0);
        assert_eq!(buf[vec2(1, 0)], 10);
        assert_eq!(buf[vec2(3, 4)], 34);

        assert_eq!(buf.get(vec2(2, 3)), Some(&23));
        assert_eq!(buf.get(vec2(4, 4)), None);
    }

    #[test]
    fn buf_mut_indexing() {
        let mut buf = Buf2::new_with(4, 5, |x, y| x * 10 + y);

        buf[[3, 4]] = 123;
        assert_eq!(buf[vec2(3, 4)], 123);

        assert_eq!(buf.get_mut(vec2(3, 4)), Some(&mut 123));
        assert_eq!(buf.get(vec2(4, 4)), None);
    }

    #[test]
    #[should_panic]
    fn buf_index_past_end_should_panic() {
        let buf = Buf2::new_default(4, 5);
        let () = buf[[4, 0]];
    }
    #[test]
    #[should_panic]
    fn buf_negative_index_should_panic() {
        let buf = Buf2::new_default(4, 5);
        let () = buf[[3, -1]];
    }

    #[test]
    #[should_panic]
    fn slice_out_of_bounds_should_panic() {
        let buf: Buf2<()> = Buf2::new_default(4, 5);
        buf.slice(&(0..11, 0..10).into());
    }

    #[test]
    #[should_panic]
    fn slice_stride_less_than_width_should_panic() {
        let _ = Slice2::new(4, 4, 3, &[0; 16]);
    }

    #[test]
    fn slice_extents() {
        let buf: Buf2<()> = Buf2::new_default(10, 10);

        let slice = buf.slice(&(1..4, 2..8).into());
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 6);
        assert_eq!(slice.stride(), 10);
        assert_eq!(slice.data().len(), 5 * 10 + 3);
    }

    #[test]
    fn slice_contiguity() {
        let buf: Buf2<()> = Buf2::new_default(10, 10);

        // Empty slice is contiguous
        assert!(buf.slice(&(2..2, 2..8).into()).is_contiguous());
        assert!(buf.slice(&(2..8, 2..2).into()).is_contiguous());
        // One-row slice is contiguous
        assert!(buf.slice(&(2..8, 2..3).into()).is_contiguous());
        // Slice spanning whole width of buf is contiguous
        assert!(buf.slice(&(0..10, 2..8).into()).is_contiguous());
        assert!(buf.slice(&(.., 2..8).into()).is_contiguous());

        assert!(!buf.slice(&(2..=2, 1..9).into()).is_contiguous());
        assert!(!buf.slice(&(2..4, 0..9).into()).is_contiguous());
        assert!(!buf.slice(&(2..4, 1..10).into()).is_contiguous());
    }

    #[test]
    #[rustfmt::skip]
    fn slice_fill() {
        let mut buf = Buf2::new_default(5, 4);
        let mut slice = buf.slice_mut(&(2.., 1..3).into());

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
        let mut buf = Buf2::new_default(5, 4);
        let mut slice = buf.slice_mut(&(2.., 1..3).into());

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
    fn slice_indexing() {
        let buf = Buf2::new_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice(&(2.., 1..3).into());

        assert_eq!(slice[vec2(0, 0)], 21);
        assert_eq!(slice[vec2(1, 0)], 31);
        assert_eq!(slice[vec2(2, 1)], 42);

        assert_eq!(slice.get(vec2(2, 1)), Some(&42));
        assert_eq!(slice.get(vec2(2, 2)), None);
    }

    #[test]
    fn slice_mut_indexing() {
        let mut buf = Buf2::new_with(5, 5, |x, y| x * 10 + y);
        let mut slice = buf.slice_mut(&(2.., 1..3).into());

        slice[[2, 1]] = 123;
        assert_eq!(slice[vec2(2, 1)], 123);

        assert_eq!(slice.get_mut(vec2(2, 1)), Some(&mut 123));
        assert_eq!(slice.get(vec2(2, 2)), None);

        buf[[2, 2]] = 321;
        let slice = buf.slice(&(1.., 2..).into());
        assert_eq!(slice[[1, 0]], 321);
    }

    #[test]
    fn slice_rows() {
        let buf = Buf2::new_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice(&(2.., 1..3).into());

        let mut rows = slice.rows();
        assert_eq!(rows.next(), Some(&[21, 31, 41][..]));
        assert_eq!(rows.next(), Some(&[22, 32, 42][..]));
        assert_eq!(rows.next(), None);
    }
}
