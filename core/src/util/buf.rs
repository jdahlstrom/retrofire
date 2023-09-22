use alloc::{vec, vec::Vec};
use core::fmt::{Debug, Formatter};
use core::ops::{Deref, DerefMut};

use inner::Inner;

pub trait AsSlice<T> {
    fn as_slice(&self) -> Slice2<T>;
}

pub trait AsSliceMut<T> {
    fn as_slice_mut(&mut self) -> Slice2Mut<T>;
}

/// A rectangular buffer backed by a `Vec`.
#[derive(Clone)]
#[repr(transparent)]
pub struct Buf2<T>(Inner<T, Vec<T>>);

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Slice2<'a, T>(Inner<T, &'a [T]>);

#[repr(transparent)]
pub struct Slice2Mut<'a, T>(Inner<T, &'a mut [T]>);

// Impls

impl<T> Buf2<T> {
    /// Returns a buffer with size `w` × `h`, with every element
    /// default initialized.
    pub fn new(w: usize, h: usize) -> Self
    where
        T: Clone + Default,
    {
        Self::init(w, h, T::default())
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized to `init`.
    pub fn init(w: usize, h: usize, init: T) -> Self
    where
        T: Clone,
    {
        Self::from_vec(w, h, vec![init; w * h])
    }
    /// Returns a buffer with size `w` × `h`, with every element
    /// initialized by calling `init_fn`.
    pub fn init_with<F>(w: usize, h: usize, mut init_fn: F) -> Self
    where
        F: Copy + FnMut(usize, usize) -> T,
    {
        let data = (0..h)
            .flat_map(move |y| (0..w).map(move |x| init_fn(x, y)))
            .collect();
        Self::from_vec(w, h, data)
    }
    /// Returns a buffer with size `w` × `h`, backed by `data`.
    ///
    /// # Panics
    /// If `w * h > data.len`.
    pub fn from_vec(w: usize, h: usize, data: Vec<T>) -> Self {
        Self(Inner::new(w, h, w, data))
    }

    pub fn data(&self) -> &[T] {
        self.0.data()
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        self.0.data_mut()
    }
}

impl<T> AsSlice<T> for Buf2<T> {
    #[inline]
    fn as_slice(&self) -> Slice2<T> {
        self.0.as_slice()
    }
}

impl<T> AsSliceMut<T> for Buf2<T> {
    #[inline]
    fn as_slice_mut(&mut self) -> Slice2Mut<T> {
        self.0.as_slice_mut()
    }
}

impl<T> Debug for Buf2<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.debug_fmt(f, "Buf2")
    }
}

impl<T> Deref for Buf2<T> {
    type Target = Inner<T, Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Buf2<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> Slice2<'a, T> {
    pub fn new(w: usize, h: usize, stride: usize, data: &'a [T]) -> Self {
        assert!((h - 1) * stride + w < data.len());
        Self(Inner::new(w, h, stride, data))
    }
}

impl<T> AsSlice<T> for Slice2<'_, T> {
    #[inline]
    fn as_slice(&self) -> Slice2<T> {
        self.0.as_slice()
    }
}

impl<T> Debug for Slice2<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.debug_fmt(f, "Slice2")
    }
}

impl<'a, T> Deref for Slice2<'a, T> {
    type Target = Inner<T, &'a [T]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> Slice2Mut<'a, T> {
    pub fn new(w: usize, h: usize, stride: usize, data: &'a mut [T]) -> Self {
        Self(Inner::new(w, h, stride, data))
    }
}

impl<T> AsSlice<T> for Slice2Mut<'_, T> {
    #[inline]
    fn as_slice(&self) -> Slice2<T> {
        self.0.as_slice()
    }
}

impl<T> AsSliceMut<T> for Slice2Mut<'_, T> {
    #[inline]
    fn as_slice_mut(&mut self) -> Slice2Mut<T> {
        self.0.as_slice_mut()
    }
}

impl<T> Debug for Slice2Mut<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.debug_fmt(f, "Slice2Mut")
    }
}

impl<'a, T> Deref for Slice2Mut<'a, T> {
    type Target = Inner<T, &'a mut [T]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> DerefMut for Slice2Mut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

mod inner {
    use core::fmt::Formatter;
    use core::marker::PhantomData;
    use core::ops::{Bound::*, Index, IndexMut, Range, RangeBounds};

    use crate::math::vec::Vec2i;
    use crate::util::buf::{Slice2, Slice2Mut};
    use crate::util::rect::Rect;

    #[derive(Copy, Clone)]
    pub struct Inner<T, D> {
        w: usize,
        h: usize,
        stride: usize,
        data: D,
        _pd: PhantomData<T>,
    }

    impl<T, D> Inner<T, D> {
        #[inline]
        pub fn width(&self) -> usize {
            self.w
        }
        #[inline]
        pub fn height(&self) -> usize {
            self.h
        }
        #[inline]
        pub fn stride(&self) -> usize {
            self.stride
        }
        #[inline]
        pub fn is_contiguous(&self) -> bool {
            self.stride == self.w || self.h <= 1 || self.w == 0
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

        fn resolve_range<R>(r: R, max: usize) -> Range<usize>
        where
            R: RangeBounds<usize>,
        {
            let start = match r.start_bound() {
                Included(&b) if b < max => b,
                Excluded(&b) if b <= max => b + 1,
                Unbounded => 0,
                b => panic!("start bound {b:?} out of bounds (max={max})",),
            };
            let end = match r.end_bound() {
                Included(&b) if b < max => b + 1,
                Excluded(&b) if b <= max => b,
                Unbounded => max,
                b => panic!("end bound {b:?} out of bounds (max={max})",),
            };
            start..end
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

    impl<T, D: AsRef<[T]>> Inner<T, D> {
        /// # Panics
        /// if `stride < w` or if the slice would overflow `data`.
        pub(super) fn new(w: usize, h: usize, stride: usize, data: D) -> Self {
            assert!(stride >= w);
            assert!(h == 0 || (h - 1) * stride + w <= data.as_ref().len());
            Self {
                w,
                h,
                stride,
                data,
                _pd: PhantomData,
            }
        }

        pub(super) fn data(&self) -> &[T] {
            self.data.as_ref()
        }

        /// Borrows `self` as a slice.
        pub fn as_slice(&self) -> Slice2<T> {
            Slice2(Inner::new(self.w, self.h, self.stride, self.data()))
        }

        /// Returns an iterator yielding the rows of `self` as &[T] slices.
        /// Each slice has length [`self.width()`](Self::width).
        pub fn rows<R>(&self, range: R) -> impl Iterator<Item = &[T]>
        where
            R: RangeBounds<usize>,
        {
            let range = Self::resolve_range(range, self.h);
            self.data()
                .chunks(self.stride)
                .map(|row| &row[..self.w])
                .skip(range.start)
                .take(range.end - range.start)
        }

        pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
            self.rows(..).flatten()
        }

        /// Returns a borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice(&self, rect: &Rect) -> Slice2<T> {
            let (x, y) = self.resolve_bounds(rect);
            let range = self.to_index(x.start, y.start)
                ..self.to_index(x.end, y.end - 1);
            Slice2(Inner::new(
                x.end - x.start,
                y.end - y.start,
                self.stride,
                &self.data()[range],
            ))
        }

        /// Returns a shared reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get(&self, pos: Vec2i) -> Option<&T> {
            self.to_index_checked(pos.x(), pos.y())
                .map(|i| &self.data()[i])
        }
    }

    impl<T, D: AsMut<[T]>> Inner<T, D> {
        pub fn as_slice_mut(&mut self) -> Slice2Mut<T> {
            Slice2Mut(Inner::new(self.w, self.h, self.stride, self.data_mut()))
        }
        /// Returns the data of `self` as a single mutable slice.
        pub(super) fn data_mut(&mut self) -> &mut [T] {
            self.data.as_mut()
        }
        /// Returns an iterator yielding the rows of this buffer
        /// as &mut [T]. Each slice has length [`self.width()`](Self::width).
        pub fn rows_mut<R: RangeBounds<usize>>(
            &mut self,
            range: R,
        ) -> impl Iterator<Item = &mut [T]> {
            let range = Self::resolve_range(range, self.h);
            self.data
                .as_mut()
                .chunks_exact_mut(self.stride)
                .map(|row| &mut row[..self.w])
                .skip(range.start)
                .take(range.end - range.start)
        }

        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.rows_mut(..).flatten()
        }

        /// Fills the buffer with clones of `val`.
        pub fn fill(&mut self, val: T)
        where
            T: Clone,
        {
            self.fill_with(|| val.clone())
        }
        /// Fills the buffer by invoking `f` for every element.
        pub fn fill_with(&mut self, mut f: impl FnMut() -> T) {
            if self.is_contiguous() {
                self.data_mut().fill_with(f)
            } else {
                self.rows_mut(..).for_each(|row| row.fill_with(&mut f))
            }
        }

        /// Returns a mutable reference to the element at `pos`,
        /// or `None` if `pos` is out of bounds.
        pub fn get_mut(&mut self, pos: Vec2i) -> Option<&mut T> {
            self.to_index_checked(pos.x(), pos.y())
                .map(|i| &mut self.data.as_mut()[i])
        }

        /// Returns a mutably borrowed rectangular slice of `self`.
        ///
        /// # Panics
        /// If any part of `rect` is outside the bounds of `self`.
        pub fn slice_mut(&mut self, rect: &Rect) -> Slice2Mut<T> {
            let (x, y) = self.resolve_bounds(rect);
            let range =
                self.to_index(x.start, y.start)..self.to_index(x.end, y.end);
            Slice2Mut(Inner::new(
                x.len(),
                y.len(),
                self.stride,
                &mut self.data_mut()[range],
            ))
        }
    }

    impl<T, D> Index<usize> for Inner<T, D>
    where
        D: AsRef<[T]>,
    {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &[T] {
            let idx = row * self.stride;
            &self.data()[idx..idx + self.w]
        }
    }

    impl<T, D> IndexMut<usize> for Inner<T, D>
    where
        Self: Index<usize, Output = [T]>,
        D: AsMut<[T]>,
    {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut [T] {
            let idx = row * self.stride;
            let w = self.w;
            &mut self.data_mut()[idx..idx + w]
        }
    }

    impl<T, D, Pos> Index<Pos> for Inner<T, D>
    where
        D: AsRef<[T]>,
        Pos: Into<Vec2i>,
    {
        type Output = T;

        #[inline]
        fn index(&self, pos: Pos) -> &T {
            let [x, y] = pos.into().0;
            // TODO Better error message in debug
            let idx = self
                .to_index_checked(x, y)
                .unwrap_or_else(|| self.position_out_of_bounds(x, y));
            &self.data.as_ref()[idx]
        }
    }

    impl<T, D, Pos> IndexMut<Pos> for Inner<T, D>
    where
        D: AsRef<[T]> + AsMut<[T]>,
        Pos: Into<Vec2i>,
    {
        #[inline]
        fn index_mut(&mut self, pos: Pos) -> &mut T {
            let [x, y] = pos.into().0;
            // TODO Better error message in debug
            let idx = self
                .to_index_checked(x, y)
                .unwrap_or_else(|| self.position_out_of_bounds(x, y));
            &mut self.data.as_mut()[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::vec::vec2;

    use super::*;

    #[test]
    fn buf_extents() {
        let buf = Buf2::init(8, 10, 0);
        assert_eq!(buf.width(), 8);
        assert_eq!(buf.height(), 10);
        assert_eq!(buf.stride(), 8);
    }

    #[test]
    fn buf_indexing() {
        let buf = Buf2::init_with(4, 5, |x, y| x * 10 + y);

        assert_eq!(buf[[0, 0]], 0);
        assert_eq!(buf[vec2(1, 0)], 10);
        assert_eq!(buf[vec2(3, 4)], 34);

        assert_eq!(buf.get(vec2(2, 3)), Some(&23));
        assert_eq!(buf.get(vec2(4, 4)), None);
    }

    #[test]
    fn buf_mut_indexing() {
        let mut buf = Buf2::init_with(4, 5, |x, y| x * 10 + y);

        buf[[3, 4]] = 123;
        assert_eq!(buf[vec2(3, 4)], 123);

        assert_eq!(buf.get_mut(vec2(3, 4)), Some(&mut 123));
        assert_eq!(buf.get(vec2(4, 4)), None);
    }

    #[test]
    #[should_panic]
    fn buf_index_past_end_should_panic() {
        let buf = Buf2::init(4, 5, 0);
        let _ = buf[[4, 0]];
    }
    #[test]
    #[should_panic]
    fn buf_negative_index_should_panic() {
        let buf = Buf2::init(4, 5, 0);
        let _ = buf[[3, -1]];
    }

    #[test]
    #[should_panic]
    fn slice_out_of_bounds_should_panic() {
        let buf = Buf2::init(10, 10, 0);
        buf.slice(&(0..11, 0..10).into());
    }

    #[test]
    fn slice_extents() {
        let buf = Buf2::init(10, 10, 0);

        let slice = buf.slice(&(1..4, 2..8).into());
        assert_eq!(slice.width(), 3);
        assert_eq!(slice.height(), 6);
        assert_eq!(slice.stride(), 10);
    }

    #[test]
    #[rustfmt::skip]
    fn slice_fill() {
        let mut buf = Buf2::new(5, 4);
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
    fn slice_indexing() {
        let buf = Buf2::init_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice(&(2.., 1..3).into());

        assert_eq!(slice[vec2(0, 0)], 21);
        assert_eq!(slice[vec2(1, 0)], 31);
        assert_eq!(slice[vec2(2, 1)], 42);

        assert_eq!(slice.get(vec2(2, 1)), Some(&42));
        assert_eq!(slice.get(vec2(2, 2)), None);
    }

    #[test]
    fn slice_mut_indexing() {
        let mut buf = Buf2::init_with(5, 5, |x, y| x * 10 + y);
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
        let buf = Buf2::init_with(5, 4, |x, y| x * 10 + y);
        let slice = buf.slice(&(2.., 1..3).into());

        let mut rows = slice.rows(..);
        assert_eq!(rows.next(), Some([21, 31, 41].as_ref()));
        assert_eq!(rows.next(), Some([22, 32, 42].as_ref()));
        assert_eq!(rows.next(), None);
    }
}
