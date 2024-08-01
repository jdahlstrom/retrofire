//! Various utility types and functions.

pub mod buf;
pub mod pnm;
pub mod rect;

pub mod dims {
    use super::rect::Rect;

    use crate::math::vec::Vec2u;

    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    pub struct Dims<T = u32>(pub T, pub T);

    impl<T: Copy> Dims<T> {
        pub fn width(&self) -> T {
            self.0
        }
        pub fn height(&self) -> T {
            self.1
        }
    }

    impl From<(Vec2u, Dims<u32>)> for Rect<u32> {
        fn from((tl, Dims(w, h)): (Vec2u, Dims<u32>)) -> Self {
            (tl.x()..tl.x() + w, tl.y()..tl.y() + h).into()
        }
    }
}
