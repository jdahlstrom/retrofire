#![allow(clippy::len_without_is_empty)]

use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut,
    Mul, MulAssign, Neg, Range, Sub, SubAssign
};

use crate::{ApproxEq, Linear};
use crate::rand::{Distrib, Random, Uniform};

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

pub const ZERO: Vec4 = dir(0.0, 0.0, 0.0);
pub const ORIGIN: Vec4 = pt(0.0, 0.0, 0.0);

pub const X: Vec4 = vec4(1.0, 0.0, 0.0, 0.0);
pub const Y: Vec4 = vec4(0.0, 1.0, 0.0, 0.0);
pub const Z: Vec4 = vec4(0.0, 0.0, 1.0, 0.0);
pub const W: Vec4 = vec4(0.0, 0.0, 0.0, 1.0);

pub const fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4 { x, y, z, w }
}

pub const fn pt(x: f32, y: f32, z: f32) -> Vec4 {
    vec4(x, y, z, 1.0)
}

pub const fn dir(x: f32, y: f32, z: f32) -> Vec4 {
    vec4(x, y, z, 0.0)
}

impl Vec4 {
    /// Returns the length of `self`.
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns self divided by `self.len()`.
    /// Precondition: `self.len() != 0.0`
    #[must_use]
    pub fn normalize(self) -> Vec4 {
        debug_assert_ne!(self, ZERO, "cannot normalize a zero vector");
        self / self.len()
    }

    /// Returns `self` component-wise mapped with `f`.
    pub fn map(self, f: impl Fn(f32) -> f32) -> Vec4 {
        vec4(f(self.x), f(self.y), f(self.z), f(self.w))
    }
    /// Returns the result of `f` applied component-wise to `self` and `rhs`.
    pub fn zip_map(self, rhs: Vec4, f: impl Fn(f32, f32) -> f32) -> Vec4 {
        vec4(f(self.x, rhs.x), f(self.y, rhs.y), f(self.z, rhs.z), f(self.w, rhs.w))
    }

    /// Returns the dot product of `self` and `rhs`.
    pub fn dot(self, rhs: Vec4) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    /// Returns the cross product of `self` and `rhs`.
    pub fn cross(self, rhs: Vec4) -> Vec4 {
        debug_assert_eq!((self.w, rhs.w), (0.0, 0.0), "a × b requires a.w = b.w = 0");
        vec4(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
            0.0,
        )
    }
    /// Returns the scalar projection of self onto the argument.
    #[must_use]
    pub fn scalar_project(self, onto: Vec4) -> f32 {
        debug_assert_ne!(onto, ZERO, "cannot project onto a zero vector");
        self.dot(onto) / onto.dot(onto)
    }
    /// Returns the vector projection of self onto the argument.
    #[must_use]
    pub fn vector_project(self, onto: Vec4) -> Vec4 {
        self.scalar_project(onto) * onto
    }

    /// Returns the reflection of `self` about the argument.
    #[must_use]
    pub fn reflect(self, about: Vec4) -> Vec4 {
        2.0 * self.vector_project(about) - self
    }

    pub fn clamp(self, min: f32, max: f32) -> Vec4 {
        self.map(|c| c.clamp(min, max))
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;

    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        [&self.x, &self.y, &self.z, &self.w][i]
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        [&mut self.x, &mut self.y, &mut self.z, &mut self.w][i]
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
        vec4(x, y, z, w)
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from([x, y, z, w]: [f32; 4]) -> Self {
        vec4(x, y, z, w)
    }
}

impl From<Vec4> for [f32; 4] {
    fn from(Vec4 { x, y, z, w }: Vec4) -> Self {
        [x, y, z, w]
    }
}

impl Add<Vec4> for Vec4 {
    type Output = Vec4;

    fn add(self, rhs: Vec4) -> Vec4 {
        self.zip_map(rhs, Add::add)
    }
}
impl AddAssign<Vec4> for Vec4 {
    fn add_assign(&mut self, rhs: Vec4) {
        *self = *self + rhs;
    }
}

impl Sub<Vec4> for Vec4 {
    type Output = Vec4;

    fn sub(self, rhs: Vec4) -> Vec4 {
        self.zip_map(rhs, Sub::sub)
    }
}
impl SubAssign<Vec4> for Vec4 {
    fn sub_assign(&mut self, rhs: Vec4) {
        *self = *self - rhs;
    }
}

impl Neg for Vec4 {
    type Output = Vec4;

    fn neg(self) -> Vec4 {
        self.map(Neg::neg)
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: f32) -> Vec4 {
        self.map(|e| e * rhs)
    }
}
impl MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: f32) -> Vec4 {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        rhs * self
    }
}

impl Linear<f32> for Vec4 {
    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, s: f32) -> Self {
        s * self
    }
}

impl ApproxEq for Vec4 {
    type Scalar = f32;

    fn epsilon(self) -> f32 {
        self.len().epsilon()
    }
    fn abs_diff(self, rhs: Self) -> f32 {
        self.zip_map(rhs, f32::abs_diff).len()
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(2);
        write!(f, "({:.p$}, {:.p$}, {:.p$}, {:.p$})",
               self.x, self.y, self.z, self.w, p = p)
    }
}

impl fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(4);
        write!(f, "Vec4({:.p$}, {:.p$}, {:.p$}, {:.p$})",
               self.x, self.y, self.z, self.w, p = p)
    }
}


pub struct UnitDir;

impl Distrib<Vec4> for UnitDir {
    fn from(&self, r: &mut Random) -> Vec4 {
        let d = Uniform(-1.0..1.0_f32);
        let x = d.from(r);
        let y = d.from(r);
        let z = d.from(r);
        dir(x, y, z).normalize()
    }
}

pub struct InUnitBall;

impl Distrib<Vec4> for InUnitBall {
    /// Generates a random `Vec4` in the unit ball.
    fn from(&self, r: &mut Random) -> Vec4 {
        let unit_box = Uniform(-X-Y-Z .. X+Y+Z);
        r.iter(unit_box)
            .skip_while(|&v| v.dot(v) > 1.0)
            .next().unwrap()
    }
}

impl Distrib<Vec4> for Uniform<Range<Vec4>> {
    /// Generates a random `Vec4` in an axis-aligned box
    /// given by the lower and upper bounds of `self.0`.
    fn from(&self, r: &mut Random) -> Vec4 {
        let Range { start, end } = &self.0;
        let x = Uniform(start.x..end.x).from(r);
        let y = Uniform(start.y..end.y).from(r);
        let z = Uniform(start.z..end.z).from(r);
        let w = Uniform(start.w..end.w).from(r);
        Vec4 { x, y, z, w }
    }
}


#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use crate::tests::util::*;

    use super::*;

    #[test]
    fn vector_len() {
        assert_eq!(ZERO.len(), 0.0);
        assert_eq!(X.len(), 1.0);
        assert_eq!((-Y).len(), 1.0);
        assert_eq!((3.14 * Y).len(), 3.14);
        assert_eq!((X + Y + Z).len(), f32::sqrt(3.0));
    }

    #[test]
    fn vector_normalize() {
        assert_eq!((10.0 * X).normalize(), X);
    }

    #[test]
    #[should_panic]
    fn normalize_zero_should_panic() {
        ZERO.normalize();
    }

    #[test]
    fn vector_add_sub_neg() {
        assert_eq!(X + X, vec4(2.0, 0.0, 0.0, 0.0));
        assert_eq!(X + Y, vec4(1.0, 1.0, 0.0, 0.0));
        assert_eq!(X - X, ZERO);
        assert_eq!(-(Y + Z), vec4(0.0, -1.0, -1.0, 0.0));
    }

    #[test]
    fn dot_product_parallel_vectors() {
        assert_eq!(X.dot(X), 1.0);
        assert_eq!(X.dot(-X), -1.0);

        assert_eq!((2.0 * Y).dot(3.0 * Y), 6.0);
        assert_eq!((-3.0 * Z).dot(Z), -3.0);
    }

    #[test]
    fn dot_product_orthogonal_vectors() {
        assert_eq!(X.dot(Y), 0.0);
        assert_eq!(Y.dot(-X), 0.0);
        assert_eq!(W.dot(Z), 0.0);
        assert_eq!((-Z).dot(Y), 0.0);
    }

    #[test]
    fn cross_product() {
        let v = vec4(2.0, 4.0, 7.0, 0.0);
        assert_eq!(v.cross(v), ZERO);
        assert_eq!(v.cross(-v), ZERO);

        assert_eq!(X.cross(Y), Z);
        assert_eq!(Y.cross(X), -Z);

        assert_eq!(X.cross(Z), -Y);
        assert_eq!(Z.cross(X), Y);

        assert_eq!(Y.cross(Z), X);
        assert_eq!(Z.cross(Y), -X);

        assert_eq!((X + Y).cross(Y + Z), X - Y + Z);
        assert_eq!((Y + Z).cross(X + Y), -X + Y - Z);
    }

    #[test]
    fn scalar_project() {
        assert_eq!(ZERO.scalar_project(Y), 0.0);
        assert_eq!(X.scalar_project(Y), 0.0);
        assert_eq!(X.scalar_project(-X), -1.0);
        assert_eq!((X + Z).scalar_project(Y), 0.0);
        assert_eq!((X + 3.0 * Y).scalar_project(Y), 3.0);
    }

    #[test]
    #[should_panic]
    fn vector_project_on_zero_vector_should_panic() {
        X.scalar_project(ZERO);
    }

    #[test]
    fn vector_project() {
        assert_eq!(X.vector_project(Y), ZERO);
        assert_eq!(X.vector_project(-X), X);
        assert_eq!((X + Z).vector_project(Y), ZERO);
        assert_eq!((X + Y).vector_project(3.0 * Y), Y);
    }

    #[test]
    fn vector_reflect() {
        assert_eq!(ZERO.reflect(Y), ZERO);
        assert_eq!(X.reflect(Y), -X);
        assert_eq!((X + Y + Z).reflect(0.01 * Y), -X + Y - Z);
        assert_eq!(Z.reflect(Y + Z), Y);
        assert_eq!(X.reflect(X + Y), Y);
        assert_approx_eq((0.9 * X + 0.1 * Y).reflect(X + Y), 0.9 * Y + 0.1 * X);
    }

    #[test]
    #[should_panic]
    fn reflect_about_zero_should_panic() {
        X.reflect(ZERO);
    }

    #[test]
    fn vector_approx_eq() {
        let v = vec4(0.0, 0.0, 0.0, 0.0);
        let u = vec4(1e-7, 0.0, 0.0, 0.0);
        let w = vec4(0.0, 0.0, 0.0, -1e-5);

        assert_approx_eq(v, v);
        assert_approx_eq(v, u);
        assert_approx_ne(v, w);
    }

    const ROUNDS: usize = 1 << 16;

    #[test]
    fn random_uniform_in_box() {
        let mut r = Random::new();
        let dist = Uniform(pt(-1.0, 1.0, 0.0)..pt(1.0, 2.0, 3.0));

        for v in r.iter(dist).take(ROUNDS) {
            assert!(-1.0 <= v.x && v.x <= 1.0, "x={}", v.x);
            assert!(1.0 <= v.y && v.y <= 2.0, "y={}", v.y);
            assert!(0.0 <= v.z && v.z <= 3.0, "z={}", v.z);
            assert_eq!(1.0, v.w);
        }
    }

    #[test]
    fn random_in_unit_ball() {
        let mut r = Random::new();

        for v in r.iter(InUnitBall).take(ROUNDS) {
            assert!(v.len() < 1.0, "v={:?}", v);
        }
    }

    #[test]
    fn random_unit_dir() {
        let mut r = Random::new();

        let avg_len = r.iter(UnitDir)
            .take(ROUNDS)
            .inspect(|v| {
                assert_eq!(0.0, v.w);
                assert_approx_eq(1.0, v.len());
            })
            .fold(ZERO, |a, b| a + b)
            .len() / ROUNDS as f32;

        assert!(avg_len < 0.05, "len={}", avg_len);
    }
}
