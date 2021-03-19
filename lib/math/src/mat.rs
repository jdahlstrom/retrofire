use core::fmt;
use core::ops::{Mul, MulAssign};

use crate::ApproxEq;
use crate::vec::*;
use std::mem::swap;

#[derive(Clone, PartialEq)]
pub struct Mat4(pub(crate) [[f32; 4]; 4]);

impl Mat4 {
    pub fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn from_rows([a, b, c, d]: [Vec4; 4]) -> Self {
        Self([a.into(), b.into(), c.into(), d.into()])
    }

    pub fn from_cols(cols: [Vec4; 4]) -> Self {
        Self::from_rows(cols).transpose()
    }

    pub fn row(&self, idx: usize) -> Vec4 {
        self.0[idx].into()
    }

    pub fn col(&self, idx: usize) -> Vec4 {
        vec4(self.0[0][idx], self.0[1][idx], self.0[2][idx], self.0[3][idx])
    }

    pub fn transpose(mut self) -> Mat4 {
        for r in 1..4 {
            let (top, bot) = self.0.split_at_mut(r);
            for c in r..4 {
                swap(&mut top[r - 1][c], &mut bot[c - r][r - 1]);
            }
        }
        self
    }
}

impl Mul<&Mat4> for &Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: &Mat4) -> Mat4 {
        let mut res = Mat4::identity();

        for r in 0..4usize {
            let row = self.row(r);
            for c in 0..4usize {
                res.0[r][c] = row.dot(rhs.col(c));
            }
        }
        res
    }
}

impl Mul<&Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: &Mat4) -> Mat4 {
        &self * rhs
    }
}

impl MulAssign<&Mat4> for Mat4 {
    fn mul_assign(&mut self, rhs: &Mat4) {
        *self = &*self * rhs
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Mat4 {
        &self * &rhs
    }
}

impl MulAssign<Mat4> for Mat4 {
    fn mul_assign(&mut self, rhs: Mat4) {
        *self *= &rhs
    }
}

impl Mul<Vec4> for &Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        vec4(self.col(0).dot(rhs),
             self.col(1).dot(rhs),
             self.col(2).dot(rhs),
             self.col(3).dot(rhs))
    }
}

impl MulAssign<&Mat4> for Vec4 {
    fn mul_assign(&mut self, rhs: &Mat4) {
        *self = rhs * *self;
    }
}

impl ApproxEq for &Mat4 {
    type Scalar = f32;

    fn epsilon(self) -> f32 {
        1e-4 // TODO
    }
    fn abs_diff(self, rhs: Self) -> f32 {
        (0..4)
            .map(|i| (self.row(i), rhs.row(i))) //
            .map(|(s, r)| s.abs_diff(r))
            .sum::<f32>()
            / 4.0
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mat4\n{:>5.2?}\n{:>5.2?}\n{:>5.2?}\n{:>5.2?}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_from_rows() {
        let m = Mat4::from_rows([
            vec4(1.1, 1.2, 1.3, 1.4),
            vec4(2.1, 2.2, 2.3, 2.4),
            vec4(3.1, 3.2, 3.3, 3.4),
            vec4(4.1, 4.2, 4.3, 4.4),
        ]);

        assert_eq!(m.0, [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ])
    }

    #[test]
    fn matrix_from_cols() {
        let m = Mat4::from_cols([
            vec4(1.1, 1.2, 1.3, 1.4),
            vec4(2.1, 2.2, 2.3, 2.4),
            vec4(3.1, 3.2, 3.3, 3.4),
            vec4(4.1, 4.2, 4.3, 4.4),
        ]);

        assert_eq!(m.0, [
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
            [1.4, 2.4, 3.4, 4.4],
        ])
    }

    #[test]
    fn matrix_rows() {
        let m = Mat4([
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ]);
        assert_eq!(m.row(1), vec4(1.0, 1.1, 1.2, 1.3));
        assert_eq!(m.row(3), vec4(3.0, 3.1, 3.2, 3.3));
    }

    #[test]
    fn matrix_cols() {
        let m = Mat4([
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ]);
        assert_eq!(m.col(0), vec4(0.0, 1.0, 2.0, 3.0));
        assert_eq!(m.col(3), vec4(0.3, 1.3, 2.3, 3.3));
    }

    #[test]
    fn matrix_transpose() {
        let m = Mat4([
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ]);
        let expected = Mat4([
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
            [1.4, 2.4, 3.4, 4.4],
        ]);

        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn matrix_identity_multiply() {
        let i = &Mat4::identity();

        assert_eq!(i * i, Mat4::identity());

        let m = &Mat4([
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ]);

        assert_eq!(m * i, m.clone());
        assert_eq!(i * m, m.clone());
    }

    #[test]
    fn matrix_matrix_multiply() {
        let a = &Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 1.0],
        ]);
        let b = &Mat4([
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        assert_eq!(a * b, Mat4([
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [15.0, 8.0, -4.0, 1.0],
        ]));

        assert_eq!(b * a, Mat4([
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [2.0, 3.0, 4.0, 1.0],
        ]));
    }

    #[test]
    fn matrix_vector_multiply() {
        let v = vec4(1.0, 2.0, 3.0, 4.0);

        let m = &Mat4::identity();
        assert_eq!(m * v, v);

        let m = &Mat4([
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(m * X, Y + Z);
        assert_eq!(m * Y, X);
        assert_eq!(m * Z, 2.0 * Z);
        assert_eq!(m * W, W);

        let m = &Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 1.0],
        ]);
        assert_eq!(m * X, X);
        assert_eq!(m * W, vec4(2.0, 3.0, 4.0, 1.0));
    }
}
