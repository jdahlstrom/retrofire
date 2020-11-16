use core::fmt;
use core::ops::{Mul, MulAssign};

use crate::ApproxEq;
use crate::vec::*;

#[derive(Clone, PartialEq)]
pub struct Mat4(pub(crate) [[f32; 4]; 4]);

impl Mat4 {
    pub const fn identity() -> Mat4 {
        Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn row(&self, idx: usize) -> Vec4 {
        self.0[idx].into()
    }

    pub fn col(&self, idx: usize) -> Vec4 {
        vec4(self.0[0][idx], self.0[1][idx], self.0[2][idx], self.0[3][idx])
    }

    pub fn transpose(mut self) -> Mat4 {
        for r in 0..4 {
            for c in (r + 1)..4 {
                let rc = self.0[r][c];
                let cr = self.0[c][r];
                self.0[c][r] = rc;
                self.0[r][c] = cr;
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

impl Mul<Vec4> for &Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        vec4(self.col(0).dot(rhs),
             self.col(1).dot(rhs),
             self.col(2).dot(rhs),
             self.col(3).dot(rhs))
    }
}

impl ApproxEq for &Mat4 {
    type Scalar = f32;
    const EPSILON: f32 = 1e-6;

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
    fn matrix_rows() {
        let m = Mat4([
            [00.0, 01.0, 02.0, 03.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]);
        assert_eq!(m.row(1), vec4(10.0, 11.0, 12.0, 13.0));
        assert_eq!(m.row(3), vec4(30.0, 31.0, 32.0, 33.0));
    }

    #[test]
    fn matrix_cols() {
        let m = Mat4([
            [00.0, 01.0, 02.0, 03.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]);
        assert_eq!(m.col(0), vec4(0.0, 10.0, 20.0, 30.0));
        assert_eq!(m.col(3), vec4(3.0, 13.0, 23.0, 33.0));
    }

    #[test]
    fn matrix_transpose() {
        let m = Mat4([
            [00.0, 01.0, 02.0, 03.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]);
        let expected = Mat4([
            [00.0, 10.0, 20.0, 30.0],
            [01.0, 11.0, 21.0, 31.0],
            [02.0, 12.0, 22.0, 32.0],
            [03.0, 13.0, 23.0, 33.0],
        ]);

        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn matrix_matrix_multiply() {
        let m = &Mat4::identity();

        assert_eq!(m * m, Mat4::identity());
    }

    #[test]
    fn matrix_vector_multiply() {
        let v = vec4(1.0, 2.0, 3.0, 4.0);

        let m = Mat4::identity();
        assert_eq!(&m * v, v);

        let m = Mat4([
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(&m * X, Y + Z);
        assert_eq!(&m * Y, X);
        assert_eq!(&m * Z, 2.0 * Z);
        assert_eq!(&m * W, W);

        let m = Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 1.0],
        ]);
        assert_eq!(&m * X, X);
        assert_eq!(&m * W, vec4(2.0, 3.0, 4.0, 1.0));
    }
}
