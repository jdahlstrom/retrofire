use core::fmt;
use core::ops::{Mul, MulAssign};

use crate::ApproxEq;
use crate::vec::*;

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

    pub fn determinant(&self) -> f32 {
        let [a, b, c, d] = self.0[0];

        let det3 = |j, k, l| {
            let [_, r, s, t] = self.0;
            let [a, b, c] = [r[j], r[k], r[l]];
            let [d, e, f] = [s[j], s[k], s[l]];
            let [g, h, i] = [t[j], t[k], t[l]];

            a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g)
        };

        a * det3(1, 2, 3) - b * det3(0, 2, 3) + c * det3(0, 1, 3) - d * det3(0, 1, 2)
    }

    pub fn transpose(&self) -> Mat4 {
        Mat4::from_rows([self.col(0), self.col(1), self.col(2), self.col(3)])
    }

    fn sub_row(&mut self, from: usize, to: usize, mul: f32) {
        self.0[to] = (self.row(to) - mul * self.row(from)).into();
    }

    fn mul_row(&mut self, row: usize, mul: f32) {
        self.0[row] = (mul * self.row(row)).into();
    }

    pub fn invert(&self) -> Mat4 {
        debug_assert_ne!(0.0, self.determinant(),
            "Singular matrix has no inverse");

        let mut inv = Mat4::identity();
        let mut this = self.clone();

        //eprintln!("start\nself={:?}", this);

        for idx in 0..4 {
            let pivot = (idx..4).max_by(|&r1, &r2| {
                let v1 = this.0[r1][idx].abs();
                let v2 = this.0[r2][idx].abs();
                v1.partial_cmp(&v2).unwrap()
            }).unwrap();

            //eprintln!("pivot={},{}, val={}", pivot, col, this.0[pivot][col]);

            if this.0[pivot][idx] != 0.0 {
                //eprintln!("swapping rows {} and {}", row, pivot);
                this.0.swap(idx, pivot);
                inv.0.swap(idx, pivot);

                let div = 1.0 / this.0[idx][idx];
                for r in idx+1..4 {
                    let x = this.0[r][idx] * div;

                    this.sub_row(idx, r, x);
                    inv.sub_row(idx, r, x);
                }
            }
            //eprintln!("idx={}\nthis={:?}\ninv={:?}", idx, this, inv);
        }

        //eprintln!("echelon\nthis={:?}\ninv={:?}", this, inv);

        // now in upper echelon form, backsubstitute variables
        for &idx in &[3, 2, 1] {
            let diag = this.0[idx][idx];
            for r in 0..idx {
                let x = this.0[r][idx] / diag;

                this.sub_row(idx, r, x);
                inv.sub_row(idx, r, x);
            }
        }

        //eprintln!("backsubst\nthis={:?}\ninv={:?}", this, inv);

        // normalize
        for r in 0..4 {
            let x = 1.0 / this.0[r][r];
            this.mul_row(r, x);
            inv.mul_row(r, x);
        }

        //eprintln!("normalized\nthis={:?}\ninv={:?}", this, inv);

        inv
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
    use crate::rand::*;
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
    fn identity_matrix_determinant_is_one() {
        assert_eq!(1.0, Mat4::identity().determinant());
    }

    #[test]
    fn singular_matrix_determinant_is_zero() {
        assert_eq!(0.0, Mat4::from_rows([X, X, Y, W]).determinant());
        assert_eq!(0.0, Mat4::from_rows([Y, Z, Z, W]).determinant());
        assert_eq!(0.0, Mat4::from_rows([W, Y, Z, W]).determinant());
    }

    #[test]
    fn scale_matrix_determinant_is_product_of_diagonals() {
        let scale = Mat4::from_rows([2.0 * X, 3.0 * Y, 4.0 * Z, W]);
        assert_eq!(2.0 * 3.0 * 4.0, scale.determinant())
    }

    #[test]
    fn rotation_matrix_determinant_is_one() {
        assert_eq!(1.0, Mat4::from_rows([Y, Z, X, W]).determinant());
        assert_eq!(1.0, Mat4::from_rows([X, Z, -Y, W]).determinant());
        assert_eq!(1.0, Mat4::from_rows([-X, Y, -Z, W]).determinant());
    }

    #[test]
    fn reflection_matrix_determinant_is_negative() {
        assert_eq!(-1.0, Mat4::from_rows([-X, Y, Z, W]).determinant());
        assert_eq!(-1.0, Mat4::from_rows([Y, X, Z, W]).determinant());
        assert_eq!(-1.0, Mat4::from_rows([X, -Y, Z, W]).determinant());
        assert_eq!(-1.0, Mat4::from_rows([X, Z, Y, W]).determinant());
    }

    #[test]
    fn identity_matrix_invert_gives_identity() {
        let id = Mat4::identity();
        assert_eq!(id, id.invert());
    }

    #[test]
    fn matrix_invert_pathological_case() {
        let m = Mat4([
            [   -0.494283,     0.164862,     0.853525,     0.000000],
            [   -0.313925,     0.235620,     0.919747,     0.000000],
            [   -0.549284,    -0.399912,    -0.733729,     0.000000],
            [    4.844985,     7.986494,    -3.569597,     1.000000],
        ]);

        let _m = Mat4([
            [   -0.5,     0.16,     0.85,     0.000000],
            [   -0.3,     0.23,     0.92,     0.000000],
            [   -0.55,    -0.4,    -0.73,     0.000000],
            [    4.84,     7.99,    -3.57,     1.000000],
        ]);

        let inv = m.invert();
        println!("m.det = {}, inv.det = {}", m.determinant(), inv.determinant());
        dbg!(&inv);
        println!("{:15.5?}", m * inv);
    }

    #[test]
    fn rot_trans_matrix_invert_stress_test() {

        let expected = Mat4::identity();
        let mut rng = Random::new();
        let dist = UnitDir;

        for i in 0..1000 {
            let x = rng.next(&dist);
            let y = x.cross(Z).normalize();
            let z = y.cross(x);
            let m = Mat4::from_rows([
                x, y, z,
                W + 1000.0 * rng.next(&dist),
            ]);
            let inv = m.invert();
            let actual = &m * &inv;
            if !expected.approx_eq_eps(&actual, 0.05) {
                eprintln!("i={} det(m)={} det(inv)={}", i, m.determinant(), inv.determinant());
                eprintln!("m={:12.6?}\ninv={:12.6?}\nm*inv={:12.6?}", m, inv, actual);
                panic!("Inaccurate inverse");
            }
        }
    }

    #[test]
    fn matrix_multiply_with_identity_is_identity_op() {
        let id = &Mat4::identity();

        assert_eq!(id * id, id.clone());

        let m = &Mat4([
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ]);

        assert_eq!(m * id, m.clone());
        assert_eq!(id * m, m.clone());
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
