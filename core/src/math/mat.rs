//! Matrices and linear transforms.

use core::array;
use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::math::approx::ApproxEq;
use crate::math::vec::{Affine, Real, Vec3, Vector};

/// A linear transform from one space (or basis) to another.
///
/// This is a tag trait with no functionality in itself. It is used to
/// statically ensure that only compatible maps can be composed, and that
/// only compatible vectors can be mapped
pub trait LinearMap {
    type From;
    type To;
}

/// Dummy LinearMap to help with generic code
impl LinearMap for () {
    type From = ();
    type To = ();
}

/// A mapping from one basis to another in real vector space of dimension `DIM`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct RealToReal<const DIM: usize, FromBasis = (), ToBasis = ()>(
    PhantomData<(FromBasis, ToBasis)>,
);

impl<const DIM: usize, F, T> LinearMap for RealToReal<DIM, F, T> {
    type From = Real<DIM, F>;
    type To = Real<DIM, T>;
}

/// A mapping from real to projective space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RealToProjective<FromBasis>(PhantomData<FromBasis>);

/// A generic matrix type.
#[repr(transparent)]
#[derive(Copy, Eq, PartialEq)]
pub struct Matrix<Repr, Map>(pub Repr, PhantomData<Map>);

/// Type alias for 3x3 square matrices.
pub type Mat3x3<Scalar = f32, Map = ()> = Matrix<[[Scalar; 3]; 3], Map>;
/// Type alias for 4x4 square matrices.
pub type Mat4x4<Scalar = f32, Map = ()> = Matrix<[[Scalar; 4]; 4], Map>;

impl<const N: usize, Map> Matrix<[[f32; N]; N], Map> {
    pub fn identity() -> Self
    where
        [[f32; N]; N]: Default,
    {
        let mut m = Matrix::from(<[[f32; N]; N]>::default());
        for i in 0..N {
            m.0[i][i] = 1.0;
        }
        m
    }
    #[inline]
    pub fn row_vec(&self, i: usize) -> Vector<[f32; N], Map::From>
    where
        Map: LinearMap,
    {
        self.0[i].into()
    }
    #[inline]
    pub fn col_vec(&self, i: usize) -> Vector<[f32; N], Map::To>
    where
        Map: LinearMap,
    {
        array::from_fn(|j| self.0[j][i]).into()
    }
    #[inline]
    pub fn to<M>(&self) -> Matrix<[[f32; N]; N], M> {
        self.0.into()
    }
}

impl<F, T> Mat4x4<f32, RealToReal<4, F, T>> {
    /// Returns the composite transform of `self` and `other`.
    ///
    /// Computes the matrix multiplication of `self` and `other`
    /// such that applying the resulting transformation is equivalent
    /// to first applying `other` and then `self`.
    pub fn compose<G>(
        &self,
        other: &Mat4x4<f32, RealToReal<4, G, F>>,
    ) -> Mat4x4<f32, RealToReal<4, G, T>> {
        let mut els = [[0.0_f32; 4]; 4];

        for j in 0..4 {
            for i in 0..4 {
                let s = self.row_vec(j);
                let o = other.col_vec(i);
                els[j][i] = s.dot(&o);
            }
        }
        Matrix(els, PhantomData)
    }

    /// Returns the composite transform of `other` and `self`.
    ///
    /// Computes the matrix multiplication of `self` and `other`
    /// such that applying the resulting transformation is equivalent to first
    /// applying `self` and then `other`. That is to say, this method is
    /// equivalent to `other.compose(self)`.
    pub fn then<U>(
        &self,
        other: &Mat4x4<f32, RealToReal<4, T, U>>,
    ) -> Mat4x4<f32, RealToReal<4, F, U>> {
        other.compose(self)
    }

    /// Maps the vector `v` from basis `F` to basis `T`.
    ///
    /// Computes the matrix–vector multiplication `Mv` where `v` is interpreted
    /// as a column vector.
    pub fn map(&self, v: &Vec3<Real<3, F>>) -> Vec3<Real<3, T>> {
        let v = Vector::from([v.x(), v.y(), v.z(), 1.0]);
        let x = self.row_vec(0).dot(&v);
        let y = self.row_vec(1).dot(&v);
        let z = self.row_vec(2).dot(&v);
        [x, y, z].into()
    }

    /// Returns the determinant of `self`.
    pub fn determinant(&self) -> f32 {
        let [a, b, c, d] = self.0[0];

        let det3 = |j, k, l| {
            let [r, s, t] = [&self.0[1], &self.0[2], &self.0[3]];
            let [a, b, c] = [r[j], r[k], r[l]];
            let [d, e, f] = [s[j], s[k], s[l]];
            let [g, h, i] = [t[j], t[k], t[l]];

            a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g)
        };

        a * det3(1, 2, 3) - b * det3(0, 2, 3) + c * det3(0, 1, 3)
            - d * det3(0, 1, 2)
    }

    /// Returns the inverse matrix of `self`.
    ///
    /// Panics in debug mode if `self` is singular or near-singular.
    /// In release mode, may give
    pub fn inverse(&self) -> Mat4x4<f32, RealToReal<4, T, F>> {
        debug_assert!(
            self.determinant().abs() > f32::EPSILON,
            "singular or near-singular matrix has no well-defined inverse"
        );

        /// Elementary row operation subtracting one row,
        /// multiplied by a scalar, from another
        fn sub_row(m: &mut Mat4x4, from: usize, to: usize, mul: f32) {
            m.0[to] = m.row_vec(to).add(&m.row_vec(from).mul(-mul)).0;
        }

        /// Elementary row operation multiplying one row with a scalar
        fn mul_row(m: &mut Mat4x4, row: usize, mul: f32) {
            m.0[row] = m.row_vec(row).mul(mul).0;
        }

        /// Elementary row operation swapping two rows
        fn swap_rows(m: &mut Mat4x4, r: usize, s: usize) {
            m.0.swap(r, s);
        }

        let mut inv = &mut Mat4x4::identity();
        let mut this = &mut self.to();

        // Applies row operations to reduce the matrix to an upper echelon form
        for idx in 0..4 {
            let pivot = (idx..4)
                .max_by(|&r1, &r2| {
                    let v1 = this.0[r1][idx].abs();
                    let v2 = this.0[r2][idx].abs();
                    v1.partial_cmp(&v2).unwrap()
                })
                .unwrap();

            if this.0[pivot][idx] != 0.0 {
                swap_rows(this, idx, pivot);
                swap_rows(inv, idx, pivot);

                let div = 1.0 / this.0[idx][idx];
                for r in (idx + 1)..4 {
                    let x = this.0[r][idx] * div;

                    sub_row(&mut this, idx, r, x);
                    sub_row(&mut inv, idx, r, x);
                }
            }
        }
        // now in upper echelon form, back-substitute variables
        for &idx in &[3, 2, 1] {
            let diag = this.0[idx][idx];
            for r in 0..idx {
                let x = this.0[r][idx] / diag;

                sub_row(this, idx, r, x);
                sub_row(inv, idx, r, x);
            }
        }
        // normalize
        for r in 0..4 {
            let x = 1.0 / this.0[r][r];
            mul_row(this, r, x);
            mul_row(inv, r, x);
        }
        inv.to()
    }
}

impl<R: Clone, M> Clone for Matrix<R, M> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<S: Debug, M: Debug + Default, const N: usize> Debug
    for Matrix<[[S; N]; N], M>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix<{:?}>[", M::default())?;
        for i in 0..N {
            writeln!(f, "    {:6.2?}", self.0[i])?;
        }
        write!(f, "]")
    }
}

impl<const DIM: usize, F, T> Debug for RealToReal<DIM, F, T>
where
    F: Debug + Default,
    T: Debug + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}->{:?}", F::default(), T::default())
    }
}

impl<const N: usize, S, M> From<[[S; N]; N]> for Matrix<[[S; N]; N], M> {
    fn from(els: [[S; N]; N]) -> Self {
        Self(els, PhantomData)
    }
}

impl<Scalar: ApproxEq, Map, const N: usize> ApproxEq<Self, Scalar>
    for Matrix<[[Scalar; N]; N], Map>
{
    fn approx_eq_eps(&self, other: &Self, eps: &Scalar) -> bool {
        self.0.approx_eq_eps(&other.0, eps)
    }
    fn relative_epsilon() -> Scalar {
        Scalar::relative_epsilon()
    }
}

pub fn scale(s: Vec3) -> Mat4x4 {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into()
}

pub fn translate(t: Vec3) -> Mat4x4 {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into()
}

pub fn orient_y(_new_y: Vec3, _x: Vec3) -> Mat4x4 {
    todo!()
}
pub fn orient_z(_new_z: Vec3, _x: Vec3) -> Mat4x4 {
    todo!()
}

pub fn viewport(left: f32, top: f32, right: f32, bottom: f32) -> Mat4x4 {
    let h = (right - left) / 2.0;
    let v = (bottom - top) / 2.0;
    [
        [h, 0.0, 0.0, h + left],
        [0.0, v, 0.0, v + top],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into()
}

#[cfg(test)]
mod tests {
    use crate::math::vec::vec3;

    use super::*;

    #[test]
    fn matrix_debug() {
        let m: Mat4x4 = [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
        .into();

        let expected = r#"Matrix<()>[
    [  0.00,   1.00,   2.00,   3.00]
    [ 10.00,  11.00,  12.00,  13.00]
    [ 20.00,  21.00,  22.00,  23.00]
    [ 30.00,  31.00,  32.00,  33.00]
]"#;

        assert_eq!(alloc::format!("{:?}", m), expected);
    }

    #[test]
    fn mat_vec_scale() {
        let m = scale(vec3(1.0, -2.0, 3.0)).to();
        let v = vec3(0.0, 4.0, -3.0);

        assert_eq!(m.map(&v), vec3(0.0, -8.0, -9.0));
    }

    #[test]
    fn mat_vec_translate() {
        let m = translate(vec3(1.0, 2.0, 3.0)).to();
        let v = vec3(0.0, 5.0, -3.0);

        assert_eq!(m.map(&v), vec3(1.0, 7.0, 0.0));
    }

    #[test]
    fn mat_times_mat_inverse_is_identity() {
        let m = translate(vec3(1.0e3, -2.0e2, 0.0))
            .to::<RealToReal<4>>()
            .then(&scale(vec3(0.5, 100.0, 42.0)).to::<RealToReal<4>>());

        let m_inv = m.inverse();

        assert_eq!(m.compose(&m_inv), Mat4x4::identity());
        assert_eq!(m_inv.compose(&m), Mat4x4::identity());
    }
}
