#![allow(clippy::needless_range_loop)]

//! Matrices and linear transforms.

use core::array;
use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::math::vec::{Affine, Proj4, Real, Vec3, Vec4, Vector};
use crate::render::{Ndc, Screen};

/// A linear transform from one space (or basis) to another.
///
/// This is a tag trait with no functionality in itself. It is used to
/// statically ensure that only compatible maps can be composed, and that
/// only compatible vectors can be transformed.
pub trait LinearMap {
    /// The source space, or domain, of `Self`.
    type Source;
    /// The destination space, or range, of `Self`.
    type Dest;
}

/// Composition of two `LinearMap`s, `Self` ∘ `Inner`.
/// If `Self` maps from `B` to `C`, and `Inner` maps from `A` to `B`,
/// `Self::Result` maps from `A` to `C`.
pub trait Compose<Inner: LinearMap>: LinearMap<Source = Inner::Dest> {
    /// The result of composing `Self` with `Inner`.
    type Result: LinearMap<Source = Inner::Source, Dest = Self::Dest>;
}

/// A mapping from one basis to another in real vector space of dimension `DIM`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct RealToReal<const DIM: usize, SrcBasis = (), DstBasis = ()>(
    PhantomData<(SrcBasis, DstBasis)>,
);

/// Mapping from NDC (normalized device coordinates) to screen space.
pub type NdcToScreen = RealToReal<3, Ndc, Screen>;

/// Mapping from real to projective space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RealToProjective<SrcBasis>(PhantomData<SrcBasis>);

/// Dummy LinearMap to help with generic code.
impl LinearMap for () {
    type Source = ();
    type Dest = ();
}

/// A generic matrix type.
#[repr(transparent)]
#[derive(Copy, Eq, PartialEq)]
pub struct Matrix<Repr, Map>(pub Repr, PhantomData<Map>);

/// Type alias for a 3x3 float matrix.
pub type Mat3x3<Map = ()> = Matrix<[[f32; 3]; 3], Map>;
/// Type alias for a 4x4 float matrix.
pub type Mat4x4<Map = ()> = Matrix<[[f32; 4]; 4], Map>;

//
// Inherent impls
//

impl<Repr: Clone, Map> Matrix<Repr, Map> {
    /// Returns a matrix equal to `self` but with mapping `M`.
    ///
    /// This method can be used to coerce a matrix to a different
    /// mapping in case it is needed to make types match.
    #[inline]
    pub fn to<M>(&self) -> Matrix<Repr, M> {
        Matrix(self.0.clone(), PhantomData)
    }
}

impl<const N: usize, Map> Matrix<[[f32; N]; N], Map> {
    /// Returns the `N`×`N` identity matrix.
    pub fn identity() -> Self
    where
        [[f32; N]; N]: Default,
    {
        let mut els = <[[f32; N]; N]>::default();
        for i in 0..N {
            els[i][i] = 1.0;
        }
        els.into()
    }
    /// Returns the row vector of `self` with index `i`.
    /// The returned vector is in space `Map::Source`.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline]
    pub fn row_vec(&self, i: usize) -> Vector<[f32; N], Map::Source>
    where
        Map: LinearMap,
    {
        self.0[i].into()
    }
    /// Returns the column vector of `self` with index `i`.
    ///
    /// The returned vector is in space `Map::Dest`.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline]
    pub fn col_vec(&self, i: usize) -> Vector<[f32; N], Map::Dest>
    where
        Map: LinearMap,
    {
        array::from_fn(|j| self.0[j][i]).into()
    }

    /// Returns whether every element of `self` is finite
    /// (ie. neither `Inf`, `-Inf`, or `NaN`).
    fn is_finite(&self) -> bool {
        self.0.iter().flatten().all(|e| e.is_finite())
    }
}

impl<M: LinearMap> Mat4x4<M> {
    /// Returns the composite transform of `self` and `other`.
    ///
    /// Computes the matrix product of `self` and `other` such that applying
    /// the resulting transformation is equivalent to first applying `other`
    /// and then `self`. More succinctly,
    /// ```text
    /// (𝗠 ∘ 𝗡)𝘃 = 𝗠(𝗡𝘃)
    /// ```
    /// for some matrices 𝗠 and 𝗡 and a vector 𝘃.
    pub fn compose<Inner>(
        &self,
        other: &Mat4x4<Inner>,
    ) -> Mat4x4<<M as Compose<Inner>>::Result>
    where
        Inner: LinearMap,
        M: Compose<Inner>,
    {
        let other: [_; 4] = array::from_fn(|i| other.col_vec(i));
        let mut res = [[0.0; 4]; 4];

        for j in 0..4 {
            let s = self.row_vec(j);
            for i in 0..4 {
                res[j][i] = s.dot(&other[i]);
            }
        }
        res.into()
    }
    /// Returns the composite transform of `other` and `self`.
    ///
    /// Computes the matrix product of `self` and `other` such that applying
    /// the resulting matrix is equivalent to first applying `self` and then
    /// `other`. The call `self.then(other)` is thus equivalent to
    /// `other.compose(self)`.
    pub fn then<Outer>(&self, other: &Mat4x4<Outer>) -> Mat4x4<Outer::Result>
    where
        Outer: Compose<M>,
    {
        other.compose(self)
    }
}

impl<Src, Dst> Mat4x4<RealToReal<3, Src, Dst>> {
    /// Maps the real 3-vector 𝘃 from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–vector multiplication 𝝡𝘃 where 𝘃 is interpreted as
    /// a column vector with an implicit 𝘃<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         / M00  ·  · \ / v0 \
    ///  Mv  =  |    ·      | | v1 |  =  ( v0' v1' v2' 1 )
    ///         |      ·    | | v2 |
    ///         \ ·  ·  M33 / \  1 /
    /// ```
    pub fn apply(&self, v: &Vec3<Real<3, Src>>) -> Vec3<Real<3, Dst>> {
        let v = Vector::from([v.x(), v.y(), v.z(), 1.0]);
        let x = self.row_vec(0).dot(&v);
        let y = self.row_vec(1).dot(&v);
        let z = self.row_vec(2).dot(&v);
        [x, y, z].into()
    }

    /// Returns the determinant of `self`.
    ///
    /// Given a matrix M,
    /// ```text
    ///         / a  b  c  d \
    ///  M  =   | e  f  g  h |
    ///         | i  j  k  l |
    ///         \ m  n  o  p /
    /// ```
    /// its determinant can be computed by recursively computing
    /// the determinants of sub-matrices on rows 1.. and multiplying
    /// them by the elements on row 0:
    /// ```text
    ///              | f g h |       | e g h |
    /// det(M) = a · | j k l | - b · | i k l |  + - ···
    ///              | n o p |       | m o p |
    /// ```
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
    /// The inverse 𝝡<sup>-1</sup> of matrix 𝝡 is a matrix that, when
    /// composed with 𝝡, results in the identity matrix:
    /// ```text
    /// 𝝡 ∘ 𝝡<sup>-1</sup> = 𝝡<sup>-1</sup> ∘ 𝝡 = 𝐈
    /// ```
    /// In other words, it applies the transform of 𝝡 in reverse.
    /// Given vectors 𝘃 and 𝘂,
    /// ```text
    /// 𝝡𝘃 = 𝘂 ⇔ 𝝡<sup>-1</sup> 𝘂 = 𝘃.
    /// ```
    /// Only matrices with a nonzero determinant have a defined inverse.
    /// A matrix without an inverse is said to be singular.
    ///
    /// Note: This method uses naive Gauss–Jordan elimination and may
    /// suffer from imprecision or numerical instability in certain cases.
    ///
    /// # Panics
    /// If `self` is singular or near-singular:
    /// * Panics in debug mode.
    /// * Does not panic in release mode, but the result may be inaccurate
    /// or contain `Inf`s or `NaN`s.
    pub fn inverse(&self) -> Mat4x4<RealToReal<3, Dst, Src>> {
        if cfg!(debug_assertions) {
            let det = self.determinant();
            assert!(
                det.abs() > f32::EPSILON,
                "a singular, near-singular, or non-finite matrix does not \
                 have a well-defined inverse (determinant = {det})"
            );
        }

        // Elementary row operation subtracting one row,
        // multiplied by a scalar, from another
        fn sub_row(m: &mut Mat4x4, from: usize, to: usize, mul: f32) {
            m.0[to] = m.row_vec(to).add(&m.row_vec(from).mul(-mul)).0;
        }

        // Elementary row operation multiplying one row with a scalar
        fn mul_row(m: &mut Mat4x4, row: usize, mul: f32) {
            m.0[row] = m.row_vec(row).mul(mul).0;
        }

        // Elementary row operation swapping two rows
        fn swap_rows(m: &mut Mat4x4, r: usize, s: usize) {
            m.0.swap(r, s);
        }

        // This algorithm attempts to reduce `this` to the identity matrix
        // by simultaneously applying elementary row operations to it and
        // another matrix `inv` which starts as the identity matrix. Once
        // `this` is reduced, the value of `inv` has become the inverse of
        // `this` and thus of `self`.

        let inv = &mut Mat4x4::identity();
        let this = &mut self.to();

        // Apply row operations to reduce the matrix to an upper echelon form
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
                    sub_row(this, idx, r, x);
                    sub_row(inv, idx, r, x);
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
        debug_assert!(inv.is_finite());
        inv.to()
    }
}

impl<B> Mat4x4<RealToProjective<B>> {
    /// Maps the real 3-vector 𝘃 from basis 𝖡 to the projective 4-space.
    ///
    /// Computes the matrix–vector multiplication 𝝡𝘃 where 𝘃 is interpreted as
    /// a column vector with an implicit 𝘃<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         / M00  ·  · \ / v0 \
    ///  Mv  =  |    ·      | | v1 |  =  ( v0' v1' v2' v3' )
    ///         |      ·    | | v2 |
    ///         \ ·  ·  M33 / \  1 /
    /// ```
    pub fn apply(&self, v: &Vec3<Real<3, B>>) -> Vec4<Proj4> {
        let v = Vector::from([v.x(), v.y(), v.z(), 1.0]);
        [
            self.row_vec(0).dot(&v),
            self.row_vec(1).dot(&v),
            self.row_vec(2).dot(&v),
            self.row_vec(3).dot(&v),
        ]
        .into()
    }
}

//
// Local trait impls
//

impl<const DIM: usize, S, D> LinearMap for RealToReal<DIM, S, D> {
    type Source = Real<DIM, S>;
    type Dest = Real<DIM, D>;
}

impl<const DIM: usize, S, I, D> Compose<RealToReal<DIM, S, I>>
    for RealToReal<DIM, I, D>
{
    type Result = RealToReal<DIM, S, D>;
}

impl<S> LinearMap for RealToProjective<S> {
    type Source = Real<3, S>;
    type Dest = Proj4;
}

impl<S, I> Compose<RealToReal<3, S, I>> for RealToProjective<I> {
    type Result = RealToProjective<S>;
}

//
// Foreign trait impls
//

impl<R: Clone, M> Clone for Matrix<R, M> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<const N: usize, Map> Default for Matrix<[[f32; N]; N], Map>
where
    [[f32; N]; N]: Default,
{
    /// Returns the `N`×`N` identity matrix.
    fn default() -> Self {
        Self::identity()
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
        write!(f, "{:?}→{:?}", F::default(), T::default())
    }
}

impl<Repr, M> From<Repr> for Matrix<Repr, M> {
    fn from(repr: Repr) -> Self {
        Self(repr, PhantomData)
    }
}

//
// Free functions
//

/// Returns a matrix applying a scaling by `s`.
///
/// Tip: use `Vec3::from(f32)` to scale uniformly:
/// ```
/// # use retrofire_core::math::mat::*;
/// let m = scale(2.0.into());
/// assert_eq!(m.0[0][0], 2.0);
/// assert_eq!(m.0[1][1], 2.0);
/// assert_eq!(m.0[2][2], 2.0);
/// ```
pub fn scale(s: Vec3) -> Mat4x4<RealToReal<3>> {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into()
}

/// Returns a matrix applying a translation by `t`.
pub fn translate(t: Vec3) -> Mat4x4<RealToReal<3>> {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into()
}

/// Returns a matrix applying a rotation such that the original y axis
/// is now parallel with `new_y`.
///
/// TODO unimplemented
pub fn orient_y(_new_y: Vec3, _x: Vec3) -> Mat4x4 {
    todo!()
}
/// Returns a matrix applying a rotation such that the original y axis
/// is now parallel with `new_y`.
///
/// TODO unimplemented
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

    #[derive(Debug, Default, Eq, PartialEq)]
    struct Basis1;
    #[derive(Debug, Default, Eq, PartialEq)]
    struct Basis2;

    #[test]
    fn matrix_debug() {
        let m: Mat4x4<RealToReal<3, Basis1, Basis2>> = [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
        .into();

        let expected = r#"Matrix<Basis1→Basis2>[
    [  0.00,   1.00,   2.00,   3.00]
    [ 10.00,  11.00,  12.00,  13.00]
    [ 20.00,  21.00,  22.00,  23.00]
    [ 30.00,  31.00,  32.00,  33.00]
]"#;

        assert_eq!(alloc::format!("{:?}", m), expected);
    }

    #[test]
    fn mat_vec_scale() {
        let m = scale(vec3(1.0, -2.0, 3.0));
        let v = vec3(0.0, 4.0, -3.0).to();

        assert_eq!(m.apply(&v), vec3(0.0, -8.0, -9.0));
    }

    #[test]
    fn mat_vec_translate() {
        let m = translate(vec3(1.0, 2.0, 3.0));
        let v = vec3(0.0, 5.0, -3.0).to();

        assert_eq!(m.apply(&v), vec3(1.0, 7.0, 0.0));
    }

    #[test]
    fn composition() {
        let t = translate(vec3(1.0, 2.0, 3.0));
        let s = scale(vec3(3.0, 2.0, 1.0));

        let ts = t.then(&s);
        let st = s.then(&t);

        assert_eq!(ts, s.compose(&t));
        assert_eq!(st, t.compose(&s));

        assert_eq!(ts.apply(&0.0.into()), vec3(3.0, 4.0, 3.0));
        assert_eq!(st.apply(&0.0.into()), vec3(1.0, 2.0, 3.0));
    }

    #[test]
    fn mat_times_mat_inverse_is_identity() {
        let m = translate(vec3(1.0e3, -2.0e2, 0.0))
            .to::<RealToReal<3>>()
            .then(&scale(vec3(0.5, 100.0, 42.0)).to::<RealToReal<3>>());

        let m_inv = m.inverse();

        assert_eq!(m.compose(&m_inv), Mat4x4::identity());
        assert_eq!(m_inv.compose(&m), Mat4x4::identity());
    }

    #[test]
    fn inverse_reverts_transform() {
        let m: Mat4x4<RealToReal<3, Basis1, Basis2>> =
            scale(vec3(1.0, 2.0, 0.5))
                .then(&translate(vec3(-2.0, 3.0, 0.0)))
                .to();
        let m_inv: Mat4x4<RealToReal<3, Basis2, Basis1>> = m.inverse();

        let v1: Vec3<Real<3, Basis1>> = vec3(1.0, -2.0, 3.0).to();
        let v2: Vec3<Real<3, Basis2>> = vec3(2.0, 0.0, -2.0).to();

        assert_eq!(m_inv.apply(&m.apply(&v1)), v1);
        assert_eq!(m.apply(&m_inv.apply(&v2)), v2);
    }
}
