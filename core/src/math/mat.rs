//! Matrices and linear and affine transforms.
//!
//! TODO Docs

#![allow(clippy::needless_range_loop)]

use core::{
    array,
    fmt::{self, Debug, Formatter},
    marker::PhantomData as Pd,
    ops::Range,
};

use crate::render::{Ndc, Screen, View};

use super::{
    approx::ApproxEq,
    float::f32,
    point::{Point2, Point2u, Point3, pt2},
    space::{Linear, Proj3, Real},
    vec::{ProjVec3, Vec2, Vec3, Vector, vec2, vec3},
};

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
///
/// If `Self` maps from `B` to `C`, and `Inner` maps from `A` to `B`,
/// `Self::Result` maps from `A` to `C`.
pub trait Compose<Inner: LinearMap>: LinearMap<Source = Inner::Dest> {
    /// The result of composing `Self` with `Inner`.
    type Result: LinearMap<Source = Inner::Source, Dest = Self::Dest>;
}

/// Trait for applying a transform to a type.
pub trait Apply<T> {
    /// The transform codomain type.
    type Output;

    /// Applies this transform to a value.
    #[must_use]
    fn apply(&self, t: &T) -> Self::Output;
}

/// A change of basis in real vector space of dimension `DIM`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct RealToReal<const DIM: usize, SrcBasis = (), DstBasis = ()>(
    Pd<(SrcBasis, DstBasis)>,
);

/// Mapping from real to projective space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RealToProj<SrcBasis>(Pd<SrcBasis>);

/// A generic matrix type.
#[repr(transparent)]
#[derive(Copy, Eq, PartialEq)]
pub struct Matrix<Repr, Map>(pub Repr, Pd<Map>);

/// Type alias for a 2x2 float matrix.
pub type Mat2<Src = (), Dst = Src, const DIM: usize = 2> =
    Matrix<[[f32; 2]; 2], RealToReal<DIM, Src, Dst>>;
/// Type alias for a 3x3 float matrix.
pub type Mat3<Src = (), Dst = Src, const DIM: usize = 2> =
    Matrix<[[f32; 3]; 3], RealToReal<DIM, Src, Dst>>;
/// Type alias for a 4x4 float matrix.
pub type Mat4<Src = (), Dst = Src, const DIM: usize = 3> =
    Matrix<[[f32; 4]; 4], RealToReal<DIM, Src, Dst>>;

pub type ProjMat3<Src = ()> = Matrix<[[f32; 4]; 4], RealToProj<Src>>;

//
// Inherent impls
//

/// Slight syntactic sugar for creating [`Matrix`] instances.
///
/// # Examples
/// ```
/// use retrofire_core::{mat, math::Mat3};
///
/// let m: Mat3 = mat![
///     0.0, 2.0, 0.0;
///     1.0, 0.0, 0.0;
///     0.0, 0.0, 3.0;
/// ];
/// assert_eq!(m.0, [
///     [0.0, 2.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [0.0, 0.0, 3.0]
/// ]);
/// ```
#[macro_export]
macro_rules! mat {
    ( $( $( $elem:expr ),+ );+ $(;)? ) => {
        $crate::math::mat::Matrix::new([
            $([$($elem),+]),+
        ])
    };
}

impl<Repr, Map> Matrix<Repr, Map> {
    /// Returns a matrix with the given elements.
    #[inline]
    pub const fn new(els: Repr) -> Self {
        Self(els, Pd)
    }

    /// Returns a matrix equal to `self` but with mapping `M`.
    ///
    /// This method can be used to coerce a matrix to a different
    /// mapping in case it is needed to make types match.
    #[inline]
    pub fn to<M>(&self) -> Matrix<Repr, M>
    where
        Repr: Clone,
    {
        Matrix(self.0.clone(), Pd)
    }

    pub fn apply<T>(&self, t: &T) -> <Self as Apply<T>>::Output
    where
        Self: Apply<T>,
    {
        Apply::apply(self, t)
    }
}

impl<Sc, const N: usize, const M: usize, Map> Matrix<[[Sc; N]; M], Map>
where
    Sc: Linear<Scalar = Sc> + Copy,
    Map: LinearMap,
{
    /// Returns the row vector of `self` with index `i`.
    ///
    /// The returned vector is in space `Map::Source`.
    ///
    /// # Panics
    /// If `i >= M`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{mat, math::{vec2, Mat2}};
    ///
    /// let m: Mat2 = mat![1.0, 2.0; 3.0, 4.0];
    /// assert_eq!(m.row_vec(0), vec2(1.0, 2.0));
    #[inline]
    pub fn row_vec(&self, i: usize) -> Vector<[Sc; N], Map::Source> {
        Vector::new(self.0[i])
    }

    /// Returns the column vector of `self` with index `i`.
    ///
    /// The returned vector is in space `Map::Dest`.
    ///
    /// # Panics
    /// If `i >= N`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{mat, math::{vec2, Mat2}};
    ///
    /// let m: Mat2 = mat![1.0, 2.0; 3.0, 4.0];
    /// assert_eq!(m.col_vec(1), vec2(2.0, 4.0));
    #[inline]
    pub fn col_vec(&self, i: usize) -> Vector<[Sc; M], Map::Dest> {
        Vector::new(self.0.map(|row| row[i]))
    }
}
impl<Sc: Copy, const N: usize, const DIM: usize, S, D>
    Matrix<[[Sc; N]; N], RealToReal<DIM, S, D>>
{
    /// Returns `self` with its rows and columns swapped.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{mat, math::{vec2, Mat2}};
    ///
    /// let m: Mat2 = mat![1.0, 2.0;
    ///                    3.0, 4.0];
    /// assert_eq!(m.transpose(), mat![1.0, 3.0;
    ///                                2.0, 4.0]);
    #[must_use]
    pub fn transpose(self) -> Matrix<[[Sc; N]; N], RealToReal<DIM, D, S>> {
        const { assert!(N >= DIM, "map dimension >= matrix dimension") }
        array::from_fn(|j| array::from_fn(|i| self.0[i][j])).into()
    }
}

impl<const N: usize, Map> Matrix<[[f32; N]; N], Map> {
    /// Returns the `N`×`N` identity matrix.
    ///
    /// An identity matrix is a square matrix with ones on the main diagonal
    /// and zeroes everywhere else:
    /// ```text
    ///         ⎛ 1  0  ⋯  0 ⎞
    ///  I  =   ⎜ 0  1       ⎟
    ///         ⎜ ⋮     ⋱  0 ⎟
    ///         ⎝ 0     0  1 ⎠
    /// ```
    /// It is the neutral element of matrix multiplication:
    /// **A · I** = **I · A** = **A**, as well as matrix-vector
    /// multiplication: **I·v** = **v**.
    pub const fn identity() -> Self {
        // Needs const traits to be more generic;
        // const array::map/from_fn for a nicer impl
        let mut els = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            els[i][i] = 1.0;
            i += 1;
        }
        Self::new(els)
    }

    /// Returns whether every element of `self` is finite
    /// (ie. neither `Inf`, `-Inf`, or `NaN`).
    fn is_finite(&self) -> bool {
        self.0.iter().flatten().all(|e| e.is_finite())
    }
}

impl<Sc, const N: usize, Map> Matrix<[[Sc; N]; N], Map>
where
    Sc: Linear<Scalar = Sc> + Copy,
    Map: LinearMap,
{
    /// Returns the composite transform of `self` and `other`.
    ///
    /// Computes the matrix product of `self` and `other` such that applying
    /// the resulting transformation is equivalent to first applying `other`
    /// and then `self`. More succinctly,
    /// ```text
    /// (𝗠 ∘ 𝗡) 𝘃 = 𝗠(𝗡 𝘃)
    /// ```
    /// for some matrices 𝗠 and 𝗡 and a vector 𝘃.
    #[must_use]
    pub fn compose<Inner: LinearMap>(
        &self,
        other: &Matrix<[[Sc; N]; N], Inner>,
    ) -> Matrix<[[Sc; N]; N], <Map as Compose<Inner>>::Result>
    where
        Map: Compose<Inner>,
    {
        let cols: [_; N] = array::from_fn(|i| other.col_vec(i));
        array::from_fn(|j| {
            let row = self.row_vec(j);
            array::from_fn(|i| row.dot(&cols[i]))
        })
        .into()
    }
    /// Returns the composite transform of `other` and `self`.
    ///
    /// Computes the matrix product of `self` and `other` such that applying
    /// the resulting matrix is equivalent to first applying `self` and then
    /// `other`. The call `self.then(other)` is thus equivalent to
    /// `other.compose(self)`.
    #[must_use]
    pub fn then<Outer: Compose<Map>>(
        &self,
        other: &Matrix<[[Sc; N]; N], Outer>,
    ) -> Matrix<[[Sc; N]; N], <Outer as Compose<Map>>::Result> {
        other.compose(self)
    }
}

impl<Src, Dest> Mat2<Src, Dest> {
    /// Returns the determinant of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::Mat2;
    ///
    /// let double: Mat2 = [[2.0, 0.0], [0.0, 2.0]].into();
    /// assert_eq!(double.determinant(), 4.0);
    ///
    /// let singular: Mat2 = [[1.0, 0.0], [2.0, 0.0]].into();
    /// assert_eq!(singular.determinant(), 0.0);
    /// ```
    #[inline]
    pub const fn determinant(&self) -> f32 {
        let [[a, b], [c, d]] = self.0;
        a * d - b * c
    }

    /// Returns the [inverse][Self::inverse] of `self`, or `None` if `self`
    /// is not invertible.
    ///
    /// A matrix is invertible if and only if its [determinant][Self::determinant]
    /// is nonzero. A non-invertible matrix is also called singular.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{Mat2, mat::RealToReal};
    ///
    /// let rotate_90: Mat2<RealToReal<2>> = [[0.0, -1.0], [1.0, 0.0]].into();
    /// let rotate_neg_90 = rotate_90.checked_inverse();
    ///
    /// assert_eq!(rotate_neg_90, Some([[0.0, 1.0], [-1.0, 0.0]].into()));
    ///
    /// let singular: Mat2<RealToReal<2>> = [[1.0, 0.0], [2.0, 0.0]].into();
    /// assert_eq!(singular.checked_inverse(), None);
    /// ```
    #[must_use]
    pub const fn checked_inverse(&self) -> Option<Mat2<Dest, Src>> {
        let det = self.determinant();
        // No approx_eq in const :/
        if det.abs() < 1e-6 {
            return None;
        }
        let r_det = 1.0 / det;
        let [[a, b], [c, d]] = self.0;
        // Inverse is transpose of cofactor matrix divided by determinant
        Some(mat![
            r_det * d, r_det * -b;
            r_det * -c, r_det * a
        ])
    }

    /// Returns the inverse of `self`, if it exists.
    ///
    /// A matrix is invertible if and only if its [determinant][Self::determinant]
    /// is nonzero. A non-invertible matrix is also called singular.
    ///
    /// # Panics
    /// If `self` is singular or near-singular.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{Mat2, mat::RealToReal, vec2};
    ///
    /// let rotate_90: Mat2<RealToReal<2>> = [[0.0, -1.0], [1.0, 0.0]].into();
    /// let rotate_neg_90 = rotate_90.inverse();
    ///
    /// assert_eq!(rotate_neg_90.0, [[0.0, 1.0], [-1.0, 0.0]]);
    /// assert_eq!(rotate_90.then(&rotate_neg_90), Mat2::identity())
    /// ```
    /// ```should_panic
    /// # use retrofire_core::math::{Mat2, mat::RealToReal};
    ///
    /// // This matrix has no inverse
    /// let singular: Mat2<RealToReal<2>> = [[1.0, 0.0], [2.0, 0.0]].into();
    ///
    /// // This will panic
    /// let _ = singular.inverse();
    /// ```
    #[must_use]
    pub const fn inverse(&self) -> Mat2<Dest, Src> {
        self.checked_inverse()
            .expect("matrix cannot be singular or near-singular")
    }
}

impl<Src, Dest> Mat3<Src, Dest, 2> {
    /// Constructs a matrix from a linear basis.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_linear(i: Vec2<Dest>, j: Vec2<Dest>) -> Self {
        Self::from_affine(i, j, Point2::origin())
    }

    /// Constructs a matrix from an affine basis, or frame.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_affine(
        i: Vec2<Dest>,
        j: Vec2<Dest>,
        o: Point2<Dest>,
    ) -> Self {
        let (i, j, o) = (i.0, j.0, o.0);
        mat![
            i[0], j[0], o[0];
            i[1], j[1], o[1];
             0.0,  0.0,  1.0;
        ]
    }

    /// Returns the linear 2x2 submatrix of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::*;
    ///
    /// // TODO translate2 does not exist (yet)
    /// /*let m = rotate2(degs(90.0)).then(&translate3(1.0, 2.0, 3.0));
    /// let lin = m.linear();
    /// assert_approx_eq!(lin.apply(&pt2(1.0, 0.0, 0.0)), pt2(0.0, 0.0, -1.0));*/
    pub const fn linear(&self) -> Mat2<Src, Dest> {
        let [r, s, _] = self.0;
        mat![r[0], r[1]; s[0], s[1]]
    }

    /// Returns the translation column vector of `self`.
    ///
    /// # Example
    /// ```
    /// use retrofire_core::math::*;
    ///
    /// // TODO translate2 does not exist (yet)
    /// /*let trans = vec2(1.0, 2.0);
    /// let m = rotate2(degs(45.0)).then(&translate(trans));
    /// assert_eq!(m.translation(), trans);*/
    pub const fn translation(&self) -> Vec2<Dest> {
        let [r, s, _] = self.0;
        vec2(r[2], s[2])
    }

    /// Returns the determinant of `self`.
    pub const fn determinant(&self) -> f32 {
        let [a, b, c] = self.0[0];

        // assert!(g == 0.0 && h == 0.0 && i == 1.0);
        // TODO If affine (as should be), reduces to:
        // a * e - b * d

        a * self.cofactor(0, 0)
            + b * self.cofactor(0, 1)
            + c * self.cofactor(0, 2)
    }

    /// Returns the cofactor of the element at the given row and column.
    ///
    /// Cofactors are used to compute the inverse of a matrix. A cofactor is
    /// calculated as follows:
    ///
    /// 1. Remove the given row and column from `self` to get a 2x2 submatrix;
    /// 2. Compute its determinant;
    /// 3. If exactly one of `row` and `col` is even, multiply by -1.
    #[inline]
    const fn cofactor(&self, row: usize, col: usize) -> f32 {
        // This automatically takes care of the negation
        let r1 = (row + 1) % 3;
        let r2 = (row + 2) % 3;
        let c1 = (col + 1) % 3;
        let c2 = (col + 2) % 3;
        self.0[r1][c1] * self.0[r2][c2] - self.0[r1][c2] * self.0[r2][c1]
    }

    /// Returns the inverse of `self`, or `None` if `self` is singular.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{mat, math::Mat3};
    ///
    /// let mat: Mat3 = mat![
    ///     2.0, 0.0, 1.0;
    ///     0.0, 4.0, 2.0;
    ///     0.0, 0.0, 1.0
    /// ];
    /// assert_eq!(mat.checked_inverse(), Some(mat![
    ///     0.5,  0.0,  -0.5;
    ///     0.0,  0.25, -0.5;
    ///     0.0,  0.0,   1.0
    /// ]));
    /// ```
    #[must_use]
    pub const fn checked_inverse(&self) -> Option<Mat3<Dest, Src, 2>> {
        let det = self.determinant();
        if det.abs() < 1e-6 {
            return None;
        }

        // Inverse is transpose of cofactor matrix divided by determinant
        let mut res = [[0.0; 3]; 3];
        let r_det = 1.0 / det;
        let mut i = 0;
        while i < 3 {
            res[i][0] = r_det * self.cofactor(0, i);
            res[i][1] = r_det * self.cofactor(1, i);
            res[i][2] = r_det * self.cofactor(2, i);
            i += 1;
        }
        /*let c_a = self.cofactor(0, 0); // = e
        let c_b = self.cofactor(0, 1); // = d
        let c_c = self.cofactor(0, 2); // = 0
        let c_d = self.cofactor(1, 0); // = b
        let c_e = self.cofactor(1, 1); // = a
        let c_f = self.cofactor(1, 2); // = 0
        let c_g = self.cofactor(2, 0); // = b * f - c * e
        let c_h = self.cofactor(2, 1); // = a * f - c * d
        let c_i = self.cofactor(2, 2); // = a * e - b * d*/

        Some(Mat3::new(res))
    }

    pub fn inverse(&self) -> Mat3<Dest, Src> {
        self.checked_inverse()
            .expect("matrix cannot be singular or near-singular")
    }
}

impl<Src, Dst> Mat4<Src, Dst> {
    /// Constructs a matrix from a linear basis.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_linear(i: Vec3<Dst>, j: Vec3<Dst>, k: Vec3<Dst>) -> Self {
        Self::from_affine(i, j, k, Point3::origin())
    }

    /// Constructs a matrix from an affine basis, or frame.
    ///
    /// A frame consists of three vectors defining a linear basis, plus a point
    /// specifying the origin point of the frame.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_affine(
        i: Vec3<Dst>,
        j: Vec3<Dst>,
        k: Vec3<Dst>,
        o: Point3<Dst>,
    ) -> Self {
        let (o, i, j, k) = (o.0, i.0, j.0, k.0);
        mat![
            i[0], j[0], k[0], o[0];
            i[1], j[1], k[1], o[1];
            i[2], j[2], k[2], o[2];
             0.0,  0.0,  0.0,  1.0
        ]
    }

    /// Returns the linear 3x3 submatrix of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::*;
    ///
    /// let m = rotate_y(degs(90.0)).then(&translate3(1.0, 2.0, 3.0));
    /// let lin = m.linear();
    /// assert_approx_eq!(lin.apply(&pt3(1.0, 0.0, 0.0)), pt3(0.0, 0.0, -1.0));
    pub const fn linear(&self) -> Mat3<Src, Dst, 3> {
        let [r, s, t, _] = self.0;
        mat![
            r[0], r[1], r[2];
            s[0], r[1], r[2];
            t[0], r[1], r[2];
        ]
    }

    /// Returns the translation column vector of `self`.
    ///
    /// # Example
    /// ```
    /// use retrofire_core::math::*;
    ///
    /// let trans = vec3(1.0, 2.0, 3.0);
    /// let m = rotate_y(degs(45.0)).then(&translate(trans));
    /// assert_eq!(m.translation(), trans);
    pub const fn translation(&self) -> Vec3<Dst> {
        vec3(self.0[0][3], self.0[1][3], self.0[2][3])
    }

    /// Returns the determinant of `self`.
    ///
    /// Given a matrix M,
    /// ```text
    ///         ⎛ a  b  c  d ⎞
    ///  M  =   ⎜ e  f  g  h ⎟
    ///         ⎜ i  j  k  l ⎟
    ///         ⎝ m  n  o  p ⎠
    /// ```
    /// its determinant can be computed by multiplying each element *e* on row 0
    /// with its *minors*: the determinant of the submatrix obtained by removing
    /// the row and column of *e*:
    /// ```text
    ///              ⎜ f g h ⎜       ⎜ e g h ⎜
    /// det(M) = a · ⎜ j k l ⎜ - b · ⎜ i k l ⎜  + c * ··· - d * ···
    ///              ⎜ n o p ⎜       ⎜ m o p ⎜
    /// ```
    pub fn determinant(&self) -> f32 {
        let [[a, b, c, d], r, s, t] = self.0;

        let det2 = |m, n| s[m] * t[n] - s[n] * t[m];
        let det3 =
            |j, k, l| r[j] * det2(k, l) - r[k] * det2(j, l) + r[l] * det2(j, k);

        a * det3(1, 2, 3) - b * det3(0, 2, 3) + c * det3(0, 1, 3)
            - d * det3(0, 1, 2)
    }

    #[must_use]
    pub fn checked_inverse(&self) -> Option<Mat4<Dst, Src>> {
        (!self.determinant().approx_eq(&0.0)).then(|| self.inverse())
    }

    /// Returns the inverse matrix of `self`.
    ///
    /// The inverse **M**<sup>-1</sup> of matrix **M** is a matrix that, when
    /// composed with **M**, results in the [identity](Self::identity) matrix:
    ///
    /// **M** ∘ **M**<sup>-1</sup> = **M**<sup>-1</sup> ∘ **M** = **I**
    ///
    /// In other words, it applies the transform of **M** in reverse.
    /// Given vectors **v** and **u**,
    ///
    /// **Mv** = **u** ⇔ **M**<sup>-1</sup> **u** = **v**.
    ///
    /// Only matrices with a nonzero determinant have a defined inverse.
    /// A matrix without an inverse is said to be singular.
    ///
    /// Note: This method uses naive Gauss–Jordan elimination and may
    /// suffer from imprecision or numerical instability in certain cases.
    ///
    /// # Panics
    /// If debug assertions are enabled, panics if `self` is singular or
    /// near-singular. If not enabled, the return value is unspecified and
    /// may contain non-finite values (infinities and NaNs).
    // TODO example
    #[must_use]
    pub fn inverse(&self) -> Mat4<Dst, Src> {
        use super::float::f32;
        if cfg!(debug_assertions) {
            let det = self.determinant();
            assert!(
                !det.approx_eq(&0.0),
                "a singular, near-singular, or non-finite matrix does not \
                 have a well-defined inverse (determinant = {det})"
            );
        }

        // Elementary row operation subtracting one row,
        // multiplied by a scalar, from another
        fn sub_row(m: &mut Mat4, from: usize, to: usize, mul: f32) {
            m.0[to] = (m.row_vec(to) - m.row_vec(from) * mul).0;
        }

        // Elementary row operation multiplying one row with a scalar
        fn mul_row(m: &mut Mat4, row: usize, mul: f32) {
            m.0[row] = (m.row_vec(row) * mul).0;
        }

        // Elementary row operation swapping two rows
        const fn swap_rows(m: &mut Mat4, r: usize, s: usize) {
            m.0.swap(r, s);
        }

        // This algorithm attempts to reduce `this` to the identity matrix
        // by simultaneously applying elementary row operations to it and
        // another matrix `inv` which starts as the identity matrix. Once
        // `this` is reduced, the value of `inv` has become the inverse of
        // `this` and thus of `self`.

        let inv = &mut Mat4::identity();
        let this = &mut self.to();

        // Apply row operations to reduce the matrix to an upper echelon form
        for idx in 0..4 {
            let pivot = (idx..4)
                .max_by(|&r1, &r2| {
                    let v1 = this.0[r1][idx].abs();
                    let v2 = this.0[r2][idx].abs();
                    v1.total_cmp(&v2)
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

impl<S> LinearMap for RealToProj<S> {
    type Source = Real<3, S>;
    type Dest = Proj3;
}

impl<S, I> Compose<RealToReal<3, S, I>> for RealToProj<I> {
    type Result = RealToProj<S>;
}

/// Dummy `LinearMap` to help with generic code.
impl LinearMap for () {
    type Source = ();
    type Dest = ();
}

impl<Repr, E, M> ApproxEq<Self, E> for Matrix<Repr, M>
where
    Repr: ApproxEq<Repr, E>,
{
    fn approx_eq_eps(&self, other: &Self, rel_eps: &E) -> bool {
        self.0.approx_eq_eps(&other.0, rel_eps)
    }

    fn relative_epsilon() -> E {
        Repr::relative_epsilon()
    }
}

// Apply trait impls

impl<Src, Dest> Apply<Vec2<Src>> for Mat2<Src, Dest> {
    type Output = Vec2<Dest>;

    /// Maps a real 2-vector from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–vector multiplication **Mv** where **v** is
    /// interpreted as a column vector:
    ///
    /// ```text
    ///  Mv  =  ⎛ M00 M01 ⎞ ⎛ v0 ⎞  =  ⎛ v0' ⎞
    ///         ⎝ M10 M11 ⎠ ⎝ v1 ⎠     ⎝ v1' ⎠
    /// ```
    fn apply(&self, v: &Vec2<Src>) -> Vec2<Dest> {
        vec2(self.row_vec(0).dot(v), self.row_vec(1).dot(v))
    }
}

impl<Src, Dest> Apply<Point2<Src>> for Mat2<Src, Dest> {
    type Output = Point2<Dest>;

    /// Maps a real 2-point from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–point multiplication **M***p* where *p* is
    /// interpreted as a column vector:
    ///
    /// ```text
    ///  Mp  =  ⎛ M00 M01 ⎞ ⎛ v0 ⎞  =  ⎛ v0' ⎞
    ///         ⎝ M10 M11 ⎠ ⎝ v1 ⎠     ⎝ v1' ⎠
    /// ```
    fn apply(&self, pt: &Point2<Src>) -> Point2<Dest> {
        self.apply(&pt.to_vec()).to_pt()
    }
}

impl<Src, Dest> Apply<Vec2<Src>> for Mat3<Src, Dest, 2> {
    type Output = Vec2<Dest>;

    /// Maps a real 2-vector from basis `Src` to basis `Dst`.
    ///
    /// Computes the affine matrix–vector multiplication **Mv**, where
    /// **v** is interpreted as a homogeneous column vector with an implicit
    /// *v*<sub>2</sub> component with value 0:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ⎞ ⎛ v0 ⎞     ⎛ v0' ⎞
    ///  Mv  =  ⎜  ·  ·  ·  ⎟ ⎜ v1 ⎟  =  ⎜ v1' ⎟
    ///         ⎝  ·  · M22 ⎠ ⎝  0 ⎠     ⎝  0  ⎠
    /// ```
    fn apply(&self, v: &Vec2<Src>) -> Vec2<Dest> {
        // TODO can't use vec3, as space has to be Real<2> to match row_vec
        let v = Vector::new([v.x(), v.y(), 0.0]);
        vec2(self.row_vec(0).dot(&v), self.row_vec(1).dot(&v))
    }
}

impl<Src, Dest> Apply<Point2<Src>> for Mat3<Src, Dest, 2> {
    type Output = Point2<Dest>;

    /// Maps a real 2-point from basis `Src` to basis `Dst`.
    ///
    /// Computes the affine matrix–point multiplication **M***p*, where
    /// *p* is interpreted as a homogeneous column vector with an implicit
    /// *p*<sub>2</sub> component with value 1:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ⎞ ⎛ p0 ⎞     ⎛ p0' ⎞
    ///  Mp  =  ⎜  ·  ·  ·  ⎟ ⎜ p1 ⎟  =  ⎜ p1' ⎟
    ///         ⎝  ·  · M22 ⎠ ⎝  1 ⎠     ⎝  1  ⎠
    /// ```
    fn apply(&self, p: &Point2<Src>) -> Point2<Dest> {
        let v = Vector::new([p.x(), p.y(), 1.0]);
        pt2(self.row_vec(0).dot(&v), self.row_vec(1).dot(&v))
    }
}

impl<Src, Dest> Apply<Vec3<Src>> for Mat3<Src, Dest, 3> {
    type Output = Vec3<Dest>;

    /// Maps a real 3-vector from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–vector multiplication **Mv** where **v** is
    /// interpreted as a column vector:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ⎞ ⎛ v0 ⎞     ⎛ v0' ⎞
    ///  Mv  =  ⎜  ·  ·  ·  ⎟ ⎜ v1 ⎟  =  ⎜ v1' ⎟
    ///         ⎝  ·  · M22 ⎠ ⎝ v2 ⎠     ⎝ v2' ⎠
    /// ```
    fn apply(&self, v: &Vec3<Src>) -> Vec3<Dest> {
        vec3(
            self.row_vec(0).dot(v),
            self.row_vec(1).dot(v),
            self.row_vec(2).dot(v),
        )
    }
}

impl<Src, Dest> Apply<Point3<Src>> for Mat3<Src, Dest, 3> {
    type Output = Point3<Dest>;

    /// Maps a real 3-point from basis `Src` to basis `Dst`.
    ///
    /// Computes the linear matrix–point multiplication **M***p* where *p* is
    /// interpreted as a column vector:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ⎞ ⎛ p0 ⎞     ⎛ p0' ⎞
    ///  Mp  =  ⎜  ·  ·  ·  ⎟ ⎜ p1 ⎟  =  ⎜ p1' ⎟
    ///         ⎝  ·  · M22 ⎠ ⎝ p2 ⎠     ⎝ p2' ⎠
    /// ```
    fn apply(&self, p: &Point3<Src>) -> Point3<Dest> {
        self.apply(&p.to_vec()).to_pt()
    }
}

impl<Src, Dst> Apply<Vec3<Src>> for Mat4<Src, Dst, 3> {
    type Output = Vec3<Dst>;

    /// Maps a real 3-vector from basis `Src` to basis `Dst`.
    ///
    /// Computes the affine matrix–vector multiplication **Mv**, where
    /// **v** is interpreted as a homogeneous column vector with an implicit
    /// *v*<sub>3</sub> component with value 0:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ·  ⎞ ⎛ v0 ⎞     ⎛ v0' ⎞
    ///  Mv  =  ⎜  ·  ·  ·  ·  ⎟ ⎜ v1 ⎟  =  ⎜ v1' ⎟
    ///         ⎜  ·  ·  ·  ·  ⎟ ⎜ v2 ⎟     ⎜ v2' ⎟
    ///         ⎝  ·  ·  · M33 ⎠ ⎝  0 ⎠     ⎝  0  ⎠
    /// ```
    fn apply(&self, v: &Vec3<Src>) -> Vec3<Dst> {
        let v = [v.x(), v.y(), v.z(), 0.0].into();
        array::from_fn(|i| self.row_vec(i).dot(&v)).into()
    }
}

impl<Src, Dst> Apply<Point3<Src>> for Mat4<Src, Dst, 3> {
    type Output = Point3<Dst>;

    /// Maps a real 3-point from basis `Src` to basis `Dst`.
    ///
    /// Computes the affine matrix–point multiplication **M***p* where *p*
    /// is interpreted as a homogeneous column vector with an implicit
    /// *p*<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         ⎛ M00 ·  ·  ·  ⎞ ⎛ p0 ⎞     ⎛ p0' ⎞
    ///  Mp  =  ⎜  ·  ·  ·  ·  ⎟ ⎜ p1 ⎟  =  ⎜ p1' ⎟
    ///         ⎜  ·  ·  ·  ·  ⎟ ⎜ p2 ⎟     ⎜ p2' ⎟
    ///         ⎝  ·  ·  · M33 ⎠ ⎝  1 ⎠     ⎝  1  ⎠
    /// ```
    fn apply(&self, p: &Point3<Src>) -> Point3<Dst> {
        let p = [p.x(), p.y(), p.z(), 1.0].into();
        array::from_fn(|i| self.row_vec(i).dot(&p)).into()
    }
}

impl<Src> Apply<Point3<Src>> for ProjMat3<Src> {
    type Output = ProjVec3;

    /// Maps the real 3-point *p* from basis B to the projective 3-space.
    ///
    /// Computes the matrix–point multiplication **M***p*, where *p*
    /// is interpreted as a homogeneous column vector with an implicit
    /// *p*<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         ⎛ M00  ·  · ⎞ ⎛ p0 ⎞     ⎛ p0' ⎞
    ///  Mp  =  ⎜    ·      ⎟ ⎜ p1 ⎟  =  ⎜ p1' ⎟
    ///         ⎜      ·    ⎟ ⎜ p2 ⎟     ⎜ p2' ⎟
    ///         ⎝ ·  ·  M33 ⎠ ⎝  1 ⎠     ⎝ p3' ⎠
    /// ```
    fn apply(&self, p: &Point3<Src>) -> ProjVec3 {
        let v = Vector::new([p.x(), p.y(), p.z(), 1.0]);
        array::from_fn(|i| self.row_vec(i).dot(&v)).into()
    }
}

//
// Foreign trait impls
//

impl<R: Clone, M> Clone for Matrix<R, M> {
    fn clone(&self) -> Self {
        self.to()
    }
}

impl<const N: usize, Map> Default for Matrix<[[f32; N]; N], Map> {
    /// Returns the `N`×`N` identity matrix.
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: Debug, Map: Debug + Default, const N: usize, const M: usize> Debug
    for Matrix<[[S; N]; M], Map>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix<{:?}>[", Map::default())?;
        for i in 0..M {
            writeln!(f, "    {:4?}", self.0[i])?;
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
        Self(repr, Pd)
    }
}

//
// Free functions
//

/// Returns a matrix applying a scaling by a vector of factors.
///
/// # Examples
/// Tip: use [`splat`][super::vec::splat] to scale uniformly:
/// ```
/// use retrofire_core::math::{Apply, scale, splat, vec3};
///
/// let m = scale(splat(2.0));
/// let scaled = m.apply(&vec3(1.0, -2.0, 3.0));
/// assert_eq!(scaled, vec3(2.0, -4.0, 6.0))
/// ```
pub const fn scale(s: Vec3) -> Mat4 {
    scale3(s.0[0], s.0[1], s.0[2])
}

/// Returns a matrix applying a scaling by the given factors.
///
/// # Examples
/// See the [`scale`] method for an example.
pub const fn scale3(x: f32, y: f32, z: f32) -> Mat4 {
    mat![
         x,  0.0, 0.0, 0.0;
        0.0,  y,  0.0, 0.0;
        0.0, 0.0,  z,  0.0;
        0.0, 0.0, 0.0, 1.0;
    ]
}

/// Returns a matrix applying a translation vector to points.
///
/// Vectors have no defined position and are unaffected by translation.
///
/// # Examples
/// ```
/// use retrofire_core::math::{Apply, pt3, vec3, translate};
///
/// let m = translate(vec3(1.0, -2.0, 3.0));
///
/// // Points are moved
/// assert_eq!(m.apply(&pt3(1.0, 1.0, 1.0)), pt3(2.0, -1.0, 4.0));
///
/// // Vectors are unaffected
/// assert_eq!(m.apply(&vec3(1.0, 1.0, 1.0)), vec3(1.0, 1.0, 1.0));
///```
pub const fn translate(t: Vec3) -> Mat4 {
    translate3(t.0[0], t.0[1], t.0[2])
}

/// Returns a matrix applying a translation to *points*.
///
/// Vectors have no defined position and are unaffected by translation.
///
/// # Examples
/// See the [`translate`] function for an example.
pub const fn translate3(x: f32, y: f32, z: f32) -> Mat4 {
    mat![
        1.0, 0.0, 0.0,  x ;
        0.0, 1.0, 0.0,  y ;
        0.0, 0.0, 1.0,  z ;
        0.0, 0.0, 0.0, 1.0;
    ]
}

#[cfg(feature = "fp")]
use super::Angle;

/// Returns a matrix applying a rotation that sends the y-axis to the given vector.
///
/// The new y-axis is chosen so that it's orthogonal to both `new_z` and `x`.
/// Returns an orthogonal basis. If `new_y` and `x` are unit vectors,
/// the basis is orthonormal.
///
/// # Panics
/// If `x` is approximately parallel to `new_y` and the basis would be
/// degenerate.
// TODO example
pub fn orient_y(new_y: Vec3, x: Vec3) -> Mat4 {
    orient(new_y, x.cross(&new_y).normalize())
}
/// Returns a matrix applying a rotation that sends the z-axis to the given vector.
///
/// The new y-axis is chosen so that it's orthogonal to both `new_z` and `x`.
/// This function returns an orthogonal basis. If `new_z` and `x` are unit
/// vectors, the basis is orthonormal.
///
/// # Panics
/// If `x` is approximately parallel to `new_z` and the basis would be
/// degenerate.
pub fn orient_z(new_z: Vec3, x: Vec3) -> Mat4 {
    orient(new_z.cross(&x).normalize(), new_z)
}

/// Constructs a linear basis, given the y and z basis vectors.
///
/// The third basis vector is the cross product of `new_y` and `new_z`.
/// If the inputs are orthogonal, the resulting basis is orthogonal.
/// If the inputs are also unit vectors, the basis is orthonormal.
///
/// # Panics
/// If `new_y` is approximately parallel to `new_z` and the basis would
/// be degenerate.
fn orient(new_y: Vec3, new_z: Vec3) -> Mat4 {
    let new_x = new_y.cross(&new_z);
    assert!(
        !new_x.len_sqr().approx_eq(&0.0),
        "{new_y:?} × {new_z:?} non-finite or too close to zero vector"
    );
    Mat4::from_linear(new_x, new_y, new_z)
}

// TODO constify rotate_* functions once we have const trig functions

/// Returns a matrix applying a 3D rotation about the x-axis (on the yz plane).
///
/// # Example
/// ```
/// use retrofire_core::assert_approx_eq;
/// use retrofire_core::math::{Apply, degs, rotate_x, vec3};
///
/// let m = rotate_x(degs(90.0));
/// assert_approx_eq!(m.apply(&vec3(0.0, 1.0, 0.0)), vec3(0.0, 0.0, 1.0));
/// ```
#[cfg(feature = "fp")]
pub fn rotate_x(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    mat![
        1.0,  0.0,  0.0,  0.0;
        0.0,  cos, -sin,  0.0;
        0.0,  sin,  cos,  0.0;
        0.0,  0.0,  0.0,  1.0;
    ]
}
/// Returns a matrix applying a 3D rotation about the y-axis (on the xz plane).
///
/// # Example
/// ```
/// use retrofire_core::assert_approx_eq;
/// use retrofire_core::math::{Apply, degs, rotate_y, vec3};
///
/// let m = rotate_y(degs(90.0));
/// assert_approx_eq!(m.apply(&vec3(1.0, 0.0, 0.0)), vec3(0.0, 0.0, -1.0));
///```
#[cfg(feature = "fp")]
pub fn rotate_y(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    mat![
        cos,  0.0,  sin, 0.0;
        0.0,  1.0,  0.0, 0.0;
       -sin,  0.0,  cos, 0.0;
        0.0,  0.0,  0.0, 1.0;
    ]
}
/// Returns a matrix applying a 3D rotation about the z axis (on the xy plane).
/// # Example
/// ```
/// use retrofire_core::assert_approx_eq;
/// use retrofire_core::math::{Apply, degs, rotate_z, vec3};
///
/// let m = rotate_z(degs(90.0));
/// assert_approx_eq!(m.apply(&vec3(1.0, 0.0, 0.0)), vec3(0.0, 1.0, 0.0));
#[cfg(feature = "fp")]
pub fn rotate_z(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    mat![
        cos, -sin,  0.0,  0.0;
        sin,  cos,  0.0,  0.0;
        0.0,  0.0,  1.0,  0.0;
        0.0,  0.0,  0.0,  1.0;
    ]
}

/// Returns a matrix applying a 2D rotation by an angle.
#[cfg(feature = "fp")]
pub fn rotate2(a: Angle) -> Mat3 {
    let (sin, cos) = a.sin_cos();
    mat![
         cos, sin, 0.0;
        -sin, cos, 0.0;
         0.0, 0.0, 1.0;
    ]
}

/// Returns a matrix applying a 3D rotation about an arbitrary axis.
#[cfg(feature = "fp")]
pub fn rotate(axis: Vec3, a: Angle) -> Mat4 {
    // 1. Change of basis such that `axis` is mapped to the z-axis,
    // 2. Rotation about the z-axis
    // 3. Change of basis back to the original
    let mut other = Vec3::X;
    if axis.cross(&other).len_sqr() < 0.25 {
        // Avoid degeneracy
        other = Vec3::Y;
    }

    let z_to_axis = orient_z(axis.normalize(), other);
    // Inverse of orthogonal matrix is its transpose
    let axis_to_z = z_to_axis.transpose();
    axis_to_z.then(&rotate_z(a)).then(&z_to_axis)
}

/// Creates a perspective projection matrix.
///
/// # Parameters
/// * `focal_ratio`: Focal length/aperture ratio. Larger values mean
///   a smaller angle of view, with 1.0 corresponding to a horizontal
///   field of view of 90 degrees.
/// * `aspect_ratio`: Viewport width/height ratio. Larger values mean
///   a wider field of view.
/// * `near_far`: Depth range between the near and far clipping planes.
///   Objects outside this range are clipped or culled.
///
/// # Panics
/// * If any parameter value is nonpositive.
/// * If `near_far` is an empty range.
pub const fn perspective(
    focal_ratio: f32,
    aspect_ratio: f32,
    near_far: Range<f32>,
) -> ProjMat3<View> {
    let (near, far) = (near_far.start, near_far.end);

    assert!(focal_ratio > 0.0, "focal ratio must be positive");
    assert!(aspect_ratio > 0.0, "aspect ratio must be positive");
    assert!(near > 0.0, "near must be positive");
    assert!(far > near, "far must be greater than near");

    let e00 = focal_ratio;
    let e11 = e00 * aspect_ratio;
    let e22 = (far + near) / (far - near);
    let e23 = 2.0 * far * near / (near - far);
    mat![
        e00, 0.0, 0.0, 0.0;
        0.0, e11, 0.0, 0.0;
        0.0, 0.0, e22, e23;
        0.0, 0.0, 1.0, 0.0;
    ]
}

/// Creates an orthographic projection matrix.
///
/// # Parameters
/// * `lbn`: The left-bottom-near corner of the projection box.
/// * `rtf`: The right-bottom-far corner of the projection box.
pub const fn orthographic(lbn: Point3, rtf: Point3) -> ProjMat3<View> {
    // Done manually due until const traits are stable
    let [x0, y0, z0] = lbn.0;
    let [x1, y1, z1] = rtf.0;
    let [dx, dy, dz] = [(x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0];
    let [cx, cy, cz] = [x0 + dx, y0 + dy, z0 + dz];
    let [idx, idy, idz] = [1.0 / dx, 1.0 / dy, 1.0 / dz];
    mat![
        idx, 0.0, 0.0, -cx * idx;
        0.0, idy, 0.0, -cy * idy;
        0.0, 0.0, idz, -cz * idz;
        0.0, 0.0, 0.0, 1.0;
    ]
}

/// Creates a viewport transform matrix with the given pixel space bounds.
///
/// A viewport matrix is used to transform points from the NDC space to
/// screen space for rasterization. NDC coordinates (-1, -1, _) are mapped
/// to `bounds.start` and NDC coordinates (1, 1, _) to `bounds.end`.
pub const fn viewport(bounds: Range<Point2u>) -> Mat4<Ndc, Screen> {
    let Range { start, end } = bounds;
    let [x0, y0] = [start.x() as f32, start.y() as f32];
    let [x1, y1] = [end.x() as f32, end.y() as f32];
    let [dx, dy] = [(x1 - x0) / 2.0 as f32, (y1 - y0) / 2.0 as f32];
    mat![
         dx, 0.0, 0.0, x0 + dx;
        0.0,  dy, 0.0, y0 + dy;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
    ]
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::pt3;

    #[cfg(feature = "fp")]
    use crate::math::degs;

    use super::*;

    #[derive(Debug, Default, Eq, PartialEq)]
    struct B1;
    #[derive(Debug, Default, Eq, PartialEq)]
    struct B2;

    type Map<const N: usize = 3> = RealToReal<N, B1, B2>;
    type InvMap<const N: usize = 3> = RealToReal<N, B2, B1>;

    const X: Vec3 = Vec3::X;
    const Y: Vec3 = Vec3::Y;
    const Z: Vec3 = Vec3::Z;
    #[allow(unused)]
    const O: Vec3 = Vec3::new([0.0; 3]);

    mod mat2x2 {
        use super::*;

        #[test]
        fn determinant_of_identity_is_one() {
            let id = <Mat2>::identity();
            assert_eq!(id.determinant(), 1.0);
        }
        #[test]
        fn determinant_of_reflection_is_negative_one() {
            let refl: Mat2 = [[0.0, 1.0], [1.0, 0.0]].into();
            assert_eq!(refl.determinant(), -1.0);
        }

        #[test]
        fn inverse_of_identity_is_identity() {
            let id = <Mat2>::identity();
            assert_eq!(id.inverse(), id);
        }
        #[test]
        fn inverse_of_inverse_is_original() {
            let m: Mat2<B1, B2> = [[0.5, 1.5], [1.0, -0.5]].into();
            let m_inv: Mat2<B2, B1> = m.inverse();
            assert_approx_eq!(m_inv.inverse(), m);
        }
        #[test]
        fn composition_of_inverse_is_identity() {
            let m: Mat2<B1, B2> = [[0.5, 1.5], [1.0, -0.5]].into();
            let m_inv: Mat2<B2, B1> = m.inverse();
            assert_approx_eq!(m.compose(&m_inv), Mat2::identity());
            assert_approx_eq!(m.then(&m_inv), Mat2::identity());
        }
    }

    mod mat3x3 {
        use super::*;

        const MAT: Mat3<B1, B2, 3> = mat![
             0.0,  1.0,  2.0;
            10.0, 11.0, 12.0;
            20.0, 21.0, 22.0;
        ];

        #[test]
        fn row_col_vecs() {
            assert_eq!(MAT.row_vec(2), vec3::<_, B1>(20.0, 21.0, 22.0));
            assert_eq!(MAT.col_vec(2), vec3::<_, B2>(2.0, 12.0, 22.0));
        }

        #[test]
        fn composition() {
            let tr: Mat3<B1, B2> = mat![
                1.0,  0.0,  2.0;
                0.0,  1.0, -3.0;
                0.0,  0.0,  1.0;
            ];
            let sc: Mat3<B2, B1> = mat![
                -1.0, 0.0, 0.0;
                 0.0, 2.0, 0.0;
                 0.0, 0.0, 1.0;
            ];

            let tr_sc = tr.then(&sc);
            let sc_tr = sc.then(&tr);

            assert_eq!(tr_sc, sc.compose(&tr));
            assert_eq!(sc_tr, tr.compose(&sc));

            assert_eq!(tr_sc.apply(&vec2(1.0, 2.0)), vec2(-1.0, 4.0));
            assert_eq!(sc_tr.apply(&vec2(1.0, 2.0)), vec2(-1.0, 4.0));

            assert_eq!(tr_sc.apply(&pt2(1.0, 2.0)), pt2(-3.0, -2.0));
            assert_eq!(sc_tr.apply(&pt2(1.0, 2.0)), pt2(1.0, 1.0));
        }

        #[test]
        fn scaling() {
            let m: Mat3 = mat![
                2.0,  0.0,  0.0;
                0.0, -3.0,  0.0;
                0.0,  0.0,  1.0;
            ];
            assert_eq!(m.apply(&vec2(1.0, 2.0)), vec2(2.0, -6.0));
            assert_eq!(m.apply(&pt2(2.0, -1.0)), pt2(4.0, 3.0));
        }

        #[test]
        fn translation() {
            let m: Mat3 = mat![
                1.0,  0.0,  2.0;
                0.0,  1.0, -3.0;
                0.0,  0.0,  1.0;
            ];
            assert_eq!(m.apply(&vec2(1.0, 2.0)), vec2(1.0, 2.0));
            assert_eq!(m.apply(&pt2(2.0, -1.0)), pt2(4.0, -4.0));
        }

        #[test]
        fn inverse_of_identity_is_identity() {
            let i = <Mat3>::identity();
            assert_eq!(i.inverse(), i);
        }
        #[test]
        fn inverse_of_scale_is_reciprocal_scale() {
            let scale: Mat3 = mat![
                2.0,  0.0,  0.0;
                0.0, -3.0,  0.0;
                0.0,  0.0,  4.0;
            ];
            assert_eq!(
                scale.inverse(),
                mat![
                    1.0/2.0, 0.0,  0.0;
                    0.0, -1.0/3.0, 0.0;
                    0.0,  0.0, 1.0/4.0;
                ]
            );
        }
        #[test]
        fn matrix_composed_with_inverse_is_identity() {
            let mat: Mat3<B1, B2> = mat![
                1.0, -2.0,  2.0;
                3.0,  4.0, -3.0;
                0.0,  0.0,  1.0;
            ];
            let composed: Mat3<B2, B2> = mat.compose(&mat.inverse());
            assert_approx_eq!(composed, Mat3::identity());
            let composed: Mat3<B1, B1> = mat.then(&mat.inverse());
            assert_approx_eq!(composed, Mat3::identity());
        }

        #[test]
        fn singular_matrix_has_no_inverse() {
            let singular: Mat3 = mat![
                1.0,  2.0,  0.0;
                0.0,  0.0,  0.0;
                0.0,  0.0,  1.0;
            ];

            assert_approx_eq!(singular.checked_inverse(), None);
        }

        #[test]
        fn matrix_debug() {
            assert_eq!(
                alloc::format!("{MAT:?}"),
                r#"Matrix<B1→B2>[
    [ 0.0,  1.0,  2.0]
    [10.0, 11.0, 12.0]
    [20.0, 21.0, 22.0]
]"#
            );
        }
    }

    mod mat4 {
        use super::*;

        const MAT: Mat4<B1, B2> = mat![
             0.0,  1.0,  2.0,  3.0;
            10.0, 11.0, 12.0, 13.0;
            20.0, 21.0, 22.0, 23.0;
            30.0, 31.0, 32.0, 33.0;
        ];

        #[test]
        fn row_col_vecs() {
            assert_eq!(MAT.row_vec(1), [10.0, 11.0, 12.0, 13.0].into());
            assert_eq!(MAT.col_vec(3), [3.0, 13.0, 23.0, 33.0].into());
        }

        #[test]
        fn composition() {
            let tr = translate3(1.0, 2.0, 3.0).to::<Map>();
            let sc = scale3(3.0, 2.0, 1.0).to::<InvMap>();

            let tr_sc = tr.then(&sc);
            let sc_tr = sc.then(&tr);

            assert_eq!(tr_sc, sc.compose(&tr));
            assert_eq!(sc_tr, tr.compose(&sc));

            let o = <Point3>::origin();
            assert_eq!(tr_sc.apply(&o.to()), pt3::<_, B1>(3.0, 4.0, 3.0));
            assert_eq!(sc_tr.apply(&o.to()), pt3::<_, B2>(1.0, 2.0, 3.0));
        }

        #[test]
        fn scaling() {
            let m = scale3(1.0, -2.0, 3.0);

            let v = vec3(0.0, 4.0, -3.0);
            assert_eq!(m.apply(&v), vec3(0.0, -8.0, -9.0));

            let p = pt3(4.0, 0.0, -3.0);
            assert_eq!(m.apply(&p), pt3(4.0, 0.0, -9.0));
        }

        #[test]
        fn translation() {
            let m = translate3(1.0, 2.0, 3.0);

            let v = vec3(0.0, 5.0, -3.0);
            assert_eq!(m.apply(&v), vec3(0.0, 5.0, -3.0));

            let p = pt3(3.0, 5.0, 0.0);
            assert_eq!(m.apply(&p), pt3(4.0, 7.0, 3.0));
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_x() {
            let m = rotate_x(degs(90.0));

            assert_eq!(m.apply(&O), O);

            // Rotates counter-clockwise on the YZ-plane as seen
            // from the direction of the positive X-axis:
            //
            //           +y
            //            ^
            //            |  <--__
            //            |       \
            //            |       |
            //            O-------+---> -z
            //          /
            //        v
            //      +x
            //
            assert_approx_eq!(m.apply(&Y), Z);
            assert_approx_eq!(
                m.apply(&pt3(0.0, 0.0, 2.0)),
                pt3(0.0, -2.0, 0.0)
            );
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_y() {
            let m = rotate_y(degs(90.0));

            assert_eq!(m.apply(&O), O);

            // Rotates counter-clockwise on the ZX-plane as seen
            // from the direction of the positive Y-axis
            //
            //           +x
            //            ^
            //            |  <--__
            //            |       \
            //            |       |
            //            O-------+---> +z
            //          /
            //        v
            //     +y
            //
            assert_approx_eq!(m.apply(&Z), X);
            assert_approx_eq!(
                m.apply(&pt3(2.0, 0.0, 0.0)),
                pt3(0.0, 0.0, -2.0)
            );
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_z() {
            let m = rotate_z(degs(90.0));

            assert_eq!(m.apply(&O), O);

            // Rotates counter-clockwise on the XY-plane as seen
            // from the direction of the positive Z-axis
            //
            //          +y
            //           ^
            //           |  <--__
            //           |       \
            //           |       |
            //           O-------+---> +x
            //         /
            //       v
            //     +z
            //
            assert_approx_eq!(m.apply(&X), Y);
            assert_approx_eq!(
                m.apply(&(pt3(0.0, 2.0, 0.0))),
                pt3(-2.0, 0.0, 0.0)
            );
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_arbitrary() {
            let m = rotate(vec3(1.0, 1.0, 0.0).normalize(), degs(180.0));

            assert_approx_eq!(m.apply(&X), Y);
            assert_approx_eq!(m.apply(&Y), X);
            assert_approx_eq!(m.apply(&Z), -Z);
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_arbitrary_x() {
            let a = rotate(X, degs(128.0));
            let b = rotate_x(degs(128.0));
            assert_eq!(a, b);
        }
        #[cfg(feature = "fp")]
        #[test]
        fn rotation_arbitrary_y() {
            let a = rotate(Y, degs(128.0));
            let b = rotate_y(degs(128.0));
            assert_eq!(a, b);
        }
        #[cfg(feature = "fp")]
        #[test]
        fn rotation_arbitrary_z() {
            let a = rotate(Z, degs(128.0));
            let b = rotate_z(degs(128.0));
            assert_eq!(a, b);
        }

        #[test]
        fn from_basis() {
            let m = Mat4::from_linear(Y, 2.0 * Z, -3.0 * X);

            assert_eq!(m.apply(&X), Y);
            assert_eq!(m.apply(&Y), 2.0 * Z);
            assert_eq!(m.apply(&Z), -3.0 * X);

            assert_eq!(m.apply(&X.to_pt()), Y.to_pt());
            assert_eq!(m.apply(&Y.to_pt()), (2.0 * Z).to_pt());
            assert_eq!(m.apply(&Z.to_pt()), (-3.0 * X).to_pt());
        }

        #[test]
        fn from_affine_basis() {
            let orig = pt3(1.0, 2.0, 3.0);
            let m = Mat4::from_affine(Y, 2.0 * Z, -3.0 * X, orig);

            assert_eq!(m.apply(&X), Y);
            assert_eq!(m.apply(&Y), 2.0 * Z);
            assert_eq!(m.apply(&Z), -3.0 * X);

            assert_eq!(m.apply(&X.to_pt()), pt3(1.0, 3.0, 3.0));
            assert_eq!(m.apply(&Y.to_pt()), pt3(1.0, 2.0, 5.0));
            assert_eq!(m.apply(&Z.to_pt()), pt3(-2.0, 2.0, 3.0));
        }

        #[test]
        fn orientation_no_op() {
            let m = orient_y(Y, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), Y);
            assert_eq!(m.apply(&Y.to_pt()), Y.to_pt());

            assert_eq!(m.apply(&Z), Z);
            assert_eq!(m.apply(&Z.to_pt()), Z.to_pt());
        }

        #[test]
        fn orientation_y_to_z() {
            let m = orient_y(Z, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), Z);
            assert_eq!(m.apply(&Y.to_pt()), Z.to_pt());

            assert_eq!(m.apply(&Z), -Y);
            assert_eq!(m.apply(&Z.to_pt()), (-Y).to_pt());
        }

        #[test]
        fn orientation_z_to_y() {
            let m = orient_z(Y, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), -Z);
            assert_eq!(m.apply(&Y.to_pt()), (-Z).to_pt());

            assert_eq!(m.apply(&Z), Y);
            assert_eq!(m.apply(&Z.to_pt()), Y.to_pt());
        }

        #[test]
        fn matrix_debug() {
            assert_eq!(
                alloc::format!("{MAT:?}"),
                r#"Matrix<B1→B2>[
    [ 0.0,  1.0,  2.0,  3.0]
    [10.0, 11.0, 12.0, 13.0]
    [20.0, 21.0, 22.0, 23.0]
    [30.0, 31.0, 32.0, 33.0]
]"#
            );
        }
    }

    #[test]
    fn transposition() {
        let m: Mat3<B1, B2> = mat![
            0.0,  1.0, 2.0;
            10.0, 11.0, 12.0;
            20.0, 21.0, 22.0
        ];
        assert_eq!(
            m.transpose(),
            Mat3::<B2, B1>::new([
                [0.0, 10.0, 20.0], //
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ])
        );
    }

    #[test]
    fn determinant_of_identity_is_one() {
        let id: Mat4 = Mat4::identity();
        assert_eq!(id.determinant(), 1.0);
    }

    #[test]
    fn determinant_of_scaling_is_product_of_diagonal() {
        let scale: Mat4 = scale3(2.0, 3.0, 4.0);
        assert_eq!(scale.determinant(), 24.0);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn determinant_of_rotation_is_one() {
        let rot = rotate_x(degs(73.0)).then(&rotate_y(degs(-106.0)));
        assert_approx_eq!(rot.determinant(), 1.0);
    }

    #[test]
    fn matrix_composed_with_inverse_is_identity() {
        let m: Mat4<B1, B2> = translate3(1.0e3, -2.0e2, 0.0)
            .then(&scale3(0.5, 100.0, 42.0))
            .to();

        let m_inv: Mat4<B2, B1> = m.inverse();

        assert_eq!(m.compose(&m_inv), Mat4::identity());
        assert_eq!(m_inv.compose(&m), Mat4::identity());
    }

    #[test]
    fn inverse_reverts_transform() {
        let m: Mat4<B1, B2> = scale3(1.0, 2.0, 0.5)
            .then(&translate3(-2.0, 3.0, 0.0))
            .to();
        let m_inv: Mat4<B2, B1> = m.inverse();

        let v1: Vec3<B1> = vec3(1.0, -2.0, 3.0);
        let v2: Vec3<B2> = vec3(2.0, 0.0, -2.0);

        assert_eq!(m_inv.apply(&m.apply(&v1)), v1);
        assert_eq!(m.apply(&m_inv.apply(&v2)), v2);
    }

    #[test]
    fn orthographic_box_maps_to_unit_cube() {
        let lbn = pt3(-20.0, 0.0, 0.01);
        let rtf = pt3(100.0, 50.0, 100.0);

        let m = orthographic(lbn, rtf);

        assert_approx_eq!(m.apply(&lbn.to()), [-1.0, -1.0, -1.0, 1.0].into());
        assert_approx_eq!(m.apply(&rtf.to()), [1.0, 1.0, 1.0, 1.0].into());
    }

    #[test]
    fn perspective_frustum_maps_to_unit_cube() {
        let left_bot_near = pt3(-0.125, -0.0625, 0.1);
        let right_top_far = pt3(125.0, 62.5, 100.0);

        let m = perspective(0.8, 2.0, 0.1..100.0);

        let lbn = m.apply(&left_bot_near);
        assert_approx_eq!(lbn / lbn.w(), [-1.0, -1.0, -1.0, 1.0].into());

        let rtf = m.apply(&right_top_far);
        assert_approx_eq!(rtf / rtf.w(), [1.0, 1.0, 1.0, 1.0].into());
    }

    #[test]
    fn viewport_maps_ndc_to_screen() {
        let m = viewport(pt2(20, 10)..pt2(620, 470));

        assert_eq!(m.apply(&pt3(-1.0, -1.0, 0.2)), pt3(20.0, 10.0, 0.2));
        assert_eq!(m.apply(&pt3(1.0, 1.0, 0.6)), pt3(620.0, 470.0, 0.6));
    }
}
