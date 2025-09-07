#![allow(clippy::needless_range_loop)]

//! Matrices and linear and affine transforms.
//!
//! TODO Docs

use core::{
    array,
    fmt::{self, Debug, Formatter},
    marker::PhantomData as Pd,
    ops::Range,
};

use crate::render::{NdcToScreen, ViewToProj};

use super::{
    float::f32,
    point::{Point2, Point2u, Point3},
    space::{Linear, Proj3, Real},
    vec::{ProjVec3, Vec2, Vec3, Vector},
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

/// Type alias for a 3x3 float matrix.
pub type Mat3x3<Map = ()> = Matrix<[[f32; 3]; 3], Map>;
/// Type alias for a 4x4 float matrix.
pub type Mat4x4<Map = ()> = Matrix<[[f32; 4]; 4], Map>;

//
// Inherent impls
//

macro_rules! mat {
    ($($i:expr),+; $($j:expr),+; $($k:expr),+; $($($l:expr),+)? $(;)?) => {
        Matrix([[$($i),+], [$($j),+], [$($k),+], $([$($l),+])?], Pd)
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
}

impl<Sc, const N: usize, const M: usize, Map> Matrix<[[Sc; N]; M], Map>
where
    Sc: Copy,
    Map: LinearMap,
{
    /// Returns the row vector of `self` with index `i`.
    ///
    /// The returned vector is in space `Map::Source`.
    ///
    /// # Panics
    /// If `i >= M`.
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
    #[inline]
    pub fn col_vec(&self, i: usize) -> Vector<[Sc; M], Map::Dest> {
        Vector::new(self.0.map(|row| row[i]))
    }
}
impl<Sc: Copy, const N: usize, const DIM: usize, S, D>
    Matrix<[[Sc; N]; N], RealToReal<DIM, S, D>>
{
    /// Returns `self` with its rows and columns swapped.
    pub fn transpose(self) -> Matrix<[[Sc; N]; N], RealToReal<DIM, D, S>> {
        const { assert!(N >= DIM, "map dimension >= matrix dimension") }
        array::from_fn(|j| array::from_fn(|i| self.0[i][j])).into()
    }
}

impl<const N: usize, Map> Matrix<[[f32; N]; N], Map> {
    /// Returns the `N`×`N` identity matrix.
    pub const fn identity() -> Self {
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

impl Mat4x4 {
    /// Constructs a matrix from a linear basis.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_linear<S, D>(
        i: Vec3<D>,
        j: Vec3<D>,
        k: Vec3<D>,
    ) -> Mat4x4<RealToReal<3, S, D>> {
        Self::from_affine(i, j, k, Point3::origin())
    }

    /// Constructs a matrix from an affine basis, or frame.
    ///
    /// The basis does not have to be orthonormal.
    pub const fn from_affine<S, D>(
        i: Vec3<D>,
        j: Vec3<D>,
        k: Vec3<D>,
        o: Point3<D>,
    ) -> Mat4x4<RealToReal<3, S, D>> {
        let (o, i, j, k) = (o.0, i.0, j.0, k.0);
        mat![
            i[0], j[0], k[0], o[0];
            i[1], j[1], k[1], o[1];
            i[2], j[2], k[2], o[2];
            0.0, 0.0, 0.0, 1.0
        ]
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
    /// (𝗠 ∘ 𝗡)𝘃 = 𝗠(𝗡𝘃)
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

impl<Src, Dst> Mat3x3<RealToReal<2, Src, Dst>> {
    /// Maps the real 2-vector 𝘃 from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–vector multiplication 𝝡𝘃 where 𝘃 is interpreted as
    /// a column vector with an implicit 𝘃<sub>2</sub> component with value 1:
    ///
    /// ```text
    ///         / M00 ·  ·  \ / v0 \     / v0' \
    ///  Mv  =  |  ·  ·  ·  | | v1 |  =  | v1' |
    ///         \  ·  · M22 / \  1 /     \  1  /
    /// ```
    #[must_use]
    pub fn apply(&self, v: &Vec2<Src>) -> Vec2<Dst> {
        let v = [v.x(), v.y(), 1.0].into(); // TODO w=0.0
        array::from_fn(|i| self.row_vec(i).dot(&v)).into()
    }

    // TODO Add trait to overload apply or similar
    #[must_use]
    pub fn apply_pt(&self, p: &Point2<Src>) -> Point2<Dst> {
        let p = [p.x(), p.y(), 1.0].into();
        array::from_fn(|i| self.row_vec(i).dot(&p)).into()
    }
}

impl<Src, Dst> Mat4x4<RealToReal<3, Src, Dst>> {
    /// Maps the real 3-vector 𝘃 from basis `Src` to basis `Dst`.
    ///
    /// Computes the matrix–vector multiplication 𝝡𝘃 where 𝘃 is interpreted as
    /// a column vector with an implicit 𝘃<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         / M00 ·  ·  ·  \ / v0 \     / v0' \
    ///  Mv  =  |  ·  ·  ·  ·  | | v1 |  =  | v1' |
    ///         |  ·  ·  ·  ·  | | v2 |     | v2' |
    ///         \  ·  ·  · M33 / \  1 /     \  1  /
    /// ```
    #[must_use]
    pub fn apply(&self, v: &Vec3<Src>) -> Vec3<Dst> {
        let v = [v.x(), v.y(), v.z(), 1.0].into(); // TODO w=0.0
        array::from_fn(|i| self.row_vec(i).dot(&v)).into()
    }

    // TODO Add trait to overload apply or similar
    #[must_use]
    pub fn apply_pt(&self, p: &Point3<Src>) -> Point3<Dst> {
        let p = [p.x(), p.y(), p.z(), 1.0].into();
        array::from_fn(|i| self.row_vec(i).dot(&p)).into()
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
        let [[a, b, c, d], r, s, t] = self.0;
        let det2 = |m, n| s[m] * t[n] - s[n] * t[m];
        let det3 =
            |j, k, l| r[j] * det2(k, l) - r[k] * det2(j, l) + r[l] * det2(j, k);

        a * det3(1, 2, 3) - b * det3(0, 2, 3) + c * det3(0, 1, 3)
            - d * det3(0, 1, 2)
    }

    /// Returns the inverse matrix of `self`.
    ///
    /// The inverse 𝝡<sup>-1</sup> of matrix 𝝡 is a matrix that, when
    /// composed with 𝝡, results in the identity matrix:
    ///
    /// 𝝡 ∘ 𝝡<sup>-1</sup> = 𝝡<sup>-1</sup> ∘ 𝝡 = 𝐈
    ///
    /// In other words, it applies the transform of 𝝡 in reverse.
    /// Given vectors 𝘃 and 𝘂,
    ///
    /// 𝝡𝘃 = 𝘂 ⇔ 𝝡<sup>-1</sup> 𝘂 = 𝘃.
    ///
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
    ///   or contain `Inf`s or `NaN`s.
    #[must_use]
    pub fn inverse(&self) -> Mat4x4<RealToReal<3, Dst, Src>> {
        use super::float::f32;
        if cfg!(debug_assertions) {
            let det = self.determinant();
            assert!(
                f32::abs(det) > f32::EPSILON,
                "a singular, near-singular, or non-finite matrix does not \
                 have a well-defined inverse (determinant = {det})"
            );
        }

        // Elementary row operation subtracting one row,
        // multiplied by a scalar, from another
        fn sub_row(m: &mut Mat4x4, from: usize, to: usize, mul: f32) {
            m.0[to] = (m.row_vec(to) - m.row_vec(from) * mul).0;
        }

        // Elementary row operation multiplying one row with a scalar
        fn mul_row(m: &mut Mat4x4, row: usize, mul: f32) {
            m.0[row] = (m.row_vec(row) * mul).0;
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

impl<Src> Mat4x4<RealToProj<Src>> {
    /// Maps the real 3-vector 𝘃 from basis 𝖡 to the projective 4-space.
    ///
    /// Computes the matrix–vector multiplication 𝝡𝘃 where 𝘃 is interpreted as
    /// a column vector with an implicit 𝘃<sub>3</sub> component with value 1:
    ///
    /// ```text
    ///         / M00  ·  · \ / v0 \     / v0' \
    ///  Mv  =  |    ·      | | v1 |  =  | v1' |
    ///         |      ·    | | v2 |     | v2' |
    ///         \ ·  ·  M33 / \  1 /     \ v3' /
    /// ```
    #[must_use]
    pub fn apply(&self, p: &Point3<Src>) -> ProjVec3 {
        let v = Vector::new([p.x(), p.y(), p.z(), 1.0]);
        array::from_fn(|i| self.row_vec(i).dot(&v)).into()
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
        Self(repr, Pd)
    }
}

//
// Free functions
//

/// Returns a matrix applying a scaling by `s`.
///
/// Tip: use [`splat`][super::vec::splat] to scale uniformly:
/// ```
/// use retrofire_core::math::{scale, splat};
/// let m = scale(splat(2.0));
/// assert_eq!(m.0[0][0], 2.0);
/// assert_eq!(m.0[1][1], 2.0);
/// assert_eq!(m.0[2][2], 2.0);
/// ```
pub const fn scale(s: Vec3) -> Mat4x4<RealToReal<3>> {
    scale3(s.0[0], s.0[1], s.0[2])
}

pub const fn scale3(x: f32, y: f32, z: f32) -> Mat4x4<RealToReal<3>> {
    mat! [
         x,  0.0, 0.0, 0.0;
        0.0,  y,  0.0, 0.0;
        0.0, 0.0,  z,  0.0;
        0.0, 0.0, 0.0, 1.0;
    ]
}

/// Returns a matrix applying a translation by `t`.
pub const fn translate(t: Vec3) -> Mat4x4<RealToReal<3>> {
    translate3(t.0[0], t.0[1], t.0[2])
}

pub const fn translate3(x: f32, y: f32, z: f32) -> Mat4x4<RealToReal<3>> {
    mat![
        1.0, 0.0, 0.0,  x ;
        0.0, 1.0, 0.0,  y ;
        0.0, 0.0, 1.0,  z ;
        0.0, 0.0, 0.0, 1.0;
    ]
}

#[cfg(feature = "fp")]
use super::{Angle, ApproxEq};

/// Returns a matrix applying a rotation such that the original y axis
/// is now parallel with `new_y` and the new z axis is orthogonal to
/// both `x` and `new_y`.
///
/// Returns an orthogonal basis. If `new_y` and `x` are unit vectors,
/// the basis is orthonormal.
///
/// # Panics
/// If `x` is approximately parallel to `new_y` and the basis would be
/// degenerate.
#[cfg(feature = "fp")]
pub fn orient_y(new_y: Vec3, x: Vec3) -> Mat4x4<RealToReal<3>> {
    orient(new_y, x.cross(&new_y).normalize())
}
/// Returns a matrix applying a rotation such that the original z axis
/// is now parallel with `new_z` and the new y axis is orthogonal to
/// both `new_z` and `x`.
///
/// Returns an orthogonal basis. If `new_z` and `x` are unit vectors,
/// the basis is orthonormal.
///
/// # Panics
/// If `x` is approximately parallel to `new_z` and the basis would be
/// degenerate.
#[cfg(feature = "fp")]
pub fn orient_z(new_z: Vec3, x: Vec3) -> Mat4x4<RealToReal<3>> {
    orient(new_z.cross(&x).normalize(), new_z)
}

/// Constructs a change-of-basis matrix given y and z basis vectors.
///
/// The third basis vector is the cross product of `new_y` and `new_z`.
/// If the inputs are orthogonal, the resulting basis is orthogonal.
/// If the inputs are also unit vectors, the basis is orthonormal.
///
/// # Panics
/// If `new_y` is approximately parallel to `new_z` and the basis would
/// be degenerate.
#[cfg(feature = "fp")]
fn orient(new_y: Vec3, new_z: Vec3) -> Mat4x4<RealToReal<3>> {
    let new_x = new_y.cross(&new_z);
    assert!(
        !new_x.len_sqr().approx_eq(&0.0),
        "{new_y:?} × {new_z:?} non-finite or too close to zero vector"
    );
    Mat4x4::from_linear(new_x, new_y, new_z)
}

// TODO constify rotate_* functions once we have const trig functions

/// Returns a matrix applying a 3D rotation about the x axis.
#[cfg(feature = "fp")]
pub fn rotate_x(a: Angle) -> Mat4x4<RealToReal<3>> {
    let (sin, cos) = a.sin_cos();
    mat![
        1.0,  0.0, 0.0, 0.0;
        0.0,  cos, sin, 0.0;
        0.0, -sin, cos, 0.0;
        0.0,  0.0, 0.0, 1.0;
    ]
}
/// Returns a matrix applying a 3D rotation about the y axis.
#[cfg(feature = "fp")]
pub fn rotate_y(a: Angle) -> Mat4x4<RealToReal<3>> {
    let (sin, cos) = a.sin_cos();
    mat![
        cos, 0.0, -sin, 0.0;
        0.0, 1.0,  0.0, 0.0;
        sin, 0.0,  cos, 0.0;
        0.0, 0.0,  0.0, 1.0;
    ]
}
/// Returns a matrix applying a 3D rotation about the z axis.
#[cfg(feature = "fp")]
pub fn rotate_z(a: Angle) -> Mat4x4<RealToReal<3>> {
    let (sin, cos) = a.sin_cos();
    mat![
         cos, sin, 0.0, 0.0;
        -sin, cos, 0.0, 0.0;
         0.0, 0.0, 1.0, 0.0;
         0.0, 0.0, 0.0, 1.0;
    ]
}

/// Returns a matrix applying a 2D rotation by an angle.
#[cfg(feature = "fp")]
pub fn rotate2(a: Angle) -> Mat3x3<RealToReal<2>> {
    let (sin, cos) = a.sin_cos();
    mat![
         cos, sin, 0.0;
        -sin, cos, 0.0;
         0.0, 0.0, 1.0;
    ]
}

/// Returns a matrix applying a 3D rotation about an arbitrary axis.
#[cfg(feature = "fp")]
pub fn rotate(axis: Vec3, a: Angle) -> Mat4x4<RealToReal<3>> {
    use crate::math::approx::ApproxEq;

    // 1. Change of basis such that `axis` is mapped to the z-axis,
    // 2. Rotation about the z-axis
    // 3. Change of basis back to the original
    let mut other = Vec3::X;
    if axis.cross(&other).len_sqr().approx_eq(&0.0) {
        // Avoid degeneracy
        other = Vec3::Y;
    }

    let z_to_axis = orient_z(axis.normalize(), other);
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
pub fn perspective(
    focal_ratio: f32,
    aspect_ratio: f32,
    near_far: Range<f32>,
) -> Mat4x4<ViewToProj> {
    let (near, far) = (near_far.start, near_far.end);

    assert!(focal_ratio > 0.0, "focal ratio must be positive");
    assert!(aspect_ratio > 0.0, "aspect ratio must be positive");
    assert!(near > 0.0, "near must be positive");
    assert!(!near_far.is_empty(), "far must be greater than near");

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
pub fn orthographic(lbn: Point3, rtf: Point3) -> Mat4x4<ViewToProj> {
    let half_d = (rtf - lbn) / 2.0;
    let [cx, cy, cz] = (lbn + half_d).0;
    let [idx, idy, idz] = half_d.map(f32::recip).0;
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
/// screen space for rasterization. NDC coordinates (-1, -1, z) are mapped
/// to `bounds.start` and NDC coordinates (1, 1, z) to `bounds.end`.
pub fn viewport(bounds: Range<Point2u>) -> Mat4x4<NdcToScreen> {
    let s = bounds.start.map(|c| c as f32);
    let e = bounds.end.map(|c| c as f32);
    let half_d = (e - s) / 2.0;
    let [dx, dy] = half_d.0;
    let [cx, cy] = (s + half_d).0;
    mat![
         dx, 0.0, 0.0,  cx;
        0.0,  dy, 0.0,  cy;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
    ]
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::{pt2, pt3, vec2, vec3};

    #[cfg(feature = "fp")]
    use crate::math::degs;

    use super::*;

    #[derive(Debug, Default, Eq, PartialEq)]
    struct Basis1;
    #[derive(Debug, Default, Eq, PartialEq)]
    struct Basis2;

    type Map<const N: usize = 3> = RealToReal<N, Basis1, Basis2>;
    type InvMap<const N: usize = 3> = RealToReal<N, Basis2, Basis1>;

    const X: Vec3 = Vec3::X;
    const Y: Vec3 = Vec3::Y;
    const Z: Vec3 = Vec3::Z;
    const O: Vec3 = Vec3::new([0.0; 3]);

    mod mat3x3 {
        use super::*;
        use crate::math::pt2;

        const MAT: Mat3x3<Map> = mat![
             0.0,  1.0,  2.0;
            10.0, 11.0, 12.0;
            20.0, 21.0, 22.0;
        ];

        #[test]
        fn row_col_vecs() {
            assert_eq!(MAT.row_vec(2), vec3::<_, Basis1>(20.0, 21.0, 22.0));
            assert_eq!(MAT.col_vec(2), vec3::<_, Basis2>(2.0, 12.0, 22.0));
        }

        #[test]
        fn composition() {
            let t: Mat3x3<Map<2>> = mat![
                1.0,  0.0,  2.0;
                0.0,  1.0, -3.0;
                0.0,  0.0,  1.0;
            ];
            let s: Mat3x3<InvMap<2>> = mat![
                -1.0, 0.0, 0.0;
                 0.0, 2.0, 0.0;
                 0.0, 0.0, 1.0;
            ];

            let ts = t.then(&s);
            let st = s.then(&t);

            assert_eq!(ts, s.compose(&t));
            assert_eq!(st, t.compose(&s));

            assert_eq!(ts.apply(&vec2(0.0, 0.0)), vec2(-2.0, -6.0));
            assert_eq!(st.apply(&vec2(0.0, 0.0)), vec2(2.0, -3.0));
        }

        #[test]
        fn scaling() {
            let m: Mat3x3<Map<2>> = mat![
                2.0,  0.0,  0.0;
                0.0, -3.0,  0.0;
                0.0,  0.0,  1.0;
            ];
            assert_eq!(m.apply(&vec2(1.0, 2.0)), vec2(2.0, -6.0));
            assert_eq!(m.apply_pt(&pt2(2.0, -1.0)), pt2(4.0, 3.0));
        }

        #[test]
        fn translation() {
            let m: Mat3x3<Map<2>> = mat![
                1.0,  0.0,  2.0;
                0.0,  1.0, -3.0;
                0.0,  0.0,  1.0;
            ];
            assert_eq!(m.apply(&vec2(1.0, 2.0)), vec2(3.0, -1.0));
            assert_eq!(m.apply_pt(&pt2(2.0, -1.0)), pt2(4.0, -4.0));
        }

        #[test]
        fn matrix_debug() {
            assert_eq!(
                alloc::format!("{MAT:?}"),
                r#"Matrix<Basis1→Basis2>[
    [  0.00,   1.00,   2.00]
    [ 10.00,  11.00,  12.00]
    [ 20.00,  21.00,  22.00]
]"#
            );
        }
    }

    mod mat4x4 {
        use super::*;
        use crate::math::pt3;

        const MAT: Mat4x4<Map> = mat![
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
            let t = translate3(1.0, 2.0, 3.0).to::<Map>();
            let s = scale3(3.0, 2.0, 1.0).to::<InvMap>();

            let ts = t.then(&s);
            let st = s.then(&t);

            assert_eq!(ts, s.compose(&t));
            assert_eq!(st, t.compose(&s));

            assert_eq!(ts.apply(&O.to()), vec3::<_, Basis1>(3.0, 4.0, 3.0));
            assert_eq!(st.apply(&O.to()), vec3::<_, Basis2>(1.0, 2.0, 3.0));
        }

        #[test]
        fn scaling() {
            let m = scale3(1.0, -2.0, 3.0);

            let v = vec3(0.0, 4.0, -3.0);
            assert_eq!(m.apply(&v), vec3(0.0, -8.0, -9.0));

            let p = pt3(4.0, 0.0, -3.0);
            assert_eq!(m.apply_pt(&p), pt3(4.0, 0.0, -9.0));
        }

        #[test]
        fn translation() {
            let m = translate3(1.0, 2.0, 3.0);

            let v = vec3(0.0, 5.0, -3.0);
            assert_eq!(m.apply(&v), vec3(1.0, 7.0, 0.0));

            let p = pt3(3.0, 5.0, 0.0);
            assert_eq!(m.apply_pt(&p), pt3(4.0, 7.0, 3.0));
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_x() {
            let m = rotate_x(degs(90.0));

            assert_eq!(m.apply(&O), O);

            assert_approx_eq!(m.apply(&Z), Y);
            assert_approx_eq!(
                m.apply_pt(&pt3(0.0, -2.0, 0.0)),
                pt3(0.0, 0.0, 2.0)
            );
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_y() {
            let m = rotate_y(degs(90.0));

            assert_eq!(m.apply(&O), O);

            assert_approx_eq!(m.apply(&X), Z);
            assert_approx_eq!(
                m.apply_pt(&pt3(0.0, 0.0, -2.0)),
                pt3(2.0, 0.0, 0.0)
            );
        }

        #[cfg(feature = "fp")]
        #[test]
        fn rotation_z() {
            let m = rotate_z(degs(90.0));

            assert_eq!(m.apply(&O), O);

            assert_approx_eq!(m.apply(&Y), X);
            assert_approx_eq!(
                m.apply_pt(&(pt3(-2.0, 0.0, 0.0))),
                pt3(0.0, 2.0, 0.0)
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
            let m = Mat4x4::from_linear(Y, 2.0 * Z, -3.0 * X);
            assert_eq!(m.apply(&X), Y);
            assert_eq!(m.apply(&Y), 2.0 * Z);
            assert_eq!(m.apply(&Z), -3.0 * X);
        }

        #[cfg(feature = "fp")]
        #[test]
        fn orientation_no_op() {
            let m = orient_y(Y, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply_pt(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), Y);
            assert_eq!(m.apply_pt(&Y.to_pt()), Y.to_pt());

            assert_eq!(m.apply(&Z), Z);
            assert_eq!(m.apply_pt(&Z.to_pt()), Z.to_pt());
        }

        #[cfg(feature = "fp")]
        #[test]
        fn orientation_y_to_z() {
            let m = orient_y(Z, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply_pt(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), Z);
            assert_eq!(m.apply_pt(&Y.to_pt()), Z.to_pt());

            assert_eq!(m.apply(&Z), -Y);
            assert_eq!(m.apply_pt(&Z.to_pt()), (-Y).to_pt());
        }

        #[cfg(feature = "fp")]
        #[test]
        fn orientation_z_to_y() {
            let m = orient_z(Y, X);

            assert_eq!(m.apply(&X), X);
            assert_eq!(m.apply_pt(&X.to_pt()), X.to_pt());

            assert_eq!(m.apply(&Y), -Z);
            assert_eq!(m.apply_pt(&Y.to_pt()), (-Z).to_pt());

            assert_eq!(m.apply(&Z), Y);
            assert_eq!(m.apply_pt(&Z.to_pt()), Y.to_pt());
        }

        #[test]
        fn matrix_debug() {
            assert_eq!(
                alloc::format!("{MAT:?}"),
                r#"Matrix<Basis1→Basis2>[
    [  0.00,   1.00,   2.00,   3.00]
    [ 10.00,  11.00,  12.00,  13.00]
    [ 20.00,  21.00,  22.00,  23.00]
    [ 30.00,  31.00,  32.00,  33.00]
]"#
            );
        }
    }

    #[test]
    fn transposition() {
        let m = Matrix::<_, Map>::new([
            [0.0, 1.0, 2.0], //
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]);
        assert_eq!(
            m.transpose(),
            Matrix::<_, InvMap>::new([
                [0.0, 10.0, 20.0], //
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ])
        );
    }

    #[test]
    fn determinant_of_identity_is_one() {
        let id: Mat4x4<Map> = Mat4x4::identity();
        assert_eq!(id.determinant(), 1.0);
    }

    #[test]
    fn determinant_of_scaling_is_product_of_diagonal() {
        let scale: Mat4x4<_> = scale3(2.0, 3.0, 4.0);
        assert_eq!(scale.determinant(), 24.0);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn determinant_of_rotation_is_one() {
        let rot = rotate_x(degs(73.0)).then(&rotate_y(degs(-106.0)));
        assert_approx_eq!(rot.determinant(), 1.0);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn mat_times_mat_inverse_is_identity() {
        let m = translate3(1.0e3, -2.0e2, 0.0)
            .then(&scale3(0.5, 100.0, 42.0))
            .to::<Map>();

        let m_inv: Mat4x4<InvMap> = m.inverse();

        assert_eq!(
            m.compose(&m_inv),
            Mat4x4::<RealToReal<3, Basis2, Basis2>>::identity()
        );
        assert_eq!(
            m_inv.compose(&m),
            Mat4x4::<RealToReal<3, Basis1, Basis1>>::identity()
        );
    }

    #[test]
    fn inverse_reverts_transform() {
        let m: Mat4x4<Map> = scale3(1.0, 2.0, 0.5)
            .then(&translate3(-2.0, 3.0, 0.0))
            .to();
        let m_inv: Mat4x4<InvMap> = m.inverse();

        let v1: Vec3<Basis1> = vec3(1.0, -2.0, 3.0);
        let v2: Vec3<Basis2> = vec3(2.0, 0.0, -2.0);

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

        assert_eq!(m.apply_pt(&pt3(-1.0, -1.0, 0.2)), pt3(20.0, 10.0, 0.2));
        assert_eq!(m.apply_pt(&pt3(1.0, 1.0, 0.6)), pt3(620.0, 470.0, 0.6));
    }
}
