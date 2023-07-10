//! Matrices and linear transforms.

use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::math::vec::{vec4, Real, Vec3, Vec4, VectorLike};

pub trait LinearMap {
    type From;
    type To;
}

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct RealToReal<const DIM: usize, FromBasis, ToBasis>(
    PhantomData<(FromBasis, ToBasis)>,
);

impl<const DIM: usize, F, T> LinearMap for RealToReal<DIM, F, T> {
    type From = Real<DIM, F>;
    type To = Real<DIM, T>;
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RealToProjective<FromBasis>(PhantomData<FromBasis>);

#[repr(transparent)]
#[derive(Copy, Eq, PartialEq)]
pub struct Matrix<Repr, Map>(Repr, PhantomData<Map>);

impl<Scalar, Map, const N: usize> Matrix<[[Scalar; N]; N], Map> {
    pub const fn new(els: [[Scalar; N]; N]) -> Self {
        Self(els, PhantomData)
    }

    pub fn repr(&self) -> &[[Scalar; N]; N] {
        &self.0
    }

    pub fn row(&self, i: usize) -> &[Scalar; N] {
        &self.0[i]
    }
}

type Mat3x3<Scalar = f32, Map = ()> = Matrix<[[Scalar; 3]; 3], Map>;
type Mat4x4<Scalar = f32, Map = ()> = Matrix<[[Scalar; 4]; 4], Map>;

impl<M> Mat4x4<f32, M> {
    pub fn identity() -> Self {
        let mut m = Self::new(Default::default());
        for i in 0..4 {
            m.0[i][i] = 1.0;
        }
        m
    }
    #[inline]
    pub fn row_vec(&self, i: usize) -> Vec4<f32> {
        self.0[i].into()
    }
    #[inline]
    pub fn col_vec(&self, i: usize) -> Vec4<f32> {
        vec4(self.0[0][i], self.0[1][i], self.0[2][i], self.0[3][i])
    }

    pub fn to<N>(&self) -> Mat4x4<f32, N> {
        self.0.into()
    }
}

impl<F, T> Mat4x4<f32, RealToReal<4, F, T>> {
    pub fn compose<G>(
        &self,
        other: &Mat4x4<f32, RealToReal<4, G, F>>,
    ) -> Mat4x4<f32, RealToReal<4, G, T>> {
        let mut els = [[0.0_f32; 4]; 4];

        for i in 0..4 {
            els[0][i] = self.row_vec(0).dot(&other.col_vec(i));
            els[1][i] = self.row_vec(1).dot(&other.col_vec(i));
            els[2][i] = self.row_vec(2).dot(&other.col_vec(i));
            els[3][i] = self.row_vec(3).dot(&other.col_vec(i));
        }
        Matrix(els, PhantomData)
    }

    pub fn then<U>(
        &self,
        other: &Mat4x4<f32, RealToReal<4, T, U>>,
    ) -> Mat4x4<f32, RealToReal<4, F, U>> {
        other.compose(self)
    }

    pub fn map(&self, v: &Vec3<f32, Real<3, F>>) -> Vec3<f32, Real<3, T>> {
        let v = &vec4(v.x(), v.y(), v.z(), 1.0);
        let x = self.row_vec(0).dot(v);
        let y = self.row_vec(1).dot(v);
        let z = self.row_vec(2).dot(v);
        [x, y, z].into()
    }

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

    pub fn invert(&self) -> Mat4x4 {
        debug_assert!(
            self.determinant().abs() > f32::EPSILON,
            "singular or near-singular matrix has no well-defined inverse"
        );

        fn sub_row(m: &mut Mat4x4, from: usize, to: usize, mul: f32) {
            m.0[to] = m.row_vec(to).add(&m.row_vec(from).mul(-mul)).0;
        }

        fn mul_row(m: &mut Mat4x4, row: usize, mul: f32) {
            m.0[row] = m.row_vec(row).mul(mul).0;
        }

        fn swap_rows(m: &mut Mat4x4, r: usize, s: usize) {
            m.0.swap(r, s);
        }

        let mut inv = &mut Mat4x4::identity();
        let mut this = &mut Mat4x4::new(self.0.clone());

        for idx in 0..4 {
            let pivot = (idx..4)
                .max_by(|&r1, &r2| {
                    let v1 = this.row(r1)[idx].abs();
                    let v2 = this.row(r2)[idx].abs();
                    v1.partial_cmp(&v2).unwrap()
                })
                .unwrap();

            if this.row(pivot)[idx] != 0.0 {
                swap_rows(this, idx, pivot);
                swap_rows(inv, idx, pivot);

                let div = 1.0 / this.row(idx)[idx];
                for r in idx + 1..4 {
                    let x = this.row(r)[idx] * div;

                    sub_row(&mut this, idx, r, x);
                    sub_row(&mut inv, idx, r, x);
                }
            }
        }

        // now in upper echelon form, backsubstitute variables
        for &idx in &[3, 2, 1] {
            let diag = this.row(idx)[idx];
            for r in 0..idx {
                let x = this.row(r)[idx] / diag;

                sub_row(this, idx, r, x);
                sub_row(inv, idx, r, x);
            }
        }
        // normalize
        for r in 0..4 {
            let x = 1.0 / this.row(r)[r];
            mul_row(this, r, x);
            mul_row(inv, r, x);
        }
        inv.0.into()
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
        self.0.approx_eq(&other.0)
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

#[cfg(feature = "std")]
mod trig {
    use core::ops::Range;

    use crate::math::angle::Angle;

    use super::*;

    pub fn rotate_x(a: Angle) -> Mat4x4 {
        let (sin, cos) = a.sin_cos();
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, sin, 0.0],
            [0.0, -sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into()
    }

    pub fn rotate_y(a: Angle) -> Mat4x4 {
        let (sin, cos) = a.sin_cos();
        [
            [cos, 0.0, -sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into()
    }

    pub fn rotate_z(a: Angle) -> Mat4x4 {
        let (sin, cos) = a.sin_cos();
        [
            [cos, sin, 0.0, 0.0],
            [-sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into()
    }

    pub fn rotate_about(_axis: Vec3, _amt: Angle) -> Mat4x4 {
        todo!()
    }

    pub fn perspective(
        fov: Angle,
        aspect: f32,
        near_far: Range<f32>,
    ) -> Mat4x4<f32, RealToProjective<() /* TODO: Ndc */>> {
        let Range {
            start: near,
            end: far,
        } = near_far;

        assert!(near > f32::EPSILON, "near cannot be 0");
        assert!(!near_far.is_empty());

        let e00 = 1.0 / (fov / 2.0).tan();
        let e11 = aspect * e00;

        let e22 = (far + near) / (far - near);
        let e23 = -near * (far + near) / (far - near) - near;

        [
            [e00, 0.0, 0.0, 0.0], //
            [0.0, e11, 0.0, 0.0], //
            [0.0, 0.0, e22, e23], //
            [0.0, 0.0, 1.0, 0.0], //
        ]
        .into()
    }
}

#[cfg(feature = "std_float")]
pub fn look_at(pos: Vec3, at: Vec3) -> CameraMatrix {
    let z = (pos - at).normalize();
    let x = Vec3::Y.cross(z).normalize();
    let y = z.cross(x);

    AffineMat3::<(), ()>::new([
        x[0], x[1], x[2], pos[0], //
        y[0], y[1], y[2], pos[1], //
        z[0], z[1], z[2], pos[2], //
        0.0, 0.0, 0.0, 1.0,
    ])
    .invert()
    .to()
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
}

#[cfg(feature = "foo")]
mod tests {
    use core::iter::zip;

    /*impl<B: Basis> ApproxEq<Self, f32> for Vector<B>
        where
            B::Scalar: ApproxEq<B::Scalar, f32>,
        {
            fn approx_eq(&self, other: &Self) -> bool {
                self.approx_eq_eps(other, &f32::EPSILON)
            }

            fn approx_eq_eps(&self, other: &Self, eps: &f32) -> bool {
                zip(self.1.as_ref(), other.1.as_ref())
                    .all(|(s, o)| s.approx_eq_eps(o, eps))
            }
        }

        impl<M: AffineMap> ApproxEq<Self, f32> for Matrix<M>
        where
            M::Scalar: ApproxEq<M::Scalar, f32>,
        {
            fn approx_eq(&self, other: &Self) -> bool {
                self.approx_eq_eps(other, &f32::EPSILON)
            }

            fn approx_eq_eps(&self, other: &Self, eps: &f32) -> bool {
                zip(self.1.as_ref(), other.1.as_ref()) //
                    .all(|(s, o)| s.approx_eq_eps(o, eps))
            }
        }
    */
    #[test]
    fn mat_width_height() {
        let m = AffineMat3::<(), ()>::identity();

        assert_eq!(m.width(), 4);
        assert_eq!(m.height(), 4);
    }

    #[test]
    fn mat_identity() {
        let m = AffineMat3::<(), ()>::identity();

        for r in 0..m.height() {
            for c in 0..m.width() {
                assert_eq!(
                    m.row(r)[c],
                    if r == c { 1.0 } else { 0.0 },
                    "row={r}, col={c}"
                );
            }
        }
    }

    #[test]
    fn mat_vec_translate() {
        let m = translate(vec3(1.0, 2.0, 3.0));
        let v = Vec3::new([0.0, 5.0, -3.0]);

        assert_eq!(m.map(&v), Vector::new([1.0, 7.0, 0.0]));
    }

    #[test]
    fn affine_mat_row_vecs() {
        let m = AffineMat3::<(), ()>::new([
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(m.row_vec(0), vec4(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.row_vec(1), vec4(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.row_vec(2), vec4(9.0, 10.0, 11.0, 12.0));
    }

    #[test]
    fn affine_mat_col_vecs() {
        let m = AffineMat3::<(), ()>::new([
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(m.col_vec(0), vec4(1.0, 5.0, 9.0, 0.0));
        assert_eq!(m.col_vec(1), vec4(2.0, 6.0, 10.0, 0.0));
        assert_eq!(m.col_vec(2), vec4(3.0, 7.0, 11.0, 0.0));
        assert_eq!(m.col_vec(3), vec4(4.0, 8.0, 12.0, 1.0));
    }

    #[test]
    fn affine3_mat_mat_mul() {
        let mw = scale(vec3(-1.0, 1.0, -1.0));

        let composed = mw.compose(&mw).compose(&mw).compose(&mw);

        //assert_approx_eq!(composed, &Matrix::identity(), eps = 0.0001);
    }

    #[test]
    fn inverse_of_identity_is_identity() {
        let i: AffineMat3 = Matrix::identity();
        let ii = i.invert();
        assert_eq!(i, ii);
    }

    #[test]
    fn inverse_of_translate_is_negative_translate() {
        let tr = vec3(1.0, 2.0, 3.0);
        assert_eq!(translate(tr).invert(), translate(-tr));
    }
}

#[cfg(all(test, feature = "stdxxx"))]
mod trig_tests {
    //use crate::assert_approx_eq;
    use crate::math::angle::turns;
    use crate::math::vec::Vec3;

    use super::super::*;

    #[test]
    fn mat_vec_rotate_x() {
        let m = super::rotate_x(turns(0.25));
        let v = Vec3::new([1.0, 2.0, 0.0]);

        assert_approx_eq!(m.map(&v), Vector::new([1.0, 0.0, -2.0]));
    }

    #[test]
    fn mat_vec_rotate_y() {
        let m = super::rotate_y(turns(0.25));
        let v = Vec3::new([1.0, 2.0, 0.0]);

        assert_approx_eq!(m.map(&v), Vector::new([0.0, 2.0, 1.0]));
    }

    #[test]
    fn mat_vec_rotate_z() {
        let m = super::rotate_z(turns(0.25));
        let v = Vec3::new([1.0, 2.0, 0.0]);

        assert_approx_eq!(m.map(&v), Vector::new([-2.0, -1.0, 0.0]));
    }

    #[test]
    fn map_vec_from_world_to_projective() {
        let v = Vector::new([200.0, 0.0, 999.0]);

        let p = super::perspective(degs(90.0), 1.0, 1.0..1000.0);

        dbg!(&p);

        dbg!(p.map(&v).project_to_real());
        dbg!(p.map(&v));
        todo!("doesn't assert anything")
    }

    #[test]
    fn inverse_of_rotate_is_negative_rotate() {
        let ng = degs(137.0);
        let i: AffineMat3 = rotate_x(ng);
        let ii = i.invert();
        assert_approx_eq!(ii, rotate_x(-ng));
    }

    #[test]
    fn inverse_of_inverse_is_original() {
        let mat = rotate_z(degs(198.0))
            .compose(&translate(vec3(1.0, 2.0, 3.0)))
            .compose(&scale(vec3(-2.0, 3.0, 4.0)))
            .compose(&rotate_x(degs(37.0)));

        assert_approx_eq!(mat, mat.invert().invert(), eps = 1e-4)
    }

    #[test]
    fn original_times_inverse_is_identity() {
        let mat = rotate_z(degs(198.0))
            .compose(&translate(vec3(1.0, 2.0, 3.0)))
            .compose(&scale(vec3(-2.0, 3.0, 4.0)))
            .compose(&rotate_x(degs(37.0)));

        assert_approx_eq!(
            mat.compose(&mat.invert()),
            Matrix::identity(),
            eps = 1e-4
        )
    }
}
