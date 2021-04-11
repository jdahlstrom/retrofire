use crate::{Angle, mat::*, vec::*};

pub trait Transform {
    fn transform(&mut self, tf: &Mat4);
}

impl Transform for Vec4 {
    fn transform(&mut self, tf: &Mat4) {
        *self = tf * *self;
    }
}

impl<T: Transform> Transform for [T] {
    fn transform(&mut self, tf: &Mat4) {
        for x in self { x.transform(tf); }
    }
}

pub fn scale(factor: f32) -> Mat4 {
    scale_axes(factor, factor, factor)
}

pub fn scale_axes(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m.0[0][0] = x;
    m.0[1][1] = y;
    m.0[2][2] = z;
    m
}

pub fn rotate_x(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    Mat4([
        [1.0, 0.0, 0.0, 0.0], //
        [0.0, cos, sin, 0.0],
        [0.0, -sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn rotate_y(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    Mat4([
        [cos, 0.0, -sin, 0.0], //
        [0.0, 1.0, 0.0, 0.0],
        [sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn rotate_z(a: Angle) -> Mat4 {
    let (sin, cos) = a.sin_cos();
    Mat4([
        [cos, sin, 0.0, 0.0], //
        [-sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn translate(offset: Vec4) -> Mat4 {
    Mat4::from_rows([X, Y, Z, offset.to_pt()])
}

pub fn orient_y(new_y: Vec4, x: Vec4) -> Mat4 {
    orient(new_y, x.cross(new_y), x)
}

pub fn orient_z(new_z: Vec4, x: Vec4) -> Mat4 {
    orient(new_z.cross(x), new_z, x)
}

fn orient(new_y: Vec4, new_z: Vec4, x: Vec4) -> Mat4 {
    assert!(new_y.w == 0.0 && new_z.w == 0.0 && x.w == 0.0);
    assert!(new_y != ZERO && new_z != ZERO && x != ZERO);

    let new_x = new_y.cross(new_z);
    Mat4::from_rows([
        new_x.normalize(),
        new_y.normalize(),
        new_z.normalize(),
        W,
    ])
}

pub fn orthogonal(low: Vec4, upp: Vec4) -> Mat4 {
    let d = upp - low;
    let (x, y, z) = (2.0 / d.x, 2.0 / d.y, 2.0 / d.z);
    let tr = pt(-x * low.x - 1.0, -y * low.y - 1.0, -z * low.z - 1.0);

    Mat4::from_rows([x * X, y * Y, z * Z, tr])
}

pub fn perspective(near: f32, far: f32, aspect: f32, fov: Angle) -> Mat4 {
    #![allow(clippy::float_cmp)]
    assert_ne!(near, 0.0, "near cannot be 0");
    assert!(near < far);

    let m11 = 1.0 / (fov.as_rad() / 2.0).tan();
    let m22 = aspect * m11;

    let m33 = (far + near) / (far - near);
    let m43 = -near * (far + near) / (far - near) - near;

    Mat4([
        [m11, 0.0, 0.0, 0.0],
        [0.0, m22, 0.0, 0.0],
        [0.0, 0.0, m33, 1.0],
        [0.0, 0.0, m43, 0.0],
    ])
}

pub fn viewport(left: f32, top: f32, right: f32, bottom: f32) -> Mat4 {
    let h = (right - left) / 2.0;
    let v = (bottom - top) / 2.0;
    Mat4([
        [h, 0.0, 0.0, 0.0],
        [0.0, v, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [h + left, v + top, 0.0, 1.0],
    ])
}

#[cfg(test)]
mod tests {
    use crate::Angle::*;
    use crate::tests::util::*;
    use crate::vec::*;

    use super::*;

    #[test]
    pub fn scale_matrix() {
        let expected = Mat4([
            [2., 0., 0., 0.],
            [0., -3., 0., 0.],
            [0., 0., 4., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = scale_axes(2., -3., 4.);
        assert_eq!(expected, actual);
    }

    #[test]
    pub fn rotate_x_matrix() {
        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_x(Tau(0.25));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_x(Tau(0.5));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_x(Tau(1.0));
        assert_approx_eq(&expected, &actual)
    }

    #[test]
    pub fn rotate_y_matrix() {
        let expected = Mat4([
            [0., 0., -1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(Tau(0.25));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [-1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(Tau(0.5));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(Tau(1.0));
        assert_approx_eq(&expected, &actual)
    }

    #[test]
    pub fn rotate_z_matrix() {
        let expected = Mat4([
            [0., 1., 0., 0.],
            [-1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_z(Tau(0.25));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_z(Tau(0.5));
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_z(Tau(1.0));
        assert_approx_eq(&expected, &actual)
    }

    #[test]
    fn compose_transforms() {
        let expected = &Mat4([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);

        let m = &rotate_z(Tau(0.25));
        assert_approx_eq(&(m * m), expected);

        let m = &rotate_z(Tau(1.0/12.0));
        assert_approx_eq(&(m * m * m * m * m * m), expected)
    }

    #[test]
    fn rotate_z_vector() {
        let m = &rotate_z(Deg(90.0));
        dbg!(m);
        assert_approx_eq(m * X, Y);
        assert_approx_eq(m * Y, -X);
        assert_eq!(m * Z, Z);
    }

    #[test]
    fn rotate_y_vector() {
        let m = &rotate_y(Deg(90.0));
        assert_approx_eq(m * Z, X);
        assert_approx_eq(m * X, -Z);
        assert_eq!(m * Y, Y);
    }

    #[test]
    fn rotate_x_vector() {
        let m = &rotate_x(Deg(90.0));
        assert_approx_eq(m * Y, Z);
        assert_approx_eq(m * Z, -Y);
        assert_eq!(m * X, X);
    }

    #[test]
    fn scale_vector() {
        let m = &scale_axes(2., 3., 4.);
        assert_eq!(m * Y, 3. * Y);
        assert_eq!(m * vec4(-2., 1., 3., 0.), vec4(-4., 3., 12., 0.));
    }

    #[test]
    fn translate_point() {
        let m = &translate(dir(2., -1., 3.));
        assert_eq!(m * (Y + W), vec4(2., 0., 3., 1.));
    }
    #[test]
    fn translate_dir_is_noop() {
        let m = &translate(dir(2., -1., 3.));
        assert_eq!(m * X, X);
        assert_eq!(m * Y, Y);
        assert_eq!(m * Z, Z);
    }

    #[test]
    fn orient_y_vector() {
        let m = &orient_y(Y+Z, X);
        assert_eq!((Y+Z).normalize(), m * Y);
        assert_eq!((Z-Y).normalize(), m * Z);
        assert_eq!(X, m * X);
        assert_eq!(ZERO, m * ZERO);
    }

    #[test]
    fn orient_z_vector() {
        let m = &orient_z(Y+Z, X);
        assert_eq!((Y-Z).normalize(), m * Y);
        assert_eq!((Y+Z).normalize(), m * Z);
        assert_eq!(X, m * X);
        assert_eq!(ZERO, m * ZERO);
    }

    #[test]
    fn orthogonal_project_points_to_unit_cube() {
        let m = &orthogonal(dir(-10.0, -5.0, 0.0), dir(10.0, 5.0, 1.0));

        assert_eq!(pt(0.0, 0.0, -1.0), m * pt(0.0, 0.0, 0.0));
        assert_eq!(pt(-1.0, 0.0, -1.0), m * pt(-10.0, 0.0, 0.0));
        assert_eq!(pt(0.0, 1.0, 0.0), m * pt(0.0, 5.0, 0.5));
        assert_eq!(pt(0.0, 0.0, 1.0), m * pt(0.0, 0.0, 1.0));

    }

    #[test]
    fn perspective_project_points_on_frustum_planes() {
        let m = &perspective(0.1, 100., 1., Deg(90.0));

        // near
        assert_approx_eq(vec4(0.0, 0.0, -0.1, 0.1), m * pt(0.0, 0.0, 0.1));
        // far
        assert_approx_eq(vec4(0.0, 0.0, 100.0, 100.0), m * pt(0.0, 0.0, 100.0));
        // left
        assert_approx_eq(vec4(-10.0, 0.0, 9.819819, 10.0), m * pt(-10.0, 0.0, 10.0));
        // right
        assert_approx_eq(vec4(10.0, 0.0, 9.819819, 10.0), m * pt(10.0, 0.0, 10.0));
        // bottom
        assert_approx_eq(vec4(0.0, -10.0, 9.819819, 10.0), m * pt(0.0, -10.0, 10.0));
        // top
        assert_approx_eq(vec4(0.0, 10.0, 9.819819, 10.0), m * pt(0.0, 10.0, 10.0));
    }

    #[test]
    fn perspective_project_aspect_ratio() {
        let m = &perspective(1., 10., 4./3., Deg(90.0));

        assert_approx_eq(vec4(4.0, 4.0, 10.0, 10.0), m * pt(4.0, 3.0, 10.0));
    }

    #[test]
    fn perspective_project_field_of_view() {
        let m = &perspective(1., 10., 1., 2. * Angle::atan(0.5));

        assert_approx_eq(vec4(1.0, 0.0, -1.0, 1.0), m * pt(0.5, 0.0, 1.0));
        assert_approx_eq(vec4(-10.0, 0.0, 10.0, 10.0), m * pt(-5.0, 0.0, 10.0));

        let m = &perspective(1., 10., 1., 2. * Angle::atan(2.0));

        assert_approx_eq(vec4(0.0, 1.0, -1.0, 1.0), m * pt(0.0, 2.0, 1.0));
        assert_approx_eq(vec4(0.0, -10.0, 10.0, 10.0), m * pt(0.0, -20.0, 10.0));
    }
}
