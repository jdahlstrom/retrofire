use crate::mat::*;

pub fn scale(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m.0[0][0] = x;
    m.0[1][1] = y;
    m.0[2][2] = z;
    m
}

pub fn rotate_x(radians: f32) -> Mat4 {
    let (sin, cos) = radians.sin_cos();
    Mat4([
        [1.0, 0.0, 0.0, 0.0], //
        [0.0, cos, sin, 0.0],
        [0.0, -sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn rotate_y(radians: f32) -> Mat4 {
    let (sin, cos) = radians.sin_cos();
    Mat4([
        [cos, 0.0, sin, 0.0], //
        [0.0, 1.0, 0.0, 0.0],
        [-sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn rotate_z(radians: f32) -> Mat4 {
    let (sin, cos) = radians.sin_cos();
    Mat4([
        [cos, sin, 0.0, 0.0], //
        [-sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m.0[3][0] = x;
    m.0[3][1] = y;
    m.0[3][2] = z;
    m
}

/*
(near, far) -> (-near, far)

(x - near) / (far - near) * (far + near) - near

 x * (far+near)/(far-near) - near*(far+near)/(far-near)-near

  ...

z' =
*/

pub fn perspective(near: f32, far: f32, aspect: f32, fov: f32) -> Mat4 {
    #![allow(clippy::float_cmp)]
    assert_ne!(near, 0.0, "near cannot be 0");

    let m11 = 1.0 / (fov / 2.0).tan();
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
    use std::f32::consts::*;

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
        let actual = scale(2., -3., 4.);
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
        let actual = rotate_x(FRAC_PI_2);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_x(PI);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_x(2. * PI);
        assert_approx_eq(&expected, &actual)
    }

    #[test]
    pub fn rotate_y_matrix() {
        let expected = Mat4([
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(FRAC_PI_2);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [-1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(PI);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_y(2. * PI);
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
        let actual = rotate_z(FRAC_PI_2);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_z(PI);
        assert_approx_eq(&expected, &actual);

        let expected = Mat4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        let actual = rotate_z(2. * PI);
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

        let m = &rotate_z(FRAC_PI_2);
        assert_approx_eq(&(m * m), expected);

        let m = &rotate_z(FRAC_PI_6);
        assert_approx_eq(&(m * m * m * m * m * m), expected)
    }

    #[test]
    fn rotate_vector() {
        let m = rotate_z(FRAC_PI_2);
        assert_approx_eq(&m * X, Y);
        assert_eq!(&m * Z, Z);
    }

    #[test]
    fn scale_vector() {
        let m = &scale(2., 3., 4.);
        assert_eq!(m * Y, 3. * Y);
        assert_eq!(m * vec4(-2., 1., 3., 0.), vec4(-4., 3., 12., 0.));
    }

    #[test]
    fn translate_vector() {
        let m = &translate(2., -1., 3.);
        assert_eq!(m * Y, Y);
        assert_eq!(m * (Y + W), vec4(2., 0., 3., 1.));
    }

    #[test]
    fn perspective_project_vec() {
        let m = &perspective(2., 100., 1., PI / 2.);
        let v = pt(0., 0., 0.5);
        let w = pt(0., 0., -0.001);

        // TODO
        assert!(false);
    }
}
