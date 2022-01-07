use math::vec::*;
use geom::mesh2::Vertex;
use crate::raster::Fragment;
use util::color::Color;

pub trait VertexShader<In, Out=In> {
    fn shade_vertex(&self, _: Vertex<In>) -> Vertex<Out>;
}

pub trait FragmentShader<F, U> {
    fn shade_fragment(&self, _: Fragment<F, U>) -> Option<Color>;
}

pub trait Shader<U, VI, VO=VI, FI=VO>: VertexShader<VI, VO> + FragmentShader<FI, U> {}

impl<T, U, VI, VO, FI> Shader<U, VI, VO, FI> for T
where T: VertexShader<VI, VO> + FragmentShader<FI, U> {}

pub struct ShaderImpl<VS, FS> {
    pub vs: VS,
    pub fs: FS,
}

impl<F, In, Out> VertexShader<In, Out> for F
where F: Fn(Vertex<In>) -> Vertex<Out>
{
    fn shade_vertex(&self, v: Vertex<In>) -> Vertex<Out> {
        self(v)
    }
}

impl<F, U, In> FragmentShader<In, U> for F
where F: Fn(Fragment<In, U>) -> Option<Color>
{
    fn shade_fragment(&self, f: Fragment<In, U>) -> Option<Color> {
        self(f)
    }
}

impl<VS, FS, In, Out> VertexShader<In, Out> for ShaderImpl<VS, FS>
where VS: VertexShader<In, Out>
{
    fn shade_vertex(&self, v: Vertex<In>) -> Vertex<Out> {
        self.vs.shade_vertex(v)
    }
}

impl<VS, FS, U, In> FragmentShader<In, U> for ShaderImpl<VS, FS>
where FS: FragmentShader<In, U>
{
    fn shade_fragment(&self, f: Fragment<In, U>) -> Option<Color> {
        self.fs.shade_fragment(f)
    }
}

pub fn lambert(normal: Vec4, light_dir: Vec4) -> f32 {
    normal.dot(light_dir).max(0.0)
}

pub fn phong(normal: Vec4, view_dir: Vec4, light_dir: Vec4, alpha: i32) -> f32 {
    light_dir.reflect(normal).dot(view_dir).max(0.0).powi(alpha)
}

pub fn expose(x: f32, tau: f32) -> f32 {
    1.0 - f32::exp(-x * tau)
}

pub fn expose_rgb(rgb: Vec4, tau: f32) -> Vec4 {
    Vec4 {
        x: expose(rgb.x, tau),
        y: expose(rgb.y, tau),
        z: expose(rgb.z, tau),
        w: rgb.w
    }
}

#[cfg(test)]
mod tests {
    use math::ApproxEq;

    use super::*;

    fn v() -> Vec4 {
        dir(0.1, -0.8, 0.3).normalize()
    }

    #[test]
    fn lambert_parallel() {
        let v = v();
        assert!(1.0.approx_eq(lambert(v, v)));
    }

    #[test]
    fn lambert_perpendicular() {
        let v = v();
        assert!(0.0.approx_eq(lambert(v, v.cross(X))));
    }

    #[test]
    fn lambert_shadowed() {
        let v = v();
        assert_eq!(0.0, lambert(v, -v));
        assert_eq!(0.0, lambert(v, Y));
    }

    #[test]
    fn phong_view_dir_parallel_to_reflect() {
        let n = Y;
        assert_eq!(1.0, phong(n, Z, -Z, 1));
        assert_eq!(1.0, phong(n, Y, Y, 1));
        assert!(1.0.approx_eq(phong(n, (Y + X).normalize(), (Y - X).normalize(), 1)));
    }

    #[test]
    fn phong_view_dir_parallel_to_light_dir() {
        let n = Y;
        assert_eq!(0.0, phong(n, X, X, 1));
        assert_eq!(0.0, phong(n, (X + Y).normalize(), (X + Y).normalize(), 1));
        let d = (0.5 * X + Y).normalize();
        assert!(0.6.approx_eq(phong(n, d, d, 1)));
    }

    #[test]
    fn expose_zero_is_zero() {
        assert_eq!(0.0, expose(0.0, 0.0));
        assert_eq!(0.0, expose(0.0, 1.0));
        assert_eq!(0.0, expose(0.0, 100.0));
    }

    #[test]
    fn expose_at_most_one() {
        assert!(1.0 >= expose(1.0, 1.0));
        assert!(1.0 >= expose(100.0, 1.0));
        assert!(1.0 >= expose(1.0, 100.0));
        assert!(1.0 >= expose(100.0, 100.0));
    }

    #[test]
    fn expose_at_least_zero() {
        assert!(0.0 <= expose(1e-10, 1e-10));
        assert!(0.0 <= expose(0.01, 0.0));
        assert!(0.0 <= expose(1.0, 0.0));
    }
}
