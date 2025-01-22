//! Debugging and visualization aids.

use alloc::vec::Vec;
use core::fmt::Debug;

use crate::geom::{Edge, Vertex, Vertex3, vertex};
use crate::math::{
    Color4, Color4f, Mat4x4, Point3, Vary, Vec3, mat::RealToProj,
    mat::RealToReal, polar, pt3, rgba, splat, turns, vec::ProjVec3, vec3,
};

use super::{Context, Frag, FragmentShader, VertexShader, scene::BBox};

pub struct Shader;

impl<'a, B> VertexShader<Vertex3<Color4f, B>, &'a Mat4x4<RealToProj<B>>>
    for Shader
{
    type Output = Vertex<ProjVec3, Color4f>;

    fn shade_vertex(
        &self,
        v: Vertex3<Color4f, B>,
        m: &'a Mat4x4<RealToProj<B>>,
    ) -> Self::Output {
        vertex(m.apply(&v.pos), v.attrib)
    }
}

impl FragmentShader<Color4f> for Shader {
    fn shade_fragment(&self, f: Frag<Color4f>) -> Option<Color4> {
        Some(f.var.to_color4())
    }
}

type DbgBatch<B> =
    super::Batch<Edge<usize>, Vertex3<Color4f, B>, (), Shader, (), Context>;

/// Returns a color visualizing the direction of a vector.
///
/// # Examples
/// ```
/// use retrofire_core::math::{rgba, vec3};
/// use retrofire_core::render::debug::dir_to_rgb;
///
/// let right = vec3(1.0, 0.0, 0.0);
/// assert_eq!(dir_to_rgb(right), rgba(1.0, 0.5, 0.5, 1.0));
///
/// let down = vec3(0.0, -1.0, 0.0);
/// assert_eq!(dir_to_rgb(down), rgba(0.5, 0.0, 0.5, 1.0));
///
/// ```
pub fn dir_to_rgb<B>(v: Vec3<B>) -> Color4f {
    let c = 0.5 * v.clamp(&splat(-1.0), &splat(1.0)) + splat(0.5);
    rgba(c.x(), c.y(), c.z(), 1.0)
}

pub fn ray<'a, B: 'static>(o: Point3<B>, dir: Vec3<B>) -> DbgBatch<B> {
    let mut b = dir.cross(&Vec3::Y);
    if b.len_sqr() < 1e-6 {
        b = dir.cross(&Vec3::X);
    }
    let b = b.normalize();
    let c = dir.cross(&b).normalize();

    let (head_w, head_h) = (0.04, 0.1);
    let len = dir.len();
    let a = o + (1.0 - head_h) * dir;
    let b = head_w * len * b;
    let c = head_w * len * c;

    let verts = [o, o + dir, a + b, a - b, a + c, a - c]
        .map(|p| vertex(p, dir_to_rgb(dir)));
    #[rustfmt::skip]
    let edges = [
        [0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5],
    ].map(Edge::from);

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}

pub fn face_normal<B: 'static + Debug + Default>(
    [a, b, c]: [Point3<B>; 3],
) -> DbgBatch<B> {
    let ab = b - a;
    let ac = c - a;
    let mid = a + (ab + ac) / 3.0;
    let n = ab.cross(&ac).normalize();
    ray(mid, n)
}

pub fn frame<S, D: 'static>(m: Mat4x4<RealToReal<3, S, D>>) -> DbgBatch<D> {
    // TODO Need a better way to get the column 3-vectors from mat4
    let [x, y, z, o] = m.transpose().0;
    let x = vec3(x[0], x[1], x[2]);
    let y = vec3(y[0], y[1], y[2]);
    let z = vec3(z[0], z[1], z[2]);
    let o = pt3(o[0], o[1], o[2]);

    let mut b = ray(o, x);
    b.append(ray(o, y));
    b.append(ray(o, z));
    b
}

/// Returns a cuboid with the given opposite vertices.
pub fn cuboid<'a, B: 'static>(v0: Point3<B>, v1: Point3<B>) -> DbgBatch<B> {
    let [x0, y0, z0] = v0.0;
    let [x1, y1, z1] = v1.0;
    #[rustfmt::skip]
    let verts = [
        v0, pt3(x0, y0, z1),
        pt3(x0, y1, z1), pt3(x0, y1, z0),
        pt3(x1, y0, z0), pt3(x1, y0, z1),
        v1, pt3(x1, y1, z0),
    ]
    .map(|p| vertex(p, dir_to_rgb(p.to_vec())));
    #[rustfmt::skip]
    let edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ].map(Edge::from);

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}

/// Returns the smallest box that contains all the vertices in `vs`.
pub fn bbox<'a, A, B: 'static>(vs: &[Vertex3<A, B>]) -> DbgBatch<B> {
    let BBox(min, max) = vs.iter().map(|v| &v.pos).collect();
    cuboid(min, max)
}

pub fn circle<B>(o: Point3<B>, r: f32) -> DbgBatch<B> {
    const RES: usize = 64; // TODO constant, use array rather than Vec

    let verts: Vec<_> = 0.0
        .vary_to(1.0, RES as u32 + 1)
        .map(|a| {
            let v = polar(r, turns(a)).to_cart().to_vec3();
            vertex(o + v, dir_to_rgb(v))
        })
        .collect();

    let edges: Vec<_> = (0..RES).map(|i| Edge(i, i + 1)).collect();

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}

/// Returns a `Batch` drawing a wireframe sphere with the given center and
/// radius.
///
/// The sphere is represented by three circles lying on the XY, XZ, and
/// YZ planes (in basis `B`).
pub fn sphere<'a, B: 'static>(o: Point3<B>, r: f32) -> DbgBatch<B> {
    const RES: usize = 64;

    let verts: Vec<_> = 0.0
        .vary_to(1.0, RES as u32 + 1)
        .flat_map(|a| {
            let xy: Vec3<B> = polar(r, turns(a)).to_cart().to_vec3();
            let xz = vec3(xy.x(), 0.0, xy.y());
            let yz = vec3(0.0, xy.x(), xy.y());
            [
                vertex(o + xy, dir_to_rgb(xy)),
                vertex(o + xz, dir_to_rgb(xz)),
                vertex(o + yz, dir_to_rgb(yz)),
            ]
        })
        .collect();

    let edges: Vec<_> = (0..RES)
        .flat_map(|i| {
            [
                Edge(3 * i, 3 * i + 3),
                Edge(3 * i + 1, 3 * i + 4),
                Edge(3 * i + 2, 3 * i + 5),
            ]
        })
        .collect();

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}
