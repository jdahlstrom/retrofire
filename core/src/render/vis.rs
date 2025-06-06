//! Visualization and debugging aids.

use alloc::vec::Vec;
use core::array::from_fn;

use crate::geom::{Vertex, Vertex3, vertex};
use crate::math::{
    Color4, Color4f, Mat4x4, Point3, Vary, Vec3, mat::RealToProj, pt3, rgba,
    spherical, splat, turns, vec::ProjVec4, vec3,
};

use super::{Context, Frag, FragmentShader, VertexShader};

pub struct Shader;

impl<'a, B> VertexShader<Vertex3<Color4f, B>, &'a Mat4x4<RealToProj<B>>>
    for Shader
{
    type Output = Vertex<ProjVec4, Color4f>;

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

type Batch<B> =
    super::Batch<[usize; 2], Vertex3<Color4f, B>, (), Shader, (), Context>;

/// Returns a color visualizing the direction of a vector.
///
/// # Examples
/// ```
/// use retrofire_core::math::{rgba, vec3};
/// use retrofire_core::render::vis::dir_to_rgb;
///
/// let right = vec3(1.0, 0.0, 0.0);
/// assert_eq!(dir_to_rgb(right), rgba(1.0, 0.5, 0.5, 1.0));
///
/// let down = vec3(0.0, -1.0, 0.0);
/// assert_eq!(dir_to_rgb(down), rgba(0.5, 0.0, 0.5, 1.0));
///
/// ```
pub fn dir_to_rgb<B>(v: Vec3<B>) -> Color4f {
    let c = 0.5 * v + splat(0.5);
    rgba(c.x(), c.y(), c.z(), 1.0)
}

pub fn ray<'a, B: 'static>(o: Point3<B>, dir: Vec3<B>) -> Batch<B> {
    let mut b = dir.cross(&Vec3::Y);
    if b.len_sqr() < 1e-6 {
        b = dir.cross(&Vec3::X);
    }
    let b = b / b.len();
    let c = dir.cross(&b);
    let c = c / c.len();

    let len = dir.len();
    let a = o + 0.84 * dir;
    let b = 0.08 * len * b;
    let c = 0.08 * len * c;

    let verts = [o, o + dir, a + b, a - b, a + c, a - c]
        .map(|p| vertex(p, dir_to_rgb(dir)));
    #[rustfmt::skip]
    let edges = [
        [0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5],
    ];
    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}

/// Returns a cuboid with the given opposite vertices.
pub fn cuboid<'a, B: 'static>(v0: Point3<B>, v1: Point3<B>) -> Batch<B> {
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
    ];

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}

/// Returns the smallest box that contains all the vertices in `vs`.
pub fn aabb<'a, A, B: 'static>(vs: &[Vertex3<A, B>]) -> Batch<B> {
    let init: (Point3<B>, Point3<B>) =
        (splat(f32::INFINITY).to_pt(), splat(-f32::INFINITY).to_pt());

    let (min, max) = vs.iter().fold(init, |(min, max), v| {
        (
            from_fn(|i| min[i].min(v.pos[i])).into(),
            from_fn(|i| max[i].max(v.pos[i])).into(),
        )
    });

    cuboid(min, max)
}

/// Returns a `Batch` drawing a wireframe sphere with the given center and
/// radius.
///
/// The sphere is represented by three circles lying on the XY, XZ, and
/// YZ planes (in the coordinate space with basis `B`).
pub fn sphere<'a, B: 'static>(o: Point3<B>, r: f32) -> Batch<B> {
    const RES: usize = 64;

    let verts: Vec<_> = 0.0
        .vary_to(1.0, RES as u32 + 1)
        .flat_map(|a| {
            let xz = spherical(r, turns(a), turns(0.0)).to_cart();
            let xy = vec3(xz.x(), xz.z(), 0.0);
            let yz = vec3(0.0, xz.z(), xz.x());
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
                [3 * i, 3 * i + 3],
                [3 * i + 1, 3 * i + 4],
                [3 * i + 2, 3 * i + 5],
            ]
        })
        .collect();

    super::Batch::new()
        .primitives(&edges)
        .vertices(&verts)
        .shader(Shader)
}
