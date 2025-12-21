//! Routines for drawing wireframe visualizations of geometric objects
//! for debugging purposes. Includes normals, bounding boxes, and more.

#[cfg(feature = "fp")]
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::geom::{Edge, Tri, Vertex, Vertex3, vertex};
use crate::math::{
    Color, Color4, Color4f, Mat4, Point3, Vec3, color::gray, mat::ProjMat3,
    pt3, vec::ProjVec3,
};
#[cfg(feature = "fp")]
use crate::math::{Vary, polar, turns, vec3};

use super::{Context, Frag, FragmentShader, VertexShader, scene::BBox};

#[derive(Copy, Clone, Default)]
pub struct Shader;

impl<'a, B> VertexShader<Vertex3<Color4f, B>, &'a ProjMat3<B>> for Shader {
    type Output = Vertex<ProjVec3, Color4f>;

    fn shade_vertex(
        &self,
        v: Vertex3<Color4f, B>,
        m: &'a ProjMat3<B>,
    ) -> Self::Output {
        vertex(m.apply(&v.pos), v.attrib)
    }
}

impl FragmentShader<Color4f> for Shader {
    fn shade_fragment(&self, f: Frag<Color4f>) -> Option<Color4> {
        Some(f.var.to_color4())
    }
}

pub type DbgBatch<B> =
    super::Batch<Edge<usize>, Vertex3<Color4f, B>, (), Shader, (), Context>;

/// Returns a color visualizing the direction of a vector.
///
/// # Examples
/// ```
/// use retrofire_core::math::{Vec3, rgba, vec3};
/// use retrofire_core::render::debug::dir_to_rgb;
///
/// let right: Vec3 = vec3(1.0, 0.0, 0.0);
/// assert_eq!(dir_to_rgb(right), rgba(1.0, 0.5, 0.5, 1.0));
///
/// let down: Vec3 = vec3(0.0, -1.0, 0.0);
/// assert_eq!(dir_to_rgb(down), rgba(0.5, 0.0, 0.5, 1.0));
///
/// ```
pub fn dir_to_rgb<B>(v: Vec3<B>) -> Color4f {
    (0.5 * Color::new(v.0) + gray(0.5))
        .clamp(&gray(0.0), &gray(1.0))
        .to_rgba()
}

/// Draws an illustration of a ray.
pub fn ray<'a, B>(o: Point3<B>, dir: Vec3<B>) -> DbgBatch<B> {
    let mut b = dir.cross(&Vec3::Y);
    if b.len_sqr() < 1e-6 {
        b = dir.cross(&Vec3::X);
    }
    let b = b.normalize_or_zero();
    let c = dir.cross(&b).normalize_or_zero();

    let (head_w, head_h) = (0.04, 0.1);
    let a = o + dir - head_h * dir.normalize_or_zero();
    let b = head_w * b;
    let c = head_w * c;

    let verts = [o, o + dir, a + b, a - b, a + c, a - c]
        .map(|p| vertex(p, dir_to_rgb(dir)));
    #[rustfmt::skip]
    let edges = [
        [0, 1], [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 4], [2, 5], [3, 4], [3, 5],
    ].map(|[i, j]| Edge(i, j));

    DbgBatch::new(&edges, &verts)
}

/// Draws a unit-length ray denoting the normal vector of a triangle.
///
/// The ray originates from the triangle's centroid.
pub fn face_normal<A, B: Debug + Default>(
    tri: Tri<Vertex3<A, B>>,
) -> DbgBatch<B> {
    ray(tri.centroid(), tri.normal().to())
}

/// Draws a visualization of an affine basis.
///
/// Draws three rays representing the coordinate axes. The rays originate
/// from the origin point of the basis.
pub fn basis<S, D>(m: Mat4<S, D>) -> DbgBatch<D> {
    let xyz = m.linear();
    let x = xyz.col_vec(0);
    let y = xyz.col_vec(1);
    let z = xyz.col_vec(2);
    let o = m.origin();

    let mut b = ray(o, x);
    b.append(ray(o, y));
    b.append(ray(o, z));
    b
}

/// Draws an axis-aligned box with the given opposite vertices.
pub fn cuboid<B>(v0: Point3<B>, v1: Point3<B>) -> DbgBatch<B> {
    let [x0, y0, z0] = v0.0;
    let [x1, y1, z1] = v1.0;
    #[rustfmt::skip]
    let verts = [
        v0, pt3(x0, y0, z1),
        pt3(x0, y1, z1), pt3(x0, y1, z0),
        pt3(x1, y0, z0), pt3(x1, y0, z1),
        v1, pt3(x1, y1, z0),
    ].map(|p| vertex(p, dir_to_rgb(p.to_vec())));
    #[rustfmt::skip]
    let edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ].map(|[i, j]| Edge(i, j));

    DbgBatch::new(&edges, &verts)
}

/// Draws the smallest axis-aligned box that contains a set of vertices.
pub fn bbox<A, B>(vs: &[Vertex3<A, B>]) -> DbgBatch<B> {
    let BBox(min, max) = vs.iter().map(|v| &v.pos).collect();
    cuboid(min, max)
}

/// Draws a circle on the XY plane with the given center and radius.
#[cfg(feature = "fp")]
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

    DbgBatch::new(&edges, &verts)
}

/// Draws a wireframe sphere with the given center and radius.
///
/// The sphere is represented by three circles lying on the XY, XZ, and
/// YZ planes.
#[cfg(feature = "fp")]
pub fn sphere<B>(o: Point3<B>, r: f32) -> DbgBatch<B> {
    const RES: usize = 64;

    let verts: Vec<_> = 0.0
        .vary_to(1.0, RES as u32 + 1)
        .flat_map(|a| {
            let [x, y] = polar::<()>(r, turns(a)).to_cart().0;
            [vec3(x, y, 0.0), vec3(x, 0.0, y), vec3(0.0, x, y)]
                .map(|v| vertex(o + v, dir_to_rgb(v)))
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

    DbgBatch::new(&edges, &verts)
}

impl<B> DbgBatch<B> {
    fn new(prims: &[Edge<usize>], verts: &[Vertex3<Color4f, B>]) -> Self {
        DbgBatch::<B>::default()
            .primitives(prims)
            .vertices(verts)
            .shader(Shader)
    }
}
