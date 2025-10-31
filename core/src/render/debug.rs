//! Debugging and visualization aids.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::{Context, Frag, FragmentShader, VertexShader, scene::BBox};
use crate::geom::{Edge, Normal3, Vertex, Vertex3, vertex};
use crate::math::{
    ApproxEq, Color, Color4, Color4f, Mat4, Point3, Vary, Vec3, color::gray,
    mat::ProjMat3, polar, pt3, rgba, turns, vec::ProjVec3,
};

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

fn batch<B>(
    prims: Vec<Edge<usize>>,
    verts: Vec<Vertex3<Color4f, B>>,
) -> DbgBatch<B> {
    DbgBatch::<B>::default()
        .primitives(prims)
        .vertices(verts)
}

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

pub fn ray<B: 'static>(orig: Point3<B>, dir: Vec3<B>) -> DbgBatch<B> {
    let b = Vec3::BASES[dir.abs().arg_min()].normalize();
    let c = dir.cross(&b).normalize();

    // TODO impl frustum/cone and use it to render the arrowhead

    let (head_w, head_h) = (0.06, 0.12);
    let tip = orig + dir;
    let a = tip - dir * head_h;
    let b = head_w * b;
    let c = head_w * c;

    let verts = [orig, tip, a - b, a - c, a + b, a + c]
        .map(|p| vertex(p, dir_to_rgb(dir)));
    #[rustfmt::skip]
    let edges = [
        [0, 1], // shaft
        [1, 2], [1, 3], [1, 4], [1, 5], // head sides
        [2, 3], [3, 4], [4, 5], [5, 2], // head base
    ].map(Edge::from);

    batch(edges.to_vec(), verts.to_vec())
}

pub fn face_normal<B: 'static + Debug + Default>(
    [a, b, c]: [Point3<B>; 3],
    len: f32,
) -> DbgBatch<B> {
    let ab = b - a;
    let ac = c - a;
    let mid = a + (ab + ac) / 3.0;
    let n = ab.cross(&ac);
    if !n.len_sqr().approx_eq(&0.0) {
        ray(mid, len * n.normalize())
    } else {
        batch(Vec::new(), Vec::new())
    }
}

pub fn vertex_normal<B: 'static>(v: Vertex3<Normal3, B>) -> DbgBatch<B> {
    ray(v.pos, v.attrib.to())
}

pub fn grid<B>(size: u32) -> DbgBatch<B> {
    let ext = size as i32 / 2;
    let c = rgba(1.0, 1.0, 1.0, 1.0);

    let mut edges = Vec::new();
    let mut verts = Vec::new();
    for i in -ext..=ext {
        let l = verts.len();
        let i = i as f32;
        let ext = ext as f32;
        verts.extend([
            vertex(pt3(i, 0.0, -ext), c),
            vertex(pt3(i, 0.0, ext), c),
            vertex(pt3(-ext, 0.0, i), c),
            vertex(pt3(ext, 0.0, i), c),
        ]);
        edges.push(Edge(l, l + 1));
        edges.push(Edge(l + 2, l + 3));
    }
    batch(edges, verts)
}

pub fn frustum<B>() -> DbgBatch<B> {
    todo!()
}

pub fn frame<S, D: 'static>(m: Mat4<S, D>) -> DbgBatch<D> {
    // TODO Need a better way to get the column 3-vectors from mat4
    let [[x @ .., _], [y @ .., _], [z @ .., _], [o @ .., _]] = m.transpose().0;
    let o = o.into();
    let mut b = ray(o, x.into());
    b.append(ray(o, y.into()));
    b.append(ray(o, z.into()));
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

    batch(edges.to_vec(), verts.to_vec())
}

/// Returns the smallest box that contains all the vertices in `vs`.
pub fn bbox<'a, A, B: 'static>(vs: &[Vertex3<A, B>]) -> DbgBatch<B> {
    let BBox(min, max) = vs.iter().map(|v| &v.pos).collect();
    cuboid(min, max)
}

const RES: usize = 64;

#[cfg(feature = "fp")]
pub fn circle<B>(o: Point3<B>, r: f32) -> DbgBatch<B> {
    let verts: Vec<_> = 0.0
        .vary_to(1.0, RES as u32)
        .map(|a| {
            let v = polar(r, turns(a)).to_cart().to_vec3();
            vertex(o + v, dir_to_rgb(v))
        })
        .collect();

    let edges: Vec<_> = (0..RES).map(|i| Edge(i, (i + 1) % RES)).collect();

    batch(edges, verts)
}

/// Returns a `Batch` drawing a wireframe sphere with the given center and
/// radius.
///
/// The sphere is represented by three circles lying on the XY, XZ, and
/// YZ planes (in basis `B`).
#[cfg(feature = "fp")]
pub fn sphere<'a, B: 'static>(o: Point3<B>, r: f32) -> DbgBatch<B> {
    let mut xy = circle(o, r);

    let mut xz = xy.clone();
    for v in &mut xz.verts {
        v.pos = pt3(v.pos.x(), 0.0, v.pos.y());
    }
    let mut yz = xy.clone();
    for v in &mut yz.verts {
        v.pos = pt3(0.0, v.pos.x(), v.pos.y());
    }

    xy.append(xz);
    xy.append(yz);
    xy
}

pub fn cylinder<B>(r: f32) -> DbgBatch<B> {
    let mut top = circle(pt3(0.0, 0.0, -1.0), r);
    let bot = circle(pt3(0.0, 0.0, 1.0), r);

    top.append(bot);
    for i in 0..4 {
        // Side edges
        top.prims
            .push(Edge(i * RES / 4, RES + i * RES / 4));
    }
    top
}

pub fn cone<B>(r: f32) -> DbgBatch<B> {
    let mut base = circle(pt3(0.0, 0.0, 1.0), r);

    base.verts
        .push(vertex(Point3::origin(), gray(1.0).to_rgba()));
    for i in 0..4 {
        // Side edges
        base.prims.push(Edge(RES, i * RES / 4));
    }
    base
}
