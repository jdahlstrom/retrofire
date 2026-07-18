//! Routines for drawing wireframe visualizations of geometric objects
//! for debugging purposes. Includes normals, bounding boxes, and more.

use alloc::vec::Vec;
use core::fmt::Write;

use crate::geom::{Edge, Mesh, Normal3, Pos, Tri, Vertex, Vertex3, vertex};
use crate::math::{
    Color, Color3, Color4, Color4f, Mat4, Point3, Vec3, color::gray,
    mat::ProjMat3, pt3, vec::ProjVec3,
};
use crate::util::{Dims, pnm::read_pnm};

use super::{
    Context, Frag, FragmentShader, Model, Text, VertexShader, scene::BBox,
    tex::Atlas,
};
#[cfg(feature = "fp")]
use crate::math::{Vary, polar, turns, vec3};
use crate::render::text::Align;

#[derive(Copy, Clone, Default)]
pub struct Shader;

#[derive(Clone, Default)]
/// A type used to draw debug visualizations of meshes.
///
/// Various mesh properties can be visualized:
/// * Edges ("wireframe" rendering)
/// * Face normals
/// * Vertex normals (if any)
/// * Bounding box
/// * Model-space origin and coordinate axes.
pub struct DbgMesh<A, B>(Mesh<A, B>, DbgBatch<B>);

pub type DbgBatch<B> = super::Batch<
    Vec<Edge<usize>>,
    Vec<Vertex3<Color4f, B>>,
    (),
    Shader,
    (),
    Context,
>;

pub static FONT_6X10: &[u8] = include_bytes!("../../assets/font_6x10.pbm");

/// Returns a 6x10 bitmap font, e.g. for rendering debug messages.
pub fn font_6x10() -> Atlas<Color3> {
    let font = read_pnm(FONT_6X10).expect("font statically included");
    Atlas::grid(Dims(6, 10), font.into())
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

pub fn text(s: &str) -> Text {
    // TODO re-reads and duplicates atlas every time...
    let mut text = Text::new(font_6x10());
    text.align = Align::Center;
    text.write_str(s).expect("cannot fail");
    text
}

/// Draws an illustration of a ray.
pub fn ray<B>(orig: Point3<B>, dir: Vec3<B>) -> DbgBatch<B> {
    let mut batch = DbgBatch::new();
    batch.ray(orig, dir);
    batch
}

/// Draws a unit-length ray denoting the normal vector of a vertex.
pub fn vertex_normal<B>(v: &Vertex3<Normal3, B>, scale: f32) -> DbgBatch<B> {
    ray(v.pos, scale * v.attrib.to())
}

/// Draws a unit-length ray denoting the normal vector of a triangle.
///
/// The ray originates from the triangle's centroid.
pub fn face_normal<A, B>(tri: &Tri<Vertex3<A, B>>, scale: f32) -> DbgBatch<B> {
    ray(tri.centroid(), scale * tri.normal().to())
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
    ].map(Edge::from);

    DbgBatch::with(edges.to_vec(), verts.to_vec())
}

/// Draws the smallest axis-aligned box that contains a set of vertices.
pub fn bbox<B>(pts: &[impl Pos<Type = Point3<B>>]) -> DbgBatch<B> {
    let BBox(min, max) = pts.iter().map(Pos::pos).collect();
    cuboid(min, max)
}

/// Creates a `DbgMesh` object from a mesh.
pub fn mesh<A: Clone, B>(mesh: Mesh<A, B>) -> DbgMesh<A, B> {
    DbgMesh(mesh, DbgBatch::default())
}

//
// Inherent impls
//

impl<A: Clone, B> DbgMesh<A, B> {
    /// Enables drawing the edges as a wireframe representation.
    #[must_use]
    pub fn edges(mut self) -> Self {
        let Mesh { faces, verts } = &self.0;

        let n_verts = self.1.verts.len();
        let edges = faces
            .iter()
            .flat_map(Tri::edges)
            .map(|Edge(a, b)| Edge(a + n_verts, b + n_verts));
        let verts = verts
            .iter()
            .map(|v| vertex(v.pos, dir_to_rgb(v.pos.to_vec())));

        self.1.prims.extend(edges);
        self.1.verts.extend(verts);

        self
    }

    /// Enables drawing the normal vectors of the faces.
    #[must_use]
    pub fn face_normals(mut self, scale: f32) -> Self {
        for tri in self.0.faces() {
            self.1.face_normal(&tri.map(Clone::clone), scale);
        }
        self
    }

    /// Enables drawing the local-space bounding box of the mesh.
    #[must_use]
    pub fn bbox(mut self) -> Self {
        self.1.append(bbox(&self.0.verts));
        self
    }

    /// Enables drawing the coordinate axes of the local space.
    #[must_use]
    pub fn basis(mut self) -> Self {
        self.1.append(basis(Mat4::<B>::identity()));
        self
    }

    /// Returns a batch for rendering the enabled visualizations.
    #[must_use]
    pub fn batch(&self) -> &DbgBatch<B> {
        &self.1
    }
}

impl<B> DbgMesh<Normal3, B> {
    #[must_use]
    pub fn vertex_normals(mut self, scale: f32) -> Self {
        for v in &self.0.verts {
            self.1.vertex_normal(v, scale);
        }
        self
    }
}

/// Draws a wireframe representation of a mesh.
///
/// This is a convenience shortcut for
/// ```text
/// # use retrofire_core::render::debug;
/// debug::mesh(mesh).edges().patch()
/// ```
#[must_use]
pub fn wireframe<A: Clone, B>(m: Mesh<A, B>) -> DbgBatch<B> {
    mesh(m).edges().batch().clone()
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

    DbgBatch::with(edges, verts)
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

    DbgBatch::with(edges, verts)
}

impl<B> DbgBatch<B> {
    fn new() -> Self {
        DbgBatch::<B>::default()
    }

    fn with(prims: Vec<Edge<usize>>, verts: Vec<Vertex3<Color4f, B>>) -> Self {
        DbgBatch::<B>::default()
            .primitives(prims)
            .vertices(verts)
    }

    fn ray(&mut self, orig: Point3<B>, dir: Vec3<B>) -> &mut Self {
        let mut b = dir.cross(&Vec3::Y);
        if b.len_sqr() < 1e-6 {
            b = dir.cross(&Vec3::X);
        }
        let b = b.normalize_or_zero();
        let c = dir.cross(&b).normalize_or_zero();

        let (head_w, head_h) = (0.02, 0.04);
        let a = orig + dir - head_h * dir.normalize_or_zero();
        let b = head_w * b;
        let c = head_w * c;

        let verts = [orig, orig + dir, a + b, a - b, a + c, a - c]
            .map(|p| vertex(p, dir_to_rgb(dir)));
        #[rustfmt::skip]
        let edges = [
            [0, 1], [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 4], [2, 5], [3, 4], [3, 5],
        ].map(Edge::from);

        let n = self.verts.len();
        self.verts.extend(verts);
        self.prims
            .extend(edges.into_iter().map(|e| Edge(e.0 + n, e.1 + n)));

        self
    }

    fn vertex_normal(
        &mut self,
        v: &Vertex3<Normal3, B>,
        scale: f32,
    ) -> &mut Self {
        self.ray(v.pos, scale * v.attrib.to())
    }

    fn face_normal<A>(
        &mut self,
        tri: &Tri<Vertex3<A, B>>,
        scale: f32,
    ) -> &mut Self {
        self.ray(tri.centroid(), scale * tri.normal().to())
    }
}

//
// Trait impls
//

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

impl<'a, B> FragmentShader<Color4f, &'a ProjMat3<B>> for Shader {
    fn shade_fragment(
        &self,
        f: Frag<Color4f>,
        _: &'a ProjMat3<B>,
    ) -> Option<Color4> {
        Some(f.var.to_color4())
    }
}
