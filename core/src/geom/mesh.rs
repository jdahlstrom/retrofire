//! Triangle meshes.

use super::{Normal3, Tri, Vertex3, tri, vertex};
use crate::{
    math::{Linear, Mat4, Point3},
    render::Model,
};
use alloc::{vec, vec::Vec};
use core::{
    fmt::{Debug, Display, Formatter},
    iter::zip,
};
use std::ops;

pub trait Index: Copy + Clone + Debug + Display + Eq + PartialEq {
    fn as_usize(self) -> usize;
    fn add(self, i: usize) -> Self;

    fn index<I: ops::Index<usize>>(self, slice: &I) -> &I::Output {
        &slice[self.as_usize()]
    }
}

/// A triangle mesh.
///
/// An object made of flat polygonal faces that typically form a contiguous
/// surface without holes or boundaries, so that every face shares each of its
/// edges with another face. For instance, a cube can be represented by a mesh
/// with 8 vertices and 12 faces. By using many faces, complex curved shapes
/// can be approximated.
#[derive(Clone)]
pub struct Mesh<Attrib, Index = u32, Basis = Model> {
    /// The faces of the mesh, with each face a triplet of indices
    /// to the `verts` vector. Several faces can share a vertex.
    pub faces: Vec<Tri<Index>>,
    /// The vertices of the mesh.
    pub verts: Vec<Vertex3<Attrib, Basis>>,
}

/// A builder type for creating meshes.
#[derive(Clone)]
pub struct Builder<Attrib = (), Index = u32, Basis = Model> {
    pub mesh: Mesh<Attrib, Index, Basis>,
}

//
// Inherent impls
//

impl<A, I: Index, B> Mesh<A, I, B> {
    /// Creates a new triangle mesh with the given faces and vertices.
    ///
    /// Each face in `faces` is a triplet of indices, referring to
    /// the vertices in `verts` that define that face.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Mesh, tri, vertex};
    /// use retrofire_core::math::pt3;
    ///
    /// let verts = [
    ///     pt3(0.0, 0.0, 0.0),
    ///     pt3(1.0, 0.0, 0.0),
    ///     pt3(0.0, 1.0, 0.0),
    ///     pt3(0.0, 0.0, 1.0)
    /// ]
    /// .map(|v| vertex(v, ()));
    ///
    /// let faces = [
    ///     // Indices point to the verts array
    ///     tri(0, 1, 2),
    ///     tri(0, 1, 3),
    ///     tri(0, 2, 3),
    ///     tri(1, 2, 3)
    /// ];
    ///
    /// // Create a mesh with a tetrahedral shape
    /// let tetra: Mesh<()> = Mesh::new(faces, verts);
    /// ```
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    pub fn new<F, V>(faces: F, verts: V) -> Self
    where
        F: IntoIterator<Item = Tri<I>>,
        V: IntoIterator<Item = Vertex3<A, B>>,
    {
        let faces: Vec<_> = faces.into_iter().collect();
        let verts: Vec<_> = verts.into_iter().collect();

        assert_indices_in_bounds(&faces, verts.len());
        Self { faces, verts }
    }

    /// Returns an iterator over the faces of `self`, mapping the vertex indices
    /// to references to the corresponding vertices.
    pub fn faces(&self) -> impl Iterator<Item = Tri<&Vertex3<A, B>>> {
        self.faces
            .iter()
            .map(|tri| tri.map(|i| i.index(&self.verts)))
    }

    /// Returns a mesh with the faces and vertices of both `self` and `other`.
    #[must_use]
    pub fn merge(mut self, Self { faces, verts }: Self) -> Self {
        let n = self.verts.len();
        self.verts.extend(verts);
        self.faces.extend(faces.into_iter().map(|tri| {
            // TODO better overflow handling
            tri.map(|i| i.add(n))
        }));
        self
    }
}

#[inline(never)]
fn assert_indices_in_bounds<I: Index>(faces: &[Tri<I>], len: usize) {
    for (Tri(vs), i) in zip(faces, 0..) {
        assert!(
            vs.iter().all(|&j| j.as_usize() < len),
            "vertex index out of bounds at faces[{i}]: {vs:?}"
        );
    }
}

impl<A, I: Index> Mesh<A, I> {
    /// Returns a new mesh builder.
    pub fn builder() -> Builder<A, I> {
        Builder::default()
    }

    /// Consumes `self` and returns a mesh builder with the faces and vertices
    ///  of `self`.
    pub fn into_builder(self) -> Builder<A, I> {
        Builder { mesh: self }
    }
}

impl<A, I: Index> Builder<A, I> {
    /// Appends a face with the given vertex indices.
    ///
    /// Invalid indices (referring to vertices not yet added) are permitted,
    /// as long as all indices are valid when the [`build`][Builder::build]
    /// method is called.
    pub fn push_face(&mut self, a: I, b: I, c: I) {
        self.mesh.faces.push(tri(a, b, c));
    }

    /// Appends all the faces yielded by the given iterator.
    ///
    /// The faces may include invalid vertex indices (referring to vertices
    ///  not yet added) are permitted, as long as all indices are valid when
    /// the [`build`][Builder::build] method is called.
    pub fn push_faces<Fs>(&mut self, faces: Fs)
    where
        Fs: IntoIterator<Item = [I; 3]>,
    {
        self.mesh.faces.extend(faces.into_iter().map(Tri));
    }

    /// Appends a vertex with the given position and attribute.
    pub fn push_vert(&mut self, pos: Point3, attrib: A) {
        self.mesh.verts.push(vertex(pos.to(), attrib));
    }

    /// Appends all the vertices yielded by the given iterator.
    pub fn push_verts<Vs>(&mut self, verts: Vs)
    where
        Vs: IntoIterator<Item = (Point3, A)>,
    {
        let vs = verts.into_iter().map(|(p, a)| vertex(p.to(), a));
        self.mesh.verts.extend(vs);
    }

    /// Returns the finished mesh containing all the added faces and vertices.
    ///
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    #[must_use]
    pub fn build(self) -> Mesh<A, I> {
        // Sanity checks done by new()
        Mesh::new(self.mesh.faces, self.mesh.verts)
    }
}

impl<A, I: Index> Builder<A, I> {
    /// Applies the given transform to the position of each vertex.
    ///
    /// This is an eager operation, that is, only vertices *currently*
    /// added to the builder are transformed.
    #[must_use]
    pub fn transform(self, tf: &Mat4<Model, Model>) -> Self {
        self.warp(|v| vertex(tf.apply(&v.pos), v.attrib))
    }

    /// Applies an arbitrary mapping to each vertex.
    ///
    /// This method can be used for various nonlinear transformations such as
    /// twisting or dilation. This is an eager operation, that is, only vertices
    /// *currently* added to the builder are transformed.
    #[must_use]
    pub fn warp(mut self, f: impl FnMut(Vertex3<A>) -> Vertex3<A>) -> Self {
        self.mesh.verts = self.mesh.verts.into_iter().map(f).collect();
        self
    }

    /// Computes a vertex normal for each vertex as an area-weighted average
    /// of normals of the faces adjacent to it.
    ///
    /// The algorithm is as follows:
    /// 1. Initialize the normal of each vertex to **0**
    /// 1. For each face:
    ///     1. Take the cross product of two of the face's edge vectors
    ///     2. Add the result to the normal of each of the face's vertices.
    /// 3. Normalize each vertex normal to unit length.
    ///
    /// This is an eager operation, that is, only vertices *currently* added
    /// to the builder are transformed. The attribute type of the result is
    /// `Normal3`; the vertex type it accepts is changed accordingly.
    #[must_use]
    pub fn with_vertex_normals(self) -> Builder<Normal3, I> {
        let Mesh { verts, faces } = self.mesh;

        // Compute weighted face normals...
        let face_normals = faces.iter().map(|tri| {
            // TODO If n-gonal faces are supported some day, the cross
            //      product is not proportional to area anymore
            let [a, b, c] = tri.map(|i| verts[i.as_usize()].pos).0;
            (b - a).cross(&(c - a)).to()
        });
        // ...initialize vertex normals to zero...
        let mut verts: Vec<_> = verts
            .iter()
            .map(|v| vertex(v.pos, Normal3::zero()))
            .collect();
        // ...accumulate normals...
        for (&Tri(vs), n) in zip(&faces, face_normals) {
            for i in vs {
                verts[i.as_usize()].attrib += n;
            }
        }
        // ...and normalize to unit length.
        for v in &mut verts {
            v.attrib = v.attrib.normalize();
        }

        // No need to sanity check again
        Mesh { faces, verts }.into_builder()
    }
}

//
// Local trait impls
//

impl Index for u8 {
    fn as_usize(self) -> usize {
        self as usize
    }
    fn add(self, i: usize) -> Self {
        self + i as Self
    }
}

impl Index for u16 {
    fn as_usize(self) -> usize {
        self as usize
    }
    fn add(self, i: usize) -> Self {
        self + i as Self
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl Index for u32 {
    fn as_usize(self) -> usize {
        self as usize
    }
    fn add(self, i: usize) -> Self {
        self + i as Self
    }
}

#[cfg(target_pointer_width = "64")]
impl Index for u64 {
    fn as_usize(self) -> usize {
        self as usize
    }
    fn add(self, i: usize) -> Self {
        self + i as Self
    }
}

impl Index for usize {
    fn as_usize(self) -> usize {
        self
    }
    fn add(self, i: usize) -> Self {
        self + i as Self
    }
}

//
// Foreign trait impls
//

impl<A: Debug, I: Debug, S: Debug + Default> Debug for Mesh<A, I, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mesh")
            .field("faces", &self.faces)
            .field("verts", &self.verts)
            .finish()
    }
}

impl<A: Debug, I: Debug, S: Debug + Default> Debug for Builder<A, I, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Builder")
            .field("faces", &self.mesh.faces)
            .field("verts", &self.mesh.verts)
            .finish()
    }
}

impl<A, I, B> Default for Mesh<A, I, B> {
    /// Returns an empty mesh.
    fn default() -> Self {
        Self { faces: vec![], verts: vec![] }
    }
}

impl<A, I, B> Default for Builder<A, I, B> {
    /// Returns an empty builder.
    fn default() -> Self {
        Self { mesh: Mesh::default() }
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_1_SQRT_2;

    use crate::{
        geom::vertex,
        math::{SQRT_3, pt3, splat, vec3},
    };

    use super::*;

    #[test]
    #[should_panic]
    fn mesh_new_panics_if_vertex_index_oob() {
        let _: Mesh<()> = Mesh::new(
            [tri(0, 1, 2), tri(1, 2, 3)],
            [
                vertex(pt3(0.0, 0.0, 0.0), ()),
                vertex(pt3(1.0, 1.0, 1.0), ()),
                vertex(pt3(2.0, 2.0, 2.0), ()),
            ],
        );
    }

    #[test]
    #[should_panic]
    fn mesh_builder_panics_if_vertex_index_oob() {
        let mut b = Mesh::builder();
        b.push_faces([[0, 1, 2], [1, 2, 3]]);
        b.push_verts([
            (pt3(0.0, 0.0, 0.0), ()),
            (pt3(1.0, 1.0, 1.0), ()),
            (pt3(2.0, 2.0, 2.0), ()),
        ]);
        _ = b.build();
    }

    #[test]
    fn vertex_normal_generation() {
        // TODO Doesn't test weighting by area

        let mut b = Mesh::builder();
        b.push_faces([[0, 2, 1], [0, 1, 3], [0, 3, 2]]);
        b.push_verts([
            (pt3(0.0, 0.0, 0.0), ()),
            (pt3(1.0, 0.0, 0.0), ()),
            (pt3(0.0, 1.0, 0.0), ()),
            (pt3(0.0, 0.0, 1.0), ()),
        ]);
        let b = b.with_vertex_normals();

        let expected = [
            splat(-1.0 / SQRT_3),
            vec3(0.0, -FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            vec3(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2),
            vec3(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
        ];

        for (a, e) in zip(b.mesh.verts, expected) {
            crate::assert_approx_eq!(a.attrib, e);
        }
    }
}
