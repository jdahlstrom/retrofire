//! Triangle meshes.

use alloc::{vec, vec::Vec};
use core::{
    fmt::{Debug, Formatter},
    iter::zip,
};

use super::{Normal3, Tri, Vertex, Vertex3, tri, vertex};
use crate::math::{Apply, Linear};
use crate::render::Model;

/// A triangle mesh.
///
/// An object made of flat polygonal faces that typically form a contiguous
/// surface without holes or boundaries, so that every face shares each of its
/// edges with another face. For instance, a cube can be represented by a mesh
/// with 8 vertices and 12 faces. By using many faces, complex curved shapes
/// can be approximated.
#[derive(Clone)]
pub struct Mesh<Vtx> {
    /// The faces of the mesh, with each face a triplet of indices
    /// to the `verts` vector. Several faces can share a vertex.
    pub faces: Vec<Tri<usize>>,
    /// The vertices of the mesh.
    pub verts: Vec<Vtx>,
}

/// A builder type for creating meshes.
#[derive(Clone)]
pub struct Builder<Vtx> {
    pub mesh: Mesh<Vtx>,
}

/// A three-dimensional mesh.
pub type Mesh3<A, B = Model> = Mesh<Vertex3<A, B>>;

//
// Inherent impls
//

impl<V> Mesh<V> {
    /// Creates a new triangle mesh with the given faces and vertices.
    ///
    /// Each face in `faces` is a triplet of indices, referring to
    /// the vertices in `verts` that define that face.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Mesh, Vertex3, tri, vertex};
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
    /// let tetra: Mesh3<()> = Mesh::new(faces, verts);
    /// ```
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    pub fn new<Fs, Vs>(faces: Fs, verts: Vs) -> Self
    where
        Fs: IntoIterator<Item = Tri<usize>>,
        Vs: IntoIterator<Item = V>,
    {
        let faces: Vec<_> = faces.into_iter().collect();
        let verts: Vec<_> = verts.into_iter().collect();

        assert_indices_in_bounds(&faces, verts.len());
        Self { faces, verts }
    }

    /// Returns a new empty mesh builder.
    pub fn builder() -> Builder<V> {
        Builder::default()
    }

    /// Returns an iterator over the faces of `self`, mapping the vertex indices
    /// to references to the corresponding vertices.
    pub fn faces(&self) -> impl Iterator<Item = Tri<&V>> {
        self.faces
            .iter()
            .map(|tri| tri.map(|i| &self.verts[i]))
    }

    /// Returns a mesh with the faces and vertices of both `self` and `other`.
    #[must_use]
    pub fn merge(mut self, Self { faces, verts }: Self) -> Self {
        let n = self.verts.len();
        self.verts.extend(verts);
        self.faces
            .extend(faces.into_iter().map(|tri| tri.map(|i| i + n)));
        self
    }

    /// Consumes `self` and returns a mesh builder with the faces and vertices
    ///  of `self`.
    pub fn into_builder(self) -> Builder<V> {
        Builder { mesh: self }
    }
}

#[inline(never)]
fn assert_indices_in_bounds(faces: &[Tri<usize>], len: usize) {
    for (Tri(vs), i) in zip(faces, 0..) {
        assert!(
            vs.iter().all(|&j| j < len),
            "vertex index out of bounds at faces[{i}]: {vs:?}"
        );
    }
}

impl<P, A> Builder<Vertex<P, A>> {
    /// Appends a face with the given vertex indices.
    ///
    /// Invalid indices (referring to vertices not yet added) are permitted,
    /// as long as all indices are valid when the [`build`][Builder::build]
    /// method is called.
    pub fn push_face(&mut self, a: usize, b: usize, c: usize) {
        self.mesh.faces.push(tri(a, b, c));
    }

    /// Appends all the faces yielded by the given iterator.
    ///
    /// The faces may include invalid vertex indices (referring to vertices
    ///  not yet added) are permitted, as long as all indices are valid when
    /// the [`build`][Builder::build] method is called.
    pub fn push_faces<Fs>(&mut self, faces: Fs)
    where
        Fs: IntoIterator<Item = [usize; 3]>,
    {
        self.mesh.faces.extend(faces.into_iter().map(Tri));
    }

    /// Appends a vertex with the given position and attribute.
    pub fn push_vert(&mut self, pos: P, attrib: A) {
        self.mesh.verts.push(vertex(pos, attrib));
    }

    /// Appends all the vertices yielded by the given iterator.
    pub fn push_verts<Vs>(&mut self, verts: Vs)
    where
        Vs: IntoIterator<Item = (P, A)>,
    {
        let vs = verts.into_iter().map(|(p, a)| vertex(p, a));
        self.mesh.verts.extend(vs);
    }

    /// Returns the finished mesh containing all the added faces and vertices.
    ///
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    #[must_use]
    pub fn build(self) -> Mesh<Vertex<P, A>> {
        // Sanity checks done by new()
        Mesh::new(self.mesh.faces, self.mesh.verts)
    }
}

impl<P, A> Builder<Vertex<P, A>> {
    /// Applies the given transform to the position of each vertex.
    ///
    /// This is an eager operation, that is, only vertices *currently*
    /// added to the builder are transformed.
    #[must_use]
    pub fn transform<T: Apply<P>>(
        self,
        tf: &T,
    ) -> Builder<Vertex<T::Output, A>> {
        self.warp(|v| vertex(tf.apply(&v.pos), v.attrib))
    }

    /// Applies an arbitrary mapping to each vertex.
    ///
    /// This method can be used for various nonlinear transformations such as
    /// twisting or dilation. This is an eager operation, that is, only vertices
    /// *currently* added to the builder are transformed.
    #[must_use]
    pub fn warp<Q>(
        self,
        f: impl FnMut(Vertex<P, A>) -> Vertex<Q, A>,
    ) -> Builder<Vertex<Q, A>> {
        Builder {
            mesh: Mesh {
                faces: self.mesh.faces,
                verts: self.mesh.verts.into_iter().map(f).collect(),
            },
        }
    }
}

impl<A> Builder<Vertex3<A>> {
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
    pub fn with_vertex_normals(self) -> Builder<Vertex3<Normal3>> {
        let Mesh { verts, faces } = self.mesh;

        // Compute weighted face normals...
        let face_normals = faces.iter().map(|tri| {
            // TODO If n-gonal faces are supported some day, the cross
            //      product is not proportional to area anymore
            let [u, v] = tri.map(|i| verts[i].pos).tangents();
            u.cross(&v)
        });
        // ...initialize vertex normals to zero...
        let mut verts: Vec<_> = verts
            .iter()
            .map(|v| vertex(v.pos, Normal3::zero()))
            .collect();
        // ...accumulate normals...
        for (&Tri(vs), n) in zip(&faces, face_normals) {
            for i in vs {
                verts[i].attrib += n.to();
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
// Foreign trait impls
//

impl<V: Debug> Debug for Mesh<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mesh")
            .field("faces", &self.faces)
            .field("verts", &self.verts)
            .finish()
    }
}

impl<V: Debug> Debug for Builder<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Builder")
            .field("faces", &self.mesh.faces)
            .field("verts", &self.mesh.verts)
            .finish()
    }
}

// Manual impls to avoid needless V: Default bound

impl<V> Default for Mesh<V> {
    /// Returns an empty mesh.
    fn default() -> Self {
        Self { faces: vec![], verts: vec![] }
    }
}

impl<V> Default for Builder<V> {
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
        let _: Mesh3<()> = Mesh::new(
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
        let mut b = Mesh::<Vertex3<_>>::builder();
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
