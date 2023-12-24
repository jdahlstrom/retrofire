//! Triangle meshes.

use core::{
    fmt::{Debug, Formatter},
    iter::zip,
};

use alloc::{vec, vec::Vec};

use crate::math::{
    mat::{Mat4x4, RealToReal},
    space::{Linear, Real},
    vec::Vec3,
};
use crate::render::Model;

use super::{vertex, Normal3, Tri};

/// Convenience type alias for a mesh vertex.
pub type Vertex<A, Sp = Real<3, Model>> = super::Vertex<Vec3<Sp>, A>;

/// A triangle mesh.
///
/// An object made of flat polygonal faces that typically form a contiguous
/// surface without holes or boundaries, so that every face shares each of its
/// edges with another face. By using many faces, complex curved shapes can be
/// approximated.
#[derive(Clone)]
pub struct Mesh<Attrib, Space = Real<3, Model>> {
    /// The faces of the mesh, with each face a triplet of indices
    /// to the `verts` vector. Several faces can share a vertex.
    pub faces: Vec<Tri<usize>>,
    /// The vertices of the mesh.
    pub verts: Vec<Vertex<Attrib, Space>>,
}

/// A builder type for creating meshes.
#[derive(Clone)]
pub struct Builder<Attrib = (), Space = Real<3, Model>> {
    pub mesh: Mesh<Attrib, Space>,
}

impl<A, S> Mesh<A, S> {
    /// Creates a new triangle mesh with the given faces and vertices.
    ///
    /// Each face in `faces` is a triplet of indices, referring to
    /// the vertices in `verts` that define that face.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::geom::{Tri, Mesh, vertex};
    /// # use retrofire_core::math::vec3;
    /// let verts = [
    ///     vec3(0.0, 0.0, 0.0),
    ///     vec3(1.0, 0.0, 0.0),
    ///     vec3(0.0, 1.0, 0.0),
    ///     vec3(0.0, 0.0, 1.0)
    /// ]
    /// .map(|v| vertex(v, ()));
    ///
    /// let faces = [
    ///     Tri([0, 1, 2]),
    ///     Tri([0, 1, 3]),
    ///     Tri([0, 2, 3]),
    ///     Tri([1, 2, 3])
    /// ];
    ///
    /// // Create a mesh with a tetrahedral shape
    /// let tetra = Mesh::new(faces, verts);
    /// ```
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    pub fn new<F, V>(faces: F, verts: V) -> Self
    where
        F: IntoIterator<Item = Tri<usize>>,
        V: IntoIterator<Item = Vertex<A, S>>,
    {
        let faces: Vec<_> = faces.into_iter().collect();
        let verts: Vec<_> = verts.into_iter().collect();

        for (i, Tri(vs)) in faces.iter().enumerate() {
            assert!(
                vs.iter().all(|&j| j < verts.len()),
                "vertex index out of bounds at faces[{i}]: {vs:?}"
            )
        }
        Self { faces, verts }
    }
}

impl<A> Mesh<A> {
    /// Returns a new mesh builder.
    pub fn builder() -> Builder<A> {
        Builder::default()
    }

    /// Consumes `self` and returns a mesh builder with the faces and vertices
    ///  of `self`.
    pub fn into_builder(self) -> Builder<A> {
        Builder { mesh: self }
    }
}

impl<A> Builder<A> {
    /// Appends a face with the given vertex indices.
    pub fn push_face(&mut self, a: usize, b: usize, c: usize) {
        self.mesh.faces.push(Tri([a, b, c]));
    }

    /// Appends all the faces yielded by the given iterator.
    pub fn push_faces<Fs>(&mut self, faces: Fs)
    where
        Fs: IntoIterator<Item = [usize; 3]>,
    {
        self.mesh.faces.extend(faces.into_iter().map(Tri));
    }

    /// Appends a vertex with the given position and attribute.
    pub fn push_vert(&mut self, pos: Vec3, attrib: A) {
        self.mesh.verts.push(vertex(pos.to(), attrib));
    }

    /// Appends all the vertices yielded by the given iterator.
    pub fn push_verts<Vs>(&mut self, verts: Vs)
    where
        Vs: IntoIterator<Item = (Vec3, A)>,
    {
        let vs = verts.into_iter().map(|(v, a)| vertex(v.to(), a));
        self.mesh.verts.extend(vs);
    }

    /// Returns the finished mesh containing all the added faces and vertices.
    ///
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    pub fn build(self) -> Mesh<A> {
        // Sanity checks done by new()
        Mesh::new(self.mesh.faces, self.mesh.verts)
    }
}

impl Builder<()> {
    /// Applies the given transform to the position of each vertex.
    ///
    /// This is an eager operation, that is, only vertices *currently*
    /// added to the builder are transformed.
    pub fn transform(
        self,
        tf: &Mat4x4<RealToReal<3, Model, Model>>,
    ) -> Builder<()> {
        let mesh = Mesh {
            faces: self.mesh.faces,
            verts: self
                .mesh
                .verts
                .into_iter()
                .map(|v| vertex(tf.apply(&v.pos), v.attrib))
                .collect(),
        };
        mesh.into_builder()
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
    pub fn with_vertex_normals(self) -> Builder<Normal3> {
        let Mesh { verts, faces } = self.mesh;

        // Compute weighted face normals...
        let face_normals = faces.iter().map(|Tri(vs)| {
            // TODO If n-gonal faces are supported some day,
            // the cross product is not proportional to area anymore
            let [a, b, c] = vs.map(|i| verts[i].pos);
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
                verts[i].attrib += n;
            }
        }
        // ...and normalize to unit length.
        for v in &mut verts {
            v.attrib = v.attrib.normalize();
        }

        Mesh::new(faces, verts).into_builder()
    }
}

//
// Foreign trait impls
//

impl<A: Debug, S: Debug + Default> Debug for Mesh<A, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mesh")
            .field("faces", &self.faces)
            .field("verts", &self.verts)
            .finish()
    }
}

impl<A: Debug, S: Debug + Default> Debug for Builder<A, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Builder")
            .field("faces", &self.mesh.faces)
            .field("verts", &self.mesh.verts)
            .finish()
    }
}

impl<A, S> Default for Mesh<A, S> {
    fn default() -> Self {
        Self { faces: vec![], verts: vec![] }
    }
}

impl<A> Default for Builder<A> {
    fn default() -> Self {
        Self { mesh: Mesh::default() }
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_1_SQRT_2;

    use crate::geom::vertex;
    use crate::math::vec3;
    use crate::prelude::splat;

    use super::*;

    #[test]
    #[should_panic]
    fn mesh_new_panics_if_vertex_index_oob() {
        _ = Mesh::new(
            [Tri([0, 1, 2]), Tri([1, 2, 3])],
            [
                vertex(vec3(0.0, 0.0, 0.0), ()),
                vertex(vec3(1.0, 1.0, 1.0), ()),
                vertex(vec3(2.0, 2.0, 2.0), ()),
            ],
        );
    }

    #[test]
    #[should_panic]
    fn mesh_builder_panics_if_vertex_index_oob() {
        let mut b = Mesh::builder();
        b.push_faces([[0, 1, 2], [1, 2, 3]]);
        b.push_verts([
            (vec3(0.0, 0.0, 0.0), ()),
            (vec3(1.0, 1.0, 1.0), ()),
            (vec3(2.0, 2.0, 2.0), ()),
        ]);
        _ = b.build();
    }

    #[test]
    fn vertex_normal_generation() {
        // TODO Doesn't test weighting by area

        let mut b = Mesh::builder();
        b.push_faces([[0, 2, 1], [0, 1, 3], [0, 3, 2]]);
        b.push_verts([
            (vec3(0.0, 0.0, 0.0), ()),
            (vec3(1.0, 0.0, 0.0), ()),
            (vec3(0.0, 1.0, 0.0), ()),
            (vec3(0.0, 0.0, 1.0), ()),
        ]);
        let b = b.with_vertex_normals();

        const SQRT_3: f32 = 1.7320508076;

        let expected = [
            splat(-1.0 / SQRT_3),
            vec3(0.0, -FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            vec3(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2),
            vec3(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
        ];

        assert_eq!(b.mesh.verts[0].attrib, expected[0]);
        assert_eq!(b.mesh.verts[1].attrib, expected[1]);
        assert_eq!(b.mesh.verts[2].attrib, expected[2]);
        assert_eq!(b.mesh.verts[3].attrib, expected[3]);
    }
}
