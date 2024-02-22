//! Triangle meshes.

use core::fmt::{Debug, Formatter};

use alloc::{vec, vec::Vec};

use crate::math::space::Real;
use crate::math::Vec3;
use crate::render::Model;

use super::{vertex, Tri};

/// Convenience type alias for a mesh vertex.
pub type Vertex<A, Sp> = super::Vertex<Vec3<Sp>, A>;

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

#[derive(Clone, Debug)]
pub struct Builder<Attrib = ()> {
    m: Mesh<Attrib>,
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
}

impl<A> Builder<A> {
    /// Appends a face with the given vertex indices.
    pub fn push_face(&mut self, a: usize, b: usize, c: usize) {
        self.m.faces.push(Tri([a, b, c]));
    }

    /// Appends all the faces yielded by the given iterator.
    pub fn push_faces<Fs>(&mut self, faces: Fs)
    where
        Fs: IntoIterator<Item = [usize; 3]>,
    {
        self.m.faces.extend(faces.into_iter().map(Tri));
    }

    /// Appends a vertex with the given position and attribute.
    pub fn push_vert(&mut self, pos: Vec3, attrib: A) {
        self.m.verts.push(vertex(pos.to(), attrib));
    }

    /// Appends all the vertices yielded by the given iterator.
    pub fn push_verts<Vs>(&mut self, verts: Vs)
    where
        Vs: IntoIterator<Item = (Vec3, A)>,
    {
        let vs = verts.into_iter().map(|(v, a)| vertex(v.to(), a));
        self.m.verts.extend(vs);
    }

    /// Returns the finished mesh containing all the added faces and vertices.
    ///
    /// # Panics
    /// If any of the vertex indices in `faces` ≥ `verts.len()`.
    pub fn build(self) -> Mesh<A> {
        // Sanity checks done by new()
        Mesh::new(self.m.faces, self.m.verts)
    }
}

impl<A: Debug, S: Debug + Default> Debug for Mesh<A, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mesh")
            .field("faces", &self.faces)
            .field("verts", &self.verts)
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
        Self { m: Mesh::default() }
    }
}

#[cfg(test)]
mod tests {
    use crate::geom::vertex;
    use crate::math::vec3;

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
}
