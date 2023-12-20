//! Triangle meshes.

use alloc::vec::Vec;
use core::fmt::Debug;

use crate::math::space::Real;
use crate::math::Vec3;
use crate::render::Model;

use super::Tri;

/// Convenience type alias for a mesh vertex.
pub type Vertex<A, Sp> = super::Vertex<Vec3<Sp>, A>;

/// A triangle mesh.
///
/// An object made of flat polygonal faces that typically form a contiguous
/// surface without holes or boundaries, so that every face shares each of its
/// edges with another face. By using many faces, complex curved shapes can be
/// approximated.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<Attrib, Space = Real<3, Model>> {
    /// The faces of the mesh, with each face a triplet of indices
    /// to the `verts` vector.
    pub faces: Vec<Tri<usize>>,
    /// The vertices of the mesh.
    pub verts: Vec<Vertex<Attrib, Space>>,
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
    /// .map(|v| vertex(v, ())).to_vec();
    ///
    /// let faces = vec![
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
    pub fn new(faces: Vec<Tri<usize>>, verts: Vec<Vertex<A, S>>) -> Self {
        let oob = faces
            .iter()
            .enumerate()
            .find(|(_, f)| f.0.iter().any(|&i| i >= verts.len()));
        if let Some((i, face)) = oob {
            panic!("vertex index out of bounds at faces[{i}]: {face:?}");
        }
        Self { faces, verts }
    }
}
