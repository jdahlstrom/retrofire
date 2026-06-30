//! Triangle meshes.

use alloc::vec::Vec;
use core::{
    fmt::{Debug, Formatter, from_fn},
    iter::zip,
    slice::ChunksExact,
};

use crate::{
    math::{Linear, Mat4, Point3},
    render::Model,
};

use super::{Normal3, Tri, Vertex3, tri, vertex};

/// A container of vertex index triplets packed to a size determined at runtime.
///
/// For example, if a mesh contains fewer than 256 vertices, a container of
/// this type can be set to encode each index in a byte, using one eighth the
/// space that a naive vector of `Tri<usize>` would use.
#[derive(Clone)]
pub struct TriIndices {
    index_type: Index,
    indices: Vec<u8>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Index {
    U8 = size_of::<u8>() as u8,
    U16 = size_of::<u16>() as u8,

    #[cfg(not(target_pointer_width = "16"))]
    U32 = size_of::<u32>() as u8,

    #[cfg(not(any(target_pointer_width = "16", target_pointer_width = "32")))]
    U64 = size_of::<u64>() as u8,

    Usize = 0,
}

impl Index {
    /// Returns the smallest index type that can fit the given value.
    pub fn from_max(max: usize) -> Self {
        use Index::*;
        if max <= U8.max() {
            U8
        } else if max <= U16.max() {
            U16
        } else if max <= U32.max() {
            U32
        } else if max <= U64.max() {
            U64
        } else {
            Usize // unreachable in practice
        }
    }

    /// Returns the maximum value of an index of this type.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::mesh::Index;
    ///
    /// assert_eq!(Index::U16.max(), 65535);
    /// ```
    #[inline]
    pub const fn max(self) -> usize {
        match self {
            Index::Usize => usize::MAX,
            _ => !0 >> 8 * (8 - self as u8),
        }
    }

    /// Returns the size in bytes of an index of this type.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::mesh::Index;
    ///
    /// assert_eq!(Index::U32.size(), 4);
    /// ```
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            Index::Usize => size_of::<usize>(),
            _ => self as u8 as usize,
        }
    }
}

impl TriIndices {
    pub const fn new(index_type: Index) -> Self {
        Self {
            index_type,
            indices: Vec::new(),
        }
    }

    /// Creates a `TriIndices` able to contain indices equal to or greater
    /// than `max`, allocating space for at least `cap` triangles.
    pub fn with_max_and_capacity(
        max_index: usize,
        triangle_cap: usize,
    ) -> Self {
        let index_type = Index::from_max(max_index);
        let byte_cap = 3 * index_type.size() * triangle_cap;
        let indices = Vec::with_capacity(byte_cap);
        Self { index_type, indices }
    }

    /// Inserts an index triple into `self`.
    ///
    /// # Panics
    /// If any of the indices is outside the range of the index type of `self`.
    pub fn push(&mut self, Tri([i, j, k]): Tri<usize>) {
        let max = self.index_type.max();
        assert!(i <= max && j <= max && k <= max);
        match self.index_type {
            Index::U8 => {
                self.indices.extend([i as u8, j as u8, k as u8]);
            }
            Index::U16 => {
                let bytes = [i, j, k].map(|i| (i as u16).to_le_bytes());
                self.indices.extend(bytes.as_flattened());
            }
            #[cfg(not(target_pointer_width = "16"))]
            Index::U32 => {
                let bytes = [i, j, k].map(|i| (i as u32).to_le_bytes());
                self.indices.extend(bytes.as_flattened());
            }
            #[cfg(not(any(
                target_pointer_width = "16",
                target_pointer_width = "32"
            )))]
            Index::U64 => {
                let bytes = [i, j, k].map(|i| (i as u64).to_le_bytes());
                self.indices.extend(bytes.as_flattened());
            }
            Index::Usize => {
                let bytes = [i, j, k].map(|i| i.to_le_bytes());
                self.indices.extend(bytes.as_flattened());
            }
        }
    }

    #[inline]
    pub fn extend<I>(&mut self, tris: I)
    where
        I: IntoIterator<Item = Tri<usize>>,
    {
        for tri in tris {
            self.push(tri)
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.indices.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.iter().len()
    }

    #[inline]
    pub fn iter(&self) -> TriIndicesIter<'_> {
        TriIndicesIter {
            index_size: self.index_type.size(),
            chunks: self.indices.chunks_exact(self.bytes_per_tri()),
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        self.indices.len()
    }

    pub fn bytes_per_tri(&self) -> usize {
        3 * self.index_type.size()
    }
}

pub struct TriIndicesIter<'a> {
    index_size: usize,
    chunks: ChunksExact<'a, u8>,
}

impl Iterator for TriIndicesIter<'_> {
    type Item = Tri<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.next()?;
        let mut le_bytes = [[0u8; size_of::<usize>()]; 3];
        let mut ics = chunk.chunks_exact(self.index_size);
        for index in &mut le_bytes {
            index[0..self.index_size].copy_from_slice(ics.next().unwrap());
        }
        Some(Tri(le_bytes.map(|i| usize::from_le_bytes(i))))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.chunks.len();
        (len, Some(len))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl ExactSizeIterator for TriIndicesIter<'_> {}

/*
impl IntoIterator for TriIndices {
    type Item = Tri<usize>;
    type IntoIter = TriIndicesIter<'static>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}*/

impl<'a> IntoIterator for &'a TriIndices {
    type Item = Tri<usize>;
    type IntoIter = TriIndicesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Debug for TriIndices {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let indices = from_fn(|f| f.debug_list().entries(self.iter()).finish());

        f.debug_struct("TriIndices")
            .field("size", &self.index_type)
            .field("indices", &indices)
            .finish()
    }
}

#[cfg(test)]
mod tri_indices_tests {
    use std::format;

    use super::*;

    #[test]
    fn index_max() {
        assert_eq!(Index::U8.max(), u8::MAX as usize);
        assert_eq!(Index::U16.max(), u16::MAX as usize);
        assert_eq!(Index::U32.max(), u32::MAX as usize);
        assert_eq!(Index::U64.max(), u64::MAX as usize);
    }

    #[test]
    fn tri_indices_debug() {
        let mut ics = TriIndices::new(Index::U16);

        ics.push(tri(0, 1, 2));
        ics.push(tri(100, 16000, 65535));

        let dbg = format!("{:?}", ics);
        assert_eq!(
            dbg,
            "TriIndices { \
                size: U16, \
                indices: [Tri([0, 1, 2]), Tri([100, 16000, 65535])] \
            }"
        );
    }

    #[test]
    fn tri_indices_u8() {
        let mut ics = TriIndices::new(Index::U8);

        ics.push(tri(0, 1, 2));
        ics.push(tri(100, 200, 255));

        assert_eq!(ics.indices, [0, 1, 2, 100, 200, 255]);

        let mut iter = ics.iter();

        assert_eq!(iter.next(), Some(tri(0, 1, 2)));
        assert_eq!(iter.next(), Some(tri(100, 200, 255)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[rustfmt::skip]
    fn tri_indices_u16() {
        let mut ics = TriIndices::new(Index::U16);

        ics.push(tri(0, 1, 2));
        ics.push(tri(0xFF, 0x20_30, 0xFF_FF));

        assert!(matches!(
            ics.indices.as_chunks(), ([
                [0x00, 0x00],
                [0x01, 0x00],
                [0x02, 0x00],
                [0xFF, 0x00],
                [0x30, 0x20],
                [0xFF, 0xFF],
            ], [])
        ));

        let mut iter = ics.iter();

        assert_eq!(iter.next(), Some(tri(0, 1, 2)));
        assert_eq!(iter.next(), Some(tri(0xFF, 0x20_30, 0xFF_FF)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[rustfmt::skip]
    fn tri_indices_u32() {
        let mut ics = TriIndices::new(Index::U32);

        ics.push(tri(0, 1, 2));
        ics.push(tri(0xFF, 0x20_30_40, 0xFF_FF_FF_FF));

        assert!(matches!(
            ics.indices.as_chunks(), ([
                [0x00, 0x00, 0x00, 0x00],
                [0x01, 0x00, 0x00, 0x00],
                [0x02, 0x00, 0x00, 0x00],
                [0xFF, 0x00, 0x00, 0x00],
                [0x40, 0x30, 0x20, 0x00],
                [0xFF, 0xFF, 0xFF, 0xFF],
            ], [])
        ));

        let mut iter = ics.iter();

        assert_eq!(iter.next(), Some(tri(0, 1, 2)));
        assert_eq!(iter.next(), Some(tri(0xFF, 0x20_30_40, 0xFF_FF_ff_FF)));
        assert_eq!(iter.next(), None);
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
pub struct Mesh<Attrib, Basis = Model> {
    /// The faces of the mesh, with each face a triplet of indices
    /// to the `verts` vector. Several faces can share a vertex.
    pub faces: TriIndices,
    /// The vertices of the mesh.
    pub verts: Vec<Vertex3<Attrib, Basis>>,
}

/// A builder type for creating meshes.
#[derive(Clone)]
pub struct Builder<Attrib = (), Basis = Model> {
    pub mesh: Mesh<Attrib, Basis>,
}

//
// Inherent impls
//

impl<A, B> Mesh<A, B> {
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
        F: IntoIterator<Item = Tri<usize>>,
        V: IntoIterator<Item = Vertex3<A, B>>,
    {
        let faces_: Vec<_> = faces.into_iter().collect();
        let verts: Vec<_> = verts.into_iter().collect();

        assert_indices_in_bounds(&faces_, verts.len());

        let mut faces =
            TriIndices::with_max_and_capacity(verts.len(), faces_.len());
        faces.extend(faces_);

        Self { faces, verts }
    }

    /// Returns an iterator over the faces of `self`, mapping the vertex indices
    /// to references to the corresponding vertices.
    pub fn faces(&self) -> impl Iterator<Item = Tri<&Vertex3<A, B>>> {
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

impl<A> Mesh<A> {
    /// Returns a new mesh builder.
    pub fn builder(index_type: Index) -> Builder<A> {
        Mesh {
            faces: TriIndices::new(index_type),
            ..Mesh::default()
        }
        .into_builder()
    }

    /// Consumes `self` and returns a mesh builder with the faces and vertices
    ///  of `self`.
    pub fn into_builder(self) -> Builder<A> {
        Builder { mesh: self }
    }
}

impl<A> Builder<A> {
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
    pub fn build(self) -> Mesh<A> {
        // Sanity checks done by new()
        Mesh::new(&self.mesh.faces, self.mesh.verts)
    }
}

impl<A> Builder<A> {
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
    pub fn with_vertex_normals(self) -> Builder<Normal3> {
        let Mesh { verts, faces } = self.mesh;

        // Compute weighted face normals...
        let face_normals = faces.iter().map(|tri| {
            // TODO If n-gonal faces are supported some day, the cross
            //      product is not proportional to area anymore
            let [a, b, c] = tri.map(|i| verts[i].pos).0;
            (b - a).cross(&(c - a)).to()
        });
        // ...initialize vertex normals to zero...
        let mut verts: Vec<_> = verts
            .iter()
            .map(|v| vertex(v.pos, Normal3::zero()))
            .collect();
        // ...accumulate normals...
        for (Tri(vs), n) in zip(&faces, face_normals) {
            for i in vs {
                verts[i].attrib += n;
            }
        }
        // ...and normalize to unit length.
        for v in &mut verts {
            v.attrib = v.attrib.normalize_or_zero();
        }

        // No need to sanity check again
        Mesh { faces, verts }.into_builder()
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
    /// Returns an empty mesh.
    fn default() -> Self {
        Self {
            faces: TriIndices::new(Index::U32),
            verts: Vec::new(),
        }
    }
}

impl<A> Default for Builder<A> {
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
