//! Triangle meshes.

use alloc::vec::Vec;
use core::{
    fmt::{Debug, Formatter},
    iter::zip,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TriIndices {
    U8(Vec<Tri<u8>>),
    U16(Vec<Tri<u16>>),
    U32(Vec<Tri<u32>>),
    U64(Vec<Tri<u64>>),
    Usize(Vec<Tri<usize>>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum IndexType {
    U8,
    U16,
    U32,
    U64,
    Usize,
}

impl IndexType {
    /// Returns the smallest index type that can hold the argument.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::mesh::IndexType;
    ///
    /// assert_eq!(IndexType::of(0), IndexType::U8);
    /// assert_eq!(IndexType::of(u8::MAX as usize), IndexType::U8);
    /// assert_eq!(IndexType::of(u8::MAX as usize + 1), IndexType::U16);
    /// assert_eq!(IndexType::of(u16::MAX as usize), IndexType::U16);
    /// assert_eq!(IndexType::of(u16::MAX as usize + 1), IndexType::U32);
    /// assert_eq!(IndexType::of(u32::MAX as usize), IndexType::U32);
    /// assert_eq!(IndexType::of(u32::MAX as usize + 1), IndexType::U64);
    /// assert_eq!(IndexType::of(u64::MAX as usize), IndexType::U64);
    /// ```
    pub fn of(i: usize) -> Self {
        use IndexType::*;
        let width_bytes = i.bit_width().div_ceil(8);
        match width_bytes {
            0..=1 => U8,
            2 => U16,
            3..=4 => U32,
            5..=8 => U64,
            _ => Usize,
        }
    }

    /// Returns the maximum value that can be held by this index type.
    pub fn max(self) -> usize {
        use IndexType::*;
        match self {
            U8 => u8::MAX as usize,
            U16 => u16::MAX as usize,
            U32 => u32::MAX as usize,
            U64 => u64::MAX as usize,
            Usize => usize::MAX,
        }
    }

    /// Returns the size of this index type in bytes.
    pub fn size(self) -> usize {
        use IndexType::*;
        match self {
            U8 => 1,
            U16 => 2,
            U32 => 4,
            U64 => 8,
            Usize => size_of::<usize>(),
        }
    }
}

macro_rules! mi {
    ($this:expr , $v:ident => $expr:expr) => {{
        use TriIndices::*;
        match $this {
            U8($v) => $expr,
            U16($v) => $expr,
            U32($v) => $expr,
            U64($v) => $expr,
            Usize($v) => $expr,
        }
    }};
}

impl TriIndices {
    pub fn new(ty: IndexType) -> Self {
        use IndexType::*;
        match ty {
            U8 => Self::U8(Vec::new()),
            U16 => Self::U16(Vec::new()),
            U32 => Self::U32(Vec::new()),
            U64 => Self::U64(Vec::new()),
            Usize => Self::Usize(Vec::new()),
        }
    }

    /// Returns a `TriIndices` that can hold indices of at least the given
    /// value, with the given capacity.
    pub fn with_max_and_capacity(max: usize, cap: usize) -> Self {
        let mut res = Self::new(IndexType::of(max));
        res.reserve(cap);
        res
    }

    fn index_type(&self) -> IndexType {
        use TriIndices::*;
        match self {
            U8(_) => IndexType::U8,
            U16(_) => IndexType::U16,
            U32(_) => IndexType::U32,
            U64(_) => IndexType::U64,
            Usize(_) => IndexType::Usize,
        }
    }

    pub fn expand(&mut self, new_max: usize) {
        let mut new = Self::with_max_and_capacity(new_max, self.capacity());
        new.extend(self.into_iter());
        *self = new;
    }

    /// Inserts an index triple into `self`.
    ///
    /// # Panics
    /// If any of the indices is outside the range of the index type of `self`.
    pub fn push(&mut self, Tri([i, j, k]): Tri<usize>) {
        let max = i.max(j).max(k);
        if max > self.index_type().max() {
            self.expand(max);
        }
        mi! { self, v => v.push(tri(
            i.try_into().expect("cannot fail"),
            j.try_into().expect("cannot fail"),
            k.try_into().expect("cannot fail")
        ))}
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = Tri<usize>>>(&mut self, tris: I) {
        for tri in tris {
            self.push(tri)
        }
    }

    /// Removes all entries from `self`.
    #[inline]
    pub fn clear(&mut self) {
        mi! { self, v => v.clear() }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        mi! { self, v => v.reserve(additional) }
    }

    /// Returns the number of entries (index triples) that `self` contains.
    #[inline]
    pub fn len(&self) -> usize {
        mi! { self, v => v.len() }
    }

    /// Returns the capacity of the index vector in number of index triples.
    #[inline]
    pub fn capacity(&self) -> usize {
        mi! { self, v => v.capacity() }
    }

    /// Returns true iff `self` contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        mi! { self, v => v.is_empty() }
    }

    pub fn get(&self, i: usize) -> Option<Tri<usize>> {
        mi! { self, v => v.get(i).map(|tri| tri.map(|i| i as usize)) }
    }

    /// Returns an iterator over the triangles contained by `self`.
    #[inline]
    pub fn iter(&self) -> TriIndicesIter<'_> {
        TriIndicesIter(self, 0)
    }

    pub fn size_in_bytes(&self) -> usize {
        todo!()
    }

    pub fn bytes_per_tri(&self) -> usize {
        todo!()
    }
}

pub struct TriIndicesIter<'a>(&'a TriIndices, usize);

impl<'a> Iterator for TriIndicesIter<'a> {
    type Item = Tri<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let tri = mi!(self.0, v => {
            let [i, j, k] = v.get(self.1)?.0;
            tri(i as _, j as _, k as _)
        });
        self.1 += 1;
        Some(tri)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
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

impl<'a> IntoIterator for &'a TriIndices {
    type Item = Tri<usize>;
    type IntoIter = TriIndicesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Extend<Tri<usize>> for TriIndices {
    fn extend<I: IntoIterator<Item = Tri<usize>>>(&mut self, iter: I) {
        self.extend(iter)
    }
}

impl FromIterator<Tri<u8>> for TriIndices {
    fn from_iter<I: IntoIterator<Item = Tri<u8>>>(iter: I) -> Self {
        Self::U8(iter.into_iter().collect())
    }
}
impl FromIterator<Tri<u16>> for TriIndices {
    fn from_iter<I: IntoIterator<Item = Tri<u16>>>(iter: I) -> Self {
        Self::U16(iter.into_iter().collect())
    }
}
impl FromIterator<Tri<u32>> for TriIndices {
    fn from_iter<I: IntoIterator<Item = Tri<u32>>>(iter: I) -> Self {
        Self::U32(iter.into_iter().collect())
    }
}
impl FromIterator<Tri<u64>> for TriIndices {
    fn from_iter<I: IntoIterator<Item = Tri<u64>>>(iter: I) -> Self {
        Self::U64(iter.into_iter().collect())
    }
}
impl FromIterator<Tri<usize>> for TriIndices {
    fn from_iter<I: IntoIterator<Item = Tri<usize>>>(iter: I) -> Self {
        Self::Usize(iter.into_iter().collect())
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
        let verts: Vec<_> = verts.into_iter().collect();
        let verts_len = verts.len();

        let faces_iter = faces
            .into_iter()
            .zip(0..)
            .inspect(|(Tri(ixs), i)| {
                assert!(
                    ixs.iter().all(|&j| j < verts_len),
                    "vertex index out of bounds at faces[{i}]:\
                one of {ixs:?} > {verts_len}"
                );
            })
            .map(|(tri, _)| tri);

        let faces = faces_iter.collect();

        Self { faces, verts }
    }

    /// Creates an empty mesh with the given index type.
    pub fn with_index(ty: IndexType) -> Self {
        Self {
            faces: TriIndices::new(ty),
            ..Default::default()
        }
    }

    /// Returns an iterator over the faces of `self`, mapping the vertex indices
    /// to references to the corresponding vertices.
    pub fn faces(&self) -> impl Iterator<Item = Tri<&Vertex3<A, B>>> {
        self.faces
            .iter()
            .map(|tri| tri.map(|i| &self.verts[i]))
    }

    /// Returns a mesh with the faces and vertices of another mesh appended
    /// to `self`.
    #[must_use]
    pub fn merge(mut self, Self { faces, verts }: Self) -> Self {
        let n = self.verts.len();
        self.verts.extend(verts);
        self.faces
            .extend(faces.into_iter().map(|tri| tri.map(|i| i + n)));
        self
    }
}

impl<A> Mesh<A> {
    /// Returns a new mesh builder.
    pub fn builder(ty: IndexType) -> Builder<A> {
        Mesh::with_index(ty).into_builder()
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
            faces: TriIndices::U32(Vec::new()),
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
        let mut b = Mesh::builder(IndexType::U8);
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

        let mut b = Mesh::builder(IndexType::U8);
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
