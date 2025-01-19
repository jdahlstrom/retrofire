//! Subdivision surfaces.

use alloc::collections::BTreeMap;

use re::geom::{mesh::Builder, Mesh, Normal3};
use re::math::Lerp;

use super::Octahedron;

/// Spherical mesh created by recursively subdividing an octahedron.
pub struct Octosphere {
    pub radius: f32,
    pub depth: u32,
}

impl Octosphere {
    pub fn build(self) -> Mesh<Normal3> {
        let coords = Octahedron::COORDS;
        #[rustfmt::skip]
        let faces = [
            [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
            [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1],
        ];
        let mut bld = Mesh::builder();
        bld.push_verts(coords.map(|c| (c, c.to_vec())));
        subdivide(&faces, &mut bld, &mut BTreeMap::new(), self.depth);
        bld.build()
    }
}

/// Recursively subdivides the given triangle faces.
///
/// The vertex indices in `faces` must be valid indices into `bld.mesh.verts`.
/// Subdivides a triangle `into four smaller triangles by creating a new vertex
/// at the midpoint of each edge.
///
/// ```text
///             a
///            /\
///          /   \
///     ab /______\ ac
///      / \      /\
///    /    \   /   \
///  /_______\/______\
/// b        bc       c
/// ```
fn subdivide(
    faces: &[[usize; 3]],
    bld: &mut Builder<Normal3>,
    // HashMap not available in `alloc` :(
    cache: &mut BTreeMap<[usize; 2], usize>,
    depth: u32,
) {
    if depth == 0 {
        bld.push_faces(faces.iter().copied());
    } else {
        for &[i, j, k] in faces {
            let mut get = |i, j| {
                *cache.entry([i, j]).or_insert_with(|| {
                    let a = bld.mesh.verts[i];
                    let b = bld.mesh.verts[j];
                    let ab = a.midpoint(&b);
                    bld.push_vert(
                        ab.pos.to_vec().normalize().to_pt().to(),
                        ab.attrib.normalize(),
                    )
                })
            };
            let [ij, ik, jk] = [get(i, j), get(i, k), get(j, k)];
            let new_faces =
                [[i, ij, ik], [j, jk, ij], [k, ik, jk], [ij, jk, ik]];

            subdivide(&new_faces, bld, cache, depth - 1);
        }
        cache.clear();
    }
}
