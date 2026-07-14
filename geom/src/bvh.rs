//! Bounding volume hierarchy.
use alloc::{boxed::Box, vec::Vec};

use retrofire_core::{
    geom::Mesh,
    render::{BBox, Model},
};

#[derive(Clone, Debug)]
pub struct Bvh {
    pub root: Node,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub bbox: BBox<Model>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,

    pub face_indices: Vec<usize>,
}

impl Node {
    fn new<A: Clone>(
        geom: &Mesh<A>,
        mut face_indices: Vec<usize>,
        lev: usize,
    ) -> Node {
        /*eprintln!(
            "{:lev$}creating node level {lev} from indices {face_indices:?}...",
            "",
        );*/

        let bbox: BBox<_> = face_indices
            .iter()
            .flat_map(|&fi| geom.faces[fi].0)
            .map(|vi| &geom.verts[vi].pos)
            .collect();

        /*eprintln!(
            "{:lev$}  bbox = {bbox:.1?}",
            "",
            lev = lev + 1,
            bbox = (bbox.0.0, bbox.1.0),
        );*/

        let cut = 64;

        let n_faces = face_indices.len();
        if n_faces < cut {
            /*eprintln!(
                "{:lev$}  n_faces < {cut} -> leaf node",
                "",
                lev = lev + 1
            );*/
            return Node {
                bbox,
                left: None,
                right: None,
                face_indices,
            };
        }

        let ext = bbox.1 - bbox.0;
        let split_i = if ext.x() >= ext.y() && ext.x() >= ext.z() {
            0
        } else if ext.y() >= ext.z() {
            1
        } else {
            2
        };

        /*eprintln!(
            "{:lev$}n_faces >= {cut} -> inner node, splitting along {split_i} (exts={exts:.1?})",
            "",
            lev = lev + 1,
            split_i = ('x' as u8 + split_i as u8) as char,
            exts = ext.0,
        );*/

        let left_len = face_indices
            .select_nth_unstable_by(n_faces / 2, |&i, &j| {
                let a = geom.faces[i]
                    .map(|k| geom.verts[k].clone())
                    .centroid();
                let b = geom.faces[j]
                    .map(|k| geom.verts[k].clone())
                    .centroid();

                a[split_i].total_cmp(&b[split_i])
            })
            .0
            .len();

        let (left, right) =
            (&face_indices[..left_len], &face_indices[left_len..]);

        let left: Option<Box<Node>> = (!left.is_empty())
            .then(|| Node::new(geom, left.to_vec(), lev + 1).into());
        let right: Option<Box<Node>> = (!right.is_empty())
            .then(|| Node::new(geom, right.to_vec(), lev + 1).into());

        if let (Some(l), Some(r)) = (&left, &right) {
            assert_eq!(bbox.merge(&l.bbox), bbox);
            assert_eq!(bbox.intersect(&l.bbox), l.bbox);
            assert_eq!(bbox.merge(&r.bbox), bbox);
            assert_eq!(bbox.intersect(&r.bbox), r.bbox);

            let overlap = l.bbox.intersect(&r.bbox);

            /*let [x, y, z] = bbox.dims().0;
            let vol_b = x * y * z;
            let [x, y, z] = overlap.dims().0;
            let vol_o = x * y * z;
            let vol = vol_o / vol_b;
            eprintln!(
                "{:lev$}  left-right overlap: {overlap:.1?} (volume = {vol:.3})",
                "",
                lev = lev + 1,
                overlap = (overlap.0.0, overlap.1.0),
            )*/
        }

        Self {
            bbox,
            left,
            right,
            face_indices: Vec::new(),
        }
    }
}

impl Bvh {
    pub fn new<A: Clone>(geom: &Mesh<A>) -> Self {
        let face_indices = (0..geom.faces.len()).collect::<Vec<_>>();
        Self {
            root: Node::new(geom, face_indices, 0),
        }
    }

    /*pub fn with_cutoff<>(geom: &Mesh<A>, cutoff: usize) -> Self {
        todo!()
    }*/
}

#[cfg(test)]
mod tests {
    use core::assert_matches;

    use retrofire_core::geom::{Mesh, tri, vertex};
    use retrofire_core::math::pt3;

    use crate::solids::{Build, Dodecahedron};

    use super::*;

    #[test]
    fn bvh_empty() {
        let mesh: Mesh<()> = Mesh::default();

        let bvh = Bvh::new(&mesh);

        assert!(bvh.root.bbox.is_empty());
        //assert_eq!(bvh.root.face_indices, []);
        assert!(bvh.root.left.is_none());
        assert!(bvh.root.right.is_none());
    }

    #[test]
    fn bvh_single_face() {
        let mesh: Mesh<()> = Mesh::new(
            [tri(0, 1, 2)],
            [
                vertex(pt3(0.0, 0.0, 0.0), ()),
                vertex(pt3(1.0, 0.0, 0.0), ()),
                vertex(pt3(0.0, 1.0, 0.0), ()),
            ],
        );

        let bvh = Bvh::new(&mesh);

        assert!(!bvh.root.bbox.is_empty());
        assert_eq!(bvh.root.face_indices, [0]);
        assert!(bvh.root.left.is_none());
        assert!(bvh.root.right.is_none());
    }

    #[test]
    fn bvh_two_faces() {
        let mesh: Mesh<()> = Mesh::new(
            [tri(0, 1, 2), tri(0, 2, 3)],
            [
                vertex(pt3(0.0, 0.0, 0.0), ()),
                vertex(pt3(1.0, 0.0, 0.0), ()),
                vertex(pt3(0.0, 1.0, 0.0), ()),
                vertex(pt3(0.0, 0.0, 1.0), ()),
            ],
        );

        let bvh = Bvh::new(&mesh);

        assert!(!bvh.root.bbox.is_empty());
        assert_eq!(bvh.root.face_indices, []);
        assert_matches!(
            bvh.root.left.as_deref(),
            Some(Node { face_indices: fi, .. }) if fi == &[0]
        );
        assert_matches!(
            bvh.root.right.as_deref(),
            Some(Node { face_indices: fi, .. }) if fi == &[1]
        );
    }

    #[test]
    fn bvh_dodecahedron() {
        let ico = Dodecahedron.build();

        let bvh = Bvh::new(&ico);

        assert!(!bvh.root.bbox.is_empty());
    }
}
