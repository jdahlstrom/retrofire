use alloc::vec::Vec;

use retrofire_core::{
    geom::{Mesh, Normal3, Polygon, tri, vertex},
    math::{Point2, pt3},
    render::Model,
};

use crate::solids::Build;
use crate::triangulate;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prism {
    pub points: Polygon<Point2<Model>>,
    pub capped: bool,
}

impl Build<()> for Prism {
    fn build(self) -> Mesh<()> {
        let verts: Vec<_> = self
            .points
            .0
            .iter()
            .flat_map(|pt| {
                [
                    vertex(pt3(pt.x(), 0.0, pt.y()), ()),
                    vertex(pt3(pt.x(), 1.0, pt.y()), ()),
                ]
            })
            .collect();

        let mut faces = Vec::new();

        let l = verts.len();
        for i in (0..l).step_by(2) {
            faces.push(tri(i, (i + 1) % l, (i + 3) % l));
            faces.push(tri(i, (i + 3) % l, (i + 2) % l));
        }

        let res = Mesh { faces, verts };

        if self.capped {
            let bottom = triangulate(&self.points);

            let mut top = bottom.clone();

            for v in &mut top.verts {
                v.pos[1] = 1.0;
            }
            for f in &mut top.faces {
                *f = tri(f.0[0], f.0[2], f.0[1]);
            }

            return res.merge(bottom).merge(top);
        }
        res
    }
}

impl Build<Normal3> for Prism {
    fn build(self) -> Mesh<Normal3> {
        Build::<()>::builder(self)
            .with_vertex_normals()
            .build()
    }
}
