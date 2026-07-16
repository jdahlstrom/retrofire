use alloc::vec::Vec;

use retrofire_core::geom::{
    Mesh, Normal2, Normal3, Polygon, Pos, Tri, Vertex2, Vertex3, tri, vertex,
};
use retrofire_core::math::{Parametric, Point2, Vary, Vec3, param, pt3, vec3};

use crate::{solids::Build, triangulate};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prism<P> {
    pub points: P,
    pub sides: usize,
    pub capped: bool,
}

impl<P> Prism<P> {
    pub fn new<V: Pos<Type = Point2>>(sides: usize, points: P) -> Self
    where
        P: Parametric<V>,
    {
        Self { points, sides, capped: true }
    }
    pub fn capped(self, capped: bool) -> Self {
        Self { capped, ..self }
    }

    fn make_caps(
        self,
        poly: &Polygon<Point2>,
        verts: &mut Vec<Vertex3<()>>,
        faces: &mut Vec<Tri<usize>>,
    ) {
        let cap_faces = triangulate(&poly);

        let bot_start = verts.len();
        for i in 0..poly.0.len() {
            verts.push(verts[i]);
        }
        // top cap verts
        let top_start = verts.len();
        for i in poly.0.len()..2 * poly.0.len() {
            verts.push(verts[i]);
        }

        for tri in &cap_faces {
            faces.push(tri.map(|i| bot_start + i));
        }
        for Tri([i, j, k]) in cap_faces {
            // Invert winding order
            faces.push(tri(i, k, j).map(|l| top_start + l));
        }
    }
}

impl<P: Parametric<Vertex2<()>>> Build<()> for Prism<P> {
    fn build(self) -> Mesh<()> {
        let pts = param::iter(&self.points, self.sides);

        let bottom_verts = pts
            .clone()
            .map(|pt| vertex(pt3(pt.pos().x(), 0.0, pt.pos().y()), ()));
        let top_verts =
            pts.map(|pt| vertex(pt3(pt.pos().x(), 1.0, pt.pos().y()), ()));

        // Bottom verts = 0..pts.len
        // Top verts = pts.len..2*pts.len
        let mut verts: Vec<_> = bottom_verts.chain(top_verts).collect();

        let mut faces = Vec::with_capacity(verts.len());

        let len = self.sides;
        for i in 0..len {
            let j = (i + 1) % len;
            let k = i + len;
            let l = j + len;
            if i % 2 == 0 {
                //  k - l
                //  | / |
                //  i - j
                faces.push(tri(i, k, l));
                faces.push(tri(i, l, j));
            } else {
                //  k - l
                //  | \ |
                //  i - j
                faces.push(tri(i, k, j));
                faces.push(tri(j, k, l));
            }
        }

        if self.capped {
            //self.make_caps(&mut verts, &mut faces);
        }

        Mesh { faces, verts }
    }
}

impl<P: Parametric<Vertex2<Normal2>>> Build<Normal3> for Prism<P> {
    fn build(self) -> Mesh<Normal3> {
        let l = self.sides;
        let pts: Vec<_> = 0f32
            .vary_to(1.0, l as u32)
            .map(|t| self.points.eval(t))
            .collect();

        let mut verts: Vec<_> = (0..l)
            .map(|i| [i, (i + 1) % l, (i + 2) % l])
            .map(|ijk| ijk.map(|i| pts[i].pos))
            .flat_map(|[a, b, c]| {
                let n = ((b - a).perp() + (c - b).perp()).normalize_or_zero();
                let n = -vec3(n.x(), 0.0, n.y());
                [
                    vertex(pt3(b.x(), 0.0, b.y()), n), // bottom
                    vertex(pt3(b.x(), 1.0, b.y()), n), // top
                ]
            })
            .collect();

        let mut faces = Vec::with_capacity(verts.len());

        todo!();

        if self.capped {
            todo!()
            //self.make_caps_n(&mut verts, &mut faces);
        }

        Mesh { faces, verts }
    }
}
