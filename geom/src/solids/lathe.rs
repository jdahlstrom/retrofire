//! Various solids of revolution.

use alloc::vec::Vec;
use core::ops::Range;

use re::geom::{vertex, Mesh, Normal2, Normal3, Polyline, Vertex, Vertex2};
use re::math::{
    polar, pt2, pt3, rotate_y, turns, vec2, vec3, Angle, Lerp, Parametric,
    Vary, Vec3,
};

/// A surface of revolution generated by rotating a 2D curve around the y-axis.
#[derive(Clone, Debug, Default)]
pub struct Lathe<P> {
    /// The curve defining the shape.
    pub points: P,
    /// The number of facets used to approximate the surface.
    pub sectors: u32,
    /// The number of vertical segments used to approximate the surface.
    pub segments: u32,
    /// Whether to add flat caps to both ends of the object.
    pub capped: bool,
    /// The range of angles over which to rotate.
    pub az_range: Range<Angle>,
}

/// TODO
pub struct Sphere {
    pub sectors: u32,
    pub segments: u32,
    pub radius: f32,
}

/// Toroidal polyhedron.
pub struct Torus {
    /// Distance from the origin to the center of the tube.
    pub major_radius: f32,
    /// Radius of the cross-section of the tube.
    pub minor_radius: f32,

    pub major_sectors: u32,
    pub minor_sectors: u32,
}

/// Right cylinder with regular *n*-gonal cross-section.
pub struct Cylinder {
    pub sectors: u32,
    pub segments: u32,
    pub capped: bool,
    pub radius: f32,
}

/// TODO
pub struct Cone {
    pub sectors: u32,
    pub segments: u32,
    pub capped: bool,
    pub base_radius: f32,
    pub apex_radius: f32,
}

/// Cylinder with hemispherical caps.
pub struct Capsule {
    pub sectors: u32,
    pub body_segments: u32,
    pub cap_segments: u32,
    pub radius: f32,
}

//
// Inherent impls
//

impl<P: Parametric<Vertex2<Normal2, ()>>> Lathe<P> {
    pub fn new(points: P, sectors: u32, segments: u32) -> Self {
        assert!(sectors >= 3, "sectors must be at least 3, was {sectors}");
        Self {
            points,
            sectors,
            segments,
            capped: false,
            az_range: turns(0.0)..turns(1.0),
        }
    }

    pub fn capped(self, capped: bool) -> Self {
        Self { capped, ..self }
    }

    /// Builds the lathe mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let secs = self.sectors as usize;
        let segs = self.segments as usize;

        // Fencepost problem: n + 1 vertices for n segments
        let verts_per_sec = segs + 1;

        let mut b = Mesh {
            verts: Vec::with_capacity(verts_per_sec * (secs + 1) + 2),
            faces: Vec::with_capacity(verts_per_sec * secs * 2),
        }
        .into_builder();

        let Range { start, end } = self.az_range;
        let rot = rotate_y((end - start) / secs as f32);
        let start = rotate_y(start);

        // Create vertices
        for Vertex { pos, attrib: n } in 0.0
            .vary_to(1.0, verts_per_sec as u32)
            .map(|t| self.points.eval(t))
        {
            let mut pos = start.apply_pt(&pt3(pos.x(), pos.y(), 0.0));
            let mut norm = start.apply(&vec3(n.x(), n.y(), 0.0));

            for _ in 0..=secs {
                b.push_vert(pos, norm);
                pos = rot.apply_pt(&pos);
                norm = rot.apply(&norm);
            }
        }
        // Create faces
        for j in 1..verts_per_sec {
            let n = secs + 1;
            for i in 1..n {
                let p = (j - 1) * n + i - 1;
                let q = (j - 1) * n + i;
                let r = j * n + i - 1;
                let s = j * n + i;
                b.push_face(p, s, q);
                b.push_face(p, r, s);
            }
        }
        // Create optional caps
        if self.capped && verts_per_sec > 0 {
            let l = b.mesh.verts.len();
            let bottom_rng = 0..=secs;
            let top_rng = (l - secs - 1)..l;

            // Duplicate the bottom ring of vertices to make the bottom cap...
            let bottom_vs: Vec<_> = b.mesh.verts[bottom_rng]
                .iter()
                .map(|v| vertex(v.pos, -Vec3::Y))
                .collect();
            b.mesh.verts.extend(bottom_vs);
            for i in 1..secs - 1 {
                b.push_face(l, l + i, l + i + 1);
            }

            // ...and the top vertices to make the top cap
            let l = b.mesh.verts.len();
            let mut top_vs: Vec<_> = b.mesh.verts[top_rng]
                .iter()
                .map(|v| vertex(v.pos, Vec3::Y))
                .collect();
            b.mesh.verts.append(&mut top_vs);
            for i in 1..secs {
                b.push_face(l, l + i + 1, l + i);
            }
        }
        b.build()
    }
}

impl Sphere {
    /// Builds the spherical mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let Self { sectors, segments, radius } = self;

        let pts = |t| {
            let a = (-0.25).lerp(&0.25, t);
            let v = polar(radius, turns(a)).to_cart();
            vertex(v.to_pt(), v.normalize())
        };
        Lathe::new(pts, sectors, segments).build()
    }
}

impl Torus {
    /// Builds the toroidal mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let pts = |t| {
            let a = 0.0.lerp(&1.0, t);
            let v = polar(self.minor_radius, turns(a)).to_cart();
            vertex(pt2(self.major_radius, 0.0) + v, v.normalize())
        };
        Lathe::new(pts, self.major_sectors, self.minor_sectors).build()
    }
}

impl Cylinder {
    /// Builds the cylindrical mesh.
    pub fn build(self) -> Mesh<Normal3> {
        #[rustfmt::skip]
        let Self { sectors, segments, capped, radius } = self;
        Cone {
            sectors,
            segments,
            capped,
            base_radius: radius,
            apex_radius: radius,
        }
        .build()
    }
}

impl Cone {
    /// Builds the conical mesh.
    pub fn build(self) -> Mesh<Normal3> {
        assert!(self.segments > 0, "segments cannot be zero");

        let base_pt = pt2(self.base_radius, -1.0);
        let apex_pt = pt2(self.apex_radius, 1.0);

        let n = apex_pt - base_pt;
        let n = vec2(n.y(), -n.x()).normalize();

        let pts = |t| {
            let pt = base_pt.lerp(&apex_pt, t);
            vertex(pt, n)
        };
        Lathe::new(pts, self.sectors, self.segments)
            .capped(self.capped)
            .build()
    }
}

impl Capsule {
    /// Builds the capsule mesh.
    pub fn build(self) -> Mesh<Normal3> {
        #[rustfmt::skip]
        let Self { sectors, body_segments, cap_segments, radius } = self;
        assert!(body_segments > 0, "body segments cannot be zero");
        assert!(cap_segments > 0, "cap segments cannot be zero");

        // Must be collected to allow rev()
        let bottom_pts: Vec<_> = turns(-0.25)
            .vary_to(turns(0.0), cap_segments + 1)
            .take(cap_segments as usize)
            .map(|alt| polar(radius, alt).to_cart())
            .map(|v| vertex(pt2(0.0, -1.0) + v, v.normalize()))
            .collect();

        let top_pts = bottom_pts
            .iter()
            .map(|Vertex { pos, attrib: n }| {
                vertex(pt2(pos.x(), -pos.y()), vec2(n.x(), -n.y()))
            })
            .rev();

        let body_pts = (-1.0)
            .vary_to(1.0, body_segments + 1)
            .map(|t| vertex(pt2(radius, t), vec2(1.0, 0.0)));

        let pts = bottom_pts
            .iter()
            .copied()
            .chain(body_pts)
            .chain(top_pts);

        let segments = 2 * cap_segments + body_segments;
        Lathe::new(Polyline::new(pts), sectors, segments).build()
    }
}
