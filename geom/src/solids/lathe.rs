use alloc::vec::Vec;
use core::ops::Range;

use re::geom::{vertex, Mesh, Normal2, Normal3, Vertex, Vertex2};
use re::math::{
    degs, polar, pt2, pt3, rotate_y, turns, vec2, vec3, Angle, Vary,
};

/// A surface-of-revolution shape generated by rotating a polyline
/// lying on the xy-plane one full revolution around the y-axis.
#[derive(Clone, Debug, Default)]
pub struct Lathe {
    /// The polyline defining the shape.
    pub points: Vec<Vertex2<Normal2, ()>>,
    /// The number of facets used to approximate the surface of revolution.
    pub sectors: u32,
    /// Whether to add flat caps to both ends of the object. Has no effect
    /// if the endpoints already lie on the y-axis.
    pub capped: bool,
    // The range of angles to rotate over.
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

impl Lathe {
    pub fn new<Pts>(points: Pts, sectors: u32) -> Self
    where
        Pts: IntoIterator<Item = Vertex2<Normal2, ()>>,
    {
        assert!(sectors >= 3, "sectors must be at least 3, was {sectors}");
        Self {
            points: points.into_iter().collect(),
            sectors,
            capped: false,
            az_range: turns(0.0)..turns(1.0),
        }
    }

    pub fn capped(self, capped: bool) -> Self {
        Self { capped, ..self }
    }

    /// Builds the lathe mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let Self { points, sectors, az_range, .. } = self;
        let secs = sectors as usize;

        let n_points = points.len();
        let mut b = Mesh {
            verts: Vec::with_capacity(n_points * (secs + 1) + 2),
            faces: Vec::with_capacity(n_points * secs * 2),
        }
        .into_builder();

        let start = rotate_y(az_range.start);
        let rot = rotate_y((az_range.end - az_range.start) / secs as f32);

        // Create vertices
        for Vertex { pos, attrib: n } in &points {
            let mut pos = start.apply_pt(&pt3(pos.x(), pos.y(), 0.0));
            let mut norm = start.apply(&vec3(n.x(), n.y(), 0.0)).normalize();

            for _ in 0..=secs {
                b.push_vert(pos, norm);
                pos = rot.apply_pt(&pos);
                norm = rot.apply(&norm);
            }
        }
        // Create faces
        for j in 1..n_points {
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
        if self.capped && n_points > 0 {
            let l = b.mesh.verts.len();
            let bottom_rng = 0..=secs;
            let top_rng = (l - secs - 1)..l;

            // Duplicate the bottom ring of vertices to make the bottom cap...
            let mut bottom_vs: Vec<_> = b.mesh.verts[bottom_rng]
                .iter()
                .map(|v| vertex(v.pos, vec3(0.0, -1.0, 0.0)))
                .collect();
            b.mesh.verts.append(&mut bottom_vs);
            for i in 1..secs {
                b.push_face(l, l + i, l + i + 1);
            }

            // ...and the top vertices to make the top cap
            let l = b.mesh.verts.len();
            let mut top_vs: Vec<_> = b.mesh.verts[top_rng]
                .iter()
                .map(|v| vertex(v.pos, vec3(0.0, 1.0, 0.0)))
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

        let pts = degs(-90.0)
            .vary_to(degs(90.0), segments)
            .map(|alt| polar(radius, alt).to_cart())
            .map(|pos| vertex(pos.to_pt(), pos));

        Lathe::new(pts, sectors).build()
    }
}

impl Torus {
    /// Builds the toroidal mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let pts = turns(0.0)
            .vary_to(turns(1.0), self.minor_sectors)
            .map(|alt| polar(self.minor_radius, alt).to_cart())
            .map(|v| vertex(pt2(self.major_radius, 0.0) + v, v));

        Lathe::new(pts, self.major_sectors).build()
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
        let n = vec2(n.y(), -n.x());

        let pts = base_pt
            .vary_to(apex_pt, self.segments)
            .map(|pt| vertex(pt, n));

        Lathe::new(pts, self.sectors)
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
        let bottom_pts: Vec<_> = degs(-90.0)
            .vary_to(degs(0.0), cap_segments)
            .map(|alt| polar(radius, alt).to_cart())
            .map(|v| vertex(pt2(0.0, -1.0) + v, v))
            .collect();

        let top_pts = bottom_pts
            .iter()
            .map(|Vertex { pos, attrib: n }| {
                vertex(pt2(pos.x(), -pos.y()), vec2(n.x(), -n.y()))
            })
            .rev();

        let body_pts = pt2(radius, -1.0)
            .vary_to(pt2(radius, 1.0), body_segments)
            .map(|pt| vertex(pt, vec2(1.0, 0.0)))
            .skip(1)
            .take(body_segments as usize - 1);

        Lathe::new(
            bottom_pts
                .iter()
                .copied()
                .chain(body_pts)
                .chain(top_pts),
            sectors,
        )
        .build()
    }
}
