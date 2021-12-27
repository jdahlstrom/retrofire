use math::{Angle, Angle::*, ApproxEq, lerp, vec::*};

use crate::bbox::BoundingBox;
use crate::mesh::{Builder, Mesh, FaceVert::New};
use crate::mesh2::{self, Face, Vertex};

pub fn unit_cube() -> Builder {
    const VERTS: [Vec4; 8] = [
        // left
        pt(-1.0, -1.0, -1.0), // 000
        pt(-1.0, -1.0, 1.0),  // 001
        pt(-1.0, 1.0, -1.0),  // 010
        pt(-1.0, 1.0, 1.0),   // 011
        // right
        pt(1.0, -1.0, -1.0), // 100
        pt(1.0, -1.0, 1.0),  // 101
        pt(1.0, 1.0, -1.0),  // 110
        pt(1.0, 1.0, 1.0),   // 111
    ];
    const FACES: [[usize; 3]; 12] = [
        // left
        [0b000, 0b001, 0b011], [0b000, 0b011, 0b010],
        // right
        [0b100, 0b111, 0b101], [0b100, 0b110, 0b111],
        // bottom
        [0b000, 0b101, 0b001], [0b000, 0b100, 0b101],
        // top
        [0b010, 0b011, 0b111], [0b010, 0b111, 0b110],
        // front
        [0b000, 0b010, 0b110], [0b000, 0b110, 0b100],
        // back
        [0b001, 0b111, 0b011], [0b001, 0b101, 0b111],
    ];
    Mesh::builder().verts(VERTS).faces(FACES)
}

pub struct UnitCube;

impl UnitCube {
    const COORDS: [Vec4; 8] = [
        // left
        pt(-1.0, -1.0, -1.0), // 000
        pt(-1.0, -1.0, 1.0),  // 001
        pt(-1.0, 1.0, -1.0),  // 010
        pt(-1.0, 1.0, 1.0),   // 011
        // right
        pt(1.0, -1.0, -1.0), // 100
        pt(1.0, -1.0, 1.0),  // 101
        pt(1.0, 1.0, -1.0),  // 110
        pt(1.0, 1.0, 1.0),   // 111
    ];
    const NORMS: [Vec4; 6] = [
        dir(-1.0, 0.0, 0.0),
        dir(1.0, 0.0, 0.0),
        dir(0.0, -1.0, 0.0),
        dir(0.0, 1.0, 0.0),
        dir(0.0, 0.0, -1.0),
        dir(0.0, 0.0, 1.0),
    ];
    const TEXCOORDS: [(f32, f32); 4] = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
    ];
    const VERTS: [(usize, [usize; 2]); 24] = [
        // left
        (0b011, [0, 0]), (0b010, [0, 1]), (0b001, [0, 2]), (0b000, [0, 3]),
        // right
        (0b110, [1, 0]), (0b111, [1, 1]), (0b100, [1, 2]), (0b101, [1, 3]),
        // bottom
        (0b000, [2, 0]), (0b100, [2, 1]), (0b001, [2, 2]), (0b101, [2, 3]),
        // top
        (0b011, [3, 0]), (0b111, [3, 1]), (0b010, [3, 2]), (0b110, [3, 3]),
        // front
        (0b010, [4, 0]), (0b110, [4, 1]), (0b000, [4, 2]), (0b100, [4, 3]),
        // back
        (0b111, [5, 0]), (0b011, [5, 1]), (0b101, [5, 2]), (0b001, [5, 3]),
    ];
    const FACES: [[usize; 3]; 12] = [
        // left
        [0, 1, 3], [0, 3, 2],
        // right
        [4, 5, 7], [4, 7, 6],
        // bottom
        [8, 9, 11], [8, 11, 10],
        // top
        [12, 13, 15], [12, 15, 14],
        // front
        [16, 17, 19], [16, 19, 18],
        // back
        [20, 21, 23], [20, 23, 22],
    ];

    pub fn build(self) -> mesh2::Mesh<(Vec4, )> {
        mesh2::Mesh {
            verts: Self::VERTS.iter()
                .map(|&(coord, [attr, _])| Vertex { coord, attr })
                .collect(),
            vertex_coords: Self::COORDS.into(),
            vertex_attrs: Self::NORMS.into(),
            faces: Self::FACES.iter()
                .map(|&verts| Face { verts, attr: 0 })
                .collect(),
            face_attrs: vec![()],
            bbox: BoundingBox::new(Self::COORDS[0], Self::COORDS[7]),
        }
    }
    pub fn with_texcoords(self) -> mesh2::Mesh<(Vec4, (f32, f32))> {
        mesh2::Mesh {
            verts: Self::VERTS.iter()
                .map(|&(coord, attr)| Vertex { coord, attr })
                .collect(),
            vertex_coords: Self::COORDS.into(),
            vertex_attrs: (Self::NORMS.into(), Self::TEXCOORDS.into()),
            faces: Self::FACES.iter().copied()
                .map(|verts| Face { verts, attr: 0 })
                .collect(),
            face_attrs: vec![()],
            bbox: BoundingBox::new(Self::COORDS[0], Self::COORDS[7]),
        }
    }
}

pub fn unit_octahedron() -> Builder {
    const VERTS: [Vec4; 6] = [
        pt(-1.0, 0.0, 0.0),
        pt(0.0, -1.0, 0.0),
        pt(0.0, 0.0, -1.0),
        pt(0.0, 1.0, 0.0),
        pt(0.0, 0.0, 1.0),
        pt(1.0, 0.0, 0.0),
    ];
    const FACES: [[usize; 3]; 8] = [
        [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
        [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1],
    ];
    Mesh::builder().verts(VERTS).faces(FACES)
}

pub struct UnitOctahedron;

impl UnitOctahedron {
    const COORDS: [Vec4; 6] = [
        pt(-1.0, 0.0, 0.0),
        pt(0.0, -1.0, 0.0),
        pt(0.0, 0.0, -1.0),
        pt(0.0, 1.0, 0.0),
        pt(0.0, 0.0, 1.0),
        pt(1.0, 0.0, 0.0),
    ];
    const NORMALS: [Vec4; 8] = [
        dir(-1.0, -1.0, -1.0),
        dir(-1.0, 1.0, -1.0),
        dir(-1.0, 1.0, 1.0),
        dir(-1.0, -1.0, 1.0),
        dir(1.0, -1.0, -1.0),
        dir(1.0, 1.0, -1.0),
        dir(1.0, 1.0, 1.0),
        dir(1.0, -1.0, 1.0),
    ];
    const VERTS: [(usize, usize); 24] = [
        (0, 0), (2, 0), (1, 0),
        (0, 1), (3, 1), (2, 1),
        (0, 2), (4, 2), (3, 2),
        (0, 3), (1, 3), (4, 3),
        (1, 4), (2, 4), (5, 4),
        (2, 5), (3, 5), (5, 5),
        (3, 6), (4, 6), (5, 6),
        (1, 7), (5, 7), (4, 7),
    ];
    const FACES: [[usize; 3]; 8] = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
        [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],
    ];

    pub fn build(self) -> mesh2::Mesh<(Vec4, )> {
        mesh2::Mesh {
            verts: Self::VERTS.map(|(coord, attr)| Vertex { coord, attr}).into(),
            vertex_coords: Self::COORDS.into(),
            vertex_attrs: Self::NORMALS.into(),
            faces: Self::FACES.map(|verts| Face { verts, attr: 0 }).into(),
            face_attrs: vec![()],
            bbox: BoundingBox::new(-X - Y - Z, X + Y + Z),
        }
    }
}

pub fn unit_sphere(parallels: usize, meridians: usize) -> Builder {
    let pts = (0..parallels)
        .map(|par| lerp(par as f32 / (parallels - 1) as f32, -90.0, 90.0))
        .map(|alt| polar(1.0, Deg(alt)))
        .map(|pt| dir(pt.z, pt.x, 0.0));

    sor(pts, meridians)
}

pub struct UnitSphere(pub usize, pub usize);

impl UnitSphere {
    pub fn build(self) -> mesh2::Mesh<()> {
        let pts = (0..self.0)
            .map(|par| lerp(par as f32 / (self.0 - 1) as f32, -90.0, 90.0))
            .map(|alt| polar(1.0, Deg(alt)))
            .map(|pt| dir(pt.z, pt.x, 0.0));

        Sor(pts.collect(), self.1, false).build()
    }
}


pub fn torus(minor_r: f32, pars: usize, mers: usize) -> Builder {
    let mut bld = Mesh::builder();

    let pars = pars as isize;
    let mers = mers as isize;

    fn angle(n: isize, max: isize) -> Angle {
        Tau(n as f32 / max as f32)
    }

    for theta in (0..mers).map(|mer| angle(mer, mers)) {
        for phi in (0..pars).map(|par| angle(par, pars)) {
            let x = theta.sin() + minor_r * theta.sin() * phi.cos();
            let z = theta.cos() + minor_r * theta.cos() * phi.cos();
            let y = minor_r * phi.sin();

            bld.add_vert(pt(x, y, z));
            if theta > Angle::ZERO && phi > Angle::ZERO {
                bld.add_face(-1, -pars - 2, -2);
                bld.add_face(-1, -pars - 1, -pars - 2);
            }
        }
        if theta > Angle::ZERO {
            bld.add_face(-pars, -1 - pars, -1);
            bld.add_face(-pars, -2 * pars, -1 - pars);
        }
    }

    // Connect the last sector to the first
    for par in 1..pars {
        bld.add_face(par, -pars + par, -1 - pars + par);
        bld.add_face(par, -1 - pars + par, par - 1);
    }
    bld.add_face(-pars, pars - 1, 0);
    bld.add_face(-pars, -1, pars - 1);

    bld
}

pub struct Torus(pub f32, pub usize, pub usize);

impl Torus {
    pub fn build(self) -> mesh2::Mesh<()> {
        let mut bld = mesh2::Builder::new();

        let minor_r = self.0;
        let pars = self.1 as isize;
        let mers = self.2 as isize;

        fn angle(n: isize, max: isize) -> Angle {
            Tau(n as f32 / max as f32)
        }

        for theta in (0..mers).map(|mer| angle(mer, mers)) {
            for phi in (0..pars).map(|par| angle(par, pars)) {
                let x = theta.sin() + minor_r * theta.sin() * phi.cos();
                let z = theta.cos() + minor_r * theta.cos() * phi.cos();
                let y = minor_r * phi.sin();

                bld.add_vert(pt(x, y, z));
                if theta > Angle::ZERO && phi > Angle::ZERO {
                    bld.add_face(-1, -pars - 2, -2);
                    bld.add_face(-1, -pars - 1, -pars - 2);
                }
            }
            if theta > Angle::ZERO {
                bld.add_face(-pars, -1 - pars, -1);
                bld.add_face(-pars, -2 * pars, -1 - pars);
            }
        }

        // Connect the last sector to the first
        for par in 1..pars {
            bld.add_face(par, -pars + par, -1 - pars + par);
            bld.add_face(par, -1 - pars + par, par - 1);
        }
        bld.add_face(-pars, pars - 1, 0);
        bld.add_face(-pars, -1, pars - 1);

        bld.build()
    }
}

pub fn unit_cone(minor_r: f32, sectors: usize) -> Builder {

    // Body
    let pts = [pt(1.0, -1.0, 0.0), pt(minor_r, 1.0, 0.0)];
    let mut bld = sor(pts, sectors);

    let azimuths = (1..sectors).map(|sec| Tau(sec as f32 / sectors as f32));

    // Bottom cap
    let c = bld.mesh.verts.len() as isize;
    bld.add_vert(pt(0.0, -1.0, 0.0));
    bld.add_vert(pt(0.0, -1.0, 1.0));
    for az in azimuths.clone() {
        bld.add_face(c, New(polar(1.0, az) - Y), -1);
    }
    bld.add_face(c, c + 1, -1);

    // Top cap
    if minor_r > 0.0 {
        let c = bld.mesh.verts.len() as isize;
        bld.add_vert(pt(0.0, 1.0, 0.0));
        bld.add_vert(pt(0.0, 1.0, minor_r));
        for az in azimuths.clone() {
            bld.add_face(c, -1, New(polar(minor_r, az) + Y));
        }
        bld.add_face(c, -1, c + 1);
    }

    bld
}

pub struct UnitCone(pub f32, pub usize);

impl UnitCone {
    pub fn build(self) -> mesh2::Mesh<()> {
        let pts = vec![
            pt(1.0, -1.0, 0.0),
            pt(self.0, 1.0, 0.0),
        ];
        Sor(pts.into(), self.1,  true).build()
    }
}

pub fn unit_cylinder(sectors: usize) -> Builder {
    unit_cone(1.0, sectors)
}

pub struct UnitCylinder(pub usize);

impl UnitCylinder {
    pub fn build(self) -> mesh2::Mesh<()> {
        UnitCone(1.0, self.0).build()
    }
}

pub fn sor(pts: impl IntoIterator<Item=Vec4>, sectors: usize) -> Builder {
    let mut bld = Mesh::builder();
    let sectors = sectors as isize;

    let mut pts = pts.into_iter();
    let circum_pts = |start, r| (start..sectors)
        .map(|sec| Tau(sec as f32 / sectors as f32))
        .map(move |az| polar(r, az));

    // TODO Clean up
    let mut p0 = if let Some(p) = pts.next() { p } else { return bld; };

    if p0.x.approx_eq(0.0) { // Start cap
        bld.add_vert(p0);
    } else {
        for p in circum_pts(0, p0.x) {
            bld.add_vert(p + p0.y * Y);
        }
    }

    for p1 in pts {
        if p0.x.approx_eq(0.0) { // Start cap
            bld.add_vert(pt(0.0, p1.y, p1.x));
            for p in circum_pts(1, p1.x) {
                bld.add_face(-1, 0, New(p + p1.y * Y));
            }
            bld.add_face(-1, 0, 1);
        } else if p1.x.approx_eq(0.0) { // End cap
            bld.add_vert(p1);
            for sec in -sectors..-1 {
                bld.add_face(-1, sec - 1, sec);
            }
            bld.add_face(-1, -2, -sectors - 1);
        } else {  // Body segment
            bld.add_vert(polar(p1.x, Tau(0.0)) + p1.y * Y);
            for p in circum_pts(1, p1.x) {
                bld.add_vert(p + p1.y * Y);
                bld.add_face(-1, -2, -sectors - 2);
                bld.add_face(-1, -sectors - 2, -sectors - 1);
            }
            bld.add_face(-1, -sectors - 1, -sectors);
            bld.add_face(-sectors, -sectors - 1, -2 * sectors);
        }
        p0 = p1;
    }
    bld
}

pub struct Sor(pub Vec<Vec4>, pub usize, pub bool);

impl Sor {
    pub fn build(self) -> mesh2::Mesh<()> {
        let Sor(pts, sectors, capped) = self;

        assert!(sectors > 2, "sectors must be at least 3, was {}", self.1);

        let mut bld = mesh2::Builder::new();

        let sectors = sectors as isize;

        let mut pts = pts.into_iter();
        let circum_pts = |start, r, y| (start..sectors)
            .map(|sec| Tau(sec as f32 / sectors as f32))
            .map(move |az| polar(r, az) + y * Y);

        // TODO Clean up
        let mut p0 = if let Some(p) = pts.next() { p } else { return bld.build(); };

        if p0.x.approx_eq(0.0) { // Start cap
            bld.add_vert(p0);
        } else {
            if capped {
                bld.add_vert(pt(0.0, p0.y, 0.0));
                bld.add_vert(pt(0.0, p0.y, p0.x));
                for p in circum_pts(1, p0.x, p0.y) {
                    bld.add_vert(p);
                    bld.add_face(0, -1, -2);
                }
                bld.add_face(0, 1, -1);
            }
            for p in circum_pts(0, p0.x, p0.y) {
                bld.add_vert(p);
            }
        }

        for p1 in pts {
            if p0.x.approx_eq(0.0) { // Start cap
                bld.add_vert(pt(0.0, p1.y, p1.x));
                for p in circum_pts(1, p1.x, p1.y) {
                    bld.add_vert(p);
                    bld.add_face(-2, 0, -1);
                }
                bld.add_face(-1, 0, 1);
            } else if p1.x.approx_eq(0.0) { // End cap
                bld.add_vert(p1);
                for sec in -sectors..-1 {
                    bld.add_face(-1, sec - 1, sec);
                }
                bld.add_face(-1, -2, -sectors - 1);
            } else {  // Body segment
                bld.add_vert(polar(p1.x, Tau(0.0)) + p1.y * Y);
                for p in circum_pts(1, p1.x, p1.y) {
                    bld.add_vert(p);
                    bld.add_face(-1, -2, -sectors - 2);
                    bld.add_face(-1, -sectors - 2, -sectors - 1);
                }
                bld.add_face(-1, -sectors - 1, -sectors);
                bld.add_face(-sectors, -sectors - 1, -2 * sectors);
            }
            p0 = p1;
        }

        if capped && !p0.x.approx_eq(0.0) { // End cap
            let a = bld.add_vert(pt(0.0, p0.y, 0.0));
            let b = bld.add_vert(pt(0.0, p0.y, p0.x));
            for p in circum_pts(1, p0.x, p0.y) {
                bld.add_vert(p);
                bld.add_face(a, -2, -1);
            }
            bld.add_face(a, -1, b);
        }

        bld.build()
    }
}


#[cfg(feature = "teapot")]
pub fn teapot() -> mesh2::Mesh<(Vec4, (f32, f32)), ()> {
    use crate::teapot::*;

    let mut verts = vec![];
    let mut faces = vec![];

    for [a, b, c, d] in FACES {
        let a = a.map(|i| i as usize - 1);
        let b = b.map(|i| i as usize - 1);
        let c = c.map(|i| i as usize - 1);

        verts.push((a[0], [a[2], a[1]]));
        verts.push((b[0], [b[2], b[1]]));
        verts.push((c[0], [c[2], c[1]]));

        faces.push(Face {
            verts: [verts.len() - 3, verts.len() - 2, verts.len() - 1],
            attr: 0
        });

        if d[0] != -1 {
            let d = d.map(|i| i as usize - 1);
            verts.push((d[0], [d[2], d[1]]));
            faces.push(Face {
                verts: [verts.len() - 4, verts.len() - 2, verts.len() - 1],
                attr: 0
            });
        }
    }

    let vertex_coords = VERTICES.iter()
        .map(|&[x, y, z]| pt(x, y, z))
        .collect();
    let bbox = BoundingBox::of(&vertex_coords);
    mesh2::Mesh {
        verts: verts.into_iter()
            .map(|(coord, attr)| Vertex { coord, attr})
            .collect(),
        vertex_coords,
        vertex_attrs: (
            VERTEX_NORMALS.iter()
                .map(|&[x, y, z]| dir(x, y, z))
                .collect(),
            TEX_COORDS.iter()
                .map(|&[u, v]| (u, v))
                .collect()
        ),
        faces,
        face_attrs: vec![()],
        bbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cube() {
        UnitCube.build().validate().unwrap();
    }

    #[test]
    fn validate_octahedron() {
        UnitOctahedron.build().validate().unwrap();
    }

    // TODO These property tests could probably use QuickCheck

    #[test]
    fn validate_sphere() {
        for par in 3..33 {
            for mer in 3..33 {
                UnitSphere(par, mer).build()
                    .validate()
                    .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn sphere_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let sph = UnitSphere(par, mer).build();
                assert_eq!(sph.faces.len(), 2 * mer + 2 * (par - 3) * mer, "par={}, mer={}", par, mer);
                assert_eq!(sph.verts.len(), 2 + (par - 2) * mer, "par={}, mer={}", par, mer);
            }
        }
    }

    #[test]
    fn validate_torus() {
        for par in 3..33 {
            for mer in 3..33 {
                Torus(0.1, par, mer).build()
                    .validate()
                    .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn torus_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let tor = Torus(0.1, par, mer).build();
                assert_eq!(tor.faces.len(), 2 * par * mer, "par={}, mer={}", par, mer);
                assert_eq!(tor.verts.len(), par * mer, "par={}, mer={}", par, mer);
            }
        }
    }

    #[test]
    fn validate_cone() {
        for sec in 3..33 {
            for &minor_r in &[0.0, 0.5, 1.0, 2.0] {
                UnitCone(minor_r, sec).build()
                    .validate()
                    .expect(&format!("sec={} m_r={}", sec, minor_r));
            }
        }
    }

    #[test]
    fn validate_sor_empty() {
        Sor(vec![], 13, false).build().validate().unwrap();
    }

    #[test]
    fn validate_sor_capped() {
        let pts = vec![-2.0*Y, X-2.0*Y, 2.0*X-Y, 0.5*X, X+2.0*Y, 3.0*Y];
        Sor(pts, 8, false).build().validate().unwrap();
    }

    #[test]
    fn validate_sor_open() {
        let pts = vec![X, 2.0*X+0.2*Y, 1.5*X+0.8*Y, X+Y];
        Sor(pts, 11, false).build().validate().unwrap();
    }

    #[test]
    fn validate_sor_capped_top() {
        let pts = vec![0.1*X-Y, 2.0*X+Y, 1.5*X+0.8*Y, Y];
        Sor(pts, 19, false).build().validate().unwrap();
    }

    #[test]
    fn validate_sor_capped_bottom() {
        let pts = vec![-Y, X, X+Y];
        Sor(pts, 3, false).build().validate().unwrap();
    }
}
