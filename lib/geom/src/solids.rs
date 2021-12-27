use math::{Angle, Angle::*, ApproxEq, lerp, vec::*};
use crate::bbox::BoundingBox;

use crate::mesh::{Builder, Mesh};
use crate::mesh::FaceVert::New;
use crate::mesh2;

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

pub fn unit_cube2() -> mesh2::Mesh<(Vec4, (f32, f32))> {
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
    mesh2::Mesh {
        verts: VERTS.iter().map(|&(ci, ai)| (ci, ai)).collect(),
        vertex_coords: COORDS.into(),
        vertex_attrs: (NORMS.into(), TEXCOORDS.into()),
        faces: FACES.iter().copied().map(|f| (f, 0)).collect(),
        face_attrs: vec![()],
        bbox: BoundingBox::of(&COORDS)
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

pub fn unit_sphere(parallels: usize, meridians: usize) -> Builder {
    let pts = (0..parallels)
        .map(|par| lerp(par as f32 / (parallels - 1) as f32, -90.0, 90.0))
        .map(|alt| polar(1.0, Deg(alt)))
        .map(|pt| dir(pt.z, pt.x, 0.0));

    sor(pts, meridians)
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

pub fn unit_cylinder(sectors: usize) -> Builder {
    unit_cone(1.0, sectors)
}

pub fn sor(pts: impl IntoIterator<Item=Vec4>, sectors: usize) -> Builder {
    let mut bld = Mesh::builder();
    let sectors = sectors as isize;

    let mut pts = pts.into_iter();
    let circum_pts = |start, r| (start..sectors)
        .map(|sec| Tau(sec as f32 / sectors as f32))
        .map(move |az| polar(r, az));

    // TODO Clean up
    let mut p0 = if let Some(p) = pts.next() { p } else { return bld };

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

#[cfg(feature = "teapot")]
pub fn teapot() -> Builder<Vec4, ()> {
    use crate::teapot::*;

    let make_faces = |&[abc @ .., d]: &[[i32; 3]; 4]| {
        let [a, b, c] = abc.map(|x| x[0] as usize - 1);
        let mut vec = vec![[a, b, c]];
        if d[0] != -1 {
            let d = d[0] as usize - 1;
            vec.push([c, d, a]);
        }
        vec
    };

    Mesh::builder()
        .verts(VERTICES.map(|[x, y, z]| pt(x, y, z)))
        .faces(FACES.iter().flat_map(make_faces))
        .vertex_attrs(VERTEX_NORMALS.map(|[x, y, z]| dir(x, y, z)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cube() {
        unit_cube().build().validate().unwrap();
    }

    #[test]
    fn validate_octahedron() {
        unit_octahedron().build().validate().unwrap();
    }

    // TODO These property tests could probably use QuickCheck

    #[test]
    fn validate_sphere() {
        for par in 3..33 {
            for mer in 3..33 {
                unit_sphere(par, mer).build()
                    .validate()
                    .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn sphere_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let sph = unit_sphere(par, mer).build();
                assert_eq!(sph.faces.len(), 2 * mer + 2 * (par - 3) * mer, "par={}, mer={}", par, mer);
                assert_eq!(sph.verts.len(), 2 + (par - 2) * mer, "par={}, mer={}", par, mer);
            }
        }
    }

    #[test]
    fn validate_torus() {
        for par in 3..33 {
            for mer in 3..33 {
                torus(0.1, par, mer).build()
                    .validate()
                    .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn torus_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let tor = torus(0.1, par, mer).build();
                assert_eq!(tor.faces.len(), 2 * par * mer, "par={}, mer={}", par, mer);
                assert_eq!(tor.verts.len(), par * mer, "par={}, mer={}", par, mer);
            }
        }
    }

    #[test]
    fn validate_cone() {
        for sec in 3..33 {
            for &minor_r in &[0.0, 0.5, 1.0, 2.0] {
                unit_cone(minor_r, sec).build()
                    .validate()
                    .expect(&format!("sec={} m_r={}", sec, minor_r));
            }
        }
    }

    #[test]
    fn validate_sor_empty() {
        sor(vec![], 13).build().validate().unwrap();
    }

    #[test]
    fn validate_sor_capped() {
        let pts = vec![-2.0*Y, X-2.0*Y, 2.0*X-Y, 0.5*X, X+2.0*Y, 3.0*Y];
        sor(pts, 8).build().validate().unwrap();
    }
    #[test]
    fn validate_sor_open() {
        let pts = vec![X, 2.0*X+0.2*Y, 1.5*X+0.8*Y, X+Y];
        sor(pts, 11).build().validate().unwrap();
    }
    #[test]
    fn validate_sor_capped_top() {
        let pts = vec![0.1*X-Y, 2.0*X+Y, 1.5*X+0.8*Y, Y];
        sor(pts, 19).build().validate().unwrap();
    }
    #[test]
    fn validate_sor_capped_bottom() {
        let pts = vec![-Y, X, X+Y];
        sor(pts, 3).build().validate().unwrap();
    }
}
