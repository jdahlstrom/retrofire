use math::{Angle, Angle::Deg, lerp, vec::*};
use math::Angle::Tau;

use crate::mesh::{Builder, Mesh};
use crate::mesh::FaceVert::New;

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
    Mesh::builder()
        .verts(VERTS.iter().copied())
        .faces(FACES.iter().copied())
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
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
    ];
    Mesh::builder()
        .verts(VERTS.iter().copied())
        .faces(FACES.iter().copied())
}

pub fn unit_sphere(parallels: usize, meridians: usize) -> Builder {
    let mut bld = Mesh::builder();

    let meridians = meridians as isize;
    let parallels = parallels - 1;

    let azimuths = (1..meridians)
        .map(|mer| lerp(mer as f32 / meridians as f32, 0.0, 360.0))
        .map(Deg);
    let mut altitudes = (1..parallels)
        .map(|par| lerp(par as f32 / parallels as f32, -90.0, 90.0))
        .map(Deg);

    // bottom cap
    bld.add_vert(-Y);
    let alt = altitudes.next().unwrap();
    bld.add_vert(spherical(1.0, Deg(0.0), alt));

    for az in azimuths.clone() {
        bld.add_vert(spherical(1.0, az, alt));
        bld.add_face(0, -1, -2);
    }
    bld.add_face(0, 1, -1);

    // body
    for alt in altitudes {
        bld.add_vert(spherical(1.0, Deg(0.0), alt));

        for az in azimuths.clone() {
            bld.add_vert(spherical(1.0, az, alt));

            bld.add_face(-1, -2, -meridians - 2);
            bld.add_face(-1, -meridians - 2, -meridians - 1);
        }

        bld.add_face(-1, -meridians - 1, -meridians);
        bld.add_face(-meridians, -meridians - 1, -2 * meridians);
    }

    // top cap
    bld.add_vert(Y);
    for mer in 1..meridians {
        bld.add_face(-2 - meridians + mer, -1 - meridians + mer, -1);
    }
    bld.add_face(-meridians - 1, -1, -2);

    bld
}

pub fn torus(minor_r: f32, pars: usize, mers: usize) -> Builder {
    let mut bld = Mesh::builder();

    let pars = pars as isize;
    let mers = mers as isize;

    fn angle(n: isize, max: isize) -> Angle {
        Tau((n % max) as f32 / max as f32)
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
    let mut bld = Mesh::builder();
    let sectors = sectors as isize;
    let azimuths = (1..sectors)
        .map(|sec| Tau(sec as f32 / sectors as f32));

    bld.add_vert(pt(0.0, 1.0, 0.0));

    if minor_r > 0.0 {
        // Top cap
        bld.add_vert(pt(0.0, 1.0, 1.0));
        for az in azimuths.clone() {
            bld.add_face(0, New(polar(minor_r, az) + Y), -1);
        }
        bld.add_face(0, 1, -1);

        // Body
        bld.add_vert(pt(0.0, 1.0, minor_r));
        bld.add_vert(pt(0.0, -1.0, 1.0));
        for az in azimuths.clone() {
            bld.add_vert(polar(minor_r, az) + Y);
            bld.add_vert(polar(1.0, az) - Y);
            bld.add_face(-4, -1, -3);
            bld.add_face(-4, -2, -1);
        }
        bld.add_face(-2, sectors + 2, -1);
        bld.add_face(-2, sectors + 1, sectors + 2);
    } else {
        // Body
        bld.add_vert(pt(0.0, -1.0, 1.0));
        for az in azimuths.clone() {
            bld.add_face(0, New(polar(1.0, az) - Y), -1);
        }
        bld.add_face(0, 1, -1);
    }

    // Bottom cap
    let c = bld.mesh.verts.len() as isize;
    bld.add_vert(pt(0.0, -1.0, 0.0));
    bld.add_vert(pt(0.0, -1.0, 1.0));

    for az in azimuths.clone().skip(1) {
        bld.add_face(c, -1, New(polar(1.0, az) - Y));
    }
    bld.add_face(c, -1, c + 1);

    bld
}

pub fn unit_cylinder(sectors: usize) -> Builder {
    unit_cone(1.0, sectors)
}

#[cfg(feature = "teapot")]
pub fn teapot() -> Builder<Vec4, ()> {
    use crate::teapot::*;

    let make_faces = |&[a, b, c, d]: &[[i32; 3]; 4]| {
        let mut vec = vec![[a[0] as usize - 1, b[0] as usize - 1, c[0] as usize - 1]];
        if d[0] != -1 {
            vec.push([c[0] as usize - 1, d[0] as usize - 1, a[0] as usize - 1]);
        }
        vec
    };

    Mesh::builder()
        .verts(VERTICES.iter().map(|&[x, y, z]| pt(x, y, z)))
        .faces(FACES.iter().flat_map(make_faces))
        .vertex_attrs(VERTEX_NORMALS.iter().map(|&[x, y, z]| dir(x, y, z)))
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
}
