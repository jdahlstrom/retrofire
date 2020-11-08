use std::f32::consts::PI;

use math::vec::*;

use crate::mesh::Mesh;

pub fn unit_cube() -> Mesh {
    Mesh {
        verts: vec![
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
        ],
        faces: vec![
            // left
            [0b000, 0b011, 0b001], [0b000, 0b010, 0b011],
            // right
            [0b100, 0b101, 0b111], [0b100, 0b111, 0b110],
            // bottom
            [0b000, 0b001, 0b101], [0b000, 0b101, 0b100],
            // top
            [0b010, 0b111, 0b011], [0b010, 0b110, 0b111],
            // front
            [0b000, 0b110, 0b010], [0b000, 0b100, 0b110],
            // back
            [0b001, 0b011, 0b111], [0b001, 0b111, 0b101],
        ],
        vertex_norms: None,
        face_norms: None,
    }
}

pub fn unit_octahedron() -> Mesh {
    Mesh {
        verts: vec![
            pt(-1.0, 0.0, 0.0),
            pt(0.0, -1.0, 0.0),
            pt(0.0, 0.0, -1.0),
            pt(0.0, 1.0, 0.0),
            pt(0.0, 0.0, 1.0),
            pt(1.0, 0.0, 0.0),
        ],
        faces: vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ],
        vertex_norms: None,
        face_norms: None,
    }
}

pub fn unit_sphere(parallels: usize, meridians: usize) -> Mesh {
    let mut verts = vec![pt(0.0, 1.0, 0.0)];
    let mut faces = vec![];
    let parallels = parallels - 1;

    let phi = PI / parallels as f32;

    // top cap
    verts.push(pt(phi.sin(), phi.cos(), 0.0));
    for mer in 1..meridians {
        let theta = 2.0 * PI * mer as f32 / meridians as f32;
        verts.push(pt(theta.cos() * phi.sin(), phi.cos(), theta.sin() * phi.sin()));
        faces.push([0, verts.len() - 1, verts.len() - 2]);
    }
    faces.push([0, 1, verts.len() - 1]);

    for par in 2..parallels {
        let phi = PI * par as f32 / parallels as f32;

        verts.push(pt(phi.sin(), phi.cos(), 0.0));
        for mer in 1..meridians {
            let theta = 2.0 * PI * mer as f32 / meridians as f32;
            verts.push(pt(theta.cos() * phi.sin(), phi.cos(), theta.sin() * phi.sin()));

            let l = verts.len();
            faces.push([l - 1, l - 2, l - meridians - 2]);
            faces.push([l - 1, l - meridians - 2, l - meridians - 1]);
        }
        let l = verts.len();
        faces.push([l - 1, l - meridians - 1, l - meridians]);
        faces.push([l - meridians, l - meridians - 1, l - 2 * meridians]);
    }

    // bottom cap
    verts.push(pt(0.0, -1.0, 0.0));

    for mer in 1..meridians {
        faces.push([verts.len() - 2 - meridians + mer, verts.len() - 1 - meridians + mer, verts.len() - 1]);
    }
    faces.push([verts.len() - meridians - 1, verts.len() - 1, verts.len() - 2]);

    Mesh {
        verts,
        faces,
        vertex_norms: None,
        face_norms: None,
    }
}

pub fn torus(minor_r: f32, pars: usize, mers: usize) -> Mesh {
    let mut verts = vec![];
    let mut faces = vec![];

    fn angle(n: usize, max: usize) -> f32 {
        2.0 * PI * (n % max) as f32 / max as f32
    }

    for mer in 0..mers {
        let theta = angle(mer, mers);
        for par in 0..pars {
            let phi = angle(par, pars);

            let x = theta.sin() + minor_r * theta.sin() * phi.cos();
            let z = theta.cos() + minor_r * theta.cos() * phi.cos();
            let y = minor_r * phi.sin();

            verts.push(pt(x, y, z));
            if mer > 0 && par > 0 {
                let l = verts.len();
                faces.push([l - 1, l - 2, l - pars - 2]);
                faces.push([l - 1, l - pars - 2, l - pars - 1]);
            }
        }
        if mer > 0 {
            let l = verts.len();
            faces.push([l - pars, l - 1, l - 1 - pars]);
            faces.push([l - pars, l - 1 - pars, l - 2 * pars]);
        }
    }

    let l = verts.len();
    for par in 1..pars {
        faces.push([l - pars + par, par, l - 1 - pars + par]);
        faces.push([l - 1 - pars + par, par, par - 1])
    }
    faces.push([l - pars, 0, pars - 1]);
    faces.push([l - pars, pars - 1, l - 1]);

    Mesh {
        verts,
        faces,
        vertex_norms: None,
        face_norms: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cube() {
        unit_cube().validate().unwrap();
    }

    #[test]
    fn validate_octahedron() {
        unit_octahedron().validate().unwrap();
    }

    // TODO These property tests could probably use QuickCheck

    #[test]
    fn validate_sphere() {
        for par in 3..33 {
            for mer in 3..33 {
                unit_sphere(par, mer).validate()
                                     .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn sphere_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let sph = unit_sphere(par, mer);
                assert_eq!(sph.faces.len(), 2 * mer + 2 * (par - 3) * mer, "par={}, mer={}", par, mer);
                assert_eq!(sph.verts.len(), 2 + (par - 2) * mer, "par={}, mer={}", par, mer);
            }
        }
    }

    #[test]
    fn validate_torus() {
        for par in 3..33 {
            for mer in 3..33 {
                torus(0.1, par, mer).validate()
                                    .expect(&format!("par={}, mer={}", par, mer));
            }
        }
    }

    #[test]
    fn torus_faces_and_vertices() {
        for par in 3..33 {
            for mer in 3..33 {
                let tor = torus(0.1, par, mer);
                assert_eq!(tor.faces.len(), 2 * par * mer, "par={}, mer={}", par, mer);
                assert_eq!(tor.verts.len(), par * mer, "par={}, mer={}", par, mer);
            }
        }
    }
}
