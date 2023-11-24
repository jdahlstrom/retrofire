use re::geom::mesh::Mesh;
use re::geom::{vertex, Tri};
use re::math::{degs, lerp, polar, turns, vec2, vec3, Linear};
use re::math::{Affine, Vec2, Vec3};
use re::render::tex::{uv, TexCoord};

pub struct Box {
    pub dimensions: Vec3,
}

impl Box {
    const COORDS: [Vec3; 8] = [
        // left
        vec3(-1.0, -1.0, -1.0), // 000
        vec3(-1.0, -1.0, 1.0),  // 001
        vec3(-1.0, 1.0, -1.0),  // 010
        vec3(-1.0, 1.0, 1.0),   // 011
        // right
        vec3(1.0, -1.0, -1.0), // 100
        vec3(1.0, -1.0, 1.0),  // 101
        vec3(1.0, 1.0, -1.0),  // 110
        vec3(1.0, 1.0, 1.0),   // 111
    ];
    #[allow(unused)]
    const NORMS: [Vec3; 6] = [
        vec3(-1.0, 0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, 0.0, 1.0),
    ];
    #[allow(unused)]
    const TEX_COORDS: [TexCoord; 4] =
        [uv(0.0, 0.0), uv(1.0, 0.0), uv(0.0, 1.0), uv(1.0, 1.0)];
    const VERTS: [(usize, [usize; 2]); 24] = [
        // left
        (0b011, [0, 0]),
        (0b010, [0, 1]),
        (0b001, [0, 2]),
        (0b000, [0, 3]),
        // right
        (0b110, [1, 0]),
        (0b111, [1, 1]),
        (0b100, [1, 2]),
        (0b101, [1, 3]),
        // bottom
        (0b000, [2, 0]),
        (0b100, [2, 1]),
        (0b001, [2, 2]),
        (0b101, [2, 3]),
        // top
        (0b011, [3, 0]),
        (0b111, [3, 1]),
        (0b010, [3, 2]),
        (0b110, [3, 3]),
        // front
        (0b010, [4, 0]),
        (0b110, [4, 1]),
        (0b000, [4, 2]),
        (0b100, [4, 3]),
        // back
        (0b111, [5, 0]),
        (0b011, [5, 1]),
        (0b101, [5, 2]),
        (0b001, [5, 3]),
    ];
    const FACES: [[usize; 3]; 12] = [
        // left
        [0, 1, 3],
        [0, 3, 2],
        // right
        [4, 5, 7],
        [4, 7, 6],
        // bottom
        [8, 9, 11],
        [8, 11, 10],
        // top
        [12, 13, 15],
        [12, 15, 14],
        // front
        [16, 17, 19],
        [16, 19, 18],
        // back
        [20, 21, 23],
        [20, 23, 22],
    ];

    pub fn build(self) -> Mesh<()> {
        Mesh {
            verts: Self::VERTS
                .iter()
                .map(|&(pos_i, _)| {
                    let [pos_x, pos_y, pos_z] = Self::COORDS[pos_i].0;
                    let [dim_x, dim_y, dim_z] = self.dimensions.mul(0.5).0;
                    vertex(
                        vec3(pos_x * dim_x, pos_y * dim_y, pos_z * dim_z).to(),
                        (),
                    )
                })
                .collect(),
            faces: Self::FACES
                .iter()
                .map(|&verts| Tri(verts))
                .collect(),
        }
    }
}

pub struct UnitOctahedron;

impl UnitOctahedron {
    const COORDS: [Vec3; 6] = [
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 0.0),
    ];
    #[allow(unused)]
    const NORMALS: [Vec3; 8] = [
        vec3(-1.0, -1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
    ];
    #[rustfmt::skip]
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
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
    ];

    pub fn build(self) -> Mesh<()> {
        Mesh {
            verts: Self::VERTS
                .iter()
                .map(|&(pos_i, _)| vertex(Self::COORDS[pos_i].to(), ()))
                .collect(),
            faces: Self::FACES
                .iter()
                .map(|&verts| Tri(verts))
                .collect(),
        }
    }
}

pub struct Sphere {
    pub sectors: usize,
    pub segments: usize,
    pub radius: f32,
}

impl Sphere {
    pub fn build(self) -> Mesh<()> {
        let Self { sectors, segments, radius } = self;
        let pts = (0..=segments)
            .map(|seg| lerp(seg as f32 / segments as f32, -90.0, 90.0))
            .map(|alt| polar(radius, degs(alt)))
            .map(Vec2::from)
            .collect();

        Sor::new(pts, sectors).build()
    }
}

pub struct Capsule {
    pub sectors: usize,
    pub segments: usize,
    pub radius: f32,
}

impl Capsule {
    pub fn build(self) -> Mesh<()> {
        let upper_pts = (0..=self.segments)
            .map(|seg| lerp(seg as f32 / (self.segments) as f32, 90.0, 0.0))
            .map(|alt| polar(self.radius, degs(alt)))
            .map(|p| vec2(0.0, 1.0).add(&p.into()));

        let lower_pts = upper_pts
            .clone()
            .map(|v| vec2(v.x(), -v.y()))
            .rev();

        let pts = upper_pts.chain(lower_pts).collect();
        Sor::new(pts, self.sectors).build()
    }
}

pub struct Torus {
    pub major_radius: f32,
    pub minor_radius: f32,
    pub major_sectors: usize,
    pub minor_sectors: usize,
}

impl Torus {
    pub fn build(self) -> Mesh<()> {
        let pts = (0..=self.minor_sectors)
            .map(|seg| lerp(seg as f32 / (self.minor_sectors) as f32, 0.0, 1.0))
            .map(|alt| polar(self.minor_radius, turns(alt)))
            .map(|p| Vec2::from(p).add(&vec2(1.0, 0.0).to()))
            .collect();

        Sor::new(pts, self.major_sectors).build()
    }
}

pub struct Cone {
    pub sectors: usize,
    pub capped: bool,
    pub base_radius: f32,
    pub cap_radius: f32,
}

impl Cone {
    pub fn build(self) -> Mesh<()> {
        let Self {
            sectors,
            base_radius,
            cap_radius,
            capped,
        } = self;
        let pts = vec![vec2(base_radius, -1.0), vec2(cap_radius, 1.0)];
        Sor::new(pts, sectors).capped(capped).build()
    }
}

pub struct Cylinder {
    pub sectors: usize,
    pub capped: bool,
    pub radius: f32,
}

impl Cylinder {
    pub fn build(self) -> Mesh<()> {
        let Self { sectors, capped, radius } = self;
        Cone {
            sectors,
            capped,
            base_radius: radius,
            cap_radius: radius,
        }
        .build()
    }
}

pub struct Sor {
    pts: Vec<Vec2>,
    sectors: usize,
    capped: bool,
}

impl Sor {
    pub fn new(pts: Vec<Vec2>, sectors: usize) -> Sor {
        assert!(sectors >= 3, "sectors must be at least 3, was {sectors}");
        Sor { pts, sectors, capped: false }
    }
    pub fn capped(self, capped: bool) -> Sor {
        Sor { capped, ..self }
    }

    pub fn build(self) -> Mesh<()> {
        let Self { pts, sectors, capped, .. } = self;

        let mut verts = vec![];
        let mut faces = vec![];

        for pt in &pts {
            let [r, y] = pt.0;

            for sec in 0..=sectors {
                let theta = lerp(sec as f32 / sectors as f32, 0.0, 1.0);
                let pt: Vec2 = polar(r, turns(theta)).into();

                verts.push(vec3(pt.x(), y, pt.y()));
            }
        }

        for j in 1..pts.len() {
            for i in 1..=sectors {
                let a = (j - 1) * (sectors + 1) + i - 1;
                let b = (j - 1) * (sectors + 1) + i;
                let c = j * (sectors + 1) + i - 1;
                let d = j * (sectors + 1) + i;
                faces.push([a, b, d]);
                faces.push([a, d, c]);
            }
        }

        if capped && !pts.is_empty() {
            verts.push(vec3(0.0, pts[0].y(), 0.0));
            verts.push(vec3(0.0, pts[pts.len() - 1].y(), 0.0));

            for i in 1..=sectors {
                let a = verts.len() - 2;
                let b = i - 1;
                let c = i;
                faces.push([a, b, c]);
            }
            for i in 1..=sectors {
                let a = verts.len() - 1;
                let b = (verts.len() - 3) - sectors + i - 1;
                let c = (verts.len() - 3) - sectors + i;
                faces.push([a, b, c]);
            }
        }

        Mesh::new(
            faces.into_iter().map(Tri).collect(),
            verts
                .into_iter()
                .map(|p| vertex(p.to(), ()))
                .collect(),
        )
    }
}

#[cfg(feature = "teapot")]
pub fn teapot() -> Mesh<()> {
    use crate::teapot::*;

    let mut verts = vec![];
    let mut faces = vec![];

    for [a, b, c, d] in FACES {
        let a = a.map(|i| i as usize - 1);
        let b = b.map(|i| i as usize - 1);
        let c = c.map(|i| i as usize - 1);

        let len = verts.len();
        faces.push(Tri([len, len + 1, len + 2]));

        verts.extend([
            (a[0], [a[2], a[1]]),
            (b[0], [b[2], b[1]]),
            (c[0], [c[2], c[1]]),
        ]);

        if d[0] != -1 {
            let d = d.map(|i| i as usize - 1);
            verts.push((d[0], [d[2], d[1]]));
            faces.push(Tri([len, len + 2, len + 3]));
        }
    }

    let vertex_coords: Vec<_> = VERTICES
        .iter()
        .map(|&v| Vec3::from(v).mul(0.1).sub(&vec3(0.0, 0.0, 0.6)))
        .collect();

    Mesh {
        faces,
        verts: verts
            .into_iter()
            .map(|(i, _)| vertex(vertex_coords[i].to(), ()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
