use std::array::from_fn;

use re::geom::{vertex, Mesh, Tri};
use re::math::vec::splat;
use re::math::{vec3, Vary, Vec3};
use re::render::tex::{uv, TexCoord};

/// A surface normal.
// TODO Use distinct type rather than alias
pub type Normal3 = Vec3;

/// A regular tetrahedron.
///
/// A Platonic solid with four vertices and four equilateral triangle faces.
/// The tetrahedron is its own dual.
///
/// `Tetrahedron`'s vertices are at:
/// * (0, 1, 0),
/// * (√(8/9), -1/3, 0),
/// * (-√(2/9), -1/3, √(2/3)), and
/// * (-√(8/9), -1/3, -√(2/3)).
#[derive(Copy, Clone, Debug)]
pub struct Tetrahedron;

#[derive(Copy, Clone, Debug)]
pub struct Box {
    /// The left bottom near corner of the box.
    pub left_bot_near: Vec3,
    /// The right top far corner of the box.
    pub right_top_far: Vec3,
}

/// A regular octahedron: a Platonic solid with six vertices and eight
/// equilateral triangle faces. The octahedron is the dual of the cube.
///
/// # Vertex coordinates
///
/// (±1, 0, 0), (0, ±1, 0), and (0, 0, ±1).
#[derive(Copy, Clone, Debug, Default)]
pub struct Octahedron;

#[derive(Copy, Clone, Debug, Default)]
pub struct Dodecahedron;

#[derive(Copy, Clone, Debug, Default)]
pub struct Icosahedron;

impl Tetrahedron {
    const FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]];

    pub fn build(self) -> Mesh<Normal3> {
        let sqrt = f32::sqrt;
        let coords = [
            vec3(0.0, 1.0, 0.0),
            vec3(sqrt(8.0 / 9.0), -1.0 / 3.0, 0.0),
            vec3(-sqrt(2.0 / 9.0), -1.0 / 3.0, sqrt(2.0 / 3.0)),
            vec3(-sqrt(2.0 / 9.0), -1.0 / 3.0, -sqrt(2.0 / 3.0)),
        ];
        let norms = [-coords[3], -coords[1], -coords[2], -coords[0]];

        let mut faces = vec![];
        let mut verts = vec![];

        for (i, [a, b, c]) in Self::FACES.into_iter().enumerate() {
            faces.push(Tri([3 * i, 3 * i + 1, 3 * i + 2]));
            verts.push(vertex(coords[a].to(), norms[i]));
            verts.push(vertex(coords[b].to(), norms[i]));
            verts.push(vertex(coords[c].to(), norms[i]));
        }
        Mesh::new(faces, verts)
    }
}

impl Box {
    const COORDS: [Vec3; 8] = [
        // left
        vec3(0.0, 0.0, 0.0), // 0b000
        vec3(0.0, 0.0, 1.0), // 0b001
        vec3(0.0, 1.0, 0.0), // 0b010
        vec3(0.0, 1.0, 1.0), // 0b011
        // right
        vec3(1.0, 0.0, 0.0), // 0b100
        vec3(1.0, 0.0, 1.0), // 0b101
        vec3(1.0, 1.0, 0.0), // 0b110
        vec3(1.0, 1.0, 1.0), // 0b111
    ];
    const NORMS: [Normal3; 6] = [
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
    #[rustfmt::skip]
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
    #[rustfmt::skip]
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

    pub fn build(self) -> Mesh<Normal3> {
        let verts = Self::VERTS
            .iter()
            .map(|&(pos_idx, [norm_idx, _uv_idx])| {
                (Self::COORDS[pos_idx], Self::NORMS[norm_idx])
            })
            .map(|(pos, norm)| {
                let pos = from_fn(|i| {
                    self.left_bot_near[i].lerp(&self.right_top_far[i], pos[i])
                });
                vertex(pos.into(), norm)
            })
            .collect();

        let faces = Self::FACES.map(Tri).to_vec();

        Mesh { verts, faces }
    }
}

impl Octahedron {
    const COORDS: [Vec3; 6] = [
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 0.0),
    ];
    const NORMS: [Normal3; 8] = [
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

    pub fn build(self) -> Mesh<Normal3> {
        Mesh {
            verts: Self::VERTS
                .iter()
                .map(|&(pos_i, norm_i)| {
                    vertex(Self::COORDS[pos_i].to(), Self::NORMS[norm_i])
                })
                .collect(),
            faces: Self::FACES.map(Tri).to_vec(),
        }
    }
}

/// The golden ratio constant φ.
const PHI: f32 = 1.61803401_f32;
/// Reciprocal of φ.
const R_PHI: f32 = 1.0 / PHI;

impl Dodecahedron {
    #[rustfmt::skip]
    const COORDS: [Vec3; 20] = [
        // -X
        vec3(-PHI, -R_PHI, 0.0),
        vec3(-PHI,  R_PHI, 0.0),
        // +X
        vec3( PHI, -R_PHI, 0.0),
        vec3( PHI,  R_PHI, 0.0),
        // -Y
        vec3(0.0, -PHI, -R_PHI),
        vec3(0.0, -PHI,  R_PHI),
        // +Y
        vec3(0.0,  PHI, -R_PHI),
        vec3(0.0,  PHI,  R_PHI),
        // -Z
        vec3(-R_PHI, 0.0, -PHI),
        vec3( R_PHI, 0.0, -PHI),
        // +Z
        vec3(-R_PHI, 0.0,  PHI),
        vec3( R_PHI, 0.0,  PHI),

        // Corner verts, corresponding to the corner faces of the icosahedron.
        vec3(-1.0, -1.0, -1.0),
        vec3(-1.0, -1.0,  1.0),
        vec3(-1.0,  1.0, -1.0),
        vec3(-1.0,  1.0,  1.0),
        vec3( 1.0, -1.0, -1.0),
        vec3( 1.0, -1.0,  1.0),
        vec3( 1.0,  1.0, -1.0),
        vec3( 1.0,  1.0,  1.0),

    ];
    const FACES: [[usize; 5]; 12] = [
        [0, 1, 14, 8, 12],
        [1, 0, 13, 10, 15],
        [3, 2, 16, 9, 18],
        [2, 3, 19, 11, 17],
        [4, 5, 13, 0, 12],
        [5, 4, 16, 2, 17],
        [7, 6, 14, 1, 15],
        [6, 7, 19, 3, 18],
        [8, 9, 16, 4, 12],
        [9, 8, 14, 6, 18],
        [11, 10, 13, 5, 17],
        [10, 11, 19, 7, 15],
    ];

    // The normals are exactly the vertices of the icosahedron, normalized.
    const NORMALS: [Vec3; 12] = Icosahedron::COORDS;

    pub fn build(self) -> Mesh<Normal3> {
        let mut faces = vec![];
        let mut verts = vec![];

        for (i, face) in Self::FACES.iter().enumerate() {
            let n = Self::NORMALS[i].normalize();
            let i5 = 5 * i;
            faces.push(Tri([i5, i5 + 1, i5 + 2]));
            faces.push(Tri([i5, i5 + 2, i5 + 3]));
            faces.push(Tri([i5, i5 + 3, i5 + 4]));
            for &j in face {
                verts.push(vertex(Self::COORDS[j].to().normalize(), n))
            }
        }

        Mesh::new(faces, verts)
    }
}

impl Icosahedron {
    #[rustfmt::skip]
    const COORDS: [Vec3; 12] = [
        vec3(-PHI, 0.0, -1.0), vec3(-PHI, 0.0, 1.0), // -X
        vec3( PHI, 0.0, -1.0), vec3( PHI, 0.0, 1.0), // +X

        vec3(-1.0, -PHI, 0.0), vec3(1.0, -PHI, 0.0), // -Y
        vec3(-1.0,  PHI, 0.0), vec3(1.0,  PHI, 0.0), // +Y

        vec3(0.0, -1.0, -PHI), vec3(0.0, 1.0, -PHI), // -Z
        vec3(0.0, -1.0,  PHI), vec3(0.0, 1.0,  PHI), // +Z
    ];
    #[rustfmt::skip]
    const FACES: [[usize; 3]; 20] = [
        [0,  4,  1], [0,  1,  6], // -X
        [2,  3,  5], [2,  7,  3], // +X
        [4,  8,  5], [4,  5, 10], // -Y
        [6,  7,  9], [6,  11, 7], // +Y
        [8,  0,  9], [8,  9,  2], // -Z
        [10, 11, 1], [10, 3, 11], // +Z

        // Corner faces, corresponding to the corner verts of the dodecahedron.
        [0, 8, 4], [1,  4, 10], // -X-Y -Z,+Z
        [0, 6, 9], [1, 11,  6], // -X+Y   "
        [2, 5, 8], [3, 10,  5], // +X-Y   "
        [2, 9, 7], [3,  7, 11], // +X+Y   "
    ];

    /// The normals are exactly the vertices of the dodecahedron, normalized.
    const NORMALS: [Vec3; 20] = Dodecahedron::COORDS;

    pub fn build(self) -> Mesh<Normal3> {
        let mut faces = vec![];
        let mut verts = vec![];

        for (i, &[a, b, c]) in Self::FACES.iter().enumerate() {
            let n = Self::NORMALS[i].normalize();
            faces.push(Tri([3 * i, 3 * i + 1, 3 * i + 2]));
            verts.push(vertex(Self::COORDS[a].to().normalize(), n));
            verts.push(vertex(Self::COORDS[b].to().normalize(), n));
            verts.push(vertex(Self::COORDS[c].to().normalize(), n));
        }

        Mesh::new(faces, verts)
    }
}

impl Default for Box {
    /// Creates a cube with unit-length edges, centered at the origin.
    fn default() -> Self {
        Self {
            left_bot_near: splat(-0.5),
            right_top_far: splat(0.5),
        }
    }
}
