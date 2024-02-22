use core::array::from_fn;

use re::geom::Mesh;
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

/// A rectangular cuboid.
///
/// Defined by the left-bottom-near and right-top-far vertices of the box.
///
/// An equilateral box is a cube, a platonic solid with six square faces.
/// The dual of the cube is the octahedron.
///
/// Assuming the two defining vertices are (l, b, n) and (r, t, f),
/// the vertices of a `Box` are at
/// * (l, b, n)
/// * (l, b, f)
/// * (l, t, n)
/// * (l, t, f)
/// * (r, b, n)
/// * (r, b, f)
/// * (r, t, n)
/// * (r, t, f)
#[derive(Copy, Clone, Debug)]
pub struct Box {
    /// The left bottom near corner of the box.
    pub left_bot_near: Vec3,
    /// The right top far corner of the box.
    pub right_top_far: Vec3,
}

/// Regular octahedron.
///
/// A Platonic solid with six vertices and eight equilateral triangle faces.
/// The octahedron is the dual of the cube.
///
/// `Octahedron`'s vertices are at (±1, 0, 0), (0, ±1, 0), and (0, 0, ±1).
#[derive(Copy, Clone, Debug, Default)]
pub struct Octahedron;

/// Regular dodecahedron.
///
/// A Platonic solid with twenty vertices and twelve regular pentagonal faces.
/// Three edges meet at every vertex. The dual of the dodecahedron is the
/// icosahedron.
///
/// `Dodecahedron`'s vertices are at:
/// * (±1, ±1, ±1)
/// * (±φ, ±1/φ, 0)
/// * (±1/φ, ±φ, 0)
/// * (±φ, 0, ±1/φ)
/// * (±1/φ, 0, ±φ)
/// * (0, ±φ, ±1/φ)
/// * (0, ±1/φ, ±φ)
///
/// where φ ≈ 1.618 is the golden ratio constant.
#[derive(Copy, Clone, Debug, Default)]
pub struct Dodecahedron;

/// Regular icosahedron.
///
/// A Platonic solid with twelve vertices and twenty equilateral triangle
/// faces. Five edges meet at every vertex. The dual of the icosahedron is
/// the dodecahedron.
///
/// `Icosahedron`'s vertices are at:
/// * (±1, 0, ±φ)
/// * (±φ, ±1, 0)
/// * (0, ±φ, ±1),
///
/// where φ ≈ 1.618 is the golden ratio constant.
#[derive(Copy, Clone, Debug, Default)]
pub struct Icosahedron;

impl Tetrahedron {
    const FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]];

    // Builds the tetrahedral mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let sqrt = f32::sqrt;
        let coords = [
            vec3(0.0, 1.0, 0.0),
            vec3(sqrt(8.0 / 9.0), -1.0 / 3.0, 0.0),
            vec3(-sqrt(2.0 / 9.0), -1.0 / 3.0, sqrt(2.0 / 3.0)),
            vec3(-sqrt(2.0 / 9.0), -1.0 / 3.0, -sqrt(2.0 / 3.0)),
        ];
        let norms = [-coords[3], -coords[1], -coords[2], -coords[0]];

        let mut b = Mesh::builder();

        for (i, vs) in Self::FACES.into_iter().enumerate() {
            b.push_face(3 * i, 3 * i + 1, 3 * i + 2);
            for v in vs {
                b.push_vert(coords[v].to(), norms[i]);
            }
        }
        b.build()
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

    /// Returns a cube centered on the origin, with the given side length.
    pub fn cube(side_len: f32) -> Self {
        Self {
            left_bot_near: splat(-0.5 * side_len),
            right_top_far: splat(0.5 * side_len),
        }
    }

    // Builds the cuboid mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let mut b = Mesh::builder();
        b.push_faces(Self::FACES);
        for (pos_i, [norm_i, _uv_i]) in Self::VERTS {
            let pos = from_fn(|i| {
                self.left_bot_near[i]
                    .lerp(&self.right_top_far[i], Self::COORDS[pos_i][i])
            });
            b.push_vert(pos.into(), Self::NORMS[norm_i]);
        }
        b.build()
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

    // Builds the octahedral mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let mut b = Mesh::builder();
        for (i, vs) in Self::FACES.iter().enumerate() {
            b.push_face(3 * i, 3 * i + 1, 3 * i + 2);
            for vi in *vs {
                let pos = Self::COORDS[Self::VERTS[vi].0];
                b.push_vert(pos, Self::NORMS[i]);
            }
        }
        b.build()
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
    #[rustfmt::skip]
    const FACES: [[usize; 5]; 12] = [
        [ 0,  1, 14, 8, 12], [ 1, 0, 13, 10, 15],
        [ 3,  2, 16, 9, 18], [ 2, 3, 19, 11, 17],
        [ 4,  5, 13, 0, 12], [ 5, 4, 16,  2, 17],
        [ 7,  6, 14, 1, 15], [ 6, 7, 19,  3, 18],
        [ 8,  9, 16, 4, 12], [ 9, 8, 14,  6, 18],
        [11, 10, 13, 5, 17], [10, 11, 19, 7, 15],
    ];

    // The normals are exactly the vertices of the icosahedron, normalized.
    const NORMALS: [Vec3; 12] = Icosahedron::COORDS;

    // Builds the dodecahedral mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let mut b = Mesh::builder();

        for (i, face) in Self::FACES.iter().enumerate() {
            let n = Self::NORMALS[i].normalize();
            // Make a pentagon from three triangles
            let i5 = 5 * i;
            b.push_face(i5, i5 + 1, i5 + 2);
            b.push_face(i5, i5 + 2, i5 + 3);
            b.push_face(i5, i5 + 3, i5 + 4);
            for &j in face {
                b.push_vert(Self::COORDS[j].to().normalize(), n);
            }
        }
        b.build()
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

    // The normals are exactly the vertices of the dodecahedron, normalized.
    const NORMALS: [Vec3; 20] = Dodecahedron::COORDS;

    // Builds the icosahedral mesh.
    pub fn build(self) -> Mesh<Normal3> {
        let mut b = Mesh::builder();
        for (i, vs) in Self::FACES.iter().enumerate() {
            let n = Self::NORMALS[i].normalize();
            b.push_face(3 * i, 3 * i + 1, 3 * i + 2);
            for vi in *vs {
                b.push_vert(Self::COORDS[vi].to().normalize(), n);
            }
        }
        b.build()
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
