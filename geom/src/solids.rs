use std::array::from_fn;

use re::geom::mesh::Mesh;
use re::geom::{vertex, Tri};
use re::math::vec::splat;
use re::math::{vec3, Vary, Vec3};
use re::render::tex::{uv, TexCoord};

/// A surface normal.
// TODO Use distinct type rather than alias
pub type Normal3 = Vec3;

/// A rectangular cuboid, defined by two opposite vertices.
#[derive(Copy, Clone, Debug)]
pub struct Box {
    /// The left bottom near corner of the box.
    pub left_bot_near: Vec3,
    /// The right top far corner of the box.
    pub right_top_far: Vec3,
}

/// A regular octahedron: a polyhedron with eight triangular faces.
///
/// Has vertices at (±1, 0, 0), (0, ±1, 0), and (0, 0, ±1).
#[derive(Copy, Clone, Debug, Default)]
pub struct Octahedron;

impl Default for Box {
    /// Creates a cube with unit-length edges, centered at the origin.
    fn default() -> Self {
        Self {
            left_bot_near: splat(-0.5),
            right_top_far: splat(0.5),
        }
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
