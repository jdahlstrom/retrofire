use crate::solids::Topology::{Flat, Spherical, Toroidal};
use retrofire_core::{
    geom::{Builder, Mesh},
    math::{Vary, pt3},
};

pub struct Lattice {
    pub rows: u32,
    pub cols: u32,
    pub topology: Topology,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Topology {
    /// With boundaries on each side.
    #[default]
    Flat,

    /// Left and right sides are identified, "wrapping around". Top vertices
    /// are identified with each other and bottom vertices with each other,
    /// forming two special points, "poles", each connected to cols triangular
    /// faces.
    Spherical,

    /// Left and right sides are identified, "wrapping around". Top and
    /// bottom sides are boundaries.
    Cylindrical,

    /// Left and right sides are identified with each other, as are top and
    /// bottom sides with each other. The lattice "wraps around" on each
    /// side, making it topologically a manifold.
    Toroidal,
}

impl Lattice {
    pub fn new(cols: u32, rows: u32) -> Self {
        Self {
            cols,
            rows,
            topology: Topology::default(),
        }
    }
    /// Creates a rectangular lattice mesh.
    ///
    /// There are `cols` × `rows` rectangular faces, each made of two triangles,
    /// with `cols+1` × `rows+1` vertices in total. The faces and vertices are
    /// stored in row-major order in the mesh.
    ///
    /// If either `cols` or `rows` is zero, returns a mesh with no faces or
    /// vertices.
    ///
    /// ```text
    ///     1     2    3..cols   cols
    /// 1,0----+-----+ · · · · +----0,1
    ///  | \   |   / |         |     |
    ///  |   \ | /   |         |     |   rows
    ///  +-----+-----+         +-----+
    ///  ·                           ·
    ///  ·                           ·   2..rows
    ///  ·                           ·
    ///  +-----+-----+         +-----+
    ///  |     |     |         |     |
    ///  |     |     |         |     |    1
    /// 0,0----+-----+ · · · · +----0,1
    /// ```
    pub fn builder(&self) -> Builder<()> {
        // TODO precompute capacity
        let mut bld = Mesh::builder();

        // TODO cols=1 or rows=1 may also be degenerate, depending on topology
        if self.cols == 0 || self.rows == 0 {
            return bld;
        }

        // Topology:
        // * Rectangular: rows+1 vertex rows, cols+1 vertex cols
        // * Cylindrical: rows+1 vertex rows, cols vertex cols
        // * Spherical: rows-1 vertex rows plus poles, cols vertex cols
        // * Toroidal: rows vertex rows, cols vertex cols

        let f_rows = self.rows;
        let f_cols = self.cols;

        use Topology::*;
        let (v_rows, v_cols) = match self.topology {
            Flat => (f_rows + 1, f_cols + 1),
            Spherical => (f_rows + 1, f_cols),
            Cylindrical => (f_rows + 1, f_cols),
            Toroidal => (f_rows, f_cols),
        };

        // Depending on topology, don't generate vertices for x=1 and/or y=1
        // if they're supposed to wrap to x=0/y=0
        // TODO change to using range vary
        for y in 0.0.vary_to(1.0, f_rows + 1).take(v_rows as usize) {
            if self.topology == Spherical && y == 0.0 {
                // "South pole"
                bld.push_vert(pt3(0.5, 0.0, 0.0), ());
            } else if self.topology == Spherical && y == 1.0 {
                // "North pole"
                bld.push_vert(pt3(0.5, 1.0, 0.0), ());
            } else {
                for x in 0.0.vary_to(1.0, f_cols + 1).take(v_cols as usize) {
                    bld.push_vert(pt3(x, y, 0.0), ())
                }
            }
        }

        for row in 0..self.rows {
            if self.topology == Spherical && row == 0 {
                // Bottom row faces are triangles connected to the south pole:
                for col in 0..self.cols {
                    let a = 0;
                    let b = (1 + (col + 1) % v_cols) as usize;
                    let c = 1 + col as usize;

                    bld.push_face(a, b, c)
                }

                continue;
            }

            // Index of first col of the current row
            let row_start = match self.topology {
                // First row only has the pole vertex
                Spherical => {
                    if row == 0 {
                        0
                    } else {
                        1 + (row - 1) * v_cols
                    }
                }
                _ => row * v_cols,
            };

            if self.topology == Spherical && row == self.rows - 1 {
                // Top row faces are triangles connected to the north pole:
                for col in 0..self.cols {
                    // First row only has the pole vertex
                    let a = row_start + col;
                    let b = row_start + (col + 1) % v_cols;
                    let c = bld.mesh.verts.len() - 1;
                    bld.push_face(a as _, b as _, c);
                }
                continue;
            }

            for col in 0..self.cols {
                // c---d
                // |   |
                // a---b
                let a = (row_start + col) as usize;

                let next_col = match self.topology {
                    Flat => col + 1,
                    // Non-flats wrap last column to first column
                    _ => (col + 1) % v_cols,
                };
                let next_row_start = match self.topology {
                    // Toroidal wraps last row to first row
                    Toroidal => (row + 1) % v_rows * v_cols,
                    _ => row_start + v_cols,
                };

                let b = (row_start + next_col) as usize;
                let c = (next_row_start + col) as usize;
                let d = (next_row_start + next_col) as usize;

                // Either four or eight edges meet at each (internal) vertex:
                if row & 1 == col & 1 {
                    //  c---d
                    //  | / |
                    //  a---b
                    bld.push_face(a, b, d);
                    bld.push_face(a, d, c);
                } else {
                    //  c---d
                    //  | \ |
                    //  a---b
                    bld.push_face(a, b, c);
                    bld.push_face(b, d, c);
                }
            }
        }
        bld
    }
}
