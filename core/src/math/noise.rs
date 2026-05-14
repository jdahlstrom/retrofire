//! Procedural noise generation.
//!
//! This module implements two- and three-dimensional Perlin noise.

use core::{array::from_fn, cell::Cell};

use super::{
    Lerp, Point, Point2, Point3, Vec2, Vec3, Vector, lerp, pt3, space::Real,
    splat, spline::smoothstep_unit, vec2, vec3,
};

/// 2D-dimensional Perlin noise generator.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Perlin2 {
    pub seed: u8,

    // Cache the grid points and gradients because usually several
    // consecutive evaluations are likely to hit the same grid square
    cache: Cell<GridCell2>,
}

/// 3-dimensional Perlin noise generator.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Perlin3 {
    pub seed: u8,

    // Cache the grid points and gradients because usually several
    // consecutive evaluations are likely to hit the same grid square
    cache: Cell<GridCell3>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct GridCell<const DIM: usize, const N: usize> {
    pts: [Point<[f32; DIM], Real<DIM>>; N],
    grads: [Vector<[f32; DIM], Real<DIM>>; N],
}

type GridCell2 = GridCell<2, 4>;
type GridCell3 = GridCell<3, 8>;

impl Perlin2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: u8) -> Self {
        Self { seed, ..Self::default() }
    }

    /// Returns the Perlin noise value corresponding to a 2D point.
    #[inline]
    pub fn eval(&self, pt: Point2) -> f32 {
        let GridCell { pts, grads } = self.grid_cell(pt);

        // Get the delta vectors from pt to the grid points
        let deltas = pts.map(|p| pt - p);

        // Compute the dot products between gradients and deltas
        let dots = from_fn(|i| grads[i].dot(&deltas[i]));

        // Smooth the interpolation variables
        let tu = deltas[0].map(smoothstep_unit).0;
        // Interpolate the final noise value at pt
        bilerp(tu, dots)
    }

    /// Returns the Perlin gradient vector corresponding to a 2D point.
    ///
    /// The vector is computed by taking the gradient vectors of all four
    /// surrounding grid points and smoothly interpolating between them.
    #[inline]
    pub fn gradient(&self, pt: Point2) -> Vec2 {
        let GridCell { pts, grads } = self.grid_cell(pt);
        let tu = (pt - pts[0]).map(smoothstep_unit).0;
        bilerp(tu, grads)
    }

    #[inline]
    fn grid_cell(&self, pt: Point2) -> GridCell2 {
        let pt0 = pt.map(f32::floor);

        let mut cached = self.cache.get();
        // Update cache if we're not in the same grid cell
        if cached.pts[0] != pt0 {
            // Find the four integer-coordinate points around pt
            cached.pts = Self::grid_pts(pt0);
            // Get the gradient vectors at the grid points
            cached.grads = cached.pts.map(|p| self.grad(p));
            self.cache.set(cached);
        }
        cached
    }

    /// Returns the four integer-coordinate grid points around a point.
    #[inline]
    fn grid_pts(pt0: Point2) -> [Point2; 4] {
        [pt0, pt0 + Vec2::X, pt0 + Vec2::Y, pt0 + Vec2::X + Vec2::Y]
    }

    /// Returns the gradient vector at a grid point.
    #[inline]
    fn grad(&self, pt: Point2) -> Vec2 {
        let hash = PERM[pt.y() as i32 as u8 as usize];
        let hash = PERM[(pt.x() as i32 as u8).wrapping_add(hash) as usize];
        let hash = PERM[self.seed.wrapping_add(hash) as usize];
        GRADS_2[hash as usize & 0x7]
    }
}
/// Gradient vectors for 2D noise.
static GRADS_2: [Vec2; 8] = const {
    const A: f32 = 1.237;
    const B: f32 = 0.513;
    [
        vec2(A, B),
        vec2(B, A),
        vec2(-B, A),
        vec2(-A, B),
        vec2(-A, -B),
        vec2(-B, -A),
        vec2(B, -A),
        vec2(A, -B),
    ]
};

impl Perlin3 {
    /// Returns the Perlin noise value corresponding to a 3D point.
    // #[inline] Benchmarks appear to indicate that this reduces perf
    pub fn eval(&self, pt: Point3) -> f32 {
        let GridCell { pts, grads } = self.grid_cell(pt);

        // Get the delta vectors from pt to the grid points
        let deltas = pts.map(|p| pt - p);

        // Compute the dot products between gradients and delts
        let dots: [f32; 8] = from_fn(|i| grads[i].dot(&deltas[i]));

        // Smooth the interpolation variables
        let tuv = deltas[0].map(smoothstep_unit).0;

        // Interpolate the final noise value at pt
        trilerp(tuv, dots)
    }

    /// Returns the Perlin gradient vector corresponding to a 3D point.
    #[inline]
    pub fn gradient(&self, pt: Point3) -> Vec3 {
        let GridCell { pts, grads } = self.grid_cell(pt);

        // Smooth the interpolation variables
        let tuv = (pt - pts[0]).map(smoothstep_unit).0;
        // Interpolate the gradient at pt
        trilerp(tuv, grads)
    }

    #[inline]
    fn grid_cell(&self, pt: Point3) -> GridCell3 {
        let pt0 = pt.map(f32::floor);

        let mut cached = self.cache.get();
        // Update cache if we're not in the same grid cell
        if cached.pts[0] != pt0 {
            // Find the four integer-coordinate points around pt
            cached.pts = Self::grid_pts(pt0);
            // Get the gradient vectors at the grid points
            cached.grads = cached.pts.map(Self::grad);
            self.cache.set(cached);
        }
        cached
    }

    #[inline]
    #[rustfmt::skip]
    fn grid_pts(pt0: Point3) -> [Point3; 8] {
        //
        //       011 +--------------+ 111
        //         / |            / |
        //       /   |          /   |
        // 010 +--------------+ 110 |
        //     |     |        |     |
        //     | 001 +--------|-----+ 101
        //     |   /          |   /       y   z
        //     | /            | /         | /
        // 000 +--------------+ 100       O--- x
        //
        let [x0, y0, z0] = pt0.0;
        let [x1, y1, z1] = (pt0 + splat(1.0)).0;
        [
            pt3(x0, y0, z0), pt3(x0, y1, z0), pt3(x0, y0, z1), pt3(x0, y1, z1),
            pt3(x1, y0, z0), pt3(x1, y1, z0), pt3(x1, y0, z1), pt3(x1, y1, z1),
        ]
    }

    #[inline]
    fn grad(pt: Point3) -> Vec3 {
        let [x, y, z] = pt.0;
        let perm = perm(x as i32 + perm(y as i32 + perm(z as i32)));
        GRADS_3[perm as usize & 0xF]
    }
}
/// Gradient vectors for 3D noise.
static GRADS_3: [Vec3; 16] = [
    // YZ plane
    vec3(0.0, -1.0, -1.0),
    vec3(0.0, -1.0, 1.0),
    vec3(0.0, 1.0, -1.0),
    vec3(0.0, 1.0, 1.0),
    // XZ plane
    vec3(-1.0, 0.0, -1.0),
    vec3(-1.0, 0.0, 1.0),
    vec3(1.0, 0.0, -1.0),
    vec3(1.0, 0.0, 1.0),
    // XY plane
    vec3(-1.0, -1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(1.0, -1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    // Pad to power of two
    vec3(1.0, 1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(0.0, -1.0, 1.0),
    vec3(0.0, -1.0, -1.0),
];

#[inline]
fn bilerp<T: Lerp>([t, u]: [f32; 2], [x00, x01, x10, x11]: [T; 4]) -> T {
    lerp(t, x00, x01).lerp(&lerp(t, x10, x11), u)
}

#[inline]
fn trilerp<V: Lerp>(
    [t, u, v]: [f32; 3],
    [v000, v001, v010, v011, v100, v101, v110, v111]: [V; 8],
) -> V {
    bilerp(
        [u, v],
        [v000, v001, v010, v011].lerp(&[v100, v101, v110, v111], t),
    )
}

#[inline]
fn perm(x: i32) -> i32 {
    PERM[(x & 0xFF) as usize] as i32
}

/// Permutation table for calculating a pseudo-random index for each grid point.
#[rustfmt::skip]
static PERM: [u8; 256] = [
    156, 2, 157, 90, 75, 199, 55, 167, 62, 92, 101, 253, 66, 134, 113, 83,
    1, 136, 78, 106, 254, 105, 248, 176, 234, 5, 195, 226, 49, 71, 87, 44,
    122, 94, 219, 140, 72, 159, 237, 212, 8, 162, 200, 124, 125, 69, 165, 74,
    245, 42, 89, 216, 158, 108, 238, 184, 217, 73, 126, 210, 14, 111, 19, 188,
    186, 45, 38, 223, 35, 112, 214, 26, 145, 95, 99, 193, 250, 189, 152, 182,
    166, 247, 148, 213, 168, 70, 96, 249, 127, 132, 4, 137, 41, 60, 102, 28,
    27, 240, 227, 155, 211, 230, 9, 80, 178, 3, 68, 153, 143, 84, 179, 181,
    12, 97, 103, 16, 225, 146, 63, 82, 203, 175, 163, 147, 11, 116, 185, 215,
    57, 120, 208, 129, 115, 198, 37, 201, 39, 98, 20, 183, 56, 118, 109, 142,
    138, 65, 117, 114, 160, 25, 43, 191, 204, 161, 22, 251, 139, 79, 131, 231,
    76, 0, 205, 206, 244, 51, 174, 13, 110, 85, 209, 77, 64, 53, 48, 221,
    133, 93, 224, 24, 33, 164, 23, 47, 171, 128, 243, 18, 52, 119, 149, 100,
    246, 233, 31, 192, 252, 190, 15, 172, 91, 229, 144, 54, 61, 58, 220, 36,
    222, 29, 50, 88, 121, 173, 232, 194, 239, 197, 32, 180, 107, 46, 7, 130,
    169, 81, 218, 67, 21, 170, 187, 59, 86, 235, 154, 123, 150, 177, 135, 228,
    104, 242, 6, 151, 255, 34, 30, 141, 202, 196, 236, 207, 241, 40, 17, 10
];

impl<const DIM: usize, const N: usize> Default for GridCell<DIM, N>
where
    [f32; DIM]: Default,
{
    fn default() -> Self {
        Self {
            pts: [Point::new([f32::NAN; DIM]); N],
            grads: [Vector::default(); N],
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use core::fmt::Write;

    use crate::math::{pt2, pt3};

    use super::*;

    #[derive(PartialEq, Debug)]
    struct Stats {
        total: f32,
        avg: f32,
        std: f32,
        min: f32,
        max: f32,
    }
    impl Stats {
        fn new() -> Self {
            Self {
                total: 0.0,
                avg: 0.0,
                std: 0.0,
                min: f32::MAX,
                max: f32::MIN,
            }
        }
        fn cum(&mut self, v: f32) {
            self.total += 1.0;
            self.avg += v;
            self.std += v * v;
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        fn finish(self) -> Self {
            Self {
                avg: self.avg / self.total,
                std: (self.std / self.total).sqrt(),
                ..self
            }
        }
    }

    #[test]
    fn perlin2_statistics() {
        let count = 1000u32;
        let scale = 10.0;
        let mut stats = Stats::new();
        let p = Perlin2::default();
        for i in 0..count {
            for j in 0..count {
                let pt = pt2(i as f32 / scale, j as f32 / scale);
                let v = p.eval(pt);
                stats.cum(v);
            }
        }
        let stats = stats.finish();

        assert_eq!(
            stats,
            Stats {
                total: 1000f32.powi(2),
                avg: -0.00023739079,
                std: 0.28690845,
                min: -0.87925243,
                max: 0.87925243
            }
        );
    }
    #[test]
    fn perlin3_statistics() {
        let count = 100u32;
        let scale = 10.0;
        let mut stats = Stats::new();
        let p = Perlin3::default();
        for i in 0..count {
            for j in 0..count {
                for k in 0..count {
                    let pt = pt3(i, j, k).map(|c| c as f32 / scale);
                    let v = p.eval(pt);
                    stats.cum(v);
                }
            }
        }
        let stats = stats.finish();
        assert_eq!(
            stats,
            Stats {
                total: 100f32.powi(3),
                avg: 0.00037712423,
                std: 0.26199436,
                min: -0.890156,
                max: 0.89380443
            }
        );
    }

    const PALETTE: &[u8] = b"   ..,:;=+*odO#%@WW";

    #[test]
    fn perlin2_pattern() {
        #[rustfmt::skip]
        static EXPECTED: &str = "\
+++===+++***++++++oooooo+++:::,,,;;;++++++;;;;;;
oooooodddOOOOOOddddddddd***;;;,,,,,,;;;======+++
ooo***ddd%%%%%%OOOooo++++++===::::::;;;+++oooddd
+++;;;+++dddOOO***;;;:::===***===;;;+++ddd###OOO
+++;;;;;;++++++;;;,,,:::+++ooo+++;;;+++dddOOOooo
***;;;:::::::::,,,,,,===oooddd+++;;;===oooooo===
+++===:::,,,,,,:::===oooOOO###***;;;+++ooo***:::
===+++;;;:::;;;+++oooooodddddd+++===***dddddd===
+++ooo+++;;;+++oooooo+++++++++;;;;;;+++dddOOOooo
oooOOOooo***oooddd***;;;;;;;;;::::::===ooo######
OOO%%%###OOOOOOOOO+++:::;;;++++++;;;;;;===ddd%%%
ddd######OOOddd***;;;;;;+++dddOOOooo===:::+++ddd
+++***ooo***+++;;;,,,:::+++OOO%%%OOO+++;;;;;;+++
;;;======;;;,,,......:::+++ddd###ddd***===::::::
;;;+++;;;.........:::+++ooooooooo+++++++++;;;,,,
+++ooo+++:::,,,;;;***dddooo***;;;:::===******===
";
        const SIZE: usize = 16;
        const SCALE: f32 = 4.0;

        let mut actual = String::new();
        let p = Perlin2::default();
        for i in 0..SIZE {
            for j in 0..SIZE {
                let pt = pt2(i as f32 / SCALE, j as f32 / SCALE);
                let val = p.eval(pt) * 0.5 + 0.5;

                let ch = PALETTE[(val * PALETTE.len() as f32) as usize];
                _ = write!(actual, "{0}{0}{0}", ch as char);
            }
            _ = writeln!(actual);
        }
        assert_eq!(&actual, EXPECTED);
    }
    #[test]
    fn perlin3_pattern() {
        #[rustfmt::skip]
        static EXPECTED: &str = "\
+++++++++ooo+++...+++###
###***+++++++++===+++OOO
+++;;;+++;;;+++ooo+++;;;
...,,,+++;;;;;;===+++***
+++++++++++++++;;;+++###
###%%%#########ooo+++===
+++ooo+++++++++ooo+++;;;
;;;===+++===+++***++++++

ddd+++===ooo***,,,+++###
###***;;;+++ooo======ddd
===,,,;;;;;;oooooo;;;:::
...,,,+++;;;;;;======+++
:::===ooo+++;;;:::===OOO
dddOOOOOOOOOOOOooo+++;;;
+++***;;;+++oooOOO***:::
++++++++++++***ooo+++===

###***+++ooo+++...+++###
OOO+++===***+++,,,===ooo
;;;,,,;;;***ooo===;;;;;;
   ...+++******+++++++++
...;;;ooo+++;;;;;;;;;+++
+++++++++oooooo***===,,,
++++++;;;ooooooOOOooo;;;
ooo*********+++******===

ddd+++***ooo===...+++###
***===+++ooo;;;...+++ooo
;;;;;;===OOO***,,,======
...,,,+++OOOOOOoooddd+++
::::::***+++===+++;;;:::
===:::,,,+++++++++;;;,,,
++++++===OOO***oooooo===
ddd*********:::===+++===

";

        const SIZE: usize = 8;
        const SCALE: f32 = 2.0;

        let mut actual = String::new();
        let p = Perlin3::default();

        for k in 0..4 {
            for i in 0..SIZE {
                for j in 0..SIZE {
                    let pt =
                        pt3(i as f32 / SCALE, j as f32 / SCALE, k as f32 / 4.0);
                    let val = p.eval(pt) * 0.5 + 0.5;

                    let ch = PALETTE[(val * PALETTE.len() as f32) as usize];
                    _ = write!(actual, "{0}{0}{0}", ch as char);
                }
                _ = writeln!(actual);
            }
            _ = writeln!(actual);
        }

        assert_eq!(actual, EXPECTED);
    }
}
