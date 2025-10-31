//! Procedural noise generation.
//!
//! This module implements two- and three-dimensional Perlin Noise.

use core::array::from_fn;

use super::{Lerp, Point2, Point3, Vec2, Vec3, lerp, smoothstep, vec2, vec3};

pub mod perlin2 {
    use super::*;

    /// Returns the Perlin noise value corresponding to the 2D point.
    pub fn noise(pt: Point2) -> f32 {
        let pt0 = pt.map(f32::floor);
        // Find the four integer-coordinate points around pt
        let grid_pts = grid_pts(pt0);
        // Get the gradient vectors at the grid points
        let grads = grid_pts.map(grad);
        // Get the delta vectors from pt to the grid points
        let deltas = grid_pts.map(|p| pt - p);

        // Compute the dot products between gradients and delts
        let dots = from_fn(|i| grads[i].dot(&deltas[i]));

        // Smooth the interpolation variables
        let tu = (pt - pt0).map(smoothstep).0;
        // Interpolate the final noise value at pt
        bilerp(tu, dots)
    }

    /// Returns the Perlin gradient vector corresponding to the 2D point.
    ///
    /// The vector is computed by taking the gradient vectors of all four
    /// surrounding grid points and smoothly interpolating between them.
    pub fn gradient(pt: Point2) -> Vec2 {
        let pt0 = pt.map(f32::floor);
        // Find the four integer-coordinate points around pt
        let grid_pts = grid_pts(pt0);
        // Get the gradient vectors at the grid points
        let grads = grid_pts.map(grad);
        // Interpolate the gradient at pt
        let tu = (pt - pt0).map(smoothstep).0;
        bilerp(tu, grads)
    }

    fn grid_pts(pt0: Point2) -> [Point2; 4] {
        let pt0 = pt0.map(f32::floor);
        [pt0, pt0 + X, pt0 + Y, pt0 + X + Y]
    }

    fn grad(pt: Point2) -> Vec2 {
        let perm = perm(pt.x() as i32 + perm(pt.y() as i32));
        GRADS_2[perm as usize & 0x7]
    }

    fn _grads(pt: Point2) -> (Vec2, [Vec2; 4]) {
        let pt0 = pt.map(f32::floor);
        let g00 = grad(pt0);
        let g01 = grad(pt0 + Y);
        let g10 = grad(pt0 + X);
        let g11 = grad(pt0 + X + Y);
        (pt - pt0, [g00, g01, g10, g11])
    }

    const X: Vec2 = Vec2::X;
    const Y: Vec2 = Vec2::Y;

    const A: f32 = 1.237;
    const B: f32 = 0.513;
    /// Gradient vectors for 2D noise.
    static GRADS_2: [Vec2; 8] = [
        vec2(A, B),
        vec2(B, A),
        vec2(-B, A),
        vec2(-A, B),
        vec2(-A, -B),
        vec2(-B, -A),
        vec2(B, -A),
        vec2(A, -B),
    ];
}

pub mod perlin3 {
    use super::*;

    /// Returns the Perlin noise value corresponding to the 3D point.
    pub fn noise(pt: Point3) -> f32 {
        let pt0 = pt.map(f32::floor);
        // Find the four integer-coordinate points around pt
        let grid_pts = grid_pts(pt0);
        // Get the gradient vectors at the grid points
        let grads = grid_pts.map(grad);
        // Get the delta vectors from pt to the grid points
        let deltas = grid_pts.map(|p| pt - p);

        // Compute the dot products between gradients and delts
        let dots: [f32; 8] = from_fn(|i| grads[i].dot(&deltas[i]));

        let ([left, right], []) = dots.as_chunks() else {
            unreachable!()
        };

        // Smooth the interpolation variables
        let tuv = (pt - pt0).map(smoothstep).0;
        // Interpolate the final noise value at pt
        trilerp(tuv, left, right)
    }

    /// Returns the Perlin gradient vector corresponding to a 3D point.
    pub fn gradient(pt: Point3) -> Vec3 {
        use crate::math::float::f32;
        let fract = pt - pt.map(f32::floor);
        let (&[left, right], []) = grads3(pt).as_chunks() else {
            unreachable!()
        };
        // Smooth the interpolation variables
        let tuv = fract.map(smoothstep).0;
        // Interpolate the gradient at pt
        trilerp(tuv, &left, &right)
    }

    #[rustfmt::skip]
    fn grid_pts(pt: Point3) -> [Point3; 8] {
        //
        //       011 +--------------+ 111
        //         / |            / |
        //       /   |          /   |
        // 010 +--------------+ 110 |
        //     |     |        |     |
        //     | 001 +--------|-----+ 101
        //     |   /          |   /
        //     | /            | /
        // 000 +--------------+ 100
        //
        let pt0 = pt.map(f32::floor);
        let pt1 = pt0 + Vec3::X;

        [
            pt0,
            pt0 + Y,
            pt0 + Z,
            pt0 + Y + Z,
            pt1,
            pt1 + Y,
            pt1 + Z,
            pt1 + Y + Z,
        ]
    }

    const Y: Vec3 = Vec3::Y;
    const Z: Vec3 = Vec3::Z;

    fn grad(pt: Point3) -> Vec3 {
        let [x, y, z] = pt.0;
        let perm = perm(x as i32 + perm(y as i32 + perm(z as i32)));
        GRADS[perm as usize & 0xF]
    }

    fn grads3(pt: Point3) -> [Vec3; 8] {
        grid_pts(pt).map(grad)
    }

    /// Gradient vectors for 3D noise.
    static GRADS: [Vec3; 16] = [
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
}

fn bilerp<T: Lerp>([t, u]: [f32; 2], [x00, x01, x10, x11]: [T; 4]) -> T {
    lerp(t, x00, x01).lerp(&lerp(t, x10, x11), u)
}
fn trilerp<V: Lerp>([x, y, z]: [f32; 3], left: &[V; 4], right: &[V; 4]) -> V {
    bilerp([y, z], left.lerp(right, x))
}

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

#[cfg(test)]
mod tests {
    use std::{eprint, eprintln, fmt::Write, string::String};

    use crate::math::{pt2, pt3};

    use super::*;

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
        for i in 0..count {
            for j in 0..count {
                let pt = pt2(i as f32 / scale, j as f32 / scale);
                let v = perlin2::noise(pt);
                stats.cum(v);
            }
        }
        let Stats { total, avg, std, min, max } = stats.finish();
        assert_eq!(avg, -0.00055865006);
        assert_eq!(std, 0.28275317);
        assert_eq!(min, -0.8853002);
        assert_eq!(max, 0.8853002);
    }
    #[test]
    fn perlin3_statistics() {
        let count = 100u32;
        let scale = 10.0;
        let mut stats = Stats::new();
        for i in 0..count {
            for j in 0..count {
                for k in 0..count {
                    let pt = pt3(i, j, k).map(|c| c as f32 / scale);
                    let v = perlin3::noise(pt);
                    stats.cum(v);
                }
            }
        }
        let Stats { total, avg, std, min, max } = stats.finish();
        assert_eq!(avg, 0.0003959533);
        assert_eq!(std, 0.24879986);
        assert_eq!(min, -0.8853568);
        assert_eq!(max, 0.87813747);
    }

    const PALETTE: &[u8] = b"   ..,:;=+*odO#%@WW";

    #[test]
    fn perlin2_pattern() {
        #[rustfmt::skip]
        static EXPECTED: &str =
";;;+++ooo***;;;::::::;;;;;;;;;;;;===;;;,,,...,,,
...:::======;;;;;;===***ooo+++===:::,,,......:::
......,,,;;;;;;===***ddddddooo;;;,,,......,,,;;;
.........:::===+++******ooo***;;;,,,,,,:::;;;;;;
;;;,,,...,,,;;;++++++===;;;;;;::::::;;;===;;;;;;
ooo+++::::::===+++===,,,.........:::===+++===;;;
dddooo+++===++++++;;;......   ...,,,;;;******+++
ooo***+++++++++***+++:::......   ...;;;***ddd***
;;;===;;;;;;;;;+++++++++;;;,,,...,,,;;;ooodddooo
...............:::+++oooooo+++;;;===+++oooooo+++
...............,,,===ddddddooo******oooooo+++:::
...............,,,===***ooo************+++:::,,,
;;;;;;;;;===;;;;;;;;;===;;;;;;;;;===;;;:::,,,:::
ooo******+++===;;;:::,,,......,,,:::,,,...,,,;;;
dddooo***+++;;;:::............,,,,,,......,,,===
ooo***+++===;;;,,,.........:::===;;;,,,,,,:::===
";

        const SIZE: usize = 16;
        const SCALE: f32 = 4.0;

        let mut actual = String::new();

        for i in 0..SIZE {
            for j in 0..SIZE {
                let pt = pt2(i as f32 / SCALE, j as f32 / SCALE);
                let val = (perlin2::noise(pt) + 1.0) * 127.0;

                let ch = PALETTE[val as usize / 16];
                _ = write!(actual, "{0}{0}{0}", ch as char);
            }
            _ = writeln!(actual);
        }
        assert_eq!(&actual, EXPECTED);
    }
    #[test]
    fn perlin3_pattern() {
        #[rustfmt::skip]
        static EXPECTED: &str =
"*********OOO***,,,***%%%
%%%ooo*********+++***###
***;;;***;;;***OOO***;;;
,,,:::***;;;;;;+++***ooo
***************;;;***%%%
%%%WWW%%%%%%%%%OOO***+++
***OOO*********OOO***;;;
;;;+++***+++***ooo******

###ooo+++OOOooo,,,***%%%
@@@ooo===***ooo===+++OOO
+++:::===+++dddooo===;;;
...,,,***++++++++++++ooo
;;;+++ddd***===;;;+++###
OOO######%%%%%%ddd***;;;
***ooo===oooddd###ooo;;;
******ooo***ooodddooo+++

%%%ooo***OOO***,,,***%%%
###***+++ooo***:::+++OOO
;;;:::;;;oooOOO+++;;;;;;
   ,,,***oooooo*********
,,,;;;OOO***;;;;;;;;;***
*********OOOOOOooo+++:::
******;;;OOOOOO###OOO;;;
OOOooooooooo***oooooo+++

###***oooOOO+++...***%%%
ddd+++***ddd===...+++OOO
======+++###ooo:::++++++
...:::***######dddOOOooo
;;;;;;ooo***+++***======
+++;;;:::oooooo***===:::
******+++%%%ooodddddd+++
###ooooooooo;;;+++***===
";
        const SIZE: usize = 8;
        const SCALE: f32 = 2.0;

        let mut actual = String::new();

        for k in 0..4 {
            for i in 0..SIZE {
                for j in 0..SIZE {
                    let pt =
                        pt3(i as f32 / SCALE, j as f32 / SCALE, k as f32 / 4.0);
                    let val = (perlin3::noise(pt) + 1.0) * 127.0;

                    let ch = PALETTE[val as usize / 12];
                    _ = write!(actual, "{0}{0}{0}", ch as char);
                }
                _ = writeln!(actual);
            }
            _ = writeln!(actual);
        }
    }
}
