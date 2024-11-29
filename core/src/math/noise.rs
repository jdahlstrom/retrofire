//! Procedural noise generation.
//!
//! This module implements two- and three-dimensional Perlin Noise.

use super::{
    spline::smoothstep,
    vary::lerp,
    vec::{splat, vec2, vec3, Vec2, Vec3},
    Vary,
};
use crate::math::vary::bilerp;
use std::array;

/// Returns the Perlin noise value corresponding to the 2D point.
pub fn perlin2(pt: Vec2) -> f32 {
    use super::float::f32;
    let fract = pt - pt.map(f32::floor);
    let [fx, fy] = fract.0;
    let [g00, g01, g10, g11] = grads2(pt);

    let d00 = g00.dot(&vec2(fx, fy));
    let d01 = g01.dot(&vec2(fx, fy - 1.0));
    let d10 = g10.dot(&vec2(fx - 1.0, fy));
    let d11 = g11.dot(&vec2(fx - 1.0, fy - 1.0));

    let t = fract.map(smoothstep);
    bilerp(t.x(), t.y(), d00, d10, d01, d11)
}

/// Returns the Perlin gradient vector corresponding to the 2D point.
///
/// The vector is computed by taking the gradient vectors of all four
/// surrounding grid points and smoothly interpolating between them.
pub fn perlin2v(pt: Vec2) -> Vec2 {
    use super::float::f32;
    let fract = pt - pt.map(f32::floor);
    let [g00, g01, g10, g11] = grads2(pt);
    let t = fract.map(smoothstep);
    bilerp(t.x(), t.y(), g00, g10, g01, g11)
}

trait ArrayExt<T> {
    /// Splits `self` to arrays of length `I` and `J`.
    fn split_to<const I: usize, const J: usize>(&self) -> (&[T; I], &[T; J]);
}

impl<T, const N: usize> ArrayExt<T> for [T; N] {
    fn split_to<const I: usize, const J: usize>(&self) -> (&[T; I], &[T; J]) {
        const {
            assert!(I + J <= N);
        }
        let (a, b) = self[..I + J].split_at(I);
        (a.try_into().unwrap(), b.try_into().unwrap())
    }
}

/// Returns the Perlin noise value corresponding to the 3D point.
pub fn perlin3(pt: Vec3) -> f32 {
    use super::float::f32;
    let pt0 = pt.map(f32::floor);
    let fract = pt - pt0;
    let grads = grads3(pt0);

    let [fx, fy, fz] = array::from_fn(|i| [fract[i], fract[i] - 1.0]);

    let yzs: [_; 4] = array::from_fn(|i| [fy[(i >> 1) & 1], fz[i & 1]]);

    let (left, right) = grads.split_to::<4, 4>();

    let left: [f32; 4] = array::from_fn(|i| {
        let [y, z] = yzs[i];
        left[i].dot(&vec3(fx[0], y, z))
    });
    let right: [f32; 4] = array::from_fn(|i| {
        let [y, z] = yzs[i];
        right[i].dot(&vec3(fx[1], y, z))
    });

    trilerp(fract, left, right)
}

/// Returns the Perlin gradient vector corresponding to the 3D point.
pub fn perlin3v(pt: Vec3) -> Vec3 {
    use super::float::f32;
    let fract = pt - pt.map(f32::floor);
    let [lbn, lbf, ltn, ltf, rbn, rbf, rtn, rtf] = grads3(pt);
    trilerp(fract, [lbn, lbf, ltn, ltf], [rbn, rbf, rtn, rtf])
}

fn trilerp<V: Vary>(pt: Vec3, left: [V; 4], right: [V; 4]) -> V {
    let [x, y, z] = pt.0.map(smoothstep);
    let [bot_near, bot_far, top_near, top_far] = lerp(x, left, right);
    bilerp(y, z, bot_near, top_near, bot_far, top_far)
}

fn perm(x: i32) -> i32 {
    PERM[(x & 0xFF) as usize] as i32
}

fn grad2(x: f32, y: f32) -> Vec2 {
    let perm = perm(x as i32 + perm(y as i32));
    GRADS_2[perm as usize & 0x7]
}

fn grads2(pt: Vec2) -> [Vec2; 4] {
    use super::float::f32;
    let [x0, y0] = pt.map(f32::floor).0;

    let g00 = grad2(x0, y0);
    let g01 = grad2(x0, y0 + 1.0);
    let g10 = grad2(x0 + 1.0, y0);
    let g11 = grad2(x0 + 1.0, y0 + 1.0);

    [g00, g01, g10, g11]
}

fn grad3(x: f32, y: f32, z: f32) -> Vec3 {
    let perm = perm(x as i32 + perm(y as i32 + perm(z as i32)));
    GRADS_3[perm as usize & 0xF]
}

fn grads3(pt0: Vec3) -> [Vec3; 8] {
    //
    //       011 +--------------+ 111
    //         / |            / |
    //       /   |          /   |
    // 010 +--------------+ 110 |
    //     |     |        |     |
    //     |     +--------|-----+ 101
    //     |   /          |   /
    //     | /            | /
    // 000 +--------------+ 100
    //

    let pts = [pt0, pt0 + splat(1.0)];
    array::from_fn(|i| {
        let x = pts[(i >> 2) & 1].x();
        let y = pts[(i >> 1) & 1].y();
        let z = pts[i & 1].z();
        grad3(x, y, z)
    })
}

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

/// Gradient vectors for 3D noise.
static GRADS_3: [Vec3; 16] = [
    //
    vec3(0.0, -1.0, -1.0),
    vec3(0.0, -1.0, 1.0),
    vec3(0.0, 1.0, -1.0),
    vec3(0.0, 1.0, 1.0),
    //
    vec3(-1.0, 0.0, -1.0),
    vec3(-1.0, 0.0, 1.0),
    vec3(1.0, 0.0, -1.0),
    vec3(1.0, 0.0, 1.0),
    //
    vec3(-1.0, -1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(1.0, -1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    //
    vec3(1.0, 1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(0.0, -1.0, 1.0),
    vec3(0.0, -1.0, -1.0),
];

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
    use super::*;

    #[test]
    fn perlin2_statistics() {
        let count = 1000u32;
        let div = 10.0;
        let [mut avg, mut min, mut max, mut var] = [0.0f32, 1.0, -1.0, 0.0];
        for i in 0..count {
            for j in 0..count {
                let v = perlin2(vec2(i as f32, j as f32) / div);
                avg += v;
                var += v * v;
                min = min.min(v);
                max = max.max(v);
            }
        }
        let c = count.pow(2) as f32;
        avg /= c;
        var /= c;

        assert_eq!(avg, -0.0005586566);
        assert_eq!(min, -0.8853002);
        assert_eq!(max, 0.8853002);
        assert_eq!(var, 0.07994927);
    }
    #[test]
    fn perlin3_statistics() {
        let count = 100u32;
        let div = 10.0;
        let [mut avg, mut min, mut max, mut var] = [0.0f32, 1.0, -1.0, 0.0];
        for i in 0..count {
            for j in 0..count {
                for k in 0..count {
                    let pt = vec3(i, j, k).map(|c| c as f32) / div;
                    let v = perlin3(pt);

                    avg += v;
                    var += v * v;
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }
        let c = count.pow(3) as f32;
        avg /= c;
        var /= c;

        assert_eq!(avg, 0.00039595732);
        assert_eq!(min, -0.8853568);
        assert_eq!(max, 0.8781377);
        assert_eq!(var, 0.06190137);
    }
}
