//! Procedural noise generation.

use super::{
    spline::smoothstep,
    vary::lerp,
    vec::{splat, vec2, vec3, Vec2, Vec3, Vector},
};

/// Returns the Perlin noise value corresponding to the 2D point.
pub fn perlin2(pt: Vec2) -> f32 {
    use super::float::f32;
    let f = pt - pt.map(f32::floor);
    let [fx, fy] = f.0;

    let [g00, g01, g10, g11] = grads2(pt);

    let d00 = g00.dot(&vec2(fx, fy));
    let d01 = g01.dot(&vec2(fx, fy - 1.0));
    let d10 = g10.dot(&vec2(fx - 1.0, fy));
    let d11 = g11.dot(&vec2(fx - 1.0, fy - 1.0));

    let t = f.map(smoothstep);
    let (a, b) = lerp(t.x(), (d00, d01), (d10, d11));
    lerp(t.y(), a, b)
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
    let (a, b) = lerp(t.x(), (g00, g01), (g10, g11));
    lerp(t.y(), a, b)
}

/// Returns the Perlin noise value corresponding to the 3D point.
pub fn perlin3(pt: Vec3) -> f32 {
    use super::float::f32;
    let pt0 = pt.map(f32::floor);
    let f = pt - pt0;
    let i_f: Vec3 = f - splat(1.0);

    let [fx, fy, fz] = f.0;
    let [ifx, ify, ifz] = i_f.0;

    let [g000, g001, g010, g011, g100, g101, g110, g111] = grads3(pt0);

    let d000 = g000.dot(&vec3(fx, fy, fz));
    let d001 = g001.dot(&vec3(fx, fy, ifz));
    let d010 = g010.dot(&vec3(fx, ify, fz));
    let d011 = g011.dot(&vec3(fx, ify, ifz));
    let d100 = g100.dot(&vec3(ifx, fy, fz));
    let d101 = g101.dot(&vec3(ifx, fy, ifz));
    let d110 = g110.dot(&vec3(ifx, ify, fz));
    let d111 = g111.dot(&vec3(ifx, ify, ifz));

    let t: Vec3 = f.map(smoothstep);

    let x0 = Vector::<_>::new([d000, d010, d001, d011]);
    let x1 = Vector::<_>::new([d100, d110, d101, d111]);
    let [y0, y1, y2, y3] = lerp(t.x(), x0, x1).0;
    let (z0, z1) = lerp(t.y(), (y0, y2), (y1, y3));
    lerp(t.z(), z0, z1)
}

/// Returns the Perlin gradient vector corresponding to the 3D point.
pub fn perlin3v(pt: Vec3) -> Vec3 {
    use super::float::f32;
    let fract = pt - pt.map(f32::floor);

    let [g000, g001, g010, g011, g100, g101, g110, g111] = grads3(pt);

    let t: Vec3 = fract.map(smoothstep);

    let x0 = ((g000, g010), (g001, g011));
    let x1 = ((g100, g110), (g101, g111));
    let ((y0, y1), (y2, y3)) = lerp(t.x(), x0, x1);
    let (z0, z1) = lerp(t.y(), (y0, y2), (y1, y3));
    lerp(t.z(), z0, z1)
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
    let pt1 = pt0 + splat(1.0);
    let [x0, y0, z0] = pt0.0;
    let [x1, y1, z1] = pt1.0;

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

    let g000 = grad3(x0, y0, z0);
    let g001 = grad3(x0, y0, z1);
    let g010 = grad3(x0, y1, z0);
    let g011 = grad3(x0, y1, z1);
    let g100 = grad3(x1, y0, z0);
    let g101 = grad3(x1, y0, z1);
    let g110 = grad3(x1, y1, z0);
    let g111 = grad3(x1, y1, z1);

    [g000, g001, g010, g011, g100, g101, g110, g111]
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
