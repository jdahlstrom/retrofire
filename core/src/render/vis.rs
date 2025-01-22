//! Visualization and debugging aids.

use alloc::vec::Vec;

use crate::{
    geom::{vertex, Vertex3},
    math::{
        mat::RealToProj, pt3, rgba, spherical, splat, turns, vec3, Color4f,
        Mat4x4, Point3, Vary, Vec3,
    },
    render::{raster::Frag, shader, Context, Shader},
};

/// Returns a color visualizing the direction of a vector.
///
/// # Examples
/// ```
/// use retrofire_core::math::{rgba, vec3};
/// use retrofire_core::render::vis::vec_to_rgb;
///
/// let right = vec3(1.0, 0.0, 0.0);
/// assert_eq!(vec_to_rgb(right), rgba(1.0, 0.5, 0.5, 1.0));
///
/// let down = vec3(0.0, -1.0, 0.0);
/// assert_eq!(vec_to_rgb(down), rgba(0.5, 0.0, 0.5, 1.0));
///
/// ```
pub fn vec_to_rgb<B>(v: Vec3<B>) -> Color4f {
    let c = 0.5 * v + splat(0.5);
    rgba(c.x(), c.y(), c.z(), 1.0)
}

type Batch<B, S> =
    super::Batch<[usize; 2], Vertex3<Color4f, B>, (), S, (), Context>;

/// Returns a `Batch` drawing a cuboid with the given opposite vertices.
pub fn cuboid<'a, B: 'static>(
    v0: Point3<B>,
    v1: Point3<B>,
) -> Batch<
    B,
    impl Shader<Vertex3<Color4f, B>, Color4f, &'a Mat4x4<RealToProj<B>>>,
> {
    let [x0, y0, z0] = v0.0;
    let [x1, y1, z1] = v1.0;
    #[rustfmt::skip]
    let vs = [
        v0, pt3(x0, y0, z1),
        pt3(x0, y1, z1), pt3(x0, y1, z0),
        pt3(x1, y0, z0), pt3(x1, y0, z1),
        v1, pt3(x1, y1, z0),
    ]
    .map(|p| vertex(p, vec_to_rgb(p.to_vec())));
    #[rustfmt::skip]
    let es = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    super::Batch::new()
        .primitives(&es)
        .vertices(&vs)
        .shader(shader())
}

/// Returns a `Batch` drawing a wireframe sphere with the given center and
/// radius.
///
/// The sphere is represented by three circles lying on the XY, XZ, and
/// YZ planes (in the coordinate space with basis `B`).
pub fn sphere<'a, B: 'static>(
    o: Point3<B>,
    r: f32,
) -> Batch<
    B,
    impl Shader<Vertex3<Color4f, B>, Color4f, &'a Mat4x4<RealToProj<B>>>,
> {
    const RES: usize = 64;

    let vs: Vec<_> = 0.0
        .vary_to(1.0, RES as u32 + 1)
        .flat_map(|a| {
            let xz = spherical(r, turns(a), turns(0.0)).to_cart();
            let xy = vec3(xz.x(), xz.z(), 0.0);
            let yz = vec3(0.0, xz.z(), xz.x());
            [
                vertex(o + xy, vec_to_rgb(xy)),
                vertex(o + xz, vec_to_rgb(xz)),
                vertex(o + yz, vec_to_rgb(yz)),
            ]
        })
        .collect();

    let es: Vec<_> = (0..RES)
        .flat_map(|i| {
            [
                [3 * i, 3 * i + 3],
                [3 * i + 1, 3 * i + 4],
                [3 * i + 2, 3 * i + 5],
            ]
        })
        .collect();

    super::Batch::new()
        .primitives(&es)
        .vertices(&vs)
        .shader(shader())
}

fn shader<'a, B>(
) -> impl Shader<Vertex3<Color4f, B>, Color4f, &'a Mat4x4<RealToProj<B>>> {
    let vsh = |v: Vertex3<Color4f, B>, m: &Mat4x4<RealToProj<B>>| {
        vertex(m.apply(&v.pos), v.attrib)
    };
    let fsh = |f: Frag<Color4f>| f.var.to_color4();

    shader::new(vsh, fsh)
}
