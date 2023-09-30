//! Clipping of rendering primitives against planes.
//!
//! In particular, this module implements clipping of geometry against
//! the six planes comprising the view frustum, in order to avoid drawing
//! things that are behind the camera or out of bounds of the viewport.

use alloc::vec;
use alloc::vec::Vec;
use core::mem::swap;

use crate::geom::{Plane, Tri, Vertex};
use crate::math::vec::{Affine, Linear, Proj4, Vec3, Vec4};

/// Trait for types that can be clipped against planes.
pub trait Clip {
    type Item;
    /// Clips `self` against `planes`, returning the resulting zero or more
    /// primitives in the out parameter `out`.
    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>);
}

/// A vector in clip space.
pub type ClipVec = Vec4<Proj4>;

pub type ClipVert<A> = Vertex<ClipVec, A>;

/// A plane in clip space.
pub type ClipPlane = Plane<ClipVec>;

impl ClipPlane {
    /// Creates a clip plane given a normal and offset from origin.
    ///
    /// TODO Floating-point arithmetic is not permitted in const functions
    ///      so the offset must be negated for now.
    pub const fn new(normal: Vec3, neg_offset: f32) -> Self {
        let [x, y, z] = normal.0;
        Self(ClipVec::new([x, y, z, neg_offset]))
    }

    /// Returns the signed distance between `pt` and `self`.
    ///
    /// The return value is positive if `pt` is "outside" the plane,
    /// defined as the half-space in the direction of the normal vector,
    /// negative if `pt` is in the other half-space, and zero if `pt`
    /// is exactly coincident with the plane:
    /// ```text
    ///            n
    ///            ^       d > 0
    ///            |         x
    ///            |         |
    ///            |-.       |
    /// -----x-----+-`------------ self
    ///    d = 0         |
    ///                  x
    ///                d < 0
    /// ```
    #[inline]
    pub fn signed_dist(&self, pt: &ClipVec) -> f32 {
        self.0.dot(pt)
    }

    /// Clips the convex polygon given by `verts_in` against `self` and
    /// returns the resulting vertices in `verts_out`.
    ///
    /// In the diagram below, clipping triangle `abc` results in quad `apqc`,
    /// where p and q are new vertices generated by interpolating between
    /// a and b and b and c, respectively.
    ///
    /// ```text
    ///          b
    ///         / \
    ///        /   \    outside
    /// ------p-----q-----self---
    ///      /       \   inside
    ///     a-__      \
    ///         `--__  \
    ///              `--c
    /// ```
    pub fn clip_simple_polygon<A>(
        &self,
        verts_in: &[ClipVert<A>],
        verts_out: &mut Vec<ClipVert<A>>,
    ) where
        A: Affine + Clone,
        A::Diff: Linear<Scalar = f32>,
    {
        let mut verts = verts_in
            .iter()
            .chain(&verts_in[..1])
            .map(|v| (v, self.signed_dist(&v.pos)));

        let Some((mut v0, mut d0)) = &verts.next() else {
            return;
        };
        for (v1, d1) in verts {
            if d0 <= 0.0 {
                // v0 is inside; emit it as-is. If v1 is also inside, we don't
                // have to do anything; it is emitted on the next iteration.
                verts_out.push((*v0).clone());
            } else {
                // v0 is outside, discard it. If v1 is also outside, we don't
                // have to do anything: it is discarded on the next iteration.
            }
            if d0 * d1 < 0.0 {
                // Edge crosses the plane surface. Split the edge in two by
                // interpolating and emitting a new vertex at intersection.
                // The new vertex becomes one of the endpoints of a new "clip"
                // edge coincident with the plane.

                // `t` is the fractional distance from `v0` to the intersection
                // point. If conditions guarantee that `d1 - d0` is nonzero.
                let t = -d0 / (d1 - d0);
                verts_out.push(ClipVert {
                    pos: v0.pos.lerp(&v1.pos, t),
                    attrib: v0.attrib.lerp(&v1.attrib, t),
                });
            }
            (v0, d0) = (v1, d1);
        }
    }
}

pub mod view_frustum {
    use crate::math::vec3;

    use super::*;

    /// The near, far, left, right, bottom, and top clipping planes,
    /// in that order.
    pub const PLANES: [ClipPlane; 6] = [
        Plane::new(vec3(0.0, 0.0, -1.0), -1.0), // Near
        Plane::new(vec3(0.0, 0.0, 1.0), -1.0),  // Far
        Plane::new(vec3(-1.0, 0.0, 0.0), -1.0), // Left
        Plane::new(vec3(1.0, 0.0, 0.0), -1.0),  // Right
        Plane::new(vec3(0.0, -1.0, 0.0), -1.0), // Bottom
        Plane::new(vec3(0.0, 1.0, 0.0), -1.0),  // Top
    ];
}

/// Clips the simple polygon given by the vertices in `verts_in` against
/// `planes`, returning the result in `verts_out`.
///
/// This function uses an out parameter rather than the return value in order
/// to avoid extra allocations. Also note that the value of `verts_in` after
/// calling the function is not specified, as it is used as temporary storage.
///
/// The algorithm used is Sutherland–Hodgman [1].
/// TODO Describe algorithm
///
/// [1]: Ivan Sutherland, Gary W. Hodgman: Reentrant Polygon Clipping.
///        Communications of the ACM, vol. 17, pp. 32–42, 1974
pub fn clip_simple_polygon<'a, A>(
    planes: &[Plane<ClipVec>],
    mut verts_in: &'a mut Vec<ClipVert<A>>,
    mut verts_out: &'a mut Vec<ClipVert<A>>,
) where
    A: Affine + Clone,
    A::Diff: Linear<Scalar = f32>,
{
    for p in planes {
        verts_out.clear();
        p.clip_simple_polygon(verts_in, verts_out);
        if verts_out.is_empty() {
            break;
        }
        swap(&mut verts_in, &mut verts_out);
    }
}

impl<A> Clip for [Tri<ClipVert<A>>]
where
    A: Affine + Clone,
    A::Diff: Linear<Scalar = f32>,
{
    type Item = Tri<ClipVert<A>>;

    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>) {
        // Avoid unnecessary allocations by reusing these
        let mut verts_in = vec![];
        let mut verts_out = vec![];

        for tri in self.iter().cloned() {
            verts_in.extend(tri.0);
            clip_simple_polygon(planes, &mut verts_in, &mut verts_out);

            if let Some((v0, vs)) = verts_out.split_first() {
                // Turn the resulting polygon into a fan of triangles
                // with v0 as the common vertex
                out.extend(
                    vs.windows(2)
                        .map(|e| Tri([v0.clone(), e[0].clone(), e[1].clone()])),
                );
            }
            verts_in.clear();
            verts_out.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::view_frustum::*;
    use super::*;

    const FAR_PLANE: ClipPlane = PLANES[1];

    fn vec(x: f32, y: f32, z: f32) -> ClipVec {
        [x, y, z, 1.0].into()
    }

    fn vtx(pos: ClipVec) -> ClipVert<f32> {
        Vertex { pos, attrib: 0.0 }
    }

    fn tri(a: ClipVec, b: ClipVec, c: ClipVec) -> Tri<ClipVert<f32>> {
        Tri([a, b, c].map(vtx))
    }

    #[test]
    fn signed_distance() {
        assert_eq!(FAR_PLANE.signed_dist(&vec(0.0, 0.0, -1.0)), -2.0);
        assert_eq!(FAR_PLANE.signed_dist(&vec(1.0, 0.0, 0.0)), -1.0);
        assert_eq!(FAR_PLANE.signed_dist(&vec(0.0, 2.0, 1.0)), 0.0);
        assert_eq!(FAR_PLANE.signed_dist(&vec(-3.0, 0.0, 2.0)), 1.0);
    }

    #[test]
    fn edge_clip_inside() {
        let e = [vec(2.0, 0.0, -1.0), vec(-1.0, 1.0, 1.0)].map(vtx);
        let mut res = vec![];
        FAR_PLANE.clip_simple_polygon(&e, &mut res);
        assert_eq!(res, e);
    }
    #[test]
    fn edge_clip_outside() {
        let e = [vec(2.0, 0.0, 1.5), vec(-1.0, 1.0, 2.0)].map(vtx);
        let mut res = vec![];
        FAR_PLANE.clip_simple_polygon(&e, &mut res);
        assert_eq!(res, []);
    }
    #[test]
    fn edge_clip_in_out() {
        let e = [vec(2.0, 0.0, 0.0), vec(-1.0, 1.0, 2.0)].map(vtx);
        let mut res = vec![];
        FAR_PLANE.clip_simple_polygon(&e, &mut res);
        // clip_simple_polygon treats a single edge as a degenerate polygon,
        // inserting an additional vertex
        assert_eq!(res[..2], [e[0], vtx(vec(0.5, 0.5, 1.0))]);
    }
    #[test]
    fn edge_clip_out_in() {
        let e = [vec(2.0, 0.0, 4.0), vec(-1.0, 1.0, 0.0)].map(vtx);
        let mut res = vec![];
        FAR_PLANE.clip_simple_polygon(&e, &mut res);
        // clip_simple_polygon treats a single edge as a degenerate polygon,
        // inserting an additional vertex
        assert_eq!(res[..2], [vtx(vec(-0.25, 0.75, 1.0)), e[1]]);
    }

    #[test]
    fn tri_clip_fully_inside() {
        let tri =
            tri(vec(0.0, -1.0, 0.0), vec(2.0, 0.0, 0.5), vec(-1.0, 1.5, 0.0));
        let res = &mut vec![];
        [tri].clip(&[FAR_PLANE], res);
        assert_eq!(res, &[tri]);
    }
    #[test]
    fn tri_clip_fully_outside() {
        let tri =
            tri(vec(0.0, -1.0, 1.5), vec(2.0, 0.0, 1.5), vec(-1.0, 1.5, 2.0));
        let res = &mut vec![];
        [tri].clip(&[FAR_PLANE], res);
        assert_eq!(res, &[]);
    }

    #[test]
    fn tri_clip_inside_on_on() {
        //
        // 1.0  --on1------------on2-- plane
        //          \           /
        //           \       /
        //            \   /
        // 0.0         ins
        //       -1.0  0.0  1.0  2.0
        let tri =
            tri(vec(0.0, -1.0, 0.0), vec(2.0, 0.0, 1.0), vec(-1.0, 1.5, 1.0));
        let res = &mut vec![];
        [tri].clip(&[FAR_PLANE], res);
        assert_eq!(res, &[tri]);
    }

    #[test]
    fn tri_clip_outside_inside_inside() {
        // 2.0      out
        //           | \
        //           |  \
        // 1.0  -----+---+----- plane
        //           |    \
        //           |     \
        // 0.0      in1----in2
        //          0.0    1.0
        let out = vec(0.0, 0.0, 2.0);
        let in1 = vec(0.0, 1.0, 0.0);
        let in2 = vec(1.0, 0.0, 0.0);
        let tr = tri(out, in1, in2);

        let res = &mut vec![];
        [tr].clip(&[FAR_PLANE], res);
        assert_eq!(
            res,
            &[
                // Clipping `out` leaves a quadrilateral
                tri(vec(0.0, 0.5, 1.0), in1, in2),
                tri(vec(0.0, 0.5, 1.0), in2, vec(0.5, 0.0, 1.0))
            ]
        );
    }
    #[test]
    fn tri_clip_outside_on_inside() {
        // 2.0      out
        //           | \
        //           |   \
        // 1.0  -----+----on--- plane
        //           |   /
        //           | /
        // 0.0    . ins .  .  .
        //          0.0   1.0
        let out = vec(0.0, 0.0, 2.0);
        let on = vec(1.0, 0.0, 1.0);
        let ins = vec(0.0, -1.0, 0.0);
        let tr = tri(out, on, ins);

        let res = &mut vec![];
        [tr].clip(&[FAR_PLANE], res);
        assert_eq!(res, &[tri(on, ins, vec(0.0, -0.5, 1.0))]);
    }
    #[test]
    fn tri_clip_outside_on_on() {
        // 2.0      out
        //           | \
        //           |   \
        // 1.0   ---on2---on1-- plane
        //           .
        //           .
        // 0.0    .  o  .  .  .
        //          0.0   1.0
        let out = vec(0.0, 0.0, 2.0);
        let on1 = vec(1.0, 0.0, 1.0);
        let on2 = vec(0.0, -1.0, 1.0);
        let tr = tri(out, on1, on2);

        let res = &mut vec![];
        [tr].clip(&[FAR_PLANE], res);
        assert_eq!(res, &[]);
    }

    #[test]
    fn tri_clip_all_planes_fully_inside() {
        let tr = tri(
            vec(-1.0, -1.0, -1.0),
            vec(1.0, 1.0, 0.0),
            vec(0.0, 1.0, 1.0),
        );
        let res = &mut vec![];
        [tr].clip(&PLANES, res);
        assert_eq!(res, &[tr]);
    }
    #[test]
    fn tri_clip_all_planes_fully_outside() {
        //    z
        //    ^
        //    2-------0
        //    · \     |
        //  - ----+   |
        //    ·   | \ |
        //    + - | - 1 - - > x
        //        |
        //  - ----+

        let tr =
            tri(vec(2.0, 2.0, 2.0), vec(2.0, -2.0, 0.0), vec(0.0, -1.0, 2.0));

        let res = &mut vec![];
        [tr].clip(&PLANES, res);
        assert_eq!(res, &[]);
    }
    #[test]
    fn tri_clip_all_planes_result_is_quad() {
        //    z
        //    ^
        //    2
        //    | \
        //  - +---+
        //    |   | \
        //    0---+---1 - - > x
        //        |
        //  - ----+

        let tr =
            tri(vec(0.0, 0.0, 0.0), vec(2.0, 0.0, 0.0), vec(0.0, 0.0, 2.0));

        let res = &mut vec![];
        [tr].clip(&PLANES, res);
        assert_eq!(
            res,
            &[
                tri(vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0), vec(1.0, 0.0, 1.0)),
                tri(vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 1.0), vec(0.0, 0.0, 1.0))
            ]
        );
    }
    #[test]
    fn tri_clip_all_planes_result_is_heptagon() {
        //        z
        //        ^       2
        //        ·   /  /
        //    +---/---+ /
        //    /   ·   |/
        //  0 | · o · / · · > x
        //    \   ·  /|
        //    +-\---/-+
        //        1

        let tr =
            tri(vec(-1.5, 0.0, 0.0), vec(0.0, 0.0, -1.5), vec(2.0, 0.0, 2.0));

        let res = &mut vec![];
        [tr].clip(&PLANES, res);
        // 7 intersection points -> clipped shape made of 5 triangles
        assert_eq!(res.len(), 5);
    }
}
