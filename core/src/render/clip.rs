//! Clipping geometric shapes against planes.
//!
//! Clipping means converting a shape into another, such that only the points
//! inside a volume enclosed by one or more planes remain; "inside" is defined
//! as the half-space that the plane's normal vector points away from.
//! In other words, clipping computes the intersection between a shape and
//! a (possibly unbounded) convex polyhedron defined by the planes.
//!
//! In particular, this module implements clipping of geometry against the six
//! planes comprising the [*view frustum*][view_frustum], in order to avoid
//! drawing objects that are behind the camera or outside the bounds of the
//! viewport. Clipping geometry before the raster stage is more efficient than
//! doing it for every scanline individually.
//!

use alloc::vec::Vec;
use core::iter::zip;

use view_frustum::{outcode, status};

use crate::geom::{Tri, Vertex, vertex};
use crate::math::{Lerp, vec::ProjVec3};

/// Trait for types that can be [clipped][self] against planes.
///
/// # Note to implementors
/// This trait is primarily meant to be implemented on slices or other
/// composites, so that several primitives can be clipped in a single call.
/// This allows reuse of temporary buffers, for instance.
///
/// Implementations should avoid creating degenerate primitives, such as
/// triangles with only two unique vertices.
pub trait Clip {
    /// Type of the clipped object. For example, `Self` if implemented for
    /// the type itself, or `T` if implemented for `[T]`.
    type Item;

    /// Clips `self` against `planes`, returning the resulting zero or more
    /// primitives in the out parameter `out`.
    ///
    /// If a primitive being clipped lies entirely within the bounding volume,
    /// it is emitted as it is. If it is entirely outside the volume, it is
    /// skipped. If it is partially inside, it is clipped such that no points
    /// outside the volume remain in the result.
    ///
    /// The result is unspecified if `out` is nonempty.
    ///
    /// TODO Investigate returning an iterator
    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>);
}

/// A vector in clip space.
pub type ClipVec = ProjVec3;

/// A vertex in clip space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClipVert<A> {
    pub pos: ClipVec,
    pub attrib: A,
    outcode: u8,
}

/// Visibility status of a polygon.
enum Status {
    /// Entirely inside view frustum
    Visible,
    /// Either outside or partly inside, needs clipping
    Clipped,
    /// Entirely outside view frustum
    Hidden,
}

#[derive(Debug, Copy, Clone)]
pub struct ClipPlane(ClipVec, u8);

impl ClipPlane {
    /// Creates a clip plane given a normal, offset, and outcode bit.
    const fn new(x: f32, y: f32, z: f32, off: f32, bit: u8) -> Self {
        Self(ClipVec::new([x, y, z, -off]), bit)
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
    ///            |_        |
    /// -----x-----+-'------------ self
    ///    d = 0         |
    ///                  x
    ///                d < 0
    /// ```
    #[inline]
    pub fn signed_dist(&self, pt: &ClipVec) -> f32 {
        self.0.dot(pt)
    }

    /// Computes the outcode bit for `pt`.
    ///
    /// The result is `self.1` if `pt` is outside this plane, 0 otherwise.
    #[inline]
    pub fn outcode(&self, pt: &ClipVec) -> u8 {
        (self.signed_dist(pt) > 0.0) as u8 * self.1
    }

    /// Checks the outcode of `v` against `self`.
    ///
    /// Returns `true` if this plane's outcode bit is 0, `false` otherwise.
    #[inline]
    pub fn is_inside<A>(&self, v: &ClipVert<A>) -> bool {
        self.1 & v.outcode == 0
    }

    pub fn intersect<A: Lerp>(
        &self,
        [v0, v1]: [&ClipVert<A>; 2],
    ) -> Option<ClipVert<A>> {
        // TODO Doesn't use is_inside because it can't distinguish the case
        //   where a vertex lies exactly on the plane. Though that's mostly
        //   a theoretical edge case (heh).
        let d0 = self.signed_dist(&v0.pos);
        let d1 = self.signed_dist(&v1.pos);
        (d0 * d1 < 0.0).then(|| {
            // The edge crosses the plane surface. Split the edge in two
            // by interpolating and emitting a new vertex at intersection.
            // The new vertex becomes one of the endpoints of a new "clip"
            // edge coincident with the plane.

            // `t` is the fractional distance from `v0` to the intersection
            // point. If condition guarantees that `d1 - d0` is nonzero.
            let t = -d0 / (d1 - d0);

            ClipVert::new(vertex(
                v0.pos.lerp(&v1.pos, t),
                v0.attrib.lerp(&v1.attrib, t),
            ))
        })
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
    ///         / \        outside
    ///        /   \
    /// ------p-----q-------self-----
    ///      /       \
    ///     a-__      \    inside
    ///         `--__  \
    ///              `--c
    /// ```
    pub fn clip_simple_polygon<A: Lerp + Clone>(
        &self,
        verts_in: &[ClipVert<A>],
        verts_out: &mut Vec<ClipVert<A>>,
    ) {
        let mut verts = verts_in.iter().chain(&verts_in[..1]);

        let Some(mut v0) = verts.next() else {
            return;
        };

        for v1 in verts {
            if self.is_inside(v0) {
                // v0 is inside; emit it as-is. If v1 is also inside, we don't
                // have to do anything; it is emitted on the next iteration.
                verts_out.push((*v0).clone());
            } else {
                // v0 is outside, discard it. If v1 is also outside, we don't
                // have to do anything; it is discarded on the next iteration.
            }

            if let Some(v) = self.intersect([v0, v1]) {
                verts_out.push(v);
            }
            v0 = v1;
        }
    }
}

/// A view frustum is a truncated, sideways pyramid representing the volume of
/// space that is visible in a viewport with perspective projection. The left,
/// top, right, and bottom sides of the frustum correspond to the edges of the
/// viewport, while the near and far sides (the top and bottom of the pyramid)
/// limit how close-up or far away objects can be drawn.
///
/// The far plane can be used to limit the amount of geometry that needs to be
/// rendered per frame, usually combined with a fog effect so objects are not
/// abruptly clipped. The near plane must have a positive offset from origin,
/// and the offset plays a large role in dictating the distribution of depth
/// values used for hidden surface removal.
///
/// TODO Describe clip space
pub mod view_frustum {
    use super::*;

    /// The near, far, left, right, bottom, and top clipping planes,
    /// in that order.
    pub const PLANES: [ClipPlane; 6] = [
        ClipPlane::new(0.0, 0.0, -1.0, 1.0, 1), // Near
        ClipPlane::new(0.0, 0.0, 1.0, 1.0, 2),  // Far
        ClipPlane::new(-1.0, 0.0, 0.0, 1.0, 4), // Left
        ClipPlane::new(1.0, 0.0, 0.0, 1.0, 8),  // Right
        ClipPlane::new(0.0, -1.0, 0.0, 1.0, 16), // Bottom
        ClipPlane::new(0.0, 1.0, 0.0, 1.0, 32), // Top
    ];

    /// Clips geometry against the standard view frustum.
    ///
    /// This is the main entry point to clipping.
    pub fn clip<G: Clip + ?Sized>(geom: &G, out: &mut Vec<G::Item>) {
        geom.clip(&PLANES, out);
    }

    /// Returns the outcode of the given point.
    ///
    /// The outcode is a bitset where the bit of each plane is 0 if the point
    /// is inside the plane, and 1 otherwise. It is used to determine whether
    /// a primitive is fully inside, partially inside, or fully outside the
    /// frustum.
    pub(super) fn outcode(pt: &ClipVec) -> u8 {
        PLANES.iter().map(|p| p.outcode(pt)).sum()
    }

    /// Returns the visibility status of the polygon given by `vs`.
    pub(super) fn status<V>(vs: &[ClipVert<V>]) -> Status {
        // The set of planes outside which all vertices are
        let all_outside = vs.iter().fold(!0, |a, b| a & b.outcode);

        // The set of planes outside which at least one vertex is.
        let any_outside = vs.iter().fold(0, |a, b| a | b.outcode);

        if all_outside != 0 {
            // If all vertices are outside at least one plane, the whole
            // polygon is hidden and can be culled. Note that they must be
            // outside the *same* lane; it isn't enough that they are all
            // outside at least *some* plane!
            Status::Hidden
        } else if any_outside == 0 {
            // If no vertex is outside any plane, the whole polygon is visible
            Status::Visible
        } else {
            // Otherwise, at least one of the vertices is outside the frustum
            // and the polygon will have to be clipped (and may end up getting
            // culled completely).
            Status::Clipped
        }
    }
}

/// Clips the simple polygon given by the vertices in `verts_in` against the
/// convex volume defined by `planes`, returning the result in `verts_out`.
/// The result is unspecified if `verts_out` is not empty.
///
/// This function uses an out parameter rather than the return value in order
/// to avoid extra allocations. Also note that the content of `verts_in` after
/// calling the function is not specified, as it is used as temporary storage.
///
/// The algorithm used is Sutherland–Hodgman [^1].
///
/// TODO Describe algorithm
///
/// [^1]: Ivan Sutherland, Gary W. Hodgman: Reentrant Polygon Clipping.
///        Communications of the ACM, vol. 17, pp. 32–42, 1974
pub fn clip_simple_polygon<'a, A: Lerp + Clone>(
    planes: &[ClipPlane],
    verts_in: &'a mut Vec<ClipVert<A>>,
    verts_out: &'a mut Vec<ClipVert<A>>,
) {
    debug_assert!(verts_out.is_empty());

    for (p, i) in zip(planes, 0..) {
        p.clip_simple_polygon(verts_in, verts_out);
        if verts_out.is_empty() {
            // Nothing left to clip; the polygon was fully outside
            break;
        } else if i < planes.len() - 1 {
            // Use the result of this iteration as the input of the next
            verts_in.clear();
            verts_in.append(verts_out);
        }
    }
}

impl<V> ClipVert<V> {
    pub fn new(Vertex { pos, attrib }: Vertex<ClipVec, V>) -> Self {
        let outcode = outcode(&pos);
        Self { pos, attrib, outcode }
    }
}

impl<A: Lerp + Clone> Clip for [[ClipVert<A>; 2]] {
    type Item = [ClipVert<A>; 2];

    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>) {
        'lines: for [v0, v1] in self {
            let both_outside = v0.outcode & v1.outcode != 0;
            let neither_outside = v0.outcode | v1.outcode == 0;

            let mut v0 = v0.clone();
            let mut v1 = v1.clone();

            if both_outside {
                continue;
            }
            if neither_outside {
                out.push([v0, v1]);
                continue;
            }

            for p in planes {
                let v0_in = p.is_inside(&v0);
                let v1_in = p.is_inside(&v1);
                // TODO Why not handled by both_outside check?
                if !v0_in && !v1_in {
                    continue 'lines;
                }
                if let Some(v) = p.intersect([&v0, &v1]) {
                    if v0_in {
                        v1 = v;
                    } else if v1_in {
                        v0 = v;
                    }
                }
            }
            out.push([v0, v1]);
        }
    }
}

impl<A: Lerp + Clone> Clip for [Tri<ClipVert<A>>] {
    type Item = Tri<ClipVert<A>>;

    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>) {
        debug_assert!(out.is_empty());

        // Avoid unnecessary allocations by reusing these
        let mut verts_in = Vec::with_capacity(10);
        let mut verts_out = Vec::with_capacity(10);

        for tri @ Tri(vs) in self {
            match status(vs) {
                Status::Visible => {
                    out.push(tri.clone());
                    continue;
                }
                Status::Hidden => continue,
                Status::Clipped => { /* go on and clip */ }
            }

            verts_in.extend(vs.clone());
            clip_simple_polygon(planes, &mut verts_in, &mut verts_out);

            if let [a, rest @ ..] = &verts_out[..] {
                // Clipping a triangle results in an n-gon, where n depends on
                // how many planes the triangle intersects. Turn the resulting
                // n-gon into a fan of triangles with common vertex `a`, for
                // example here the polygon `abcd` is divided into triangles
                // `abc` and `acd`:
                //
                //    _ _ _c____________d_ _
                //         | \         /
                //         |   \      /
                //         |     \   /
                //    _ _ _|_______\/_ _ _ _
                //         b        a
                //
                out.extend(
                    rest.windows(2)
                        .map(|e| Tri([a.clone(), e[0].clone(), e[1].clone()])),
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

    use crate::{geom::vertex, math::Vary};

    use super::{view_frustum::*, *};

    const FAR_PLANE: ClipPlane = PLANES[1];

    fn vec(x: f32, y: f32, z: f32) -> ClipVec {
        [x, y, z, 1.0].into()
    }

    fn vtx(pos: ClipVec) -> ClipVert<f32> {
        ClipVert::new(vertex(pos, 0.0))
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
    fn outcode_inside() {
        assert_eq!(outcode(&vec(0.0, 0.0, 0.0)), 0);
        assert_eq!(outcode(&vec(1.0, 0.0, 0.0)), 0);
        assert_eq!(outcode(&vec(0.0, -1.0, 0.0)), 0);
        assert_eq!(outcode(&vec(0.0, 1.0, 1.0)), 0);
    }

    #[test]
    fn outcode_outside() {
        // Top Btm Rgt Lft Far Near
        //  32  16  8   4   2   1

        // Outside near == 1
        assert_eq!(outcode(&vec(0.0, 0.0, -1.5)), 0b00_0_01);
        // Outside right == 8
        assert_eq!(outcode(&vec(2.0, 0.0, 0.0)), 0b00_10_00);
        // Outside bottom == 16
        assert_eq!(outcode(&vec(0.0, -1.01, 0.0)), 0b01_00_00);
        // Outside far left == 2|4
        assert_eq!(outcode(&vec(-2.0, 0.0, 2.0)), 0b00_01_10);
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
    fn tri_clip_against_frustum_fully_inside() {
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
    fn tri_clip_against_frustum_fully_outside() {
        //    z
        //    ^
        //    2-------0
        //    · \     |
        //  --1---+   |
        //    ·   | \ |
        //    + - 1 - 2 - - > x
        //        |
        //  ------+

        let tr =
            tri(vec(2.0, 2.0, 2.0), vec(2.0, -2.0, 0.0), vec(0.0, -1.0, 2.0));

        let res = &mut vec![];
        [tr].clip(&PLANES, res);
        assert_eq!(res, &[]);
    }
    #[test]
    fn tri_clip_against_frustum_result_is_quad() {
        //    z
        //    ^
        //    2
        //    | \
        //  - 1---+
        //    |   | \
        //    0---1---2 - - > x
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
    fn tri_clip_against_frustum_result_is_heptagon() {
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
        assert!(res.iter().all(in_bounds));
    }

    #[test]
    #[allow(unused)]
    fn tri_clip_against_frustum_all_cases() {
        // Methodically go through every possible combination of every
        // vertex inside/outside every plane, including degenerate cases.

        let xs = || (-2.0).vary(1.0, Some(5));

        let pts: Vec<_> = xs()
            .flat_map(move |x| {
                xs().flat_map(move |y| xs().map(move |z| vec(x, y, z)))
            })
            .collect();

        let tris = pts.iter().flat_map(|a| {
            pts.iter()
                .flat_map(|b| pts.iter().map(|c| tri(*a, *b, *c)))
        });

        let mut in_tris = 0;
        let mut in_degen = 0;
        let mut out_tris = [0; 8];
        let mut out_degen = 0;
        let mut out_total = 0;
        for tr in tris {
            let res = &mut vec![];
            [tr].clip(&PLANES, res);
            assert!(
                res.iter().all(in_bounds),
                "clip returned oob vertex:\n\
                    input: {:#?}\n\
                    output: {:#?}",
                tr,
                &res
            );
            in_tris += 1;
            in_degen += is_degenerate(&tr) as u32;
            out_tris[res.len()] += 1;
            out_total += res.len();
            out_degen += res.iter().filter(|t| is_degenerate(t)).count()
        }
        #[cfg(feature = "std")]
        {
            use std::dbg;
            dbg!(in_tris);
            dbg!(in_degen);
            dbg!(out_degen);
            dbg!(out_total);
        }
        assert_eq!(in_tris, 5i32.pow(9));
        assert_eq!(
            out_tris,
            [559754, 536199, 537942, 254406, 58368, 6264, 192, 0]
        );
    }

    fn is_degenerate(Tri([a, b, c]): &Tri<ClipVert<f32>>) -> bool {
        a.pos == b.pos || a.pos == c.pos || b.pos == c.pos
    }

    fn in_bounds(Tri(vs): &Tri<ClipVert<f32>>) -> bool {
        vs.iter()
            .flat_map(|v| (v.pos / v.pos.w()).0)
            .all(|a| a.abs() <= 1.00001)
    }
}
