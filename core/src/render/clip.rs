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
use core::{
    fmt::{Debug, Formatter},
    iter,
    mem::swap,
};

use crate::geom::{Edge, Tri, Vertex, vertex};
use crate::math::{Lerp, ProjVec3};

use view_frustum::{outcode, status};

/// Trait for types that can be [clipped][self] against convex volumes.
///
/// # Note to implementors
/// Implementations should avoid creating degenerate primitives, such as
/// triangles with only two unique vertices.
pub trait Clip {
    /// Clips primitives against a set of planes, returning the resulting
    /// zero or more primitives as an iterator.
    ///
    /// If a primitive being clipped lies entirely within the bounding volume,
    /// it is emitted as it is. If it is entirely outside the volume, it is
    /// skipped. If it is partially inside, it is clipped such that no points
    /// outside the volume remain in the result.
    #[must_use]
    fn clip<'a, I: IntoIterator<Item = Self, IntoIter: 'a>>(
        items: I,
        planes: &'a [ClipPlane],
    ) -> impl Iterator<Item = Self>;
}

/// A vector in clip space.
pub type ClipVec = ProjVec3;

/// A vertex in clip space.
#[derive(Copy, Clone, PartialEq)]
pub struct ClipVert<A> {
    pub pos: ClipVec,
    pub attrib: A,
    outcode: u8,
}

impl<A: Debug> Debug for ClipVert<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClipVert")
            .field("pos", &self.pos.0)
            .field("attrib", &self.attrib)
            .field("outcode", &format_args!("{:06b}", self.outcode))
            .finish()
    }
}

/// Visibility of a shape in the view frustum.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Status {
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
    #[inline]
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

    /// Computes this plane's outcode bit for a point.
    ///
    /// The result is `self.1` if `pt` is outside this plane, 0 otherwise.
    #[inline]
    pub fn outcode(&self, pt: &ClipVec) -> u8 {
        (self.signed_dist(pt) > 0.0) as u8 * self.1
    }

    /// Checks the outcode of a clip vertex against `self`.
    ///
    /// Returns `true` if this plane's outcode bit in `v` is 0, `false` otherwise.
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

    /// Clips a convex polygon against `self`.
    ///
    /// Returns the resulting vertices in the out parameter `verts_out`.
    ///
    /// In the diagram below, clipping triangle ABC results in quad ABPQ,
    /// where P and Q are new vertices generated by interpolating between
    /// A and C, and B and C, respectively.
    ///
    /// ```text
    ///
    ///     n
    ///     ^            C
    ///     |           / \         outside
    ///     |         /    \
    /// ----+-------Q-------P--------self-----
    ///           /          \
    ///         A--___        \     inside
    ///               `---__   \
    ///                     `---B
    /// ```
    pub fn clip_simple_polygon<A: Lerp>(
        &self,
        verts_in: &[ClipVert<A>],
        verts_out: &mut Vec<ClipVert<A>>,
    ) {
        let [fst, .., lst] = verts_in else {
            return;
        };
        let edges = verts_in
            .array_windows()
            .map(|[v0, v1]| Edge(v0, v1))
            .chain([Edge(lst, fst)]);

        for Edge(v0, v1) in edges {
            if self.is_inside(v0) {
                // v0 is inside; emit it as-is. If v1 is also inside, no need
                // to do anything; it will be emitted on the next iteration.
                verts_out.push(v0.clone());
            } else {
                // v0 is outside, discard it. If v1 is also outside, no need
                // to do anything; it will be discarded on the next iteration.
            }

            if let Some(v) = self.intersect([v0, v1]) {
                verts_out.push(v);
            }
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
    #[rustfmt::skip]
    pub const PLANES: [ClipPlane; 6] = [
        ClipPlane::new( 0.0,  0.0, -1.0,  1.0,  0x01), // Near
        ClipPlane::new( 0.0,  0.0,  1.0,  1.0,  0x02), // Far
        ClipPlane::new(-1.0,  0.0,  0.0,  1.0,  0x04), // Left
        ClipPlane::new( 1.0,  0.0,  0.0,  1.0,  0x08), // Right
        ClipPlane::new( 0.0, -1.0,  0.0,  1.0,  0x10), // Bottom
        ClipPlane::new( 0.0,  1.0,  0.0,  1.0,  0x20), // Top
    ];

    /// Returns the outcode of the given point.
    ///
    /// The outcode is a bitset where the bit of each plane is 0 if the point
    /// is inside the plane, and 1 otherwise. It is used to determine whether
    /// a primitive is fully inside, partially inside, or fully outside the
    /// frustum.
    #[inline]
    pub fn outcode(pt: &ClipVec) -> u8 {
        PLANES.iter().map(|p| p.outcode(pt)).sum()
    }

    /// Returns the visibility status of the convex hull given by `vs`.
    pub fn status<V>(vs: &[ClipVert<V>]) -> Status {
        // The set of planes outside which all vertices are
        let all_outside = vs.iter().fold(!0, |code, v| code & v.outcode);

        // The set of planes outside which at least one vertex is.
        let any_outside = vs.iter().fold(0, |code, v| code | v.outcode);

        if all_outside != 0 {
            // If all vertices are outside at least one plane, the whole
            // object is hidden and can be culled. Note that they must be
            // outside the *same* plane; it isn't enough that they are all
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

/// Computes the intersection of a simple polygon and a convex volume.
///
/// Returns the part of the polygon that is inside all planes, if any, in
/// `verts_out`. This function uses an out parameter rather than the return
/// value in order to avoid extra allocations. The result is unspecified if
/// `verts_out` is not empty. `verts_in` is left empty by the function.
///
/// The algorithm used is Sutherland–Hodgman [^1].
///
/// TODO Describe algorithm
///
/// [^1]: Ivan Sutherland, Gary W. Hodgman: Reentrant Polygon Clipping.
///        Communications of the ACM, vol. 17, pp. 32–42, 1974
pub fn clip_simple_polygon<'a, A: Lerp>(
    planes: &[ClipPlane],
    verts_in: &'a mut Vec<ClipVert<A>>,
    verts_out: &'a mut Vec<ClipVert<A>>,
) {
    debug_assert!(verts_out.is_empty());

    for (plane, i) in iter::zip(planes, 0..) {
        plane.clip_simple_polygon(verts_in, verts_out);
        verts_in.clear();
        if verts_out.is_empty() {
            // Nothing left to clip; the polygon was fully outside
            break;
        } else if i < planes.len() - 1 {
            // Use the result of this iteration as the input of the next
            swap(verts_in, verts_out);
        }
    }
}

impl<V> ClipVert<V> {
    #[inline]
    pub fn new(Vertex { pos, attrib }: Vertex<ClipVec, V>) -> Self {
        let outcode = outcode(&pos);
        Self { pos, attrib, outcode }
    }
}

impl<A: Lerp> Clip for Edge<ClipVert<A>> {
    fn clip<'a, I>(
        edges: I,
        planes: &'a [ClipPlane],
    ) -> impl Iterator<Item = Self>
    where
        I: IntoIterator<Item = Self, IntoIter: 'a>,
    {
        let mut out = Vec::new();

        'edges: for edge in edges {
            // Bitset of planes that both ends are outside
            let both_outside = edge.0.outcode & edge.1.outcode;
            // Bitset of planes that at least one end is outside
            let either_outside = edge.0.outcode | edge.1.outcode;

            if both_outside != 0 {
                // There is at least one plane outside which both ends are.
                // The edge must be fully outside and can be discarded.
                continue;
            }
            if either_outside == 0 {
                // Neither end is outside any plane. The edge is fully visible,
                // no clipping needed.
                out.push(edge.clone());
                continue;
            }

            // Otherwise, either only one endpoint is inside all planes, *or*
            // the two endpoints are outside *different* planes. Clipping is
            // needed to compute the fully-inside segment of the edge *if any*.
            let Edge(mut v0, mut v1) = edge.clone();
            for plane in planes {
                let v0_inside = plane.is_inside(&v0);
                let v1_inside = plane.is_inside(&v1);

                if !v0_inside && !v1_inside {
                    // We know the original edge was not outside any *single*
                    // plane, but if it *was* outside the bounding volume as
                    // a whole, clipping will eventually result in a remainder
                    // that is fully outside a plane and can be discarded.
                    continue 'edges;
                }
                if let Some(v) = plane.intersect([&v0, &v1]) {
                    if v0_inside {
                        v1 = v;
                    } else if v1_inside {
                        v0 = v;
                    }
                }
            }
            out.push(Edge(v0, v1));
        }
        out.into_iter()
    }
}

impl<A: Lerp + 'static> Clip for Tri<ClipVert<A>> {
    fn clip<'a, I>(
        tris: I,
        planes: &'a [ClipPlane],
    ) -> impl Iterator<Item = Self>
    where
        I: IntoIterator<Item = Self, IntoIter: 'a>,
    {
        // Avoid unnecessary allocations by reusing these.
        //
        // Clipping an N-gon against M planes can result in a polygon of at most
        // N+M sides, so for a triangle against a frustum the result is at most
        // a 9-gon and when triangulated, at most seven triangles.
        let mut verts_in = Vec::with_capacity(9);
        let mut verts_out = Vec::with_capacity(9);
        let mut tris_out = Vec::with_capacity(7);

        let mut tris_in = tris.into_iter();
        // TODO Could use a custom named iterator type
        iter::from_fn(move || {
            'next_tri: loop {
                // If there are buffered output triangles pending, just pop and
                // return one. The order does not matter, LIFO is fine.
                if let tri_out @ Some(_) = tris_out.pop() {
                    return tri_out;
                }

                // Otherwise, get the next input tri; if there are none,
                // we're done because there are no pending output tris either.
                let Some(tri_in) = tris_in.next() else {
                    return None;
                };

                // TODO This status check could be moved to clip_simple_polygon
                match status(&tri_in.0) {
                    // If the input tri is fully visible, just return it
                    Status::Visible => return Some(tri_in),
                    // Otherwise if it's fully hidden, ignore it and try the
                    // next one
                    Status::Hidden => continue 'next_tri,
                    // Otherwise, it needs clipping
                    Status::Clipped => { /* go ahead and clip */ }
                }

                verts_in.extend(tri_in.0);
                clip_simple_polygon(planes, &mut verts_in, &mut verts_out);

                // Clipping a triangle results in an n-gon, where n depends on
                // how many planes the triangle intersects. For example, here
                // clipping triangle ABC generated three new vertices, resulting
                // in quad APQR. Convert the quad into triangles PRA and PQR
                // (or APQ and AQR, it is arbitrary):
                //
                //              B
                //
                //           /     \
                //             Q_____P___________
                //        /    |..../..\
                //             |.../.....\
                //     /       |./.........\
                //             |/............\
                //  C  _  _  _ R_______________A
                //             |
                //             |
                //
                if let [p, rest @ ..] = &verts_out[..] {
                    tris_out.extend(
                        rest.array_windows()
                            .cloned()
                            .map(|[a, b]| Tri([p.clone(), a, b])),
                    );
                }
                verts_out.clear();
            }
        })
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
        Tri([a, b, c]).map(vtx)
    }

    fn poly<const N: usize>(pts: [ClipVec; N]) -> [ClipVert<f32>; N] {
        pts.map(vtx)
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
        assert_eq!(outcode(&vec(0.0, 0.0, -1.5)), 0b00_00_01);
        // Outside right == 8
        assert_eq!(outcode(&vec(2.0, 0.0, 0.0)), 0b00_10_00);
        // Outside bottom == 16
        assert_eq!(outcode(&vec(0.0, -1.01, 0.0)), 0b01_00_00);
        // Outside far left == 2|4
        assert_eq!(outcode(&vec(-2.0, 0.0, 2.0)), 0b00_01_10);
    }

    /// Clipping against a single plane
    mod far_plane {
        use super::*;

        // Clipping degenerate polys (single edge)
        // TODO Probably not very useful tests, should rather test
        //      the actual Clip impl for Edge...

        #[test]
        fn edge_inside() {
            let e = [vec(2.0, 0.0, -1.0), vec(-1.0, 1.0, 1.0)].map(vtx);
            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&e, &mut res);
            assert_eq!(res, e);
        }
        #[test]
        fn edge_outside() {
            let e = poly([vec(2.0, 0.0, 1.5), vec(-1.0, 1.0, 2.0)]);
            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&e, &mut res);
            assert_eq!(res, []);
        }
        #[test]
        fn edge_in_out() {
            let e = poly([vec(2.0, 0.0, 0.0), vec(-1.0, 1.0, 2.0)]);
            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&e, &mut res);
            // clip_simple_polygon treats a single edge as a degenerate polygon,
            // inserting an additional vertex
            assert_eq!(res[..2], [e[0], vtx(vec(0.5, 0.5, 1.0))]);
        }
        #[test]
        fn edge_out_in() {
            let e = poly([vec(2.0, 0.0, 4.0), vec(-1.0, 1.0, 0.0)]);
            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&e, &mut res);
            // clip_simple_polygon treats a single edge as a degenerate polygon,
            // inserting an additional vertex
            assert_eq!(res[..2], [vtx(vec(-0.25, 0.75, 1.0)), e[1]]);
        }

        #[test]
        fn tri_fully_inside() {
            let tr = poly([
                vec(0.0, -1.0, 0.0),
                vec(2.0, 0.0, 1.0),
                vec(-1.0, 1.5, -1.0),
            ]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            assert_eq!(res, tr);
        }

        #[test]
        fn tri_fully_outside() {
            let tr = poly([
                vec(0.0, -1.0, 1.5),
                vec(2.0, 0.0, 1.0 + 1e-6),
                vec(-1.0, 1.5, 2.0),
            ]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            assert_eq!(res, []);
        }

        #[test]
        fn tri_inside_on_on() {
            //
            // 1.0  --on1------------on2-- plane
            //          \           /
            //           \       /
            //            \   /
            // 0.0         ins
            //       -1.0  0.0  1.0  2.0
            let ins = vec(0.0, -1.0, 0.0);
            let on1 = vec(2.0, 0.0, 1.0);
            let on2 = vec(-1.0, 1.5, 1.0);

            let tr = poly([ins, on1, on2]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            assert_eq!(res, tr);
        }

        #[test]
        fn tri_outside_inside_inside() {
            // 2.0      out
            //           | \
            //           |  \
            // 1.0  -----p---q----- plane
            //           |    \
            //           |     \
            // 0.0      in1----in2
            //          0.0    1.0
            let out = vec(0.0, 0.0, 2.0);
            let in1 = vec(0.0, 1.0, 0.0);
            let in2 = vec(1.0, 0.0, 0.0);

            let tr = poly([out, in1, in2]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            let p = vec(0.0, 0.5, 1.0);
            let q = vec(0.5, 0.0, 1.0);
            // Clipping `out` leaves a quadrilateral
            assert_eq!(res, poly([p, in1, in2, q]));
        }
        #[test]
        fn tri_outside_on_inside() {
            // 2.0      out
            //           | \
            //           |   \
            // 1.0  -----p----on--- plane
            //           |   /
            //           | /
            // 0.0    . ins .  .  .
            //          0.0   1.0
            let out = vec(0.0, 0.0, 2.0);
            let on = vec(1.0, 0.0, 1.0);
            let ins = vec(0.0, -1.0, 0.0);

            let tr = poly([out, on, ins]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            let p = vec(0.0, -0.5, 1.0);
            assert_eq!(res, poly([on, ins, p]));
        }
        #[test]
        fn tri_outside_on_on() {
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
            let tr = poly([out, on1, on2]);

            let mut res = vec![];
            FAR_PLANE.clip_simple_polygon(&tr, &mut res);
            // Returns a degenerate poly with a single vertex, such polys are
            // discarded by the Tri Clip impl which only returns full triangles
            assert_eq!(res, poly([on1, on2]));
        }
    }

    /// Clipping against the whole view frustum (six planes)
    mod frustum {
        use super::*;

        #[test]
        fn tri_fully_inside_result_no_change() {
            let tr = tri(
                vec(-1.0, -1.0, -1.0),
                vec(1.0, 1.0, 0.0),
                vec(0.0, 1.0, 1.0),
            );
            let mut res = Tri::clip([tr], &PLANES);
            assert_eq!(res.next(), Some(tr));
            assert_eq!(res.next(), None);
        }
        #[test]
        fn tri_fully_outside_result_empty() {
            //    z
            //    ^
            //    2-------0
            //    · \     |
            //  --1---+   |
            //    ·   | \ |
            //    + - 1 - 2 - - > x
            //        |
            //  ------+

            let tr = tri(
                vec(2.0, 2.0, 2.0),
                vec(2.0, -2.0, 0.0),
                vec(0.0, -1.0, 2.0),
            );
            assert_eq!(Tri::clip([tr], &PLANES).next(), None);
        }
        #[test]
        fn tri_result_is_quad() {
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

            let mut res = Tri::clip([tr], &PLANES);
            assert_eq!(
                res.next(),
                Some(tri(
                    vec(0.0, 0.0, 0.0),
                    vec(1.0, 0.0, 1.0),
                    vec(0.0, 0.0, 1.0)
                ))
            );
            assert_eq!(
                res.next(),
                Some(tri(
                    vec(0.0, 0.0, 0.0),
                    vec(1.0, 0.0, 0.0),
                    vec(1.0, 0.0, 1.0)
                ))
            );
            assert_eq!(res.next(), None);
        }

        #[test]
        fn tri_result_is_heptagon() {
            //        z
            //        ^       2
            //        ·   /  /
            //    +---/---+ /
            //    /   ·   |/
            //  0 | · o · / · · > x
            //    \   ·  /|
            //    +-\---/-+
            //        1

            let tr = tri(
                vec(-1.5, 0.0, 0.0),
                vec(0.0, 0.0, -1.5),
                vec(2.0, 0.0, 2.0),
            );

            let res = Tri::clip([tr], &PLANES).collect::<Vec<_>>();

            // 7 intersection points -> clipped shape made of 5 triangles
            assert_eq!(res.len(), 5);
            assert!(res.iter().all(in_bounds));
        }

        #[test]
        #[allow(unused)]
        fn tri_all_cases() {
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
                let res = Tri::clip([tr], &PLANES).collect::<Vec<_>>();
                assert!(
                    res.iter().all(in_bounds),
                    "clip returned oob vertex:\n\
                    input: {:?}\n\n\
                    output: {:?}",
                    tr,
                    &res
                );
                in_tris += 1;
                in_degen += is_degenerate(&tr) as u32;
                out_tris[res.len()] += 1;
                out_total += res.len();
                out_degen += res.iter().filter(|t| is_degenerate(t)).count();
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
}
