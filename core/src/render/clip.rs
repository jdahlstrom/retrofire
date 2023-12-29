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

use alloc::vec;
use alloc::vec::Vec;

use crate::geom::{Plane, Tri, Vertex};
use crate::math::space::Proj4;
use crate::math::vary::Vary;
use crate::math::vec::{Vec3, Vec4};
use crate::render::clip::view_frustum::{outcode, status};

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
pub type ClipVec = Vec4<Proj4>;

/// A plane in clip space.
pub type ClipPlane = Plane<ClipVec>;

/// A vertex in clip space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClipVert<A> {
    pub pos: ClipVec,
    pub attrib: A,
    outcode: Outcode,
}

/// Records whether a point is inside or outside of each frustum plane.
///
/// Each plane is represented by a single bit, 1 meaning "inside".
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Outcode(u8);

/// Visibility status of a polygon.
enum Status {
    /// Entirely inside view frustum
    Visible,
    /// Only partly inside, needs clipping
    Clipped,
    /// Entirely outside view frustum
    Hidden,
}

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
    pub fn clip_simple_polygon<A: Vary>(
        &self,
        verts_in: &[ClipVert<A>],
        verts_out: &mut Vec<ClipVert<A>>,
    ) {
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
                // have to do anything; it is discarded on the next iteration.
            }
            if d0 * d1 < 0.0 {
                // Edge crosses the plane surface. Split the edge in two by
                // interpolating and emitting a new vertex at intersection.
                // The new vertex becomes one of the endpoints of a new "clip"
                // edge coincident with the plane.

                // `t` is the fractional distance from `v0` to the intersection
                // point. If condition guarantees that `d1 - d0` is nonzero.
                let t = -d0 / (d1 - d0);

                verts_out.push(ClipVert {
                    pos: v0.pos.lerp(&v1.pos, t),
                    attrib: v0.attrib.lerp(&v1.attrib, t),
                    outcode: Outcode(0b111111), // inside!
                });
            }
            (v0, d0) = (v1, d1);
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
    use crate::geom::Plane;
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

    /// Returns the outcode of the given point.
    pub(super) fn outcode(pt: &ClipVec) -> Outcode {
        // Top Btm Rgt Lft Far Near
        //  1   2   4   8   16  32
        let code = PLANES
            .iter()
            .fold(0, |code, p| code << 1 | (p.signed_dist(pt) <= 0.0) as u8);

        Outcode(code)
    }

    /// Returns the visibility status of the polygon given by `vs`.
    pub(super) fn status<V>(vs: &[ClipVert<V>]) -> Status {
        let (all, any) = vs.iter().fold((!0, 0), |(all, any), v| {
            (all & v.outcode.0, any | v.outcode.0)
        });
        if any != 0b111111 {
            // If there's at least one plane that all vertices are outside of,
            // then the whole polygon is hidden
            Status::Hidden
        } else if all == 0b111111 {
            // If each vertex is inside all planes, the polygon is fully visible
            Status::Visible
        } else {
            // Otherwise the polygon may have to be clipped
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
pub fn clip_simple_polygon<'a, A: Vary>(
    planes: &[Plane<ClipVec>],
    verts_in: &'a mut Vec<ClipVert<A>>,
    verts_out: &'a mut Vec<ClipVert<A>>,
) {
    debug_assert!(verts_out.is_empty());

    for (i, p) in planes.iter().enumerate() {
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
    pub fn new(v: Vertex<ClipVec, V>) -> Self {
        ClipVert {
            pos: v.pos,
            attrib: v.attrib,
            outcode: outcode(&v.pos),
        }
    }
}

impl<A: Vary> Clip for [Tri<ClipVert<A>>] {
    type Item = Tri<ClipVert<A>>;

    fn clip(&self, planes: &[ClipPlane], out: &mut Vec<Self::Item>) {
        debug_assert!(out.is_empty());

        // Avoid unnecessary allocations by reusing these
        let mut verts_in = vec![];
        let mut verts_out = vec![];

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

            if let Some((a, rest)) = verts_out.split_first() {
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
    use crate::geom::vertex;
    use alloc::vec;

    use crate::math::vary::Vary;

    use super::view_frustum::*;
    use super::*;

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
        assert_eq!(outcode(&vec(0.0, 0.0, 0.0)).0, 0b111111);
        assert_eq!(outcode(&vec(1.0, 0.0, 0.0)).0, 0b111111);
        assert_eq!(outcode(&vec(0.0, -1.0, 0.0)).0, 0b111111);
        assert_eq!(outcode(&vec(0.0, 1.0, 1.0)).0, 0b111111);
    }

    #[test]
    fn outcode_outside() {
        // Top Btm Rgt Lft Far Near
        //  1   2   4   8   16  32

        // Outside near == 32
        assert_eq!(outcode(&vec(0.0, 0.0, -1.5)).0, 0b011111);
        // Outside right == 4
        assert_eq!(outcode(&vec(2.0, 0.0, 0.0)).0, 0b111011);
        // Outside bottom == 2
        assert_eq!(outcode(&vec(0.0, -1.01, 0.0)).0, 0b111101);
        // Outside far left == 16|8
        assert_eq!(outcode(&vec(-2.0, 0.0, 2.0)).0, 0b100111);
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
    fn tri_clip_all_planes_result_is_quad() {
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
        assert!(res.iter().all(in_bounds));
    }

    #[test]
    fn tri_clip_all_cases() {
        // Methodically go through every possible combination of every
        // vertex inside/outside of every plane, including degenerate cases.

        let xs = || (-1.5).vary(1.5, Some(5));

        let pts = || {
            xs().flat_map(move |x| {
                xs().flat_map(move |y| xs().map(move |z| vec(x, y, z)))
            })
        };

        let tris = pts().flat_map(|a| {
            pts().flat_map(move |b| pts().map(move |c| tri(a, b, c)))
        });

        let mut in_tris = 0;
        let mut out_tris = 0;
        for tr in tris {
            let res = &mut vec![];
            [tr].clip(&PLANES, res);
            assert!(
                res.iter().all(in_bounds),
                "clip returned oob vertex:\n  from: {:#?}\n  to: {:#?}",
                tr,
                &res
            );
            in_tris += 1;
            out_tris += res.len();
        }
        assert_eq!(in_tris, 5i32.pow(9));
        assert_eq!(out_tris, 1_033_639);
    }

    fn in_bounds(tri: &Tri<ClipVert<f32>>) -> bool {
        tri.0
            .iter()
            .flat_map(|v| v.pos.project_to_real().0)
            .all(|a| a.abs() <= 1.00001)
    }
}
