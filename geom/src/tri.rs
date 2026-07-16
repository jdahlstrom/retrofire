use alloc::vec::Vec;
use core::hint::cold_path;

use retrofire_core::geom::{Polygon, Pos, Tri, Winding, tri};
use retrofire_core::math::Point2;

/// Converts a polygon into a triangle mesh that partitions the polygon.
///
/// In other words, returns a set of triangles that exactly covers the input
/// polygon with no overlap. Given a polygon with *n* vertices, the returned
/// mesh will have exactly *n* vertices and *n*-2 faces.
///
/// This function is based on the so-called ear-clipping algorithm [TODO ref].
#[must_use]
pub fn triangulate<B, V>(poly: &Polygon<V>) -> Vec<Tri<usize>>
where
    V: Pos<Type = Point2<B>> + Clone,
{
    let points = &poly.0;
    let len = points.len();
    if len < 3 {
        cold_path();
        return Default::default();
    }
    let mut res = Vec::with_capacity(len - 2);

    // A very fast path for convex polygons (O(n) vs O(n²))
    if poly.is_convex() {
        res.extend((2..len).map(|i| tri(0, i - 1, i)));
        return res;
    }

    let mut clipper = Clipper {
        points: points.as_slice(),
        indices: (0..len).collect(),
        winding: poly.winding(),
    };
    for n in 0..len * len {
        if let Some(tri) = clipper.clip_if_ear(n % clipper.indices.len()) {
            res.push(tri);
            if clipper.indices.is_empty() {
                return res;
            }
        }
    }
    panic!("triangulation stuck, polygon is likely self-intersecting");

    /* eprintln!(
            "Triangulated {}-gon -> {} triangles in {count} iters ({:.3} ms)",
            poly.0.len(),
            faces.len(),
            start.elapsed().as_secs_f32() * 1000.0
        );

        debug_assert_eq!(res.faces.len(), pts.len() - 2);
        debug_assert_eq!(res.faces.capacity(), pts.len() - 2);
    */
}

/// Helper type for implementing the ear clipping triangulation algorithm.
struct Clipper<'a, V> {
    points: &'a [V],
    indices: Vec<usize>,
    winding: Winding,
}

impl<B, V: Pos<Type = Point2<B>>> Clipper<'_, V> {
    fn clip_if_ear(&mut self, i: usize) -> Option<Tri<usize>> {
        let j = self.next(i);
        if self.is_convex(j) && !self.contains_point(j) {
            let k = self.next(j);
            // Valid ear -> clip it
            let res = tri(self.indices[i], self.indices[j], self.indices[k]);
            self.indices.remove(j);
            Some(res)
        } else {
            None
        }
    }
    /// Returns true iff the vertex at ixs[j] is convex, that is, its winding
    /// is equal to the winding of the entire polygon. Convex vertices are
    /// candidates for clipping, non-convex (aka reflex) vertices never are.
    fn is_convex(&self, j: usize) -> bool {
        let i = self.prev(j);
        let k = self.next(j);
        self.winding(i, j, k) == self.winding
    }

    /// Returns the winding of the triangle given by the indices i, j, and k.
    /// Assumes that the indices are valid.
    fn winding(&self, i: usize, j: usize, k: usize) -> Winding {
        let Self { points: pts, indices: ixs, .. } = &self;
        tri(&pts[ixs[i]], &pts[ixs[j]], &pts[ixs[k]]).winding()
    }

    /// Returns true iff the triangle at ixs[j] contains any other
    /// vertex.
    /// Only vertices that are reflex need to be considered, because if *any*
    /// vertices are contained, at least one of them must be reflex.
    fn contains_point(&self, j: usize) -> bool {
        let pts = &self.points;
        let ixs = &self.indices;

        let i = self.prev(j);
        let k = self.next(j);

        let tri = Tri([i, j, k].map(|h| *pts[ixs[h]].pos()));

        for h in 0..ixs.len() {
            if [i, j, k].contains(&h) {
                continue;
            }
            // TODO this should be precomputed
            // TODO once all are convex, fast path out
            if self.is_convex(h) {
                continue;
            }
            if tri.contains(pts[ixs[h]].pos().clone()) {
                return true;
            }
        }
        false
    }

    #[inline]
    fn next(&self, i: usize) -> usize {
        if i + 1 == self.indices.len() {
            0
        } else {
            i + 1
        }
    }

    #[inline]
    fn prev(&self, i: usize) -> usize {
        (if i == 0 { self.indices.len() } else { i }) - 1
    }
}

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    use retrofire_core::math::pt2;

    #[test]
    fn triangulate_tri() {
        let pts: [Point2; _] = [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0)];
        let p = Polygon::new(pts);
        let tris = triangulate(&p);
        assert_eq!(tris, [tri(0, 1, 2),]);
    }
    #[test]
    fn triangulate_quad() {
        let [a, b, c, d]: [Point2; _] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0), pt2(0.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        // FIXME assert_eq!(t, [tri(a, b, c), tri(a, c, d)]);
    }
    #[test]
    fn triangulate_concave_quad() {
        let [a, b, c, d]: [Point2; _] =
            [pt2(0.0, 0.0), pt2(0.5, 0.2), pt2(1.0, 0.0), pt2(0.5, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        // FIXME assert_eq!(t, [tri(b, c, d), tri(a, b, d)]);
    }
    #[test]
    fn triangulate_concave_6gon() {
        let [a, b, c, d, e, f]: [Point2; _] = [
            pt2(0.0, 0.0),
            pt2(1.0, 0.0),
            pt2(0.4, 0.8),
            pt2(1.0, 1.0),
            pt2(0.0, 1.0),
            pt2(0.6, 0.2),
        ];
        let p = Polygon::new([a, b, c, d, e, f]);
        let t = triangulate(&p);
        // FIXME assert_eq!(t, [tri(c, d, e), tri(f, a, b), tri(c, e, f), tri(b, c, f)]);
    }
    #[test]
    #[should_panic = "triangulation stuck, polygon is likely self-intersecting"]
    fn triangulate_self_intersecting() {
        let [a, b, c, d]: [Point2; _] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let _ = triangulate(&p);
    }
}
