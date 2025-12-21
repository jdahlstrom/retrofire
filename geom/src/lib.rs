#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod io;
pub mod isect;
pub mod solids;

pub use isect::Intersect;

use alloc::vec::Vec;
use core::fmt::Debug;
use retrofire_core::geom::{Edge, Polygon, Tri, tri};
use retrofire_core::math::{ApproxEq, Point2};
use std::eprintln;

pub fn triangulate<B: Debug + Default>(
    poly: &Polygon<Point2<B>>,
) -> Vec<Tri<Point2<B>>> {
    let mut pts = poly.0.clone();

    if pts.len() < 3 {
        return Vec::new();
    }

    let mut res = Vec::with_capacity(pts.len() - 2);
    let mut n = 0;
    'outer: loop {
        //eprintln!("n={n}:");

        if n > poly.0.len() * poly.0.len() {
            break;
        }

        if pts.len() == 3 {
            //eprintln!("only one triangle left, add to result and return");
            res.push(tri(pts[0], pts[1], pts[2]));
            break;
        }
        let i = (n + 0) % pts.len();
        let j = (n + 1) % pts.len();
        let k = (n + 2) % pts.len();

        let a = pts[i];
        let b = pts[j];
        let c = pts[k];

        //eprintln!("points left: {}, {pts:?}", pts.len());

        /*eprintln!(
            "Considering tri {i}={:?}, {j}={:?}, {k}={:?}...",
            a.0, b.0, c.0
        );*/

        n += 1;

        if (b - a).perp_dot(c - a) <= 0.0 {
            //eprintln!("-> wrong winding, skip");
            continue;
        }

        let ac = Edge(a, c);

        for e in poly.edges() {
            let e = Edge(*e.0, *e.1);
            if let Some(pt) = ac.intersect(&e)
                && !pt.approx_eq(&e.0)
                && !pt.approx_eq(&e.1)
                && !pt.approx_eq(&a)
                && !pt.approx_eq(&c)
            {
                /*eprintln!(
                    "-> edge {:?},{:?} intersects {:?},{:?} at {:?}\n-> skip",
                    a.0, c.0, e.0.0, e.1.0, pt.0
                );*/
                continue 'outer;
            }
        }
        // ac is inside and does not intersect any edge -> clip the ear
        pts.remove(j);
        //eprintln!("-> is an ear, clip it");
        res.push(tri(a, b, c));
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrofire_core::math::pt2;

    #[test]
    fn triangulate_tri() {
        let pts = [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0)];
        let p = Polygon::new(pts);
        let t = triangulate(&p);
        assert_eq!(t, [Tri(pts)]);
    }
    #[test]
    fn triangulate_quad() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0), pt2(0.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(a, b, c), tri(a, c, d)]);
    }
    #[test]
    fn triangulate_concave_quad() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(0.5, 0.2), pt2(1.0, 0.0), pt2(0.5, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(b, c, d), tri(a, b, d)]);
    }
    #[test]
    fn triangulate_concave_6gon() {
        let [a, b, c, d, e, f] = [
            pt2(0.0, 0.0),
            pt2(1.0, 0.0),
            pt2(0.4, 0.8),
            pt2(1.0, 1.0),
            pt2(0.0, 1.0),
            pt2(0.6, 0.2),
        ];
        let p = Polygon::new([a, b, c, d, e, f]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(c, d, e), tri(f, a, b), tri(c, e, f), tri(b, c, f)]);
    }
    #[test]
    fn triangulate_self_intersecting() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, []);
    }
}
