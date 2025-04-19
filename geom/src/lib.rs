#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod intersect;
pub mod io;
pub mod solids;

/*pub fn triangulate(poly: &[Point2]) -> Vec<Tri<Point2>> {
    let mut res = Vec::new();

    for pt in poly {}

    res
}
*/
