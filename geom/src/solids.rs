//! Mesh approximations of various geometric shapes.

#[cfg(feature = "std")]
mod lathe;
mod platonic;

use alloc::vec::Vec;

use retrofire_core::geom::{Mesh, Normal3, Tri, mesh::Builder, tri, vertex};
use retrofire_core::math::{BezierSpline, Lerp, Mat4, Point2, Point3, Vec3};

#[cfg(feature = "std")]
pub use lathe::*;

pub use platonic::*;
use retrofire_core::math::spline::{BSpline, CatmullRomSpline};

pub trait Build<A>: Sized {
    fn build(self) -> Mesh<A>;

    fn builder(self) -> Builder<A> {
        self.build().into_builder()
    }
}

pub struct Icosphere(pub f32, pub u8);

impl Build<Normal3> for Icosphere {
    fn build(self) -> Mesh<Normal3> {
        #[derive(Default)]
        struct Tessellator {
            coords: Vec<Vec3>,
            faces: Vec<Tri<usize>>,
            #[cfg(feature = "std")]
            map: std::collections::HashMap<(usize, usize), usize>,
            #[cfg(not(feature = "std"))]
            map: alloc::collections::BTreeMap<(usize, usize), usize>,
        }
        impl Tessellator {
            fn map_get(&mut self, i: usize, j: usize) -> usize {
                *self.map.entry((i, j)).or_insert_with(|| {
                    let a: Vec3 = self.coords[i];
                    let ab = a.midpoint(&self.coords[j]);
                    self.coords.push(ab);
                    self.coords.len() - 1
                })
            }
            fn recurse(&mut self, d: u8, i: usize, j: usize, k: usize) {
                if d == 0 {
                    self.faces.push(tri(i, j, k));
                } else {
                    let ij = self.map_get(i, j);
                    let ik = self.map_get(i, k);
                    let jk = self.map_get(j, k);

                    self.recurse(d - 1, i, ij, ik);
                    self.recurse(d - 1, j, jk, ij);
                    self.recurse(d - 1, k, ik, jk);
                    self.recurse(d - 1, ij, jk, ik);
                }
            }
        }

        let mut recurser = Tessellator {
            coords: Icosahedron::COORDS.to_vec(),
            ..Default::default()
        };

        for [i, j, k] in Icosahedron::FACES {
            recurser.recurse(self.1, i, j, k);
        }

        let verts = recurser.coords.iter().map(|&p| {
            vertex((p.normalize() * self.0).to().to_pt(), p.normalize())
        });

        Mesh::new(recurser.faces, verts)
    }
}

pub fn extrude<B>(
    p: impl IntoIterator<Item = Point2>,
    frames: impl IntoIterator<Item = Mat4<B, B>>,
) -> Builder<(), B> {
    let mut b = Mesh::builder();
    let mut frames = frames.into_iter();
    let p: Vec<_> = p.into_iter().map(|p| p.to_pt3()).collect();
    let n = p.len();

    // first
    {
        let Some(f) = frames.next() else { return b };
        let p = p.iter().map(|pt| (f.apply(&pt.to()).to(), ()));
        b.push_verts(p);
    }

    for f in frames {
        let l = b.mesh.verts.len();
        for i in 0..n {
            let j = (i + 1) % n;
            b.push_face(l + i, l + i - n, l + j - n);
            b.push_face(l + i, l + j - n, l + j)
        }
        let p = p.iter().map(|pt| (f.apply(&pt.to()).to(), ()));
        b.push_verts(p);
    }
    b
}
