//! Mesh approximations of various geometric shapes.

#[cfg(feature = "std")]
mod lathe;
mod platonic;

use alloc::vec::Vec;

use retrofire_core::geom::{Mesh, Normal3, Tri, mesh::Builder, tri, vertex};
use retrofire_core::math::{Lerp, Vec3};

#[cfg(feature = "std")]
pub use lathe::*;

pub use platonic::*;
use retrofire_core::geom::mesh::Index;

pub trait Build<A, I: Index = u32>: Sized {
    fn build(self) -> Mesh<A, I>;

    fn builder(self) -> Builder<A, I> {
        self.build().into_builder()
    }
}

pub struct Icosphere {
    pub radius: f32,
    pub depth: u8,
}

impl Build<Normal3, u32> for Icosphere {
    fn build(self) -> Mesh<Normal3> {
        #[derive(Default)]
        struct Tessellator {
            coords: Vec<Vec3>,
            faces: Vec<Tri<u32>>,
            #[cfg(feature = "std")]
            map: std::collections::HashMap<(u32, u32), u32>,
            #[cfg(not(feature = "std"))]
            map: alloc::collections::BTreeMap<(u32, u32), u32>,
        }
        impl Tessellator {
            fn map_get(&mut self, i: u32, j: u32) -> u32 {
                *self.map.entry((i, j)).or_insert_with(|| {
                    let a: Vec3 = *i.index(&self.coords);
                    let ab = a.midpoint(j.index(&self.coords));
                    self.coords.push(ab);
                    self.coords.len() as u32 - 1
                })
            }
            fn recurse(&mut self, d: u8, i: u32, j: u32, k: u32) {
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

        let mut tess = Tessellator {
            coords: Icosahedron::COORDS.to_vec(),
            ..Default::default()
        };
        for [i, j, k] in Icosahedron::FACES.map(|ixs| ixs.map(u32::from)) {
            tess.recurse(self.depth, i, j, k);
        }
        let verts = tess.coords.iter().map(|&p| {
            vertex((p.normalize() * self.radius).to().to_pt(), p.normalize())
        });
        Mesh::new(tess.faces, verts)
    }
}
