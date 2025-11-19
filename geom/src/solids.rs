//! Mesh approximations of various geometric shapes.

#[cfg(feature = "std")]
mod lathe;
mod platonic;

use alloc::vec::Vec;
use retrofire_core::geom::{
    Mesh, Normal3, Tri, Vertex3, mesh::Builder, tri, vertex,
};
use retrofire_core::math::{Color3, Lerp, ProjMat3, Vec3, pt3};

#[cfg(feature = "std")]
pub use lathe::*;

pub use platonic::*;
use retrofire_core::prelude::Frag;
use retrofire_core::render::tex::{Sample, SamplerClamp, cube_map};
use retrofire_core::render::{Shader, TexCoord, Texture, World, shader};
use retrofire_core::util::Buf2;

pub trait Build<A>: Sized {
    fn build(self) -> Mesh<A>;

    fn builder(self) -> Builder<A> {
        self.build().into_builder()
    }
}

pub struct Icosphere(pub f32, pub u8);

#[derive(Clone, Debug)]
pub struct Skybox(pub Mesh<Normal3>, pub Texture<Buf2<Color3>>);

impl Skybox {
    pub fn new(tex: impl Into<Texture<Buf2<Color3>>>) -> Self {
        let cube = Box {
            left_bot_near: pt3(-1.0, -1.0, -1.0),
            right_top_far: pt3(1.0, 1.0, 1.0),
        }
        .build();
        Self(cube, tex.into())
    }

    pub fn shader(
        &self,
    ) -> impl Shader<Vertex3<Normal3>, TexCoord, &ProjMat3<World>> {
        shader::new(
            |v: Vertex3<Normal3>, tf: &ProjMat3<World>| {
                vertex(
                    tf.apply(&v.pos.to()),
                    // TODO could bake cubemap texcoords to the cube
                    cube_map(v.pos.to(), v.attrib.to()),
                )
            },
            |frag: Frag<TexCoord>, _: &_| {
                //SamplerBilinear
                (SamplerClamp).sample(&self.1, frag.var).to_rgba()

                //dir_to_rgb(frag.var.to_vec().normalize_approx())
                //    .to_color4()
            },
        )
    }
}

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
