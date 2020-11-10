use std::time::Instant;

use geom::mesh::{Mesh, VertexAttr};
use math::Linear;
use math::mat::Mat4;
use math::vec::*;
pub use stats::Stats;

use crate::raster::*;

pub mod raster;
pub mod vary;
pub mod stats;
mod hsr;

pub type Shader<'a, Vary, Uniform> = &'a dyn Fn(Fragment<Vary>, Uniform) -> Vec4;
pub type Plotter<'a> = &'a mut dyn FnMut(usize, usize, Vec4);

pub struct Renderer {
    transform: Mat4,
    projection: Mat4,
    viewport: Mat4,
    stats: Stats,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            transform: Mat4::default(),
            projection: Mat4::default(),
            viewport: Mat4::default(),
            stats: Stats::default(),
        }
    }

    pub fn set_transform(&mut self, mat: Mat4) {
        self.transform = mat;
    }

    pub fn set_projection(&mut self, mat: Mat4) {
        self.projection = mat;
    }

    pub fn set_viewport(&mut self, mat: Mat4) {
        self.viewport = mat;
    }

    pub fn render<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, sh: Shader<(Vec4, VA), FA>, pl: Plotter) -> Stats
    where VA: VertexAttr,
          FA: Copy + Default
    {
        let clock = Instant::now();

        self.transform(&mut mesh);
        self.projection(&mut mesh.verts);
        self.hidden_surface_removal(&mut mesh);
        Self::z_sort(&mut mesh);

        self.rasterize(mesh, sh, pl);

        self.stats.time_used += Instant::now() - clock;
        self.stats.frames += 1;
        self.stats
    }

    fn transform<VA: VertexAttr, FA>(&self, mesh: &mut Mesh<VA, FA>) {
        let tf = &self.transform;
        let Mesh { verts, vertex_attrs, .. } = mesh;

        for v in verts {
            *v = tf * *v;
        }
        for va in vertex_attrs.iter_mut().flatten() {
            va.transform(tf);
        }
    }

    fn projection(&self, verts: &mut Vec<Vec4>) {
        let proj = &self.projection;
        for v in verts {
            *v = proj * *v;
            *v = *v / v.w;
        };
    }

    pub fn viewport(&self, verts: &mut Vec<Vec4>) {
        let view = &self.viewport;
        for v in verts {
            *v = view * *v;
        }
    }

    fn hidden_surface_removal<VA, FA>(&mut self, mut mesh: &mut Mesh<VA, FA>)
    where VA: Copy + Linear<f32>, FA: Copy {
        self.stats.faces_in += mesh.faces.len();
        hsr::hidden_surface_removal(&mut mesh);
        self.stats.faces_out += mesh.faces.len();
    }

    pub fn z_sort<VA, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
        let Mesh { verts, faces, face_attrs, .. } = mesh;

        if let Some(ref attrs) = face_attrs {
            let mut v = faces.iter().zip(attrs).collect::<Vec<_>>();

            v.sort_unstable_by(|&(a, _), &(b, _)| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });

            let (f, a): (Vec<_>, Vec<_>) = v.into_iter()
                                            .unzip();

            *faces = f.into_iter().copied().collect();
            *face_attrs = Some(a.into_iter().copied().collect());
        } else {
            faces.sort_unstable_by(|&a, &b| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });
        }
    }

    pub fn rasterize<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, shade: Shader<(Vec4, VA), FA>, plot: Plotter)
    where VA: VertexAttr,
          FA: Copy + Default,
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs } = &mut mesh;

        let orig_verts = verts.clone();

        self.viewport(verts);

        for (i, &[a, b, c]) in faces.iter().enumerate() {
            let (av, bv, cv) = (verts[a], verts[b], verts[c]);
            let (ao, bo, co) = (orig_verts[a], orig_verts[b], orig_verts[c]);
            let (ava, bva, cva) = if let Some(attrs) = vertex_attrs {
                (attrs[a], attrs[b], attrs[c])
            } else {
                Default::default()
            };

            let fa = if let Some(attrs) = face_attrs {
                attrs[i]
            } else {
                FA::default()
            };

            tri_fill(Fragment { coord: av, varying: (ao, ava) },
                     Fragment { coord: bv, varying: (bo, bva) },
                     Fragment { coord: cv, varying: (co, cva) },
                     |frag| {
                         let col = shade(frag, fa);
                         plot(frag.coord.x as usize, frag.coord.y as usize, col);
                         self.stats.pixels += 1;
                     });
        }
    }
}
