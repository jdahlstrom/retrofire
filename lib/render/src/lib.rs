use std::time::Instant;

use geom::mesh::{Mesh, VertexAttr};
use math::Linear;
use math::mat::Mat4;
use math::vec::*;
pub use stats::Stats;

use crate::raster::*;

mod hsr;
pub mod raster;
pub mod shade;
pub mod stats;
pub mod vary;

#[derive(Default, Clone)]
pub struct Obj<VA, FA> {
    pub tf: Mat4,
    pub mesh: Mesh<VA, FA>,
}

pub struct Scene<VA, FA> {
    pub objects: Vec<Obj<VA, FA>>,
    pub camera: Mat4,
}

#[derive(Default, Clone)]
pub struct Renderer {
    transform: Mat4,
    projection: Mat4,
    viewport: Mat4,
    zbuf: (Vec<f32>, usize),
    pub stats: Stats,
}

impl Renderer {
    pub fn new() -> Renderer {
        Self::default()
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

    pub fn set_z_buffer(&mut self, zbuf: Vec<f32>, pitch: usize) {
        self.zbuf = (zbuf, pitch);
    }

    pub fn render_scene<VA, FA, Shade, Plot>(
        &mut self, scene: Scene<VA, FA>, sh: &Shade, pl: &mut Plot
    ) where
        VA: VertexAttr,
        FA: Copy,
        Shade: Fn(Fragment<(Vec4, VA)>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        // TODO use .fill() once stable
        for x in &mut self.zbuf.0 { *x = f32::INFINITY; }

        for obj in scene.objects {
            self.set_transform(&obj.tf * &scene.camera);
            self.render(obj.mesh, sh, pl);
        }
    }

    pub fn render<VA, FA, Shade, Plot>(
        &mut self, mut mesh: Mesh<VA, FA>, shade: &Shade, plot: &mut Plot
    ) -> Stats
    where
        VA: VertexAttr,
        FA: Copy,
        Shade: Fn(Fragment<(Vec4, VA)>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        let clock = Instant::now();

        self.transform(&mut mesh);
        self.projection(&mut mesh.verts);
        self.hidden_surface_removal(&mut mesh);

        if !mesh.faces.is_empty() {
            //Self::z_sort(&mut mesh);
            self.perspective_divide(&mut mesh.verts);
            self.rasterize(mesh, shade, plot);
        }

        self.stats.time_used += Instant::now() - clock;
        self.stats
    }

    fn transform<VA: VertexAttr, FA>(&self, mesh: &mut Mesh<VA, FA>) {
        for v in &mut mesh.verts {
            *v = &self.transform * *v;
        }
    }

    fn projection(&self, verts: &mut Vec<Vec4>) {
        for v in verts {
            *v = &self.projection * *v;
        };
    }

    fn perspective_divide(&self, verts: &mut Vec<Vec4>) {
        for v in verts {
            *v = *v / v.w;
        };
    }

    fn viewport(&self, verts: &mut Vec<Vec4>) {
        for v in verts {
            *v = &self.viewport * *v;
        }
    }

    fn hidden_surface_removal<VA, FA>(&mut self, mut mesh: &mut Mesh<VA, FA>)
    where VA: Copy + Linear<f32>, FA: Copy
    {
        self.stats.faces_in += mesh.faces.len();
        hsr::hidden_surface_removal(&mut mesh);
        self.stats.faces_out += mesh.faces.len();
    }

    // TODO Replace with z-buffering (or s-buffering!)
    pub fn z_sort<VA, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
        let (faces, attrs) = {
            let Mesh { verts, faces, face_attrs, .. } = &*mesh;

            let mut v = faces.iter().zip(face_attrs).collect::<Vec<_>>();

            v.sort_unstable_by(|&(a, _), &(b, _)| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });

            v.into_iter().unzip()
        };

        mesh.faces = faces;
        mesh.face_attrs = attrs;
    }

    pub fn rasterize<VA, FA, Shade, Plot>(
        &mut self,
        mut mesh: Mesh<VA, FA>,
        shade: &Shade,
        plot: &mut Plot
    ) where
        VA: VertexAttr,
        FA: Copy,
        Shade: Fn(Fragment<(Vec4, VA)>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs } = &mut mesh;

        let orig_verts = verts.clone();

        self.viewport(verts);

        for (i, &[a, b, c]) in faces.iter().enumerate() {
            // TODO Clean up this mess
            let (av, bv, cv) = (verts[a], verts[b], verts[c]);
            let (ao, bo, co) = (orig_verts[a], orig_verts[b], orig_verts[c]);
            let (ava, bva, cva) = (vertex_attrs[a], vertex_attrs[b], vertex_attrs[c]);

            let fa = face_attrs[i];

            tri_fill(Fragment { coord: av, varying: (ao, ava) },
                     Fragment { coord: bv, varying: (bo, bva) },
                     Fragment { coord: cv, varying: (co, cva) },
                     |frag| {
                         let (x, y) = (frag.coord.x as usize, frag.coord.y as usize);
                         if frag.coord.z < self.zbuf.0[self.zbuf.1 * y + x] {
                             let col = shade(frag, fa);
                             plot(x, y, col);
                             self.zbuf.0[self.zbuf.1 * y + x] = frag.coord.z;
                             self.stats.pixels += 1;
                         }
                     });
        }
    }
}
