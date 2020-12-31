use std::time::Instant;

use color::Color;
use geom::bbox::BoundingBox;
use geom::mesh::Mesh;
use math::Linear;
use math::mat::Mat4;
use math::vec::*;
pub use stats::Stats;

use crate::hsr::Visibility;
use crate::raster::*;

mod hsr;
pub mod raster;
pub mod shade;
pub mod stats;
pub mod vary;
pub mod color;
pub mod tex;

#[derive(Default, Clone)]
pub struct Obj<VA, FA> {
    pub tf: Mat4,
    pub mesh: Mesh<VA, FA>,
}

#[derive(Default, Clone)]
pub struct Scene<VA, FA> {
    pub objects: Vec<Obj<VA, FA>>,
    pub camera: Mat4,
}

#[derive(Default, Clone)]
pub struct Renderer {
    transform: Mat4,
    projection: Mat4,
    viewport: Mat4,
    pub stats: Stats,
    pub options: Options,
}

#[derive(Copy, Clone, Default)]
pub struct Options {
    pub perspective_correct: bool,
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

    pub fn render_scene<VA, FA, Shade, Plot>(
        &mut self, scene: &Scene<VA, FA>, sh: &Shade, pl: &mut Plot
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Color,
        Plot: FnMut(usize, usize, Color),
    {
        for obj in &scene.objects {
            self.set_transform(&obj.tf * &scene.camera);
            self.render(&obj.mesh, sh, pl);
        }
        self.stats
    }

    pub fn render<VA, FA, Shade, Plot>(
        &mut self, mesh: &Mesh<VA, FA>, shade: &Shade, plot: &mut Plot
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Color,
        Plot: FnMut(usize, usize, Color),
    {
        let clock = Instant::now();

        self.stats.objs_in += 1;
        self.stats.faces_in += mesh.faces.len();

        let bbox_vis = self.bbox_visibility(mesh.bbox);
        if bbox_vis != Visibility::Hidden {
            let mut mesh = mesh.clone();

            self.transform(&mut mesh.verts);
            self.projection(&mut mesh.verts);

            self.hidden_surface_removal(&mut mesh, bbox_vis);

            if !mesh.faces.is_empty() {
                self.stats.faces_out += mesh.faces.len();
                self.stats.objs_out += 1;

                Self::z_sort(&mut mesh);

                self.perspective_divide(&mut mesh);

                self.rasterize(mesh, shade, plot);
            }
        }

        self.stats.time_used += Instant::now() - clock;
        self.stats
    }

    fn bbox_visibility(&self, bbox: BoundingBox) -> Visibility {
        let vs = &mut bbox.verts();
        self.transform(vs);
        self.projection(vs);

        hsr::bbox_visibility(vs)
    }

    fn transform(&self, verts: &mut [Vec4]) {
        for v in verts {
            *v = &self.transform * *v;
        }
    }

    fn projection(&self, verts: &mut [Vec4]) {
        for v in verts {
            *v = &self.projection * *v;
        };
    }

    fn viewport(&self, verts: &mut [Vec4]) {
        for v in verts {
            *v = &self.viewport * *v;
        }
    }

    fn hidden_surface_removal<VA, FA>(
        &mut self, mut mesh: &mut Mesh<VA, FA>, bbox_vis: Visibility
    )
    where VA: Copy + Linear<f32>, FA: Copy
    {
        hsr::hidden_surface_removal(&mut mesh, bbox_vis);
    }

    // TODO Replace with z-buffering (or s-buffering!)
    fn z_sort<VA, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
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

    fn perspective_divide<VA, FA>(&self, mesh: &mut Mesh<VA, FA>)
    where VA: Linear<f32> + Copy
    {
        if self.options.perspective_correct {
            for (v, a) in mesh.verts.iter_mut().zip(mesh.vertex_attrs.iter_mut()) {
                let w = 1.0 / v.w;
                *v = v.mul(w);
                *a = a.mul(w);
            }
        } else {
            for v in mesh.verts.iter_mut() {
                let w = 1.0 / v.w;
                *v = v.mul(w);
            }
        }
    }

    pub fn rasterize<VA, FA, Shade, Plot>(
        &mut self,
        mut mesh: Mesh<VA, FA>,
        shade: &Shade,
        plot: &mut Plot
    ) where
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Color,
        Plot: FnMut(usize, usize, Color),
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs, .. } = &mut mesh;

        self.viewport(verts);

        for (i, face) in faces.iter().enumerate() {
            let frags = face.iter().map(|&vi| Fragment {
                coord: verts[vi],
                varying: vertex_attrs[vi]
            }).collect::<Vec<_>>();

            let fa = face_attrs[i];

            tri_fill(frags[0], frags[1], frags[2], |frag| {
                plot(
                    frag.coord.x as usize,
                    frag.coord.y as usize,
                    shade(frag, fa)
                );
                self.stats.pixels += 1;
            });
        }
    }
}
