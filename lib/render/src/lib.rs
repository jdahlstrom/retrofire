use std::time::Instant;

use geom::mesh::Mesh;
use math::Linear;
use math::mat::Mat4;
use math::vec::*;
pub use stats::Stats;

use crate::raster::*;
use crate::hsr::frontface;

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
        &mut self, scene: Scene<VA, FA>, sh: &Shade, pl: &mut Plot
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        for obj in scene.objects {
            self.set_transform(&obj.tf * &scene.camera);
            self.render(obj.mesh, sh, pl);
        }
        self.stats
    }

    pub fn render<VA, FA, Shade, Plot>(
        &mut self, mut mesh: Mesh<VA, FA>, shade: &Shade, plot: &mut Plot
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        let clock = Instant::now();

        self.transform(&mut mesh.verts);
        mesh.bbox = mesh.bbox.transform(&self.transform);
        self.projection(&mut mesh.verts);

        self.hidden_surface_removal(&mut mesh);

        if !mesh.faces.is_empty() {
            Self::z_sort(&mut mesh);
            self.perspective_divide(&mut mesh.verts);
            self.rasterize(mesh, shade, plot);
        }

        self.stats.time_used += Instant::now() - clock;
        self.stats
    }

    fn transform(&self, verts: &mut Vec<Vec4>) {
        for v in verts {
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

        let mut vs = mesh.bbox.verts();

        for v in &mut vs {
            *v = &self.projection * *v;
        }

        match hsr::bbox_test(&vs) {
            0 => {
                //eprintln!("Culled");
                mesh.faces.clear();
            }, // culled
            1 => {
                //eprintln!("Clipped");
                hsr::hidden_surface_removal(&mut mesh);
            }
            _ => {
                let Mesh { faces, verts, face_attrs, .. } = mesh;
                let mut visible_faces = Vec::with_capacity(faces.len() / 2);
                let mut visible_attrs = Vec::with_capacity(faces.len() / 2);

                for (i, &mut [a,b,c]) in faces.into_iter().enumerate() {
                    if frontface(&[verts[a], verts[b], verts[c]]) {
                        visible_faces.push([a,b,c]);
                        visible_attrs.push(face_attrs[i]);
                    }
                }
                mesh.faces = visible_faces;
                mesh.face_attrs = visible_attrs;
                //eprintln!("Inside");
            },
        }
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
        VA: Copy + Linear<f32>,
        FA: Copy,
        Shade: Fn(Fragment<VA>, FA) -> Vec4,
        Plot: FnMut(usize, usize, Vec4),
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs, .. } = &mut mesh;

        self.viewport(verts);

        for (i, face) in faces.iter().enumerate() {
            let frags = face.iter()
                            .map(|&vi| Fragment {
                                coord: verts[vi],
                                varying: vertex_attrs[vi]
                            })
                            .collect::<Vec<_>>();

            let fa = face_attrs[i];

            tri_fill(frags[0], frags[1], frags[2],
                     |frag| {
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
