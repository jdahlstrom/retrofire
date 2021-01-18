use std::time::Instant;

use geom::mesh::{Face, Mesh, Vertex};
use math::Linear;
use math::mat::Mat4;
use math::transform::Transform;
pub use stats::Stats;
use util::Buffer;
use util::color::Color;

use crate::hsr::Visibility;
use crate::raster::*;

mod hsr;
pub mod raster;
pub mod shade;
pub mod stats;
pub mod tex;
pub mod vary;

pub trait RasterOps<VA: Copy, FA> {
    fn shade(&mut self, frag: Fragment<VA>, uniform: FA) -> Color;

    fn test(&mut self, _frag: Fragment<(f32, VA)>) -> bool { true }

    // TODO
    // fn blend(&mut self, _x: usize, _y: usize, c: Color) -> Color { c }

    fn output(&mut self, coord: (usize, usize), c: Color);
}

pub struct Raster<Shade, Test, Output> {
    pub shade: Shade,
    pub test: Test,
    pub output: Output,
}

impl<VA, FA, Shade, Test, Output> RasterOps<VA, FA>
for Raster<Shade, Test, Output>
where
    VA: Copy,
    Shade: FnMut(Fragment<VA>, FA) -> Color,
    Test: FnMut(Fragment<(f32, VA)>) -> bool,
    Output: FnMut((usize, usize), Color),
{
    fn shade(&mut self, frag: Fragment<VA>, uniform: FA) -> Color {
        (self.shade)(frag, uniform)
    }
    fn test(&mut self, frag: Fragment<(f32, VA)>) -> bool {
        (self.test)(frag)
    }
    fn output(&mut self, coord: (usize, usize), c: Color) {
        (self.output)(coord, c);
    }
}

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

pub struct DepthBuf {
    buf: Buffer<f32>,
}
impl DepthBuf {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            buf: Buffer::new(width, height, f32::INFINITY),
        }
    }

    pub fn test<V: Copy>(&mut self, frag: Fragment<(f32, V)>) -> bool {
        let Fragment { coord: (x, y), varying: (z, _) } = frag;
        let d = &mut self.buf.data[y * self.buf.width + x];
        if *d > z { *d = z; true } else { false }
    }

    pub fn clear(&mut self) {
        self.buf.data.fill(f32::INFINITY);
    }
}

#[derive(Default, Clone)]
pub struct Renderer {
    pub modelview: Mat4,
    pub projection: Mat4,
    pub viewport: Mat4,
    pub options: Options,
    pub stats: Stats,
}

#[derive(Copy, Clone, Default)]
pub struct Options {
    pub perspective_correct: bool,
    pub depth_sort: bool,
}

impl Renderer {
    pub fn new() -> Renderer {
        Self::default()
    }

    pub fn render_scene<VA, FA>(
        &mut self,
        Scene { objects, camera }: &Scene<VA, FA>,
        raster: &mut impl RasterOps<VA, FA>,
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        for Obj { tf, mesh } in objects {
            self.modelview = tf * camera;
            self.render(mesh, raster);
        }
        self.stats
    }

    pub fn render<VA, FA>(
        &mut self,
        mesh: &Mesh<VA, FA>,
        raster: &mut impl RasterOps<VA, FA>,
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        let Self {
            ref modelview, ref projection, ref viewport,
            ref options, stats
        } = self;

        let clock = Instant::now();

        stats.objs_in += 1;
        stats.faces_in += mesh.faces.len();

        let mvp = modelview * projection;

        let bbox_vis = {
            let mut vs = mesh.bbox.verts();
            vs.transform(&mvp);
            hsr::vertex_visibility(vs.iter())
        };

        if bbox_vis != Visibility::Hidden {
            let mut mesh = mesh.clone();

            mesh.verts.transform(&mvp);

            hsr::hidden_surface_removal(&mut mesh, bbox_vis);

            if !mesh.faces.is_empty() {
                stats.objs_out += 1;
                stats.faces_out += mesh.faces.len();

                if options.depth_sort {
                    Self::depth_sort(&mut mesh);
                }

                Self::perspective_divide(
                    &mut mesh, options.perspective_correct);

                mesh.verts.transform(viewport);

                Self::rasterize(mesh.faces(), raster, stats);
            }
        }
        stats.time_used += clock.elapsed();

        *stats
    }

    fn depth_sort<VA: Copy, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
        let (faces, attrs) = {
            let mut faces = mesh.faces().collect::<Vec<_>>();

            faces.sort_unstable_by(|a, b| {
                let az: f32 = a.verts.iter().map(|&v| v.coord.z).sum();
                let bz: f32 = b.verts.iter().map(|&v| v.coord.z).sum();
                bz.partial_cmp(&az).unwrap()
            });

            faces.into_iter().map(|f| (f.indices, f.attr)).unzip()
        };
        mesh.faces = faces;
        mesh.face_attrs = attrs;
    }

    fn perspective_divide<VA, FA>(mesh: &mut Mesh<VA, FA>, pc: bool)
    where VA: Linear<f32> + Copy
    {
        let Mesh { verts, vertex_attrs, .. } = mesh;
        if pc {
            for (v, a) in verts.iter_mut().zip(vertex_attrs) {
                let w = 1.0 / v.w;
                *v = v.mul(w);
                *a = a.mul(w);
            }
        } else {
            for v in verts.iter_mut() {
                let w = 1.0 / v.w;
                *v = v.mul(w);
            }
        }
    }

    pub fn rasterize<VA, FA>(
        faces: impl Iterator<Item=Face<VA, FA>>,
        raster: &mut impl RasterOps<VA, FA>,
        stats: &mut Stats
    ) where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        for Face { verts: [a, b, c], attr, .. } in faces {
            let verts = [
                Vertex { coord: a.coord, attr: (a.coord.z, a.attr) },
                Vertex { coord: b.coord, attr: (b.coord.z, b.attr) },
                Vertex { coord: c.coord, attr: (c.coord.z, c.attr) },
            ];
            tri_fill(verts, |frag: Fragment<_>| {
                if raster.test(frag) {
                    let frag = Fragment { coord: frag.coord, varying: frag.varying.1 };
                    let color = raster.shade(frag, attr);
                    // TODO let color = raster.blend(x, y, color);
                    raster.output(frag.coord, color);
                }
                stats.pixels += 1;
            });
        }
    }
}
