use std::ops::DerefMut;
use std::time::Instant;

use color::Color;
use hsr::Visibility;
use raster::Fragment;
use raster::tri_fill;
pub use stats::Stats;

use crate::{
    geom::{mesh::Mesh, Face, Vertex},
    math::{Linear, mat::Mat4, transform::Transform},
};

mod hsr;
pub mod color;
pub mod raster;
pub mod shade;
pub mod stats;
pub mod tex;

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

#[derive(Clone)]
pub struct Buffer<T, B: DerefMut<Target=[T]> = Vec<T>> {
    pub width: usize,
    pub height: usize,
    pub data: B,
}
impl<B: DerefMut<Target=[u8]>> Buffer<u8, B> {
    pub fn plot(&mut self, x: usize, y: usize, c: Color) {
        let idx = 4 * (self.width * y + x);
        let [_, r, g, b] = c.to_argb();
        self.data[idx + 0] = b;
        self.data[idx + 1] = g;
        self.data[idx + 2] = r;
    }
}
impl<T> Buffer<T, Vec<T>> {
    pub fn new(width: usize, height: usize, init: T) -> Self
    where T: Clone {
        Self {
            width, height,
            data: vec![init; width * height],
        }
    }
}
impl<'a, T> Buffer<T, &'a mut [T]> {
    pub fn borrow(width: usize, data: &'a mut [T]) -> Self {
        let height = data.len() / width;
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }
}

pub struct DepthBuf {
    buf: Buffer<f32>,
}
impl DepthBuf {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            buf: Buffer {
                width, height,
                data: vec![f32::INFINITY; width * height],
            }
        }
    }

    pub fn test<V: Copy>(&mut self, frag: Fragment<(f32, V)>) -> bool {
        let Fragment { coord: (x, y), varying: (z, _) } = frag;
        let d = &mut self.buf.data[y as usize * self.buf.width + x as usize];
        if *d > z { *d = z; true } else { false }
    }

    pub fn clear(&mut self) {
        let buf = &mut self.buf;
        buf.data.clear();
        buf.data.resize(buf.width * buf.height, f32::INFINITY);
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
        scene: &Scene<VA, FA>,
        raster: &mut impl RasterOps<VA, FA>,
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        for obj in &scene.objects {
            self.modelview = &obj.tf * &scene.camera;
            self.render(&obj.mesh, raster);
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
        let clock = Instant::now();

        self.stats.objs_in += 1;
        self.stats.faces_in += mesh.faces.len();

        let mvp = &self.modelview * &self.projection;

        let bbox_vis = {
            let vs = &mut mesh.bbox.verts();
            vs.transform(&mvp);
            hsr::vertex_visibility(vs.iter())
        };

        if bbox_vis != Visibility::Hidden {
            let mut mesh = mesh.clone();

            mesh.verts.transform(&mvp);

            hsr::hidden_surface_removal(&mut mesh, bbox_vis);

            if !mesh.faces.is_empty() {
                self.stats.faces_out += mesh.faces.len();
                self.stats.objs_out += 1;

                if self.options.depth_sort {
                    Self::depth_sort(&mut mesh);
                }

                self.perspective_divide(&mut mesh);

                self.rasterize(mesh, raster);
            }
        }

        self.stats.time_used += Instant::now() - clock;
        self.stats
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

    fn perspective_divide<VA, FA>(&self, mesh: &mut Mesh<VA, FA>)
    where VA: Linear<f32> + Copy
    {
        let Mesh { verts, vertex_attrs, .. } = mesh;
        if self.options.perspective_correct {
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
        &mut self,
        mut mesh: Mesh<VA, FA>,
        raster: &mut impl RasterOps<VA, FA>,
    ) where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        mesh.verts.transform(&self.viewport);

        for Face { verts: [a, b, c], attr, .. } in mesh.faces() {
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
                    let coord = (frag.coord.0 as usize, frag.coord.1 as usize);
                    raster.output(coord, color);
                }
                self.stats.pixels += 1;
            });
        }
    }
}
