use std::time::Instant;

use geom::{LineSeg, Polyline, Sprite};
use geom::mesh::*;
use math::Linear;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use scene::{Obj, Scene};
pub use stats::Stats;
use util::Buffer;
use util::color::Color;

use crate::hsr::Visibility;
use crate::raster::*;

mod hsr;
pub mod raster;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod tex;
pub mod vary;

pub trait RasterOps<VA: Copy, FA> {
    fn shade(&mut self, frag: Fragment<VA>, uniform: FA) -> Color;

    #[inline(always)]
    fn test(&mut self, _frag: Fragment<(f32, VA)>) -> bool { true }

    // TODO
    // fn blend(&mut self, _x: usize, _y: usize, c: Color) -> Color { c }

    fn output(&mut self, coord: (usize, usize), c: Color);

    #[inline(always)]
    fn rasterize(&mut self, frag: Fragment<(f32, VA)>, uniform: FA) -> bool {
        if self.test(frag) {
            let frag = without_depth(frag);
            let color = self.shade(frag, uniform);
            // TODO let color = self.blend(x, y, color);
            self.output(frag.coord, color);
            true
        } else { false }
    }
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

pub trait Render<VA: Copy, FA: Copy> {
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<VA, FA>;
}

impl<VA, FA> Render<VA, FA> for Mesh<VA, FA>
where
    VA: Linear<f32> + Copy,
    FA: Copy,
{
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<VA, FA>
    {
        let Renderer {
            ref modelview, ref projection, ref viewport, stats, ..
        } = rdr;
        let options = rdr.options;

        stats.faces_in += self.faces.len();

        let mvp = modelview * projection;

        let bbox_vis = {
            let mut vs = self.bbox.verts();
            vs.transform(&mvp);
            hsr::vertex_visibility(vs.iter())
        };

        if bbox_vis != Visibility::Hidden {
            let mut mesh = (*self).clone();

            mesh.verts.transform(&mvp);

            hsr::hidden_surface_removal(&mut mesh, bbox_vis);

            if !mesh.faces.is_empty() {
                stats.objs_out += 1;
                stats.faces_out += mesh.faces.len();

                if options.depth_sort { depth_sort(&mut mesh); }
                perspective_divide(&mut mesh, options.perspective_correct);

                mesh.verts.transform(viewport);

                for Face { verts: [a, b, c], attr, .. } in mesh.faces() {
                    let verts = [with_depth(a), with_depth(b), with_depth(c)];
                    tri_fill(verts, |frag: Fragment<_>| {
                        raster.rasterize(frag, attr);
                        stats.pixels += 1;
                    });
                }
            }

            let mut render_edges = |edges: Vec<[Vec4; 2]>, color| {
                for edge in edges.into_iter()
                    .map(|[a, b]| [vertex(a, ()), vertex(b, ())])
                {
                    LineSeg(edge).render(rdr, &mut Raster {
                        shade: |_, _| color,
                        test: |_| true,
                        output: |crd, clr| raster.output(crd, clr),
                    });
                }
            };

            if let Some(color) = options.wireframes {
                render_edges(self.edges(), color);
            }
            if let Some(color) = options.bounding_boxes {
                render_edges(self.bbox.edges(), color);
            }
        }
    }
}

impl<VA> Render<VA, ()> for LineSeg<VA>
where
    VA: Linear<f32> + Copy
{
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<VA, ()>
    {
        rdr.stats.faces_in += 1;

        let mut this = (*self).clone();
        let mvp = &rdr.modelview * &rdr.projection;
        this.transform(&mvp);

        if let Some(&[a, b]) = hsr::clip(&this.0).get(0..2) {
            rdr.stats.faces_out += 1;
            let verts = [
                clip_to_screen(a, &rdr.viewport),
                clip_to_screen(b, &rdr.viewport)
            ];
            line(verts, |frag: Fragment<_>| {
                if raster.rasterize(frag, ()) {
                    rdr.stats.pixels += 1;
                }
            });
        }
    }
}

impl<VA> Render<VA, ()> for Polyline<VA>
where
    VA: Linear<f32> + Copy
{
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<VA, ()>
    {
        for seg in self.edges().map(LineSeg) {
            seg.render(rdr, raster);
        }
    }
}

impl<A> Render<A, ()> for Sprite<A>
where
    A: Linear<f32> + Copy
{
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<A, ()>
    {
        rdr.stats.faces_in += 1;

        let mut this = *self;
        this.center.transform(&rdr.modelview);

        let vs: Vec<_> = this.verts()
                .map(|mut v| { v.coord.transform(&rdr.projection); v })
                .collect();

        let vs: Vec<_> = hsr::clip(&vs).into_iter()
            .map(|v| clip_to_screen(v, &rdr.viewport))
            .collect();

        if !vs.is_empty() {
            rdr.stats.faces_out += 1;
            tri_fill([vs[0], vs[1], vs[2]], |frag|
                if raster.rasterize(frag, ()) {
                    rdr.stats.pixels += 1;
                });
            tri_fill([vs[0], vs[2], vs[3]], |frag|
                if raster.rasterize(frag, ()) {
                    rdr.stats.pixels += 1;
                });
        }
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
    pub wireframes: Option<Color>,
    pub bounding_boxes: Option<Color>,
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
        let clock = Instant::now();
        for Obj { tf, mesh } in objects {
            self.modelview = tf * camera;
            mesh.render(self, raster);
        }
        self.stats.objs_in += objects.len();
        self.stats.time_used += clock.elapsed();
        self.stats
    }

    pub fn render<VA, FA>(
        &mut self,
        mesh: &Mesh<VA, FA>,
        raster: &mut impl RasterOps<VA, FA>
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        let clock = Instant::now();

        mesh.render(self, raster);
        
        self.stats.objs_in += 1;
        self.stats.time_used += clock.elapsed();
        self.stats
    }
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

#[inline(always)]
fn with_depth<VA>(v: Vertex<VA>) -> Vertex<(f32, VA)> {
    Vertex { coord: v.coord, attr: (v.coord.z, v.attr) }
}

#[inline(always)]
fn without_depth<V>(f: Fragment<(f32, V)>) -> Fragment<V> {
    Fragment { coord: f.coord, varying: f.varying.1 }
}

fn clip_to_screen<A>(mut v: Vertex<A>, viewport: &Mat4) -> Vertex<(f32, A)> {
    v.coord /= v.coord.w;
    v.coord.transform(viewport);
    with_depth(v)
}
