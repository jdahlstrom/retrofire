use std::time::Instant;

use geom::{LineSeg, mesh::*, Polyline, Sprite};
use math::Linear;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use scene::{Obj, Scene};
pub use stats::Stats;
use util::{Buffer, color::Color};

use crate::hsr::Visibility;
use crate::raster::*;
use crate::vary::Varying;

mod hsr;
pub mod raster;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod tex;
pub mod text;
pub mod vary;

pub trait RasterOps<VA: Copy, FA> {
    fn shade(&mut self, _: Fragment<VA>, uniform: FA) -> Color;

    #[inline(always)]
    fn test(&mut self, _: Fragment<(f32, VA)>) -> bool { true }

    // TODO
    // fn blend(&mut self, _x: usize, _y: usize, c: Color) -> Color { c }

    fn output(&mut self, coord: (usize, usize), c: Color);

    #[inline(always)]
    fn rasterize(&mut self, frag: Fragment<(f32, VA)>, uniform: FA) -> bool {
        self.test(frag) && {
            let frag = without_depth(frag);
            let color = self.shade(frag, uniform);
            // TODO let color = self.blend(x, y, color);
            self.output(frag.coord, color);
            true
        }
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
        let d = self.buf.get_mut(x, y);
        if *d > z { *d = z; true } else { false }
    }

    pub fn clear(&mut self) {
        self.buf.data_mut().fill(f32::INFINITY);
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
        rdr.stats.faces_in += self.faces.len();

        let mvp = &rdr.modelview * &rdr.projection;

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
                rdr.stats.objs_out += 1;
                rdr.stats.faces_out += mesh.faces.len();

                if rdr.options.depth_sort { depth_sort(&mut mesh); }
                perspective_divide(&mut mesh, rdr.options.perspective_correct);

                mesh.verts.transform(&rdr.viewport);

                for Face { verts: [a, b, c], attr, .. } in mesh.faces() {
                    let verts = [with_depth(a), with_depth(b), with_depth(c)];
                    tri_fill(verts, |frag: Fragment<_>| {
                        raster.rasterize(frag, attr);
                        rdr.stats.pixels += 1;
                    });
                }
            }

            let mut render_edges = |rdr: &mut Renderer, edges: Vec<[Vec4; 2]>, color| {
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

            if let Some(color) = rdr.options.wireframes {
                render_edges(rdr, self.edges(), color);
            }
            if let Some(color) = rdr.options.bounding_boxes {
                render_edges(rdr, self.bbox.edges(), color);
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

impl<VA, FA> Render<VA, FA> for Sprite<VA, FA>
where
    VA: Linear<f32> + Copy,
    FA: Copy,
{
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<VA, FA>
    {
        rdr.stats.faces_in += 1;

        let mut this = *self;
        this.center.transform(&rdr.modelview);
        let scale = &rdr.modelview.row(0).len();
        this.width *= scale;
        this.height *= scale;

        let vs: Vec<_> = this.verts()
                .map(|mut v| { v.coord.transform(&rdr.projection); v })
                .collect();

        let vs: Vec<_> = hsr::clip(&vs).into_iter()
            .map(|v| clip_to_screen(v, &rdr.viewport))
            .collect();

        if let &[v0, v1, v2, v3] = &vs[..] {
            rdr.stats.faces_out += 1;

            let (x0, y0) = (v0.coord.x.round(), v0.coord.y.round());
            let (x1, y1) = (v2.coord.x.round(), v2.coord.y.round());
            let v = Varying::between((v0.attr, v1.attr), (v3.attr, v2.attr), y1 - y0);

            for (y, (v0, v1)) in (y0 as usize .. y1 as usize).zip(v) {
                let v = Varying::between(v0, v1, x1 - x0);
                for (x, v) in (x0 as usize .. x1 as usize).zip(v) {
                    let frag = Fragment { coord: (x, y), varying: v };
                    if raster.rasterize(frag, self.face_attr) {
                        rdr.stats.pixels += 1;
                    }
                }
            }
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

    pub fn render_scene<'a, VA, FA, Re, Ra>(
        &'a mut self,
        Scene { objects, camera }: &Scene<Re>,
        raster: &mut Ra,
    ) -> Stats
    where
        VA: Copy,
        FA: Copy,
        Re: Render<VA, FA>,
        Ra: RasterOps<VA, FA> + 'a
    {
        let clock = Instant::now();
        for Obj { tf, geom, .. } in objects {
            self.modelview = tf * camera;
            geom.render(self, raster);
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
