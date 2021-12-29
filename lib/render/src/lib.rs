use std::mem::swap;
use std::time::Instant;

use geom::{LineSeg, mesh::*, Polyline, Sprite};
use math::Linear;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use scene::{Obj, Scene};
pub use stats::Stats;
use util::color::Color;
use util::buf::Buffer;

use crate::hsr::Visibility;
use crate::raster::*;
use crate::shade::{Shader, ShaderImpl};
use crate::vary::Varying;

mod hsr;
pub mod fx;
pub mod raster;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod text;
pub mod vary;

pub trait RasterOps {
    #[inline(always)]
    fn test<U>(&self, _: Fragment<f32, U>) -> bool { true }

    // TODO
    // fn blend(&mut self, _x: usize, _y: usize, c: Color) -> Color { c }

    fn output<U>(&mut self, _: Fragment<(f32, Color), U>);
}

pub struct Raster<Test, Output> {
    pub test: Test,
    pub output: Output,
}

impl<Test, Output> RasterOps for Raster<Test, Output>
where
    Test: Fn(Fragment<f32>) -> bool,
    Output: FnMut(Fragment<(f32, Color)>),
{
    #[inline(always)]
    fn test<U>(&self, frag: Fragment<f32, U>) -> bool {
        (self.test)(frag.uniform(()))
    }
    #[inline(always)]
    fn output<U>(&mut self, frag: Fragment<(f32, Color), U>) {
        (self.output)(frag.uniform(()));
    }
}


pub struct Framebuf<'a> {
    // TODO Support other color buffer formats
    pub color: Buffer<u8, &'a mut [u8]>,
    pub depth: &'a mut Buffer<f32>,
}

impl<'a> Framebuf<'a> {

}

impl<'a> RasterOps for Framebuf<'a> {

    fn test<U>(&self, f: Fragment<f32, U>) -> bool {
        f.varying < *self.depth.get(f.coord.0, f.coord.1)
    }

    fn output<U>(&mut self, f: Fragment<(f32, Color), U>) {
        let ((x, y), (z, col)) = (f.coord, f.varying);
        let [_, r, g, b] = col.to_argb();
        let idx = 4 * (self.color.width() * y + x);
        let data = self.color.data_mut();
        data[idx + 0] = b;
        data[idx + 1] = g;
        data[idx + 2] = r;

        *self.depth.get_mut(x, y) = z;
    }
}

pub trait Render<U, VI, FI=VI> {

    fn render<S, R>(&self, rdr: &mut Renderer, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VI, FI>,
        R: RasterOps;

    #[inline(always)]
    fn rasterize<S, R, VO>(
        &self, shade: &mut S, raster: &mut R,
        frag: Fragment<(f32, VO), U>,
    ) -> bool
    where
        U: Copy,
        VO: Copy,
        S: Shader<U, VI, VO>,
        R: RasterOps,
    {
        let (z, a) = frag.varying;
        raster.test(frag.varying(z)) && {
            shade
                .shade_fragment(frag.varying(a))
                .map(|col| {
                    // TODO let c = self.blend(x, y, c);
                    raster.output(frag.varying((z, col)))
                })
                .is_some()
        }
    }
}

impl<VI, U> Render<U, VI> for Mesh<VI, U>
where
    VI: Linear<f32> + Copy,
    U: Copy,
{
    fn render<S, R>(&self, rdr: &mut Renderer, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI>,
        R: RasterOps
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

            // TODO
            for (c, a) in mesh.verts.iter_mut().zip(mesh.vertex_attrs.iter_mut()) {
                let v = shader.shade_vertex(vertex(*c, *a));
                *c = v.coord;
                *a = v.attr;
            }

            hsr::hidden_surface_removal(&mut mesh, bbox_vis);

            if !mesh.faces.is_empty() {
                rdr.stats.objs_out += 1;
                rdr.stats.faces_out += mesh.faces.len();

                if rdr.options.depth_sort { depth_sort(&mut mesh); }
                perspective_divide(&mut mesh, rdr.options.perspective_correct);

                mesh.verts.transform(&rdr.viewport);

                for Face { verts, attr, .. } in mesh.faces() {
                    let verts = verts.map(with_depth);
                    tri_fill(verts, |frag| {
                        if self.rasterize(shader, raster, frag.uniform(attr)) {
                            rdr.stats.pixels += 1;
                        }
                    });
                }
            }

            let mut render_edges = |rdr: &mut _,
                                    edges: Vec<[Vec4; 2]>,
                                    col: Color| {
                for edge in edges.into_iter()
                    .map(|[a, b]| [vertex(a, col), vertex(b, col)])
                    .map(LineSeg)
                {
                    edge.render(rdr, &mut ShaderImpl {
                        vs: |a| a,
                        fs: |f: Fragment<_>| Some(f.varying),
                    }, &mut Raster {
                        test: |_| true,
                        output: |f| raster.output(f),
                    });
                }
            };
            if let Some(col) = rdr.options.wireframes {
                render_edges(rdr, self.edges(), col);
            }
            if let Some(col) = rdr.options.bounding_boxes {
                render_edges(rdr, self.bbox.edges(), col);
            }
        }
    }
}

impl<VI> Render<(), VI> for LineSeg<VI>
where
    VI: Linear<f32> + Copy
{
    fn render<S, R>(&self, rdr: &mut Renderer, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), VI>,
        R: RasterOps
    {
        rdr.stats.faces_in += 1;

        let mut this = (*self).clone();
        let mvp = &rdr.modelview * &rdr.projection;
        this.transform(&mvp);

        for v in &mut this.0 {
            *v = shader.shade_vertex(*v);
        }
        let mut clip_out = Vec::new();
        hsr::clip(&mut this.0.to_vec(), &mut clip_out);
        if let &[a, b] = clip_out.as_slice() {
            rdr.stats.faces_out += 1;
            let verts = [
                clip_to_screen(a, &rdr.viewport),
                clip_to_screen(b, &rdr.viewport)
            ];
            line(verts, |frag: Fragment<_>| {
                if self.rasterize(shader, raster, frag) {
                    rdr.stats.pixels += 1;
                }
            });
        }
    }
}

impl<V> Render<(), V> for Polyline<V>
where
    V: Linear<f32> + Copy
{
    fn render<S, R>(&self, rdr: &mut Renderer, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V>,
        R: RasterOps,
    {
        for seg in self.edges() {
            seg.render(rdr, shader, raster);
        }
    }
}

impl<U, V> Render<U, V> for Sprite<V, U>
where
    U: Copy,
    V: Linear<f32> + Copy,
{
    fn render<S, R>(&self, rdr: &mut Renderer, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, V>,
        R: RasterOps,
    {
        rdr.stats.faces_in += 1;

        let mut this = *self;
        this.anchor.transform(&rdr.modelview);
        let scale = &rdr.modelview.row(0).len();
        this.width *= scale;
        this.height *= scale;

        let mut vs: Vec<_> = this.verts()
                .map(|v| {
                    let mut v = shader.shade_vertex(v);
                    v.coord.transform(&rdr.projection);
                    v
                })
                .collect();

        let mut clip_out = Vec::new();
        hsr::clip(&mut vs, &mut clip_out);
        let mut vs: Vec<_> = clip_out.into_iter()
            .map(|v| clip_to_screen(v, &rdr.viewport))
            .collect();

        match vs.as_mut_slice() {
            [] => {}
            [v0, v1, v2, v3] => {
                rdr.stats.faces_out += 1;

                if v0.coord.y > v2.coord.y { swap(v0, v3); swap(v1, v2); }
                if v0.coord.x > v1.coord.x { swap(v0, v1); swap(v2, v3); }

                // TODO extract to fn rect_fill
                let (x0, y0) = (v0.coord.x.round(), v0.coord.y.round());
                let (x1, y1) = (v2.coord.x.round(), v2.coord.y.round());
                let v = Varying::between((v0.attr, v1.attr), (v3.attr, v2.attr), y1 - y0);

                for (y, (v0, v1)) in (y0 as usize..y1 as usize).zip(v) {
                    let v = Varying::between(v0, v1, x1 - x0);
                    for (x, v) in (x0 as usize..x1 as usize).zip(v) {
                        let frag = Fragment {
                            coord: (x, y),
                            varying: v,
                            uniform: this.face_attr
                        };
                        if self.rasterize(shader, raster, frag) {
                            rdr.stats.pixels += 1;
                        }
                    }
                }
            }
            _ => debug_assert!(false, "should not happen: vs.len()={}", vs.len())
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

    pub fn render_scene<'a, V, F, Re, Sh, Ra>(
        &'a mut self,
        Scene { objects, camera }: &Scene<Re>,
        shader: &mut Sh,
        raster: &mut Ra
    ) -> Stats
    where
        V: Copy,
        F: Copy,
        Re: Render<F, V>,
        Sh: Shader<F, V>,
        Ra: RasterOps + 'a
    {
        let clock = Instant::now();
        for Obj { tf, geom, .. } in objects {
            self.modelview = tf * camera;
            geom.render(self, shader, raster);
        }
        self.stats.objs_in += objects.len();
        self.stats.time_used += clock.elapsed();
        self.stats
    }

    // FIXME Ensure correct stats in Render impls, then delete this method
    #[deprecated]
    #[allow(unused)]
    fn render<VA, FA>(
        &mut self,
        _: &Mesh<VA, FA>,
        _: &mut impl RasterOps
    ) -> Stats
    where
        VA: Copy + Linear<f32>,
        FA: Copy,
    {
        unimplemented!();

        let clock = Instant::now();

        self.stats.objs_in += 1;
        self.stats.time_used += clock.elapsed();
        self.stats
    }
}

fn depth_sort<VA: Copy, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
    let (faces, attrs) = {
        let mut faces = mesh.faces().collect::<Vec<_>>();

        faces.sort_unstable_by(|a, b| {
            let az: f32 = a.verts.iter().map(|v| v.coord.z).sum();
            let bz: f32 = b.verts.iter().map(|v| v.coord.z).sum();
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

fn clip_to_screen<A>(mut v: Vertex<A>, viewport: &Mat4) -> Vertex<(f32, A)> {
    v.coord /= v.coord.w;
    v.coord.transform(viewport);
    with_depth(v)
}
