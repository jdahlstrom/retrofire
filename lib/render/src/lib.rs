use std::fmt::Debug;
use std::mem::swap;
use std::time::Instant;

use geom::{LineSeg, mesh, mesh2, Polyline, Sprite};
use geom::mesh::{Face, Vertex};
use geom::mesh2::Soa;
use math::Linear;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use scene::{Obj, Scene};
pub use stats::Stats;
use util::buf::Buffer;
use util::color::Color;

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

impl<'a> RasterOps for Framebuf<'a> {
    #[inline]
    fn test<U>(&self, f: Fragment<f32, U>) -> bool {
        f.varying < *self.depth.get(f.coord.0, f.coord.1)
    }
    #[inline]
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

    #[inline]
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

impl<G, U, V> Render<U, V> for Scene<G>
where
    G: Render<U, V>,
    U: Copy,
    V: Linear<f32> + Copy,
{
    fn render<S, R>(&self, rdr: &mut Renderer, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, V>,
        R: RasterOps
    {
        let clock = Instant::now();
        let Self { objects, camera } = self;
        for Obj { tf, geom, .. } in objects {
            rdr.modelview = tf * camera;
            geom.render(rdr, shade, raster);
        }
        rdr.stats.objs_in += objects.len();
        rdr.stats.time_used += clock.elapsed();
    }
}

impl<VI, U> Render<U, VI> for mesh2::Mesh<VI, U>
where
    VI: Soa + Linear<f32> + Copy + Debug,
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
            let vs = self.bbox.verts().transform(&mvp);
            hsr::vertex_visibility(vs.iter())
        };

        if bbox_vis != Visibility::Hidden {
            let vcs: Vec<_> = self.vertex_coords.iter()
                // TODO do transform in vertex shader?
                .map(|&v| v.transform(&mvp))
                .collect();

            let verts: Vec<_> = self.verts.iter()
                .map(|v| Vertex {
                    coord:  vcs[v.coord],
                    attr: VI::get(&self.vertex_attrs, &v.attr)
                })
                .collect();

            let faces: Vec<_> = self.faces.iter()
                .map(|f| Face {
                    indices: f.verts,
                    verts: f.verts.map(|i| verts[i]),
                    attr: self.face_attrs[f.attr]
                })
                .collect();

            let (mut verts, mut faces) = hsr::hidden_surface_removal(&verts, &faces, bbox_vis);

            if !faces.is_empty() {
                rdr.stats.objs_out += 1;
                rdr.stats.faces_out += faces.len();

                if rdr.options.depth_sort { depth_sort(&mut faces); }
                perspective_divide(&mut verts, rdr.options.perspective_correct);

                for v in &mut verts {
                    v.coord.transform_mut(&rdr.viewport);
                }

                for Face { indices, attr, .. } in faces {
                    let verts = indices.map(|i| verts[i]).map(with_depth);

                    tri_fill(verts, |frag| {
                        if self.rasterize(shader, raster, frag.uniform(attr)) {
                            rdr.stats.pixels += 1;
                        }
                    });

                    if let Some(col) = rdr.options.wireframes {
                        let [a, b, c] = verts;
                        for e in [a, b, c, a].windows(2) {
                            line([e[0], e[1]], |frag| {
                                if raster.test(frag.varying(frag.varying.0-0.001)) {
                                    raster.output(frag.varying((0.0, col)));
                                }
                            });
                        }
                    }
                }
            }
        }

        /* TODO Wireframe and bounding box debug rendering
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
         */
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

        let mvp = &rdr.modelview * &rdr.projection;

        let mut verts = self.0.map(|mut v| {
            v.coord.transform_mut(&mvp);
            shader.shade_vertex(v)
        }).to_vec();
        let mut clip_out = Vec::new();
        hsr::clip(&mut verts, &mut clip_out);
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
        this.anchor.transform_mut(&rdr.modelview);
        let scale = &rdr.modelview.row(0).len();
        this.width *= scale;
        this.height *= scale;

        let this = Self {
            anchor: self.anchor.transform(&rdr.modelview),
            width: self.width * scale,
            height: self.height * scale,
            ..*self
        };

        let mut vs: Vec<_> = this.verts()
                .map(|v| {
                    let mut v = shader.shade_vertex(v);
                    v.coord.transform_mut(&rdr.projection);
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
}

fn depth_sort<VA: Copy, FA: Copy>(_faces: &mut Vec<Face<VA, FA>>) {
    todo!()
}

fn perspective_divide<VA>(verts: &mut Vec<Vertex<VA>>, pc: bool)
where VA: Linear<f32> + Copy
{
    for Vertex { coord, attr } in verts {
        let w = 1.0 / coord.w;
        *coord = coord.mul(w);
        if pc {
            *attr = attr.mul(w);
        }
    }
}

#[inline(always)]
fn with_depth<VA>(v: Vertex<VA>) -> Vertex<(f32, VA)> {
    Vertex { coord: v.coord, attr: (v.coord.z, v.attr) }
}

fn clip_to_screen<A>(mut v: Vertex<A>, viewport: &Mat4) -> Vertex<(f32, A)> {
    v.coord = (v.coord / v.coord.w).transform(viewport);
    with_depth(v)
}
