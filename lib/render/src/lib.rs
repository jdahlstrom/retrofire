use std::time::Instant;

use geom::{LineSeg, Polyline};
use geom::mesh::{Face, Mesh, Vertex};
use math::{Angle, Linear};
use math::mat::Mat4;
use math::transform::{rotate_x, rotate_y, Transform, translate};
use math::vec::*;
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


#[derive(Clone, Debug, Default)]
pub struct FpsCamera {
    pub pos: Vec4,
    pub azimuth: Angle,
    pub altitude: Angle,
}

impl FpsCamera {
    pub fn new(pos: Vec4, azimuth: Angle) -> Self {
        Self { pos, azimuth, ..Self::default() }
    }

    pub fn translate(&mut self, dir: Vec4) {
        let fwd = &rotate_y(self.azimuth) * Z;
        let right = Y.cross(fwd);
        self.pos += Vec4::lincomb(fwd, dir.z, right, dir.x);
    }

    pub fn rotate(&mut self, az: Angle, alt: Angle) {
        self.azimuth = (self.azimuth + az)
            .wrap(-Angle::STRAIGHT, Angle::STRAIGHT);
        self.altitude = (self.altitude + alt)
            .clamp(-Angle::RIGHT, Angle::RIGHT);
    }

    pub fn world_to_view(&self) -> Mat4 {

        let rot_y = &rotate_y(self.azimuth);
        let fwd = rot_y * (&rotate_x(self.altitude) * Z);
        let fwd_move = rot_y * Z;
        let right = Y.cross(fwd_move);
        let up = fwd.cross(right);

        let orient = Mat4::from_cols([right, up, fwd, W]);
        let transl = translate(-self.pos.x, -self.pos.y, -self.pos.z);

        transl * &orient
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
            ref modelview, ref projection, ref viewport, ref options, stats,
        } = rdr;

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
        let Renderer {
            ref modelview, ref projection, ref viewport, stats, ..
        } = rdr;

        stats.faces_in += 1;

        let mut this = (*self).clone();
        let mvp = modelview * projection;
        this.transform(&mvp);

        if let Some(&[mut a, mut b]) = hsr::clip(&this.0).get(0..2) {
            stats.faces_out += 1;

            a.coord /= a.coord.w;
            b.coord /= b.coord.w;
            a.coord.transform(&viewport);
            b.coord.transform(&viewport);
            let verts = [with_depth(a), with_depth(b)];
            line(verts, |frag: Fragment<_>| {
                if raster.rasterize(frag, ()) {
                    stats.pixels += 1;
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
        let Renderer {
            ref modelview, ref projection, ref viewport, stats, ..
        } = rdr;

        stats.faces_in += 1;

        let mut this = self.clone();
        let mvp = modelview * projection;
        this.transform(&mvp);

        for edge in this.edges() {
            if let Some(&[mut a, mut b]) = hsr::clip(&edge).get(0..2) {
                stats.faces_out += 1;

                a.coord /= a.coord.w;
                b.coord /= b.coord.w;
                a.coord.transform(&viewport);
                b.coord.transform(&viewport);
                let verts = [with_depth(a), with_depth(b)];
                line(verts, |frag: Fragment<_>| {
                    if raster.rasterize(frag, ()) {
                        stats.pixels += 1;
                    }
                });
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
fn with_depth<VA: Copy>(v: Vertex<VA>) -> Vertex<(f32, VA)> {
    Vertex { coord: v.coord, attr: (v.coord.z, v.attr) }
}

#[inline(always)]
fn without_depth<V: Copy>(f: Fragment<(f32, V)>) -> Fragment<V> {
    Fragment { coord: f.coord, varying: f.varying.1 }
}
