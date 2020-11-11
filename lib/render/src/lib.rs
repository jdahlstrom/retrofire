use std::f32::EPSILON;
use std::fmt::{Display, Formatter, Result};
use std::mem::swap;
use std::time::{Duration, Instant};

use geom::mesh::{Mesh, VertexAttr};
use math::{lerp, Linear};
use math::mat::Mat4;
use math::vec::*;
use raster::*;

pub mod raster;
pub mod vary;

pub type Shader<'a, Vary, Uniform> = &'a dyn Fn(Fragment<Vary>, Uniform) -> Vec4;
pub type Plotter<'a> = &'a mut dyn FnMut(usize, usize, Vec4);

pub struct Renderer {
    transform: Mat4,
    projection: Mat4,
    viewport: Mat4,
    stats: Stats,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    pub frames: usize,
    pub faces_in: usize,
    pub faces_out: usize,
    pub pixels: usize,
    pub time_used: Duration,
}

impl Stats {
    pub fn avg_per_frame(&self) -> Stats {
        Stats {
            frames: 1,
            faces_in: self.faces_in / self.frames,
            faces_out: self.faces_out / self.frames,
            pixels: self.pixels / self.frames,
            time_used: self.time_used / self.frames as u32,
        }
    }

    pub fn avg_per_sec(&self) -> Stats {
        let secs = self.time_used.as_secs_f32();
        Stats {
            frames: (self.frames as f32 / secs) as usize,
            faces_in: (self.faces_in as f32 / secs) as usize,
            faces_out: (self.faces_out as f32 / secs) as usize,
            pixels: (self.pixels as f32 / secs) as usize,
            time_used: Duration::from_secs(1),
        }
    }
}

fn human(n: usize) -> String {
    if n < 1_000 { format!("{:6}", n) }
    else if n < 1_000_000 { format!("{:5.1}k", n as f32 / 1_000.) }
    else if n < 1_000_000_000 { format!("{:5.1}M", n as f32 / 1_000_000.) }
    else if n < 1_000_000_000_000 { format!("{:5.1}M", n as f32 / 1_000_000.) }
    else { format!("{:5.1e}", n) }
}

fn human_time(d: Duration) -> String {
    let s = d.as_secs_f32();
    if s < 1.0 { format!("{:4.2}msec", s * 1000.) }
    else { format!("{:.2}sec ", s) }
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "frames: {} │ faces in: {} │ \
                   faces out: {} │ pixels: {} │ \
                   time used: {:>9}",
               human(self.frames), human(self.faces_in), human(self.faces_out),
               human(self.pixels), human_time(self.time_used))
    }
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            transform: Mat4::default(),
            projection: Mat4::default(),
            viewport: Mat4::default(),
            stats: Stats::default(),
        }
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

    pub fn render<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, sh: Shader<(Vec4, VA), FA>, pl: Plotter) -> Stats
    where VA: VertexAttr,
          FA: Copy + Default
    {
        let clock = Instant::now();

        self.transform(&mut mesh);
        self.projection(&mut mesh.verts);
        Self::hidden_surface_removal(&mut self.stats, &mut mesh);
        Self::z_sort(&mut mesh);

        self.rasterize(mesh, sh, pl);

        self.stats.time_used += Instant::now() - clock;
        self.stats.frames += 1;
        self.stats
    }

    fn transform<VA: VertexAttr, FA>(&self, mesh: &mut Mesh<VA, FA>) {
        let tf = &self.transform;
        let Mesh { verts, vertex_attrs, .. } = mesh;

        for v in verts {
            *v = tf * *v;
        }
        for va in vertex_attrs.iter_mut().flatten() {
            va.transform(tf);
        }
    }

    fn projection(&self, verts: &mut Vec<Vec4>) {
        let proj = &self.projection;
        for v in verts {
            *v = proj * *v;
            *v = *v / v.w;
        };
    }

    pub fn viewport(&self, verts: &mut Vec<Vec4>) {
        let view = &self.viewport;
        for v in verts {
            *v = view * *v;
        }
    }

    fn hidden_surface_removal<VA, FA>(stats: &mut Stats, mesh: &mut Mesh<VA, FA>)
    where VA: VertexAttr,
          FA: Copy
    {
        stats.faces_in += mesh.faces.len();

        let Mesh { verts, faces, vertex_attrs, face_attrs, .. } = mesh;

        let vertex_attrs = vertex_attrs.as_mut().unwrap();

        let mut visible_faces = Vec::with_capacity(faces.len() / 2);
        let mut visible_attrs = Vec::with_capacity(faces.len() / 2);

        for (&[a, b, c], &fa) in faces.iter().zip(face_attrs.as_ref().unwrap()) {

            let face_verts = [(verts[a], vertex_attrs[a]),
                              (verts[b], vertex_attrs[b]),
                              (verts[c], vertex_attrs[c])];

            match face_visibility(&[verts[a], verts[b], verts[c]]) {
                FaceVis::Hidden => {
                    continue
                },
                FaceVis::Unclipped => {
                    visible_faces.push([a, b, c]);
                    visible_attrs.push(fa);
                },
                FaceVis::Clipped => {
                    let clipped_verts = Self::clip(&face_verts);
                    if clipped_verts.is_empty() {
                        continue;
                    }
                    let cn = clipped_verts.len();
                    let vn = verts.len();
                    verts.extend(clipped_verts.iter().map(|v| v.0));
                    vertex_attrs.extend(clipped_verts.into_iter().map(|v| v.1));

                    for i in 1..cn - 1 {
                        visible_faces.push([vn, vn + i, vn + i + 1]);
                        visible_attrs.push(fa);
                    }
                }
            }
        }

        mesh.faces = visible_faces;
        mesh.face_attrs = Some(visible_attrs);

        stats.faces_out += mesh.faces.len();
    }

    fn clip<VA>(verts: &[(Vec4, VA)]) -> Vec<(Vec4, VA)>
    where VA: Linear<f32> + Copy
    {
        let mut verts = verts.to_vec();
        let mut verts2 = Vec::with_capacity(8);

        for i in 0..3 {
            for &o in &[-1.0, 1.0] {
                verts2.clear();
                for (&a, &b) in edges(&verts) {
                    let vs = Self::intersect(a, b, a.0[i], b.0[i], o, ['x', 'y', 'z'][i]);
                    verts2.extend(vs.iter().flatten());
                }
                swap(&mut verts, &mut verts2);
            }
        }

        verts
    }

    fn intersect<V>(a: V, b: V, ac: f32, bc: f32, oc: f32, _c: char) -> [Option<V>; 2]
    where V: Copy + Linear<f32>
    {
        //eprint!("Intersecting {} = {} .. {} with {}: ", c, ac, bc, oc);
        let mut res = [None, None];
        if inside(ac, oc) {
            //eprint!("a = {:?} ", a);
            res[0] = Some(a);
        }
        if inside(ac, oc) != inside(bc, oc) {
            let t = (oc - ac) / (bc - ac);
            let o = lerp(t, a, b);
            //eprint!("o = {:?}", o);
            res[1] = Some(o);
        }
        //eprintln!();
        res
    }

    pub fn z_sort<VA, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
        let Mesh { verts, faces, face_attrs, .. } = mesh;

        if let Some(ref attrs) = face_attrs {

            let mut v = faces.iter().zip(attrs).collect::<Vec<_>>();

            v.sort_unstable_by(|&(a, _), &(b, _)| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });

            let (f, a): (Vec<_>, Vec<_>) = v.into_iter()
                .unzip();

            *faces = f.into_iter().copied().collect();
            *face_attrs = Some(a.into_iter().copied().collect());

        } else {
            faces.sort_unstable_by(|&a, &b| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });
        }
    }

    pub fn rasterize<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, shade: Shader<(Vec4, VA), FA>, plot: Plotter)
    where VA: VertexAttr,
          FA: Copy + Default,
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs } = &mut mesh;

        let orig_verts = verts.clone();

        self.viewport(verts);

        for (i, &[a, b, c]) in faces.iter().enumerate() {
            let (av, bv, cv) = (verts[a], verts[b], verts[c]);
            let (ao, bo, co) = (orig_verts[a], orig_verts[b], orig_verts[c]);
            let (ava, bva, cva) = if let Some(attrs) = vertex_attrs {
                (attrs[a], attrs[b], attrs[c])
            } else {
                Default::default()
            };

            let fa = if let Some(attrs) = face_attrs {
                attrs[i]
            } else {
                FA::default()
            };

            tri_fill(Fragment { coord: av, varying: (ao, ava) },
                     Fragment { coord: bv, varying: (bo, bva) },
                     Fragment { coord: cv, varying: (co, cva) },
                     |frag| {
                         let col = shade(frag, fa);
                         plot(frag.coord.x as usize, frag.coord.y as usize, col);
                         self.stats.pixels += 1;
                     });
        }
    }
}

enum FaceVis {
    Unclipped,
    Clipped,
    Hidden
}

fn face_visibility(face: &[Vec4; 3]) -> FaceVis {
    if !frontface(face) {
        FaceVis::Hidden
    } else if face.iter().all(vertex_in_frustum) {
        FaceVis::Unclipped
    } else {
        FaceVis::Clipped
    }
}

fn edges<T>(ts: &[T]) -> impl Iterator<Item=(&T, &T)> {
    (0..ts.len()).map(move |i| (&ts[i], &ts[(i + 1) % ts.len()]))
}

fn frontface(&[a, b, c]: &[Vec4; 3]) -> bool {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0.0
}

fn vertex_in_frustum(v: &Vec4) -> bool {
    inside(v.x.abs(), 1.0)
        && inside(v.y.abs(), 1.0)
        && inside(v.z.abs(), 1.0)
}

fn inside(a: f32, o: f32) -> bool {
    if o > 0.0 {
        a <= o + EPSILON
    } else {
        a >= o - EPSILON
    }
}

#[cfg(test)]
mod tests {
    use math::ApproxEq;

    use super::*;

    // TODO Test interpolation of vertex attributes

    type Vtx = (Vec4, ());

    fn assert_approx_eq(expected: Vec<Vtx>, actual: Vec<Vtx>) {
        assert_eq!(expected.len(), actual.len(), "expected: {:#?}\nactual: {:#?}", expected, actual);
        for (&e, &a) in expected.iter().zip(&actual) {
            assert!(e.0.approx_eq(a.0), "expected: {:?}, actual: {:?}", e, a);
            // TODO assert!(e.1.approx_eq(a.1), "expected: {:?}, actual: {:?}", e, a);
        }
    }

    fn v(v: Vec4) -> Vtx { (v, ()) }

    fn vs(vs: &[Vec4]) -> Vec<Vtx> {
        vs.iter().copied().map(v).collect()
    }

    #[test]
    fn clip_fully_outside_triangle() {
        let expected = Vec::<Vtx>::new();
        let actual = Renderer::clip(&vs(&[2.0 * Y, -X + 3.0 * Y, X + 3.0 * Y]));

        assert_eq!(expected, actual);
    }

    #[test]
    fn clip_all_vertices_inside() {
        let expected = vs(&[Y, -X, X]);
        let actual = Renderer::clip(&expected);

        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn clip_vertices_on_bounds() {
        let expected = vs(&[-X, Y, X - Y]);
        let actual = Renderer::clip(&expected);
        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn clip_all_vertices_outside() {
        let expected = vs(&[
            0.25 * X + Y,
            X - 0.5 * Y,
            X - Y,
            0.5 * X - Y,
            -X - 0.25 * Y,
            -X + 0.5 * Y,
            -0.5 * X + Y]
        );
        let actual = Renderer::clip(&vs(&[
            1.5 * Y,
            1.5 * (X - Y),
            -1.5 * X]
        ));

        assert_approx_eq(expected, actual)
    }


    #[test]
    fn clip_screen_filling_triangle() {
        let expected = vs(&[X + Y, X - Y, -X - Y, -X + Y]);
        let actual = Renderer::clip(&vs(&[-20.0 * (X + Y), 20.0 * Y, 20.0 * (X - Y)]));

        assert_approx_eq(expected, actual)
    }
}
