use std::fmt::{Display, Formatter, Result};
use std::time::{Duration, Instant};

use geom::mesh::Mesh;
use math::lerp;
use math::mat::Mat4;
use math::vec::*;
use raster::*;
use std::f32::EPSILON;

pub mod raster;
pub mod vary;

pub type Shader<'a> = &'a dyn Fn(Fragment<Vec4>) -> Vec4;
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

    pub fn render(&mut self, mut mesh: Mesh, sh: Shader, pl: Plotter) -> Stats {
        let clock = Instant::now();

        self.transform(&mut mesh);
        self.projection(&mut mesh);
        self.hidden_surface_removal(&mut mesh);
        self.z_sort(&mut mesh);
        self.viewport(&mut mesh);
        self.rasterize(mesh, sh, pl);

        self.stats.time_used += Instant::now() - clock;
        self.stats.frames += 1;
        self.stats
    }

    fn transform(&self, mesh: &mut Mesh) {
        let tf = &self.transform;
        let Mesh { verts, vertex_norms, .. } = mesh;

        for v in verts {
            *v = tf * *v;
        }
        for n in vertex_norms.iter_mut().flatten() {
            *n = (tf * *n).normalize();
        }
    }

    fn projection(&self, mesh: &mut Mesh) {
        let proj = &self.projection;
        for v in &mut mesh.verts {
            *v = proj * *v;
            *v = *v / v.w;
        };
    }

    pub fn viewport(&self, mesh: &mut Mesh) {
        let view = &self.viewport;
        for v in &mut mesh.verts {
            *v = view * *v;
        }
    }

    fn hidden_surface_removal(&mut self, mesh: &mut Mesh) {
        self.stats.faces_in += mesh.faces.len();

        let mut visible_faces = Vec::with_capacity(mesh.faces.len() / 2);

        for face in &mesh.faces {
            let &[a, b, c] = face;
            let verts = [mesh.verts[a], mesh.verts[b], mesh.verts[c]];

            match face_visibility(&verts) {
                FaceVis::Hidden => {
                    continue
                },
                FaceVis::Unclipped => {
                    visible_faces.push([a, b, c]);
                },
                FaceVis::Clipped => {
                    let mut clipped_verts = Self::clip(&verts);
                    if clipped_verts.is_empty() {
                        continue;
                    }
                    let cn = clipped_verts.len();
                    let vn = mesh.verts.len();
                    mesh.verts.append(&mut clipped_verts);
                    for i in 1..cn - 1 {
                        visible_faces.push([vn, vn + i, vn + i + 1]);
                    }
                }
            }
        }

        mesh.faces = visible_faces;

        self.stats.faces_out += mesh.faces.len();
    }

    fn clip(verts: &[Vec4]) -> Vec<Vec4> {

        let mut verts = verts.to_vec();
        let mut verts2 = Vec::with_capacity(8);
        for (&a, &b) in edges(&verts) {
            let [v, u] = Self::intersect(a, b, a.x, b.x, -1.0, "x");
            if let Some(v) = v { verts2.push(v); }
            if let Some(u) = u { verts2.push(u); }
        }

        verts.clear();
        for (&a, &b) in edges(&verts2) {
            let [v, u] = Self::intersect(a, b, a.x, b.x, 1.0, "x");
            if let Some(v) = v { verts.push(v); }
            if let Some(u) = u { verts.push(u); }
        }

        verts2.clear();
        for (&a, &b) in edges(&verts) {
            let [v, u] = Self::intersect(a, b, a.y, b.y, -1.0, "y");
            if let Some(v) = v { verts2.push(v); }
            if let Some(u) = u { verts2.push(u); }
        }

        verts.clear();
        for (&a, &b) in edges(&verts2) {
            let [v, u] = Self::intersect(a, b, a.y, b.y, 1.0, "y");
            if let Some(v) = v { verts.push(v); }
            if let Some(u) = u { verts.push(u); }
        }

        verts2.clear();
        for (&a, &b) in edges(&verts) {
            let [v, u] = Self::intersect(a, b, a.z, b.z, -1.0, "z");
            if let Some(v) = v { verts2.push(v); }
            if let Some(u) = u { verts2.push(u); }
        }

        verts.clear();
        for (&a, &b) in edges(&verts2) {
            let [v, u] = Self::intersect(a, b, a.z, b.z, 1.0, "z");
            if let Some(v) = v { verts.push(v); }
            if let Some(u) = u { verts.push(u); }
        }

        verts
    }

    fn intersect(a: Vec4, b: Vec4, ac: f32, bc: f32, oc: f32, _c: &str) -> [Option<Vec4>; 2] {
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

    pub fn z_sort(&self, mesh: &mut Mesh) {
        let Mesh { verts, faces, .. } = mesh;
        faces.sort_unstable_by(|a, b| {
            let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
            let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
            bz.partial_cmp(&az).unwrap()
        });
    }

    pub fn rasterize(&mut self, mesh: Mesh, shade: Shader, plot: Plotter) {
        let Mesh { faces, verts, vertex_norms: norms, .. } = &mesh;
        for &[a, b, c] in faces {
            let (av, bv, cv) = (verts[a], verts[b], verts[c]);
            let (an, bn, cn) = if let Some(ns) = norms {
                (ns[a], ns[b], ns[c])
            } else {
                (ZERO, ZERO, ZERO)
            };
            tri_fill(frag(av, an), frag(bv, bn), frag(cv, cn), |frag| {
                let col = shade(frag);
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

fn edges(vs: &[Vec4]) -> impl Iterator<Item=(&Vec4, &Vec4)> {
    (0..vs.len()).map(move |i| (&vs[i], &vs[(i + 1) % vs.len()]))
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

fn frag(v: Vec4, n: Vec4) -> Fragment<Vec4> {
    Fragment { coord: v, varying: n }
}

#[cfg(test)]
mod tests {
    use math::ApproxEq;

    use super::*;

    #[test]
    fn clip_fully_outside_triangle() {
        let expected = Vec::<Vec4>::new();
        let actual = Renderer::clip(&vec![2.0 * Y, -X + 3.0 * Y, X + 3.0 * Y]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn clip_all_vertices_inside() {
        let expected = vec![Y, -X, X];
        let actual = Renderer::clip(&expected.clone());

        assert_eq!(expected, actual);
    }

    #[test]
    fn clip_vertices_on_bounds() {
        let expected = vec![-X, Y, X - Y];
        let actual = Renderer::clip(&expected.clone());
        assert_eq!(expected, actual);
    }

    #[test]
    fn clip_all_vertices_outside() {
        let expected = vec![0.25 * X + Y, X - 0.5 * Y, X - Y, 0.5 * X - Y, -X - 0.25 * Y, -X + 0.5 * Y, -0.5 * X + Y];
        let actual = Renderer::clip(&vec![1.5 * Y, 1.5 * (X - Y), -1.5 * X]);

        dbg!(&actual);

        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual) {
            assert!(e.approx_eq(a));
        }
    }

    #[test]
    fn clip_screen_filling_triangle() {
        let expected = vec![X + Y, X - Y, -X - Y, -X + Y];
        let actual = Renderer::clip(&vec![-20.0 * (X + Y), 20.0 * Y, 20.0 * (X - Y)]);

        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual) {
            assert!(e.approx_eq(a));
        }
    }
}
