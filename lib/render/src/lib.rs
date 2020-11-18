use std::time::Instant;

use geom::mesh::{Mesh, VertexAttr};
use math::Linear;
use math::mat::Mat4;
use math::vec::*;
pub use stats::Stats;

use crate::raster::*;

mod hsr;
pub mod raster;
pub mod shade;
pub mod stats;
pub mod vary;

pub type Shader<'a, Vary, Uniform> = &'a dyn Fn(Fragment<Vary>, Uniform) -> Vec4;
pub type Plotter<'a> = &'a mut dyn FnMut(usize, usize, Vec4);

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

    pub fn render<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, sh: Shader<(Vec4, VA), FA>, pl: Plotter) -> Stats
    where VA: VertexAttr,
          FA: Copy
    {
        let clock = Instant::now();

        self.transform(&mut mesh);
        self.projection(&mut mesh.verts);
        self.hidden_surface_removal(&mut mesh);
        Self::z_sort(&mut mesh);

        self.perspective_divide(&mut mesh.verts);

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

        // TODO this should only be done for normals
        for va in vertex_attrs.iter_mut() {
            va.transform(tf);
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
        let view = &self.viewport;
        for v in verts {
            *v = view * *v;
        }
    }

    fn hidden_surface_removal<VA, FA>(&mut self, mut mesh: &mut Mesh<VA, FA>)
    where VA: Copy + Linear<f32>, FA: Copy {
        //print!("HSR faces {} -> ", mesh.faces.len());
        self.stats.faces_in += mesh.faces.len();
        hsr::hidden_surface_removal(&mut mesh);
        self.stats.faces_out += mesh.faces.len();
        //println!("{}", mesh.faces.len());
    }

    // TODO Replace with z-buffering (or s-buffering!)
    pub fn z_sort<VA, FA: Copy>(mesh: &mut Mesh<VA, FA>) {
        let (faces, attrs): (Vec<_>, Vec<_>) = {
            let Mesh { verts, faces, face_attrs, .. } = &*mesh;

            let mut v = faces.iter().zip(face_attrs).collect::<Vec<_>>();

            v.sort_unstable_by(|&(a, _), &(b, _)| {
                let az = verts[a[0]].z + verts[a[1]].z + verts[a[2]].z;
                let bz = verts[b[0]].z + verts[b[1]].z + verts[b[2]].z;
                bz.partial_cmp(&az).unwrap()
            });

            (v.iter().map(|(&faces, _)| faces).collect(),
             v.iter().map(|(_, &attrs)| attrs).collect())
        };

        mesh.faces = faces;
        mesh.face_attrs = attrs;
    }

    pub fn rasterize<VA, FA>(&mut self, mut mesh: Mesh<VA, FA>, shade: Shader<(Vec4, VA), FA>, plot: Plotter)
    where VA: VertexAttr,
          FA: Copy,
    {
        let Mesh { faces, verts, vertex_attrs, face_attrs } = &mut mesh;

        let orig_verts = verts.clone();

        self.viewport(verts);

        for (i, &[a, b, c]) in faces.iter().enumerate() {
            // TODO Clean up this mess
            let (av, bv, cv) = (verts[a], verts[b], verts[c]);
            let (ao, bo, co) = (orig_verts[a], orig_verts[b], orig_verts[c]);
            let (ava, bva, cva) = (vertex_attrs[a], vertex_attrs[b], vertex_attrs[c]);

            let fa = face_attrs[i];

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


#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use math::transform::perspective;

    use super::*;

    #[test]
    fn clip_projected_face() {
        let mut r = Renderer::new();
        let p = perspective(0.1, 10., 1., 2. * PI / 3.);
        r.set_projection(p);

        let z = 10.*Z+W;
        let x = 10.*3.0f32.sqrt()*X;
        let y = 10.*3.0f32.sqrt()*Y;

        let mut m: Mesh<(), ()> = Mesh {
            verts: vec![z+x, z, z+y],
            faces: vec![[0, 1, 2]],
            vertex_attrs: vec![(), (), ()],
            face_attrs: vec![()]
        };
        dbg!("world", &m.verts);

        r.projection(&mut m.verts);

        dbg!("clip", &m.verts);

        r.perspective_divide(&mut m.verts);

        dbg!("ndc", &m.verts);

        r.hidden_surface_removal(&mut m);

        dbg!((&m.verts, &m.faces));
    }
}