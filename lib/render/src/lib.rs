use geom::mesh::Mesh;
use math::mat::Mat4;
use math::vec::*;
use raster::*;

pub mod raster;
pub mod vary;

pub type Shader<'a> = &'a dyn Fn(Fragment<Vec4>) -> Vec4;
pub type Plotter<'a> = &'a mut dyn FnMut(usize, usize, Vec4);

pub struct Renderer {
    transform: Mat4,
    projection: Mat4,
    viewport: Mat4,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            transform: Mat4::default(),
            projection: Mat4::default(),
            viewport: Mat4::default(),
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

    pub fn render(&mut self, mut mesh: Mesh, sh: Shader, pl: Plotter) {
        self.transform(&mut mesh);
        self.projection(&mut mesh);
        self.z_sort(&mut mesh);
        self.viewport(&mut mesh);
        self.rasterize(mesh, sh, pl);
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
            });
        }
    }
}

fn frag(v: Vec4, n: Vec4) -> Fragment<Vec4> {
    Fragment { coord: v, varying: n }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
}
