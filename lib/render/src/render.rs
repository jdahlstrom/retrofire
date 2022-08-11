use std::convert::identity;
use std::fmt::Debug;
use std::mem::{replace, swap};
use std::time::Instant;

use geom::{LineSeg, Polyline, Sprite};
use geom::bbox::BoundingBox;
use geom::mesh::{Face, Mesh, Soa, SoaVecs, SubMesh, Vertex, vertex};
use math::Linear;
use math::mat::Mat4;
use math::transform::Transform;
use math::vec::{vec4, X, Y, Z};
use util::color::Color;

use crate::{Batch, hsr, Rasterize, State};
use crate::hsr::Visibility;
use crate::scene::{Obj, Scene};
use crate::shade::{Shader, ShaderImpl};
use crate::Visibility::Hidden;

pub trait Render<Uni, VtxIn, FragIn> {
    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<Uni, VtxIn, FragIn, VtxOut=VtxIn>,
        R: Rasterize;
}

impl<G, U, VtxIn, FragIn> Render<U, VtxIn, FragIn> for Scene<G>
where
    G: Render<U, VtxIn, FragIn>,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, VtxIn, FragIn, VtxOut=VtxIn>,
        R: Rasterize
    {
        let clock = Instant::now();
        let Self { objects, camera } = self;
        for Obj { tf, geom, .. } in objects {
            // TODO transform stack
            let prev_mv = replace(&mut st.modelview, tf * camera);
            geom.render(st, shade, raster);
            st.modelview = prev_mv;
        }
        st.stats.objs_in += objects.len();
        st.stats.time_used += clock.elapsed();
    }
}

impl<VI, U> Render<U, VI, VI> for Mesh<VI, U>
where
    VI: Soa + Linear<f32> + Copy + 'static,
    U: Copy + Debug,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VI, VtxOut=VI>,
        R: Rasterize,
    {
        let mvp = &st.modelview * &st.projection;
        if bbox_visibility(&self.bbox, &mvp) == Hidden {
            return;
        }

        st.stats.objs_out += 1;

        let prims = self.faces.iter()
            .map(|face| Face {
                verts: face.verts,
                attr: self.face_attrs[face.attr],
            })
            .collect();

        let verts = self.verts.iter()
            .map(|v| Vertex {
                coord: self.vertex_coords[v.coord],
                attr: self.vertex_attrs.get(&v.attr),
            })
            .collect();

        let batch = Batch { prims, verts };
        batch.render(st, shader, raster);

        if let Some(col) = st.options.bounding_boxes {
            render_bbox(&self.bbox, st, raster, col);
        }

        return;
    }
}

impl<VI, U> Render<U, VI, VI> for SubMesh<'_, VI, U>
where
    VI: Soa + Linear<f32> + Copy + Debug,
    U: Copy + Debug,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VtxOut=VI>,
        R: Rasterize
    {
        let mvp = &st.modelview * &st.projection;
        if bbox_visibility(&self.bbox, &mvp) == Hidden {
            return;
        }

        st.stats.objs_out += 1;

        let prims = self.face_indices.iter()
            .map(|&fi| self.mesh.faces[fi])
            .map(|face| Face {
                verts: face.verts,
                attr: self.mesh.face_attrs[face.attr],
            })
            .collect();

        let verts = self.mesh.verts.iter()
            .map(|v| Vertex {
                coord: self.mesh.vertex_coords[v.coord],
                attr: self.mesh.vertex_attrs.get(&v.attr),
            })
            .collect();

        // TODO Mechanism to avoid shade/transform of unused vertices
        let batch = Batch { prims, verts };
        batch.render(st, shader, raster);

        if let Some(col) = st.options.bounding_boxes {
            render_bbox(&self.bbox, st, raster, col);
        }
    }
}

fn bbox_visibility(bbox: &BoundingBox, mvp: &Mat4) -> Visibility {
    let vs = bbox.verts().transform(&mvp);
    hsr::vertex_visibility(vs.iter())
}

fn render_bbox(
    bbox: &BoundingBox,
    st: &mut State,
    raster: &mut impl Rasterize,
    col: Color,
) {
    for edge in bbox.edges() {
        let edge = LineSeg(edge.map(|v| vertex(v, ())));
        edge.render(st, &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(col),
        }, raster);
    }
}

impl<V> Render<(), V, V> for LineSeg<Vertex<V>>
where
    V: Linear<f32> + Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V, VtxOut=V>,
        R: Rasterize
    {
        let batch = Batch {
            prims: vec![LineSeg([0usize, 1usize])],
            verts: self.0.to_vec()
        };
        batch.render(st, shader, raster)
    }
}

impl<V: Linear<f32> + Copy> Render<(), V, V> for Polyline<Vertex<V>> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V, VtxOut=V>,
        R: Rasterize,
    {
        let batch = Batch {
            prims: (0..self.0.len() - 1)
                .map(|i| LineSeg([i, i + 1]))
                .collect(),
            verts: self.0.clone(),
        };
        batch.render(st, shader, raster);
    }
}

impl<U, V> Render<U, V, V> for Sprite<Vertex<V>, U>
where
    U: Copy + Debug,
    V: Linear<f32> + Copy
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, V, VtxOut=V>,
        R: Rasterize,
    {
        let s = Sprite {
            verts: [0usize, 1, 2, 3],
            face_attr: self.face_attr
        };
        let b = Batch {
            prims: vec![s],
            verts: self.verts.to_vec()
        };
        let [x, y, z, _] = st.modelview.cols();
        let mv = Mat4::from_rows([
            x.to_dir().len() * X,
            y.to_dir().len() * Y,
            z.to_dir().len() * Z,
            vec4(x.w, y.w, z.w, 1.0)
        ]);

        let prev_mv = replace(&mut st.modelview, mv);
        b.render(st, shader, raster);
        st.modelview = prev_mv;
    }
}
