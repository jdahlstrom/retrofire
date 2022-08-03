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

use crate::{Batch, hsr, Rasterize, Span, State};
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

impl<V: Copy> Render<(), V, V> for Polyline<V> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V, VtxOut=V>,
        R: Rasterize,
    {
        for seg in self.edges() {
            seg.render(st, shader, raster);
        }
    }
}

impl<U: Copy, V: Copy> Render<U, V, V> for Sprite<V, U> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, V>,
        R: Rasterize,
    {
        st.stats.prims_in += 1;
        st.stats.verts_in += 1;

        let mut this = *self;
        this.anchor.transform_mut(&st.modelview);
        let scale = &st.modelview.row(0).len();
        this.width *= scale;
        this.height *= scale;

        let this = Self {
            anchor: self.anchor.transform(&st.modelview),
            width: self.width * scale,
            height: self.height * scale,
            ..*self
        };

        let mut vs: Vec<_> = this.verts()
            .map(|v| {
                let mut v = shader.shade_vertex(v);
                v.coord.transform_mut(&st.projection);
                v
            })
            .collect();

        let mut clip_out = Vec::new();
        hsr::clip(&mut vs, &mut clip_out);
        let mut vs: Vec<_> = clip_out.into_iter()
            .map(|v| clip_to_screen(v, &st.viewport))
            .collect();

        match vs.as_mut_slice() {
            [] => {}
            [v0, v1, v2, v3] => {
                st.stats.prims_out += 1;
                st.stats.verts_out += 1;

                if v0.coord.y > v2.coord.y {
                    swap(v0, v3);
                    swap(v1, v2);
                }
                if v0.coord.x > v1.coord.x {
                    swap(v0, v1);
                    swap(v2, v3);
                }

                // TODO extract to fn rect_fill
                let (x0, y0) = (v0.coord.x.round(), v0.coord.y.round());
                let (x1, y1) = (v2.coord.x.round(), v2.coord.y.round());
                let v = (v0.attr, v1.attr)
                    .vary(&(v3.attr, v2.attr), y1 - y0);

                for (y, (v0, v1)) in (y0 as usize..y1 as usize).zip(v) {
                    raster.rasterize_span(shader, Span {
                        y,
                        xs: (x0 as usize, x1 as usize),
                        vs: (v0, v1),
                        uni: self.face_attr,
                    });
                }
            }
            _ => debug_assert!(false, "should not happen: vs.len()={}", vs.len())
        }
    }
}
