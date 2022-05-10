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
use math::vec::ZERO;
use util::color::Color;

use crate::{Fragment, hsr, line, Rasterize, Span, State, tri_fill};
use crate::hsr::Visibility::{self, Hidden};
use crate::scene::{Obj, Scene};
use crate::shade::{Shader, ShaderImpl};
use math::vary::Varying;

pub trait Render<Uni, VtxIn, FragIn> {

    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<Uni, VtxIn, FragIn, VtxOut = VtxIn>,
        R: Rasterize;
}

impl<G, U, VtxIn, FragIn> Render<U, VtxIn, FragIn> for Scene<G>
where
    G: Render<U, VtxIn, FragIn>,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, VtxIn, FragIn, VtxOut = VtxIn>,
        R: Rasterize
    {
        let clock = Instant::now();
        let Self { objects, camera } = self;
        for Obj { tf, geom, .. } in objects {
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
    VI: Soa + Linear<f32> + Copy,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VI, VtxOut=VI>,
        R: Rasterize,
    {
        st.stats.prims_in += self.faces.len();
        st.stats.verts_in += self.verts.len();

        let mvp = &st.modelview * &st.projection;

        let bbox_vis = bbox_visibility(&self.bbox, &mvp);
        if bbox_vis != Hidden {

            let verts: Vec<_> = self.verts.iter()
                .map(|v| Vertex {
                    // TODO do transform in vertex shader?
                    coord: self.vertex_coords[v.coord].transform(&mvp),
                    attr: self.vertex_attrs.get(&v.attr),
                })
                .map(|v| shader.shade_vertex(v))
                .collect();

            let faces: Vec<_> = self.faces.iter()
                .map(|f| Face {
                    verts: f.verts,
                    attr: self.face_attrs[f.attr]
                })
                .collect();

            let (verts, faces) = hsr::hidden_surface_removal(verts, faces, bbox_vis);

            render_faces(verts, faces, st, shader, raster);

            if let Some(col) = st.options.bounding_boxes {
                render_bbox(&self.bbox, st, raster, col);
            }
        }
    }
}

impl<VI, U> Render<U, VI, VI> for SubMesh<'_, VI, U>
where
    VI: Soa + Linear<f32> + Copy + Debug,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VtxOut=VI>,
        R: Rasterize
    {
        st.stats.prims_in += self.face_indices.len();
        st.stats.verts_in += self.mesh.verts.len();

        let mesh = self.mesh;
        let mvp = &st.modelview * &st.projection;

        let bbox_vis = bbox_visibility(&self.bbox, &mvp);
        if bbox_vis != Hidden {
            let faces: Vec<_> = self.face_indices.iter()
                .map(|&fi| mesh.faces[fi])
                .collect();

            let mut verts: Vec<_> = mesh.verts.iter()
                .map(|v| Vertex {
                    coord: ZERO,
                    attr: mesh.vertex_attrs.get(&v.attr),
                })
                .collect();

            // Shade and transform only coords that are actually used
            for vi in faces.iter().flat_map(|f| f.verts) {
                verts[vi].coord = mesh.vertex_coords[mesh.verts[vi].coord].transform(&mvp);
                verts[vi] = shader.shade_vertex(verts[vi]);
            }

            let faces: Vec<_> = faces.into_iter()
                .map(|f| Face {
                    verts: f.verts,
                    attr: mesh.face_attrs[f.attr]
                })
                .collect();

            let (verts, faces) = hsr::hidden_surface_removal(verts, faces, bbox_vis);

            render_faces(verts, faces, st, shader, raster);

            if let Some(col) = st.options.bounding_boxes {
                render_bbox(&self.bbox, st, raster, col);
            }
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
    col: Color
) {
    for edge in bbox.edges() {
        let edge = LineSeg(edge.map(|v| vertex(v, ())));
        edge.render(st, &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(col)
        }, raster);
    }
}

fn render_faces<V, U, S, R>(
    mut verts: Vec<Vertex<V>>,
    mut faces: Vec<Face<usize, U>>,
    st: &mut State,
    shader: &mut S,
    raster: &mut R
)
where
    U: Copy,
    V: Linear<f32> + Copy,
    S: Shader<U, V, V, VtxOut = V>,
    R: Rasterize
{
    if faces.is_empty() { return; }

    st.stats.objs_out += 1;
    st.stats.prims_out += faces.len();
    st.stats.verts_out += verts.len();

    if st.options.depth_sort { depth_sort(&mut faces); }
    perspective_divide(&mut verts, st.options.perspective_correct);

    for v in &mut verts {
        v.coord.transform_mut(&st.viewport);
    }

    for f in faces {
        let verts = f.verts.map(|i| with_depth(verts[i]));

        // TODO Dispatching based on whether PC is enabled causes a major
        // performance regression in checkers.rs but not in benchmarks.
        // Should be investigated.
        if true /* st.options.perspective_correct */ {
            tri_fill(verts, f.attr, |span| {
                st.stats.pix_in += span.xs.1 - span.xs.0;
                st.stats.pix_out += raster.rasterize_span(shader, span)
            });
        } else {
            /*tri_fill(verts, f.attr, |span| {
                st.stats.pixels += raster.rasterize_span(shader, span)
            });*/
        }

        if let Some(col) = st.options.wireframes {
            let [a, b, c] = verts;
            for edge in [[a, b], [b, c], [c, a]] {
                line(edge, col, |mut frag| {
                    // Avoid Z fighting
                    frag.varying.0 -= 0.01;
                    raster.rasterize_frag(&mut ShaderImpl {
                        vs: identity,
                        fs: |_| Some(frag.uniform)
                    }, frag);
                });
            }
        }
    }
}

impl<V: Copy> Render<(), V, V> for LineSeg<V> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V>,
        R: Rasterize
    {
        st.stats.prims_in += 1;
        st.stats.verts_in += 2;

        let mvp = &st.modelview * &st.projection;

        let mut verts = self.0.map(|mut v| {
            v.coord.transform_mut(&mvp);
            shader.shade_vertex(v)
        }).to_vec();
        let mut clip_out = Vec::new();
        hsr::clip(&mut verts, &mut clip_out);
        if let &[a, b, ..] = clip_out.as_slice() {

            st.stats.prims_out += 1;
            st.stats.verts_out += 2;
            let verts = [
                clip_to_screen(a, &st.viewport),
                clip_to_screen(b, &st.viewport)
            ];
            line(verts, (), |frag: Fragment<_>| {
                st.stats.pix_in += 1;
                st.stats.pix_out += raster.rasterize_frag(shader, frag);
            })
        }
    }
}

impl<V: Copy> Render<(), V, V> for Polyline<V> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V, VtxOut = V>,
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

                if v0.coord.y > v2.coord.y { swap(v0, v3); swap(v1, v2); }
                if v0.coord.x > v1.coord.x { swap(v0, v1); swap(v2, v3); }

                // TODO extract to fn rect_fill
                let (x0, y0) = (v0.coord.x.round(), v0.coord.y.round());
                let (x1, y1) = (v2.coord.x.round(), v2.coord.y.round());
                let v = Varying::between((v0.attr, v1.attr), (v3.attr, v2.attr), y1 - y0);

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


fn depth_sort<VA: Copy, FA: Copy>(_faces: &mut Vec<Face<VA, FA>>) {
    todo!()
}

fn perspective_divide<A>(verts: &mut Vec<Vertex<A>>, pc: bool)
where
    A: Linear<f32> + Copy
{
    for Vertex { coord, attr } in verts {
        let w = 1.0 / coord.w;
        *coord = coord.mul(w);
        if pc {
            *attr = attr.mul(w);
        }
    }
}

#[inline]
fn with_depth<A>(v: Vertex<A>) -> Vertex<(f32, A)> {
    v.attr_with(|v| (v.coord.z, v.attr))
}

#[inline]
fn clip_to_screen<A>(mut v: Vertex<A>, viewport: &Mat4) -> Vertex<(f32, A)> {
    v.coord = (v.coord / v.coord.w).transform(viewport);
    with_depth(v)
}
