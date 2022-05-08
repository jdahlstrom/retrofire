use std::fmt::Debug;
use std::mem::{replace, swap};
use std::time::Instant;

use geom::{LineSeg, Polyline, Sprite};
use geom::bbox::BoundingBox;
use geom::mesh::{Face, Mesh, Soa, SubMesh, Vertex};
use math::Linear;
use math::mat::Mat4;
use math::transform::Transform;
use math::vec::ZERO;

use crate::{Fragment, hsr, line, Rasterize, State, tri_fill};
use crate::hsr::Visibility;
use crate::hsr::Visibility::Hidden;
use crate::scene::{Obj, Scene};
use crate::shade::Shader;
use crate::vary::Varying;

pub trait Render<U, VI, FI=VI> {

    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, VI, VI, FI>,
        R: Rasterize;
}

impl<G, U, V> Render<U, V> for Scene<G>
where
    G: Render<U, V>,
    U: Copy,
    V: Linear<f32> + Copy,
{
    fn render<S, R>(&self, st: &mut State, shade: &mut S, raster: &mut R)
    where
        S: Shader<U, V>,
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

impl<VI, U> Render<U, VI> for Mesh<VI, U>
where
    VI: Soa + Linear<f32> + Copy + Debug,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI>,
        R: Rasterize
    {
        st.stats.faces_in += self.faces.len();

        let mvp = &st.modelview * &st.projection;

        let bbox_vis = bbox_visibility(&self.bbox, &mvp);
        if bbox_vis != Hidden {

            let verts: Vec<_> = self.verts.iter()
                .map(|v| Vertex {
                    // TODO do transform in vertex shader?
                    coord: self.vertex_coords[v.coord].transform(&mvp),
                    attr: VI::get(&self.vertex_attrs, &v.attr),
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

            render_faces(verts, faces, st, shader, raster)
        }
    }
}

impl<VI, U> Render<U, VI> for SubMesh<'_, VI, U>
where
    VI: Soa + Linear<f32> + Copy + Debug,
    U: Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, VI>,
        R: Rasterize
    {
        st.stats.faces_in += self.face_indices.len();

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
                    attr: VI::get(&mesh.vertex_attrs, &v.attr),
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
        }
    }
}

fn bbox_visibility(bbox: &BoundingBox, mvp: &Mat4) -> Visibility {
    let vs = bbox.verts().transform(&mvp);
    hsr::vertex_visibility(vs.iter())
}

fn render_faces<VI, U, S, R>(
    mut verts: Vec<Vertex<VI>>,
    mut faces: Vec<Face<usize, U>>,
    st: &mut State,
    shader: &mut S,
    raster: &mut R
) where
    U: Copy,
    VI: Copy + Debug + Linear<f32> + Soa,
    S: Shader<U, VI>,
    R: Rasterize
{
    if faces.is_empty() { return; }

    st.stats.objs_out += 1;
    st.stats.faces_out += faces.len();

    if st.options.depth_sort { depth_sort(&mut faces); }
    perspective_divide(&mut verts, st.options.perspective_correct);

    for v in &mut verts {
        v.coord.transform_mut(&st.viewport);
    }

    for f in faces {
        let verts = f.verts.map(|i| with_depth(verts[i]));

        tri_fill(verts, |frag| {
            if raster.rasterize(shader, frag.uniform(f.attr)) {
                st.stats.pixels += 1;
            }
        });

        if let Some(col) = st.options.wireframes {
            let [a, b, c] = verts;
            for e in [a, b, c, a].windows(2) {
                line([e[0], e[1]], |frag| {
                    if raster.test(frag.varying(frag.varying.0 - 0.001)) {
                        raster.output(frag.varying((0.0, col)));
                    }
                });
            }
        }
    }

    /* TODO Wireframe and bounding box debug rendering
    let mut render_edges = |st: &mut _,
                            edges: Vec<[Vec4; 2]>,
                            col: Color| {
        for edge in edges.into_iter()
            .map(|[a, b]| [vertex(a, col), vertex(b, col)])
            .map(LineSeg)
        {
            edge.render(st, &mut ShaderImpl {
                vs: |a| a,
                fs: |f: Fragment<_>| Some(f.varying),
            }, &mut Raster {
                test: |_| true,
                output: |f| raster.output(f),
            });
        }
    };
    if let Some(col) = st.options.wireframes {
        render_edges(rdr, self.edges(), col);
    }
    if let Some(col) = rdr.options.bounding_boxes {
        render_edges(rdr, self.bbox.edges(), col);
    }
    */
}


impl<VI> Render<(), VI> for LineSeg<VI>
where
    VI: Linear<f32> + Copy
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), VI>,
        R: Rasterize
    {
        st.stats.faces_in += 1;

        let mvp = &st.modelview * &st.projection;

        let mut verts = self.0.map(|mut v| {
            v.coord.transform_mut(&mvp);
            shader.shade_vertex(v)
        }).to_vec();
        let mut clip_out = Vec::new();
        hsr::clip(&mut verts, &mut clip_out);
        if let &[a, b, ..] = clip_out.as_slice() {
            st.stats.faces_out += 1;
            let verts = [
                clip_to_screen(a, &st.viewport),
                clip_to_screen(b, &st.viewport)
            ];
            line(verts, |frag: Fragment<_>| {
                if raster.rasterize(shader, frag) {
                    st.stats.pixels += 1;
                }
            });
        }
    }
}

impl<V> Render<(), V> for Polyline<V>
where
    V: Linear<f32> + Copy
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), V>,
        R: Rasterize,
    {
        for seg in self.edges() {
            seg.render(st, shader, raster);
        }
    }
}

impl<U, V> Render<U, V> for Sprite<V, U>
where
    U: Copy,
    V: Linear<f32> + Copy,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, V>,
        R: Rasterize,
    {
        st.stats.faces_in += 1;

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
                st.stats.faces_out += 1;

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
                        if raster.rasterize(shader, frag) {
                            st.stats.pixels += 1;
                        }
                    }
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

#[inline(always)]
fn with_depth<A>(v: Vertex<A>) -> Vertex<(f32, A)> {
    v.attr_with(|v| (v.coord.z, v.attr))
}

fn clip_to_screen<A>(mut v: Vertex<A>, viewport: &Mat4) -> Vertex<(f32, A)> {
    v.coord = (v.coord / v.coord.w).transform(viewport);
    with_depth(v)
}
