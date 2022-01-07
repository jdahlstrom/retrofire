use std::ops::ControlFlow::*;

use sdl2::keyboard::Scancode;

use front::sdl::*;
use geom::mesh2::{Mesh, Vertex as Vertex2};
use geom::mesh::Vertex;
use geom::solids::teapot;
use math::Angle::{Deg, Rad};
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::raster::*;
use render::scene::{Obj, Scene};
use render::shade::*;
use util::color::Color;
use util::tex::TexCoord;

type CoordNormAndTexCrd = (Vec4, Vec4, TexCoord);
type Vert = Vertex<CoordNormAndTexCrd>;
type Frag = Fragment<CoordNormAndTexCrd>;

struct Shd(Vec4);

impl VertexShader<CoordNormAndTexCrd> for Shd {
    fn shade_vertex(&self, v: Vert) -> Vert {
        v
    }
}
impl FragmentShader<CoordNormAndTexCrd, ()> for Shd {
    fn shade_fragment(&self, f: Frag) -> Option<Color> {
        let (coord, n, _uv) = f.varying;

        let light_dir = (self.0 - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        let ambient = vec4(0.05, 0.05, 0.08, 0.0);
        let diffuse = 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(n, light_dir);
        let specular = 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(n, view_dir, light_dir, 5);

        Some(expose_rgb(ambient + diffuse + specular, 3.).into())
    }
}

fn main() {
    let w = 800;
    let h = 600;
    let margin = 50;

    let teapot = teapot();

    let teapot = Mesh::<CoordNormAndTexCrd> {
        verts: teapot.verts.into_iter()
            .map(|Vertex2 { coord, attr: [a0, a1] }|
                Vertex2 { coord, attr: [coord, a0, a1] })
            .collect(),
        vertex_coords: teapot.vertex_coords.clone(),
        vertex_attrs: (teapot.vertex_coords, teapot.vertex_attrs.0, teapot.vertex_attrs.1),
        faces: teapot.faces,
        face_attrs: teapot.face_attrs,
        bbox: teapot.bbox,
    };

    let model_tf = rotate_x(Deg(-90.0)) * translate(-5.0 * Y);
    let light = pt(-1.0, 2.0, 2.0);

    let mut theta = Rad(0.);
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut rdr = Renderer::new();
    rdr.projection = perspective(1., 60., w as f32 / h as f32, Deg(60.0));
    rdr.viewport = viewport(margin as f32, (h - margin) as f32,
                            (w - margin) as f32, margin as f32);

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|Frame { mut buf, pressed_keys, delta_t, .. }| {

        let obj_tf = &model_tf
            * &rotate_x(-0.57 * theta)
            * &rotate_y(theta);

        let view_tf = &translate(trans) * &view_dir;

        let tf = &obj_tf * &view_tf;

        let mut geom = teapot.clone();
        geom.vertex_attrs.0.transform_mut(&tf);
        geom.vertex_attrs.1.transform_mut(&tf);

        let scene = Scene {
            objects: vec![Obj { tf, geom }],
            camera: Mat4::identity(),
        };

        scene.render(&mut rdr, &mut Shd(light), &mut buf);

        for scancode in pressed_keys {
            use Scancode::*;
            let t = 15. * delta_t;
            let r = Rad(delta_t);
            match scancode {
                W => trans.z += t,
                S => trans.z -= t,
                D => trans.x += t,
                A => trans.x -= t,

                Left => view_dir *= rotate_y(r),
                Right => view_dir *= &rotate_y(-r),

                _ => {},
            }
        }

        theta = theta + Rad(delta_t);
        rdr.stats.frames += 1;

        Continue(())
    }).unwrap();

    runner.print_stats(rdr.stats);
}
