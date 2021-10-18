use std::convert::identity;
use std::ops::ControlFlow::*;

use sdl2::keyboard::Scancode;

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
use front::sdl::*;

fn main() {
    let w = 800;
    let h = 600;
    let margin = 50;

    let teapot = teapot();

    let vattrs: Vec<_> = teapot.mesh.verts.iter().copied()
        .zip(teapot.mesh.vertex_attrs.iter().copied())
        .collect();

    let mesh = teapot.vertex_attrs(vattrs).build();

    let model_tf = rotate_x(Deg(-90.0)) * translate(-5.0 * Y);

    fn shade(frag: Fragment<(Vec4, Vec4)>) -> Option<Color> {
        let (coord, normal) = frag.varying;
        let light_dir = (pt(-1.0, 2., -2.) - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        let ambient = vec4(0.05, 0.05, 0.08, 0.0);
        let diffuse = 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(normal, light_dir);
        let specular = 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(normal, view_dir, light_dir, 5);

        Some(expose_rgb(ambient + diffuse + specular, 3.).into())
    }

    let mut theta = Rad(0.);
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut rdr = Renderer::new();
    rdr.projection = perspective(1., 60., w as f32 / h as f32, Deg(60.0));
    rdr.viewport = viewport(margin as f32, (h - margin) as f32,
                            (w - margin) as f32, margin as f32);

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|Frame { mut buf, pressed_keys, delta_t, .. }| {

        let tf = &model_tf
            * &rotate_x(-0.57 * theta)
            * &rotate_y(theta)
            * &translate(trans)
            * &view_dir;

        let mut geom = mesh.clone();
        for (c, n) in &mut geom.vertex_attrs {
            *c *= &tf;
            *n *= &tf;
        }
        let scene = Scene {
            objects: vec![Obj { tf, geom }],
            camera: Mat4::identity(),
        };

        rdr.render_scene(&scene, &mut ShaderImpl {
            vs: identity,
            fs: shade,
        }, &mut buf);

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
