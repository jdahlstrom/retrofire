use std::f32::consts::PI;

use sdl2::keyboard::Scancode;

use geom::solids::teapot;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::color::Color;
use render::raster::*;
use render::shade::*;
use runner::*;

mod runner;

fn main() {
    let w = 800;
    let h = 600;
    let margin = 50;

    let mesh = teapot().gen_normals().validate().expect("Invalid mesh!");

    let vattrs = mesh.verts.iter().copied()
        .zip(mesh.vertex_attrs.iter().copied())
        .collect::<Vec<_>>();

    let mesh = mesh.with_vertex_attrs(vattrs);

    let model_tf = rotate_x(-PI / 2.) * &translate(0., -5., 0.);

    fn shade(frag: Fragment<(Vec4, Vec4)>, _face_n: Vec4) -> Color {
        let (coord, normal) = frag.varying;
        let light_dir = (pt(-1.0, 2., -2.) - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        let ambient = vec4(0.05, 0.05, 0.08, 0.0);
        let diffuse = 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(normal, light_dir);
        let specular = 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(normal, view_dir, light_dir, 5);

        expose_rgb(ambient + diffuse + specular, 3.).into()
    }

    let mut theta = 0.;
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1., 60., w as f32 / h as f32, PI / 3.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|frame| {
        let tf = &model_tf
            * &rotate_x(theta * -0.57)
            * &rotate_y(theta)
            * &translate(trans.x, trans.y, trans.z)
            * &view_dir;

        let mut mesh = mesh.clone();

        for (c, n) in &mut mesh.vertex_attrs {
            *c = &tf * *c;
            *n = &tf * *n;
        }

        let scene = Scene {
            objects: vec![Obj { tf, mesh }],
            camera: Mat4::identity(),
        };

        rdr.render_scene(&scene, &shade, &mut |x, y, color| {
            let idx = 4 * (w as usize * y + x);
            let [_, r, g, b] = color.to_argb();
            frame.buf[idx + 0] = b;
            frame.buf[idx + 1] = g;
            frame.buf[idx + 2] = r;
        });

        for scancode in frame.pressed_keys {
            use Scancode::*;
            let dt = frame.delta_t;
            match scancode {
                W => trans.z += 15. * dt,
                S => trans.z -= 15. * dt,
                D => trans.x += 15. * dt,
                A => trans.x -= 15. * dt,
                Left => view_dir *= &rotate_y(dt),
                Right => view_dir *= &rotate_y(dt),
                _ => {},
            }
        }

        theta += frame.delta_t;
        rdr.stats.frames += 1;

        Ok(Run::Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}