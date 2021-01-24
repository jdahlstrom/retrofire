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
use math::Angle::{Rad, Deg};

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

    let model_tf = rotate_x(Deg(-90.0)) * &translate(0., -5., 0.);

    fn shade(frag: Fragment<(Vec4, Vec4)>, _face_n: Vec4) -> Color {
        let (coord, normal) = frag.varying;
        let light_dir = (pt(-1.0, 2., -2.) - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        let ambient = vec4(0.05, 0.05, 0.08, 0.0);
        let diffuse = 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(normal, light_dir);
        let specular = 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(normal, view_dir, light_dir, 5);

        expose_rgb(ambient + diffuse + specular, 3.).into()
    }

    let mut theta = Rad(0.);
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut rdr = Renderer::new();
    rdr.projection = perspective(1., 60., w as f32 / h as f32, Deg(60.0));
    rdr.viewport = viewport(margin as f32, (h - margin) as f32,
                            (w - margin) as f32, margin as f32);

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|Frame { mut buf, zbuf, pressed_keys, delta_t, .. }| {

        let tf = &model_tf
            * &rotate_x(-0.57 * theta)
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

        rdr.render_scene(&scene, &mut Raster {
            shade,
            test: |frag| zbuf.test(frag),
            output: |(x, y), col| buf.plot(x, y, col)
        });

        for scancode in pressed_keys {
            use Scancode::*;
            let t = 15. * delta_t;
            let r = Rad(delta_t);
            match scancode {
                W => trans.z += t,
                S => trans.z -= t,
                D => trans.x += t,
                A => trans.x -= t,
                Left => view_dir *= &rotate_y(r),
                Right => view_dir *= &rotate_y(-r),
                _ => {},
            }
        }

        theta = theta + Rad(delta_t);
        rdr.stats.frames += 1;

        Ok(Run::Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}