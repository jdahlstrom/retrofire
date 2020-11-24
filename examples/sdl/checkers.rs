use std::f32::consts::PI;

use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use geom::solids::unit_cube;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::{Renderer, Obj, Scene};
use render::raster::*;
use Run::*;

use crate::runner::*;

mod runner;

fn checkers() -> Mesh<(), Vec4> {
    let size: usize = 20;
    let isize = size as i32;

    let mut vs = vec![];
    for j in -isize..=isize {
        for i in -isize..=isize {
            vs.push(pt(i as f32, 0.0, j as f32));
        }
    }
    let mut fs = vec![];
    let mut colors = vec![];
    for j in 0..2 * size {
        for i in 0..2 * size {
            let w = 2 * size + 1;
            fs.push([w * j + i, w * (j + 1) + i + 1, w * j + i + 1]);
            fs.push([w * j + i, w * (j + 1) + i, w * (j + 1) + i + 1]);

            let c = 255. * (j & 1 ^ i & 1) as f32 * vec4(1.0, 1.0, 1.0, 0.0);
            colors.push(c);
            colors.push(c);
        }
    }
    Mesh::from_verts_and_faces(vs.clone(), fs)
        .with_face_attrs(colors)
        .validate().unwrap()
}

fn main() {
    let margin = 50;
    let w = 800;
    let h = 600;
    let mut camera = Mat4::identity();

    let mut objects = vec![];
    objects.push(Obj { tf: translate(0., -1., 0.), mesh: checkers() });
    let face_colors = vec![X, X, Y, Y, ZERO, ZERO, ZERO, ZERO, Z, Z, X+Y, X+Y];
    for j in -4..=4 {
        for i in -4..=4 {
            let mesh = unit_cube().with_face_attrs(
                face_colors.iter().map(|&c| 255.*c));
            let tf = translate(4. * i as f32, 0., 4. * j as f32);
            objects.push(Obj { tf, mesh })
        }
    }

    let mut rdr = Renderer::new();
    rdr.set_z_buffer(vec![f32::INFINITY; w * h], w);
    rdr.set_projection(perspective(0.1, 50., w as f32 / h as f32, PI / 2.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|frame| {
        let shade = |_, color| color;
        let mut plot = |x, y, col: Vec4| {
            let idx = 4 * (w as usize * y + x);
            frame.buf[idx + 0] = col.z as u8;
            frame.buf[idx + 1] = col.y as u8;
            frame.buf[idx + 2] = col.x as u8;
        };

        let scene = Scene {
            objects: objects.clone(),
            camera: camera.clone(),
        };
        rdr.render_scene(scene, &shade, &mut plot);

        for scancode in frame.pressed_keys {
            let t = -8. * frame.delta_t;
            let r = -2. * frame.delta_t;
            use Scancode::*;
            match scancode {
                W => camera *= &translate(0.0, 0.0, t),
                A => camera *= &translate(-t, 0.0, 0.0),
                S => camera *= &translate(0.0, 0.0, -t),
                D => camera *= &translate(t, 0.0, 0.0),

                Left => camera *= &rotate_y(r),
                Right => camera *= &rotate_y(-r),

                _ => {},
            }
        }
        rdr.stats.frames += 1;
        Ok(Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}
