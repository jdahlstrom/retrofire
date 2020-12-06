use std::f32::consts::PI;

use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use geom::solids::unit_sphere;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::color::*;
use Run::*;

use crate::runner::*;

mod runner;

fn checkers() -> Mesh<(), Color> {
    let size: usize = 40;
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

            let c = gray(255 * (j & 1 ^ i & 1) as u8);
            colors.extend(&[c, c]);
        }
    }
    Mesh::from_verts_and_faces(vs.clone(), fs)
        .with_face_attrs(colors).validate().unwrap()
}

fn main() {
    let margin = 50;
    let w = 800;
    let h = 600;

    let camera = Mat4::identity();
    let mut objects = vec![];
    objects.push(Obj { tf: translate(0., -1., 0.), mesh: checkers() });

    for j in -10..=10 {
        for i in -10..=10 {
            let mesh = unit_sphere(9, 9);
            let flen = mesh.faces.len();
            let mesh = mesh.with_face_attrs(
                [RED, GREEN, BLUE].iter().copied().cycle().take(flen));
            let tf = translate(4. * i as f32, 0., 4. * j as f32);
            objects.push(Obj { tf, mesh });
        }
    }

    let mut scene = Scene { objects, camera };

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(0.1, 50., w as f32 / h as f32, PI / 2.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|frame| {
        let shade = |_, color| color;
        let mut plot = |x, y, c: Color| {
            let idx = 4 * (w as usize * y + x);
            let [_, r, g, b] = c.to_argb();
            frame.buf[idx + 0] = b;
            frame.buf[idx + 1] = g;
            frame.buf[idx + 2] = r;
        };

        rdr.render_scene(&scene, &shade, &mut plot);

        for scancode in frame.pressed_keys {
            let t = -8. * frame.delta_t;
            let r = -2. * frame.delta_t;
            use Scancode::*;
            let cam = &mut scene.camera;
            match scancode {
                W => *cam *= &translate(0.0, 0.0, t),
                A => *cam *= &translate(-t, 0.0, 0.0),
                S => *cam *= &translate(0.0, 0.0, -t),
                D => *cam *= &translate(t, 0.0, 0.0),

                Left => *cam *= &rotate_y(r),
                Right => *cam *= &rotate_y(-r),

                _ => {},
            }
        }
        rdr.stats.frames += 1;
        Ok(Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}
