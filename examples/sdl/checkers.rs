use std::f32::consts::PI;

use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::Renderer;
use render::raster::*;
use Run::*;

use crate::runner::*;

mod runner;

fn checkers() -> Mesh<Vec4, ()> {
    let size: usize = 4;
    let isize = size as i32;

    let mut vs = vec![];
    for j in -isize..=isize {
        for i in -isize..=isize {
            vs.push(pt(i as f32, 0.0, j as f32));
        }
    }
    let mut fs = vec![];
    for j in 0..2 * size {
        for i in 0..2 * size {
            let w = 2 * size + 1;
            fs.push([w * j + i, w * (j + 1) + i + 1, w * j + i + 1]);
            fs.push([w * j + i, w * (j + 1) + i, w * (j + 1) + i + 1]);
        }
    }
    Mesh::from_verts_and_faces(vs.clone(), fs)
        .with_vertex_attrs(vs)
        .validate().unwrap()
}

fn shade(frag: Fragment<(Vec4, Vec4)>, _: ()) -> Vec4 {
    let v = frag.varying.1 * 1.;
    let bw = v.x.floor() as i32 & 1 ^ v.z.floor() as i32 & 1;
    let bw = bw as f32;

    vec4(bw, bw, bw, 0.0)
}

fn main() {
    let margin = 50;
    let w = 800;
    let h = 600;

    let mesh = checkers();
    let mut camera = Mat4::identity();

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1., 50., w as f32 / h as f32, PI / 2.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let model_to_world = scale(8.0, 8.0, 8.0) * &translate(0., -4., 20.);

    let mut runner = SdlRunner::new(w, h).unwrap();

    runner.run(|frame| {
        rdr.set_transform(&model_to_world * &camera);
        rdr.render(mesh.clone(), shade, |x, y, col| {
            let idx = 4 * (w as usize * y + x);
            let col = 255. * col;
            frame.buf[idx + 0] = col.z as u8;
            frame.buf[idx + 1] = col.y as u8;
            frame.buf[idx + 2] = col.x as u8;
        });

        for scancode in frame.pressed_keys {
            let t = -10. * frame.delta_t;
            let r = -frame.delta_t;
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
        Ok(Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}
