use std::f32::consts::PI;
use std::time::*;

use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::{Renderer, Stats};
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

    let start = Instant::now();
    SdlRunner::new(w, h).unwrap().run(|r| {
        rdr.set_transform(&model_to_world * &camera);
        rdr.render(mesh.clone(), &shade, &mut |x, y, col| r.plot(x, y, col));

        for scancode in r.keystate() {
            use Scancode::*;
            match scancode {
                W => camera *= &translate(0.0, 0.0, -0.2),
                A => camera *= &translate(0.2, 0.0, 0.0),
                S => camera *= &translate(0.0, 0.0, 0.2),
                D => camera *= &translate(-0.2, 0.0, 0.0),

                Left => camera *= &rotate_y(-0.03),
                Right => camera *= &rotate_y(0.03),

                _ => {},
            }
        }
        Ok(Continue)
    }).unwrap();

    let stats = rdr.stats;
    let Stats { frames, pixels, faces_in, faces_out, .. } = stats;
    let elapsed = start.elapsed().as_secs_f32();
    println!("\n S  T  A  T  S");
    println!("═══════════════╕");
    println!(" Total         │ {}", stats);
    println!(" Per sec       │ {}", stats.avg_per_sec());
    println!(" Per frame     │ {}", stats.avg_per_frame());
    println!("───────────────┤");
    println!(" Avg pix/face  │ {}", pixels.checked_div(faces_out).unwrap_or(0));
    println!(" Avg vis faces │ {}%", (100 * faces_out).checked_div(faces_in).unwrap_or(0));
    println!(" Elapsed time  │ {:.2}s", elapsed);
    println!(" Average fps   │ {:.2}\n", frames as f32 / elapsed);
}
