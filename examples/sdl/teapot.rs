use std::f32::consts::PI;
use std::time::*;

use sdl2::keyboard::Scancode;

use geom::solids::teapot;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::{Renderer, Stats};
use render::raster::*;
use render::shade::*;
use runner::*;

mod runner;

fn main() {
    let w = 800;
    let h = 600;
    let margin = 50;

    let mesh = teapot().gen_normals().validate().expect("Invalid mesh!");

    fn vs(tf: &Mat4, (coord, norm): (Vec4, Vec4)) -> (Vec4, Vec4) {
        (tf * coord, tf * norm)
    }

    fn fs(frag: Fragment<(Vec4, Vec4)>, _face_n: Vec4) -> Vec4 {
        let Fragment { varying: (coord, normal), .. } = frag;
        let light_dir = (pt(-1.0, 2., -2.) - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        let ambient = vec4(0.05, 0.05, 0.08, 0.0);
        let diffuse = 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(normal, light_dir);
        let specular = 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(normal, view_dir, light_dir, 5);

        255. * expose_rgb(ambient + diffuse + specular, 3.)
    }

    let mut theta = 0.;
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1., 60., w as f32 / h as f32, PI / 3.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let start = Instant::now();

    SdlRunner::new(w, h).unwrap().run(|r| {
        let tf = scale(1.0, 1.0, 1.0)
            * &rotate_x(-PI / 2.)
            * &translate(0., -5., 0.)
            * &rotate_x(theta * -0.57)
            * &rotate_y(theta)
            * &translate(trans.x, trans.y, trans.z)
            * &view_dir;
        rdr.set_transform(tf);

        let mut surf = r.surface().unwrap();
        let buf = surf.without_lock_mut().unwrap();

        rdr.render(mesh.clone(), vs, fs, |x, y, col: Vec4| {
            let idx = 4 * (w as usize * y + x);
            buf[idx + 0] = col.z as u8;
            buf[idx + 1] = col.y as u8;
            buf[idx + 2] = col.x as u8;
        });

        for scancode in r.keystate() {
            use Scancode::*;
            match scancode {
                W => trans.z += 0.2,
                S => trans.z -= 0.2,
                D => trans.x += 0.3,
                A => trans.x -= 0.3,
                Left => view_dir *= &rotate_y(0.01),
                Right => view_dir *= &rotate_y(-0.01),
                _ => {},
            }
        }

        theta += 0.02;

        Ok(Run::Continue)
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