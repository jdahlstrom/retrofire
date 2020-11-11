use std::f32::consts::PI;
use std::time::*;

use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color;
use sdl2::rect::Rect;

use geom::solids::teapot;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::{Renderer, Stats};
use render::raster::*;
use render::shade::*;

fn main() {
    let sdl = sdl2::init().unwrap();

    let w = 800;
    let h = 600;
    let margin = 100;
    let window = sdl.video().unwrap()
                    .window("retrofire-sdl demo", w, h)
                    .position_centered()
                    .build().unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    //let mesh = torus(0.3, 11, 23).gen_normals();
    let mesh = teapot().gen_normals().validate().expect("Invalid mesh!");

    fn shade(frag: Fragment<(Vec4, Vec4)>, _face_n: Vec4) -> Vec4 {
        let Fragment { varying: (coord, normal), .. } = frag;
        let light_dir = (pt(-1.0, 2., -2.) - coord).normalize();
        let view_dir = (pt(0.0, 0.0, 0.0) - coord).normalize();

        255. * (
            expose_rgb(
                vec4(0.05, 0.05, 0.08, 0.0)
                    + 0.6 * vec4(1.0, 0.9, 0.6, 0.0) * lambert(normal, light_dir)
                    + 0.6 * vec4(1.0, 1.0, 1.0, 0.0) * phong(normal, view_dir, light_dir, 5)
                , 3.)
        )
    }

    let mut theta = 0.;
    let mut view_dir = Mat4::identity();
    let mut trans = dir(0.0, 0.0, 40.0);

    let mut stats;

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1., 60., 4. / 3., PI / 3.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let start = Instant::now();
    'running: loop {
        let mut surface = window.surface(&event_pump).unwrap();
        surface.fill_rect(Rect::new(0, 0, w, h), Color::GREY).unwrap();

        let tf = scale(1.0, 1.0, 1.0)
            * &rotate_x(-PI / 2.)
            * &translate(0., -5., 0.)
            * &rotate_x(theta * -0.57)
            * &rotate_y(theta)
            * &translate(trans.x, trans.y, trans.z)
            * &view_dir;
        rdr.set_transform(tf);

        let buf = surface.without_lock_mut().unwrap();
        let mut plot = |x: usize, y: usize, col: Vec4| {
            let idx = 4 * (w as usize * y + x);

            //let col = 255.0 * col;
            buf[idx + 0] = col.z as u8;
            buf[idx + 1] = col.y as u8;
            buf[idx + 2] = col.x as u8;
        };

        stats = rdr.render(mesh.clone(), &shade, &mut plot);

        surface.update_window().unwrap();

        for event in event_pump.poll_iter() {
            use Event::*;
            use Keycode::*;
            match event {
                Quit { .. } | KeyDown { keycode: Some(Escape), .. } => break 'running,
                _ => {}
            }
        }

        for scancode in event_pump.keyboard_state().pressed_scancodes() {
            use Scancode::*;
            match scancode {
                W => trans.z += 0.2,
                S => trans.z -= 0.2,
                D => trans.x += 0.3,
                A => trans.x -= 0.3,
                Left => view_dir = view_dir * &rotate_y(0.01),
                Right => view_dir = view_dir * &rotate_y(-0.01),
                _ => {},
            }
        }

        theta += 0.02;
    }

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