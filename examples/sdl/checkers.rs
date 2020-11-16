use std::f32::consts::PI;
use std::time::*;

use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color;
use sdl2::rect::Rect;

use geom::mesh::Mesh;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::{Renderer, Stats};
use render::raster::*;

fn main() {
    let sdl = sdl2::init().unwrap();

    let w = 800;
    let h = 600;
    let margin = 50;
    let window = sdl.video().unwrap()
                    .window("retrofire-sdl demo", w, h)
                    .position_centered()
                    .build().unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

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

    let mesh = Mesh::from_verts_and_faces(vs.clone(), fs)
        .with_vertex_attrs(vs)
        .validate().unwrap();

    fn shade(frag: Fragment<(Vec4, Vec4)>, _: ()) -> Vec4 {
        let v = frag.varying.1 * 1.;
        let bw = v.x.floor() as i32 & 1 ^ v.z.floor() as i32 & 1;
        let bw = bw as f32;

        vec4(bw, bw, bw, 0.0)
    }

    let mut camera = Mat4::identity();

    let mut stats;

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1., 50., 4. / 3., PI / 2.0));
    rdr.set_viewport(viewport(margin as f32, (h - margin) as f32,
                              (w - margin) as f32, margin as f32));

    let start = Instant::now();
    'running: loop {
        let mut surface = window.surface(&event_pump).unwrap();
        surface.fill_rect(Rect::new(0, 0, w, h), Color::GREY).unwrap();

        let model_to_world = scale(8.0, 8.0, 8.0) * &translate(0., -4., 20.);

        rdr.set_transform(&model_to_world * &camera);

        let buf = surface.without_lock_mut().unwrap();

        let mut plot = |x: usize, y: usize, col: Vec4| {
            let idx = 4 * (w as usize * y + x);

            let col = 255.0 * col;
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
                S => camera *= &translate(0.0, 0.0, 0.2),
                W => camera *= &translate(0.0, 0.0, -0.2),
                A => camera *= &translate(0.2, 0.0, 0.0),
                D => camera *= &translate(-0.2, 0.0, 0.0),

                Left => camera *= &rotate_y(-0.03),
                Right => camera *= &rotate_y(0.03),

                _ => {},
            }
        }
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