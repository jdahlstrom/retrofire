use std::thread;
use std::time::Duration;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color as SdlColor;
use sdl2::rect::Rect;

use geom::mesh::Vertex;
use math::vec::*;
use render::raster::{Fragment, tri_fill};
use util::color::*;

fn vert(x: f32, y: f32, c: Color) -> Vertex<Color> {
    Vertex { coord: vec4(x, y, 0.0, 0.0), attr: c }
}

fn main() {
    println!("Hello SDL");

    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();

    let window = video.window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .build()
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    let mut xs = [(100., 1.), (500., 2.), (300., -2.5)];
    let mut ys = [(300., -2.), (100., -1.2), (400., 4.)];
    let mut a = 0.0f32;

    'running: loop {
        let mut surface = window.surface(&event_pump).unwrap();
        let (width, height) = surface.size();

        surface.fill_rect(Rect::new(0, 0, width, height), SdlColor::BLACK).unwrap();

        let buf = surface.without_lock_mut().unwrap();
        tri_fill(
            [
                vert(xs[0].0, ys[0].0, RED),
                vert(xs[1].0, ys[1].0, GREEN),
                vert(xs[2].0, ys[2].0, BLUE) //Z + 0.25 * (X + Y))
            ],
            |Fragment { coord, varying }| {
                let pos = 4 * (coord.1 * width as usize  + coord.0) as usize;
                let [_, r, g, b] = Color::from(varying).to_argb();
                buf[pos + 0] = b; //(127. + 128. * varying.cos()) as u8;
                buf[pos + 1] = g; // (127. + 128. * (varying * 0.7).cos()) as u8;
                buf[pos + 2] = r; // (127. + 128. * (varying * 1.6).sin()) as u8;
            });

        for (x, dx) in &mut xs {
            if *x < 1.0 || *x as u32 >= width { *dx = -*dx }
            *x += *dx;
        }
        for (y, dy) in &mut ys {
            if *y < 1.0 || *y as u32 >= height { *dy = -*dy }
            *y += *dy;
        }

        surface.update_window().unwrap();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                },
                _ => {}
            }
        }
        a += 0.01;
        thread::sleep(Duration::from_millis(1_000 / 60));
    }
}