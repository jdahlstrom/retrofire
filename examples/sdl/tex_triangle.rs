use std::time::{Duration, Instant};

use sdl2::{
    event::Event,
    keyboard::Keycode,
    pixels::Color
};

use geom::mesh::Vertex;
use math::vec::vec4;
use render::{
    Framebuf,
    raster::{Fragment, tex::tex_fill, tri_fill},
    Rasterize,
    tex::{TexCoord, Texture}
};
use util::Buffer;
use util::color::{BLACK, WHITE};
use render::tex::Sampler;

fn vert(x: f32, y: f32, u: f32, v: f32, w: f32) -> Vertex<TexCoord> {
    Vertex { coord: vec4(x, y, 0.0, 0.0), attr: TexCoord { u, v, w } }
}

fn main() {

    const TOGGLE: bool = true;

    println!("Hello SDL");

    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();

    let window = video.window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .build()
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    let xs = [(50., 1.), (700., 2.), (600., -2.5)];
    let ys = [(400., -2.), (50., -1.2), (550., 4.)];

    //let texbuf = load_pnm("examples/sdl/crate.ppm").unwrap();
    let texbuf = Buffer::from_vec(4, vec![
        BLACK, WHITE, BLACK, WHITE,
        WHITE, BLACK, WHITE, BLACK,
        BLACK, WHITE, BLACK, WHITE,
        WHITE, BLACK, WHITE, BLACK,
    ]);
    let tex = &Texture::from(texbuf.clone());

    let mut frames = 0;
    let mut elapsed = Duration::ZERO;

    let zbuf = &mut Buffer::new(800, 600, f32::INFINITY);

    'running: loop {
        let mut surface = window.surface(&event_pump).unwrap();
        let w = surface.width() as usize;

        surface.fill_rect(None, Color::GRAY).unwrap();
        zbuf.fill(f32::INFINITY);

        let buf = surface.without_lock_mut().unwrap();

        let fb = &mut Framebuf {
            color: Buffer::borrow(w, buf),
            depth: zbuf,
            depth_test: true,
            depth_write: true,
        };

        let verts = [
            vert(xs[0].0, ys[0].0, 0.0, 0.0, 1.0),
            vert(xs[1].0, ys[1].0, 1.5, 0.0, 1.5),
            vert(xs[2].0, ys[2].0, 0.0, 2.2, 2.2),
        ];

        let start = Instant::now();
        if TOGGLE {
            tex_fill(verts, &texbuf, fb);
        } else {
            let verts = verts.map(|Vertex { coord, attr }| {
                Vertex { coord, attr: (coord.z, Sampler::new(&tex, attr)) }
            });
            tri_fill(verts, |sc| {
                fb.rasterize(sc, |frag: Fragment<(f32, Sampler)>| {
                    Some(frag.varying.1.sample())
                });
            });
        }
        elapsed += start.elapsed();

        /*for (x, dx) in &mut xs {
            if *x < 11.0 || *x as u32 >= 788 { *dx = -*dx }
            *x += *dx;
        }
        for (y, dy) in &mut ys {
            if *y < 11.0 || *y as u32 >= 588 { *dy = -*dy }
            *y += *dy;
        }*/

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
        //thread::sleep(Duration::from_millis(1_000 / 60));

        frames += 1;
    }

    println!("Avg: {} fps", frames as f32 / elapsed.as_secs_f32());
}
