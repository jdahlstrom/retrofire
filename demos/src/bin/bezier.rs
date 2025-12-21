use core::ops::ControlFlow::Continue;
use minifb::{MouseButton, MouseMode};
use re::prelude::*;

use re::core::geom::Polygon;
use re::core::{
    geom::{Edge, Ray},
    math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64},
    math::spline::approximate,
    render::raster::line,
};
use re::front::{Frame, dims, minifb::Window};
use re::geom::triangulate;

fn main() {
    let dims @ (w, h) = dims::SVGA_800_600;

    let mut win = Window::builder()
        .title("retrofire//bezier")
        .dims(dims)
        .target_fps(Some(30))
        .build()
        .expect("should create window");

    let (min, max) =
        (pt2(100.0, 100.0), pt2(w as f32 - 100.0, h as f32 - 100.0));

    let rng = &mut Xorshift64::from_time();
    let pos = Uniform::<Point2>(min..max);
    let vel = VectorsOnUnitDisk;

    let mut pos_vels: Vec<(Point2, Vec2)> =
        (pos, vel).samples(rng).take(32).collect();

    // Disable some unneeded things
    //win.ctx.color_clear = None;
    //win.ctx.depth_clear = None;

    let mut poly = Polygon::default();
    let mut down = false;
    win.run(|Frame { dt, buf, win, .. }| {
        let buf = &mut buf.borrow_mut().color_buf.buf;

        if !down && win.imp.get_mouse_down(MouseButton::Left) {
            down = true;
            let (mx, my) = win.imp.get_mouse_pos(MouseMode::Clamp).unwrap();
            poly.0.push(pt2(mx, my));

            eprintln!("{}", poly.0.len());
        }
        if !win.imp.get_mouse_down(MouseButton::Left) {
            down = false;
        }

        let tris = triangulate(&poly);

        for tri in tris {
            for Edge(&p, &q) in tri.edges() {
                let a = vertex(p.to_pt3(), ());
                let b = vertex(q.to_pt3(), ());
                line([a, b], |sl| {
                    buf[sl.y][sl.xs].fill(0xFF);
                });
            }
        }

        for Edge(&p, &q) in poly.edges() {
            let a = vertex(p.to_pt3(), ());
            let b = vertex(q.to_pt3(), ());
            line([a, b], |sl| {
                buf[sl.y][sl.xs].fill(!0);
            });
        }

        Continue(())
    });
}
