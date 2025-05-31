//! A demo that displays an animated Bézier curve on the screen.

use std::ops::{ControlFlow, ControlFlow::*};

use re::prelude::*;

use re::{
    geom::{Edge, Ray},
    math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64},
    render::raster::line,
};
use re_front::{
    Frame, dims,
    minifb::{Framebuf, Window},
};

struct Bezier {
    pos_vels: Vec<(Point2, Vec2)>,
    bounds: (Point2, Point2),
}

fn main() {
    let dims @ (w, h) = dims::SVGA_800_600;

    let mut win = Window::builder()
        .title("retrofire//bezier")
        .dims(dims)
        .build()
        .expect("should create window");

    let (min, max) =
        (pt2(100.0, 100.0), pt2(w as f32 - 100.0, h as f32 - 100.0));

    let rng = &mut Xorshift64::from_time();
    let pos = Uniform::<Point2>(min..max);
    let vel = VectorsOnUnitDisk;

    let pos_vels = (pos, vel).samples(rng).take(32).collect();

    let mut bez = Bezier { pos_vels, bounds: (min, max) };

    win.run(|frame| bez.do_frame(frame));
}

impl Bezier {
    fn do_frame(
        &mut self,
        frame: &mut Frame<Window, Framebuf>,
    ) -> ControlFlow<()> {
        // Setup
        let Self { pos_vels, bounds: (min, max) } = self;
        let rays: Vec<Ray<_>> = pos_vels
            .chunks(2)
            .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
            .collect();

        let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        let approx = b.approximate(|err| err.len_sqr() < 1.0);

        // Render
        let mut cbuf = frame.buf.color_buf.as_mut_slice2();
        for Edge(p0, p1) in approx.edges() {
            let p0 = p0.to_pt3().to();
            let p1 = p1.to_pt3().to();
            line([vertex(p0, ()), vertex(p1, ())], |sl| {
                cbuf[sl.y][sl.xs].fill(0xFF_FF_FF);
            })
        }

        // Update
        let secs = frame.dt.as_secs_f32();
        for (pos, vel) in pos_vels.iter_mut() {
            *pos = (*pos + 40.0 * secs * *vel).clamp(&min, &max);
            let [dx, dy] = &mut vel.0;
            if pos.x() == min.x() || pos.x() == max.x() {
                *dx = -*dx;
            }
            if pos.y() == min.y() || pos.y() == max.y() {
                *dy = -*dy;
            }
        }
        Continue(())
    }
}
