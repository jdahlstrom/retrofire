use core::ops::ControlFlow::Continue;

use re::prelude::*;

use re::geom::{Edge, Ray};
use re::math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64};
use re::render::raster::line;
use re_front::{Frame, dims, minifb::Window};

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

    let mut pos_vels: Vec<(Point2, Vec2)> =
        (pos, vel).samples(rng).take(32).collect();

    // Disable some unneeded things
    win.ctx.color_clear = None;
    win.ctx.depth_clear = None;

    win.run(|Frame { dt, buf, .. }| {
        // Fade out previous frame a bit
        buf.color_buf
            .buf
            .iter_mut()
            .for_each(|c| *c = c.saturating_sub(0x08_08_02));

        let rays: Vec<Ray<_>> = pos_vels
            .chunks(2)
            .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
            .collect();

        let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        let approx = b.approximate(|err| err.len_sqr() < 1.0);

        for Edge(p0, p1) in approx.edges() {
            let vs = [p0, p1].map(|p| vertex(p.to_pt3().to(), ()));
            line(vs, |sl| {
                buf.color_buf.buf[sl.y][sl.xs].fill(0xFF_FF_FF);
            })
        }

        let dt = dt.as_secs_f32();
        for (pos, vel) in pos_vels.iter_mut() {
            *pos = (*pos + 80.0 * *vel * dt).clamp(&min, &max);
            let [dx, dy] = &mut vel.0;
            if pos.x() == min.x() || pos.x() == max.x() {
                *dx = -*dx;
            }
            if pos.y() == min.y() || pos.y() == max.y() {
                *dy = -*dy;
            }
        }
        Continue(())
    });
}
