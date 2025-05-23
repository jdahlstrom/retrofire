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

    win.run(|Frame { dt, buf, .. }| {
        let rays: Vec<Ray<_>> = pos_vels
            .chunks(2)
            .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
            .collect();

        let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        let approx = b.approximate(|err| err.len_sqr() < 1.0);

        for Edge(p0, p1) in approx.edges() {
            let p0 = p0.to_pt3().to();
            let p1 = p1.to_pt3().to();
            line([vertex(p0, ()), vertex(p1, ())], |sl| {
                buf.color_buf[sl.y][sl.xs].fill(0xFF_FF_FF);
            })
        }

        let secs = dt.as_secs_f32();
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
    });
}
