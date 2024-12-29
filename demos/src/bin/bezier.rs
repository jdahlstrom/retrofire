use std::{mem::swap, ops::ControlFlow::Continue};

use re::prelude::*;

use re::geom::Ray;
use re::math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64};

use re_front::{dims::SVGA_800_600, minifb::Window, Frame};

fn line([mut p0, mut p1]: [Point2; 2]) -> impl Iterator<Item = Point2u> {
    if p0.y() > p1.y() {
        swap(&mut p0, &mut p1);
    }
    let [dx, dy] = (p1 - p0).0;
    let abs_dx = dx.abs();

    let (step, n) = if abs_dx > dy {
        (vec2(dx.signum(), dy / abs_dx), abs_dx)
    } else {
        (vec2(dx / dy, 1.0), dy)
    };

    p0.vary(step, Some(n as u32 + 1))
        .map(|p| p.map(|c| c as u32))
}

fn main() {
    let dims @ (w, h) = SVGA_800_600;

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
        let rays: Vec<Ray<_, _>> = pos_vels
            .chunks(2)
            .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
            .collect();

        let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        let apx = b.approximate(|err| err.len_sqr() < 1.0);

        for seg in apx.windows(2) {
            for pt in line([seg[0], seg[1]]) {
                // The curve can't go out of bounds if the control points don't
                buf.color_buf[pt] = 0xFF_FF_FF;
            }
        }

        let secs = dt.as_secs_f32();
        for (pos, vel) in pos_vels.iter_mut() {
            *pos = (*pos + *vel * 200.0 * secs).clamp(&min, &max);
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
