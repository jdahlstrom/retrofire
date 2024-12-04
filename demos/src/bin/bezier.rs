use std::mem::swap;
use std::ops::ControlFlow::Continue;

use re::prelude::*;

use re::math::{
    point::{pt2, Point2, Point2u},
    rand::{Distrib, Uniform, UnitDisk, Xorshift64},
    spline::BezierSpline,
};

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

    // TODO Fix once points are Vary
    p0.to_vec()
        .vary(step, Some(n as u32))
        .map(|p| p.map(|c| c as u32).to_pt())
}

fn main() {
    let dims @ (w, h) = SVGA_800_600;

    let mut win = Window::builder()
        .title("retrofire//bezier")
        .dims(dims)
        .build()
        .expect("should create window");

    let rng = Xorshift64::from_time();
    let pos = Uniform(vec2(0.0, 0.0)..vec2(w as f32, h as f32));
    let vel = UnitDisk;

    let (mut pts, mut deltas): (Vec<Point2>, Vec<Vec2>) = (pos, vel)
        .samples(rng)
        .take(4)
        .map(|(p, v)| (p.to_pt(), v))
        .unzip();

    win.run(|Frame { dt, buf, .. }| {
        let b = BezierSpline::new(&pts);
        // Stop once error is less than one pixel
        let apx = b.approximate(|err| err.len_sqr() < 1.0);

        for seg in apx.windows(2) {
            for pt in line([seg[0], seg[1]]) {
                // The curve can't go out of bounds if the control points don't
                buf.color_buf[pt] = 0xFF_FF_FF;
            }
        }

        let max = pt2((w - 1) as f32, (h - 1) as f32);
        let secs = dt.as_secs_f32();
        for (p, d) in pts.iter_mut().zip(deltas.iter_mut()) {
            *p = (*p + *d * 200.0 * secs).clamp(&pt2(0.0, 0.0), &max);
            let [dx, dy] = &mut d.0;
            if p.x() == 0.0 || p.x() == max.x() {
                *dx = -*dx;
            }
            if p.y() == 0.0 || p.y() == max.y() {
                *dy = -*dy;
            }
        }
        Continue(())
    });
}
