use std::mem::swap;
use std::ops::ControlFlow::Continue;

use re::math::rand::{Distrib, Uniform, UnitDisk, Xorshift64};
use re::math::spline::BezierSpline;
use re::prelude::*;

use re_front::minifb::Window;
use re_front::Frame;

fn line([mut p0, mut p1]: [Vec2; 2]) -> impl Iterator<Item = Vec2u> {
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

    p0.vary(step, Some(n as u32))
        .map(|p| vec2(p.x() as u32, p.y() as u32))
}

const W: u32 = 640;
const H: u32 = 480;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//bezier")
        .size(W, H)
        .build();

    let gen = Xorshift64::from_time();

    let (mut pts, mut deltas): (Vec<Vec2>, Vec<Vec2>) =
        Uniform(gen, vec2(0.0, 0.0)..vec2(W as f32, H as f32))
            .iter()
            .zip(UnitDisk(gen).iter())
            .take(4)
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

        let max = vec2((W - 1) as f32, (H - 1) as f32);
        for (p, d) in pts.iter_mut().zip(deltas.iter_mut()) {
            *p = (*p + *d * 200.0 * dt.as_secs_f32()) //
                .clamp(&Vec2::zero(), &max);

            if p[0] == 0.0 || p[0] == max.x() {
                d[0] = -d[0];
            }
            if p[1] == 0.0 || p[1] == max.y() {
                d[1] = -d[1];
            }
        }

        Continue(())
    });
}
