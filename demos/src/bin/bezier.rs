use std::array::from_fn;
use std::mem::swap;
use std::ops::ControlFlow::Continue;

use re::math::rand::{Distrib, Uniform, UnitDisk, XorShift64};
use re::math::spline::BezierSpline;
use re::math::{vec2, Affine, Linear, Vary, Vec2, Vec2i};
use rf::minifb::Window;

fn line([mut p0, mut p1]: [Vec2; 2]) -> impl Iterator<Item = Vec2i> {
    if p0.y() > p1.y() {
        swap(&mut p0, &mut p1);
    }
    let [dx, dy] = p1.sub(&p0).0;
    let abs_dx = dx.abs();

    let (step, n) = if abs_dx > dy {
        let dy_dx = dy / abs_dx;
        (vec2(dx.signum(), dy_dx), abs_dx)
    } else {
        let dx_dy = dx / dy;
        (vec2(dx_dy, 1.0), dy)
    };

    p0.vary(step, Some(n as u32))
        .map(|p| vec2(p.x() as i32, p.y() as i32))
}

const W: usize = 640;
const H: usize = 480;

fn main() {
    let mut win = Window::builder()
        .title("Bezier")
        .size(W, H)
        .build();

    let gen = XorShift64::from_time();

    let mut pts: [Vec2; 4] = {
        let mut d = Uniform(gen, [0.0, 0.0]..[W as f32, H as f32]);
        from_fn(|_| d.next().into())
    };
    let mut deltas: [Vec2; 4] = {
        let mut d = UnitDisk(gen);
        from_fn(|_| d.next().into())
    };

    win.run(|frame| {
        let b = BezierSpline::new(&pts);
        // Stop once error is less than one pixel
        let apx = b.approximate(|err| err.len() < 1.0);

        for seg in apx.windows(2) {
            for pt in line([seg[0], seg[1]]) {
                // The curve can't go out of bounds if the control points don't
                frame.buf.color_buf[pt] = 0xFF_FF_FF;
            }
        }

        let max = vec2((W - 1) as f32, (H - 1) as f32);
        for (p, d) in pts.iter_mut().zip(deltas.iter_mut()) {
            *p = p
                .add(&d.mul(200.0 * frame.dt.as_secs_f32()))
                .clamp(&vec2(0.0, 0.0), &max);

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
