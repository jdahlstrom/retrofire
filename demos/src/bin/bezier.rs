use std::mem::swap;
use std::ops::ControlFlow::Continue;

use re::math::spline::BezierSpline;
use re::math::vary::Vary;
use re::math::{vec2, Affine, Linear, Vec2, Vec2i};
use rf::minifb::Window;

fn line([mut p0, mut p1]: [Vec2; 2]) -> impl Iterator<Item = Vec2i> {
    if p0.y() > p1.y() {
        swap(&mut p0, &mut p1);
    }
    let [dx, dy] = p1.sub(&p0).0;
    let dxa = dx.abs();

    let (step, n) = if dxa > dy {
        let dy_dx = dy / dxa;
        (vec2(dx.signum(), dy_dx), dxa)
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
        .title("minifb front demo")
        .size(W, H)
        .build();

    let mut pts = [
        vec2(200.0, 100.0),
        vec2(0.0, H as f32),
        vec2(W as f32, 100.0),
        vec2(500.0, 300.0),
    ];

    let mut deltas = [
        vec2(1.0, 0.2),
        vec2(-0.2, -0.7),
        vec2(0.8, 0.4),
        vec2(-0.6, 0.6),
    ];

    win.run(|frame| {
        let b = BezierSpline::new(&pts);
        // Stop once error is less than one pixel
        let apx = b.approximate(|a| a.len() < 1.0);
        let buf = &mut frame.buf.color_buf;

        for seg in apx.windows(2) {
            for pt in line(seg.try_into().unwrap()) {
                if let Some(p) = buf.get_mut(pt) {
                    *p = 0xFF_FF_FF
                };
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
