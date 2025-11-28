use core::ops::ControlFlow::Continue;
use re::prelude::*;
use std::process::exit;

use re::core::geom::{Edge, Polyline, Ray};
use re::core::math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64};
use re::core::math::spline::CatmullRomSpline;
use re::core::render::raster::line;

use re::front::{Frame, dims, minifb::Window};
use re::prelude::spline::BSpline;

fn main() {
    let dims @ (w, h) = dims::SVGA_800_600;

    let mut win = Window::builder()
        .title("retrofire//bezier")
        .dims(dims)
        .build()
        .expect("should create window");

    let (min, max) = (
        pt2(100.0, 80.0),
        pt2(w as f32, h as f32) - vec2(100.0, 80.0),
    );

    let rng = &mut Xorshift64::from_time();
    let pos = Uniform::<Point2>(min..max);
    let vel = VectorsOnUnitDisk;

    let mut pos_vels: Vec<(Point2, Vec2)> =
        (pos, vel).samples(rng).take(10).collect();

    // Disable some unneeded things
    win.ctx.color_clear = None;
    win.ctx.depth_clear = None;

    win.run(|Frame { dt, buf, .. }| {
        let buf = &mut buf.borrow_mut().color_buf.buf;

        // Fade out previous frame a bit
        buf.iter_mut()
            .for_each(|c| *c = c.saturating_sub(!0));

        // let rays: Vec<Ray<_>> = pos_vels
        //     .chunks(2)
        //     .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
        //     .collect();

        let pts: Vec<Point2<_>> = pos_vels.iter().map(|(p, _)| *p).collect();

        //let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        //let approx = b.approximate(1.0);

        let &[p0, p1, .., pm, pn] = pts.as_slice() else {
            panic!()
        };

        let mut pts2 = vec![p0 + (p0 - p1)];

        pts2.extend(pts.clone());
        pts2.push(pn + (pn - pm));

        let spl = CatmullRomSpline::new(pts2.as_slice());
        let approx = Polyline::<Point2>::new(
            0.0.vary_to(1.0, 1000).map(|t| spl.eval(t)),
        );

        for Edge(p0, p1) in approx.edges() {
            let vs = [p0, p1].map(|p| vertex(p.to_pt3().to(), ()));
            line(vs, |sl| {
                buf[sl.y][sl.xs].fill(!0);
            })
        }

        for t in 0.0.vary_to(1.0, 400) {
            let p = spl.eval(t);
            let v = spl.velocity(t) * 0.05;
            // let vs = [p, p + v].map(|p| vertex(p.to_pt3().to(), ()));
            // line(vs, |sl| {
            //     buf[sl.y][sl.xs].fill(0x00_FF_00);
            // });
            let vs = [p - v.perp(), p + v.perp()]
                .map(|p| vertex(p.to_pt3().to(), ()));
            line(vs, |sl| {
                buf[sl.y][sl.xs].fill(0x66_66_FF);
            })
        }

        // let spl = BSpline::new(pts2.as_slice());
        // let approx = Polyline::<Point2>::new(
        //     0.0.vary_to(1.0, 1000).map(|t| spl.eval(t)),
        // );
        //
        // for Edge(p0, p1) in approx.edges() {
        //     let vs = [p0, p1].map(|p| vertex(p.to_pt3().to(), ()));
        //     line(vs, |sl| {
        //         buf[sl.y][sl.xs].fill(0x00_99_FF);
        //     })
        // }

        for pt in pts {
            let x = pt.x() as u32;
            let y = pt.y() as u32;
            buf[[x, y]] = !0;
            buf[[x - 1, y]] = !0;
            buf[[x, y - 1]] = !0;
            buf[[x + 1, y]] = !0;
            buf[[x, y + 1]] = !0;
        }

        let dt = dt.as_secs_f32();
        for (pos, vel) in &mut pos_vels {
            *pos = (*pos + 20.0 * *vel * dt).clamp(&min, &max);
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
