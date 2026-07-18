use core::ops::ControlFlow::Continue;

use re::prelude::*;

use re::core::render::View;
use re::core::{
    geom::{Polyline, Ray},
    math::color::gray,
    math::rand::{Distrib, Uniform, VectorsOnUnitDisk, Xorshift64},
    math::spline::approximate,
    render::{render, shader},
    util::dims,
};
use re::front::{Frame, minifb::Window};

fn main() {
    let dims @ Dims(w, h) = dims::SVGA_800_600;

    let mut win = Window::builder()
        .title("retrofire//bezier")
        .dims(dims)
        .build()
        .expect("should create window");

    let (min, max) = (pt2(0.0, 0.0), pt2(w as f32, h as f32));

    let rng = &mut Xorshift64::from_time();
    let pos = Uniform::<Point2<_>>(min..max);
    let vel = VectorsOnUnitDisk;

    let mut pos_vels: Vec<(Point2<_>, Vec2<View>)> = (pos, vel)
        .samples(rng)
        .take(32)
        .map(|(p, v)| (p, v.to()))
        .collect();

    // Disable some unneeded things
    win.ctx.color_clear = None;
    win.ctx.depth_clear = None;

    let proj = orthographic(
        pt3(100.0, h as f32 - 100.0, -1.0),
        pt3(w as f32 - 100.0, 100.0, 1.0),
    );
    let vp = viewport(pt2(100, 100)..pt2(w - 100, h - 100));

    win.run(|Frame { dt, buf, ctx, .. }| {
        let buf = &mut buf.borrow_mut().color_buf;

        // Fade out previous frame a bit
        buf.buf
            .slice_mut(pt2(100, 100)..pt2(w - 100, h - 100))
            .iter_mut()
            .for_each(|c| *c = c.saturating_sub(0x10_10_04));

        let rays: Vec<Ray<_>> = pos_vels
            .chunks(2)
            .map(|ch| Ray(ch[0].0, (ch[1].0 - ch[0].0) * 0.4))
            .collect();

        let b = BezierSpline::from_rays(rays);
        // Stop once error is less than one pixel
        let approx = approximate(&b, 1.0);

        let approx = Polyline::new(
            approx
                .0
                .into_iter()
                .map(|v| vertex(v.to_pt3(), ())),
        );

        let shader = shader::new(
            |v: Vertex<_, _>, tf: &ProjMat3<_>| vertex(tf.apply(&v.pos), ()),
            |_f, _| gray(0xFFu8).to_rgba(),
        );
        render(approx, &shader, &proj, vp, buf, ctx);

        let dt = dt.as_secs_f32();
        for (pos, vel) in &mut pos_vels {
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
