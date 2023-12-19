use std::ops::ControlFlow::Continue;

use cam::FpsCamera;
use re::geom::{vertex, Tri, Vertex};
use re::math::color::{rgb, rgba};
use re::math::mat::{scale, translate, RealToReal};
use re::math::space::{Affine, Linear, Real};
use re::math::vec::splat;
use re::math::{vec2, vec3, Mat4x4, Vec3};
use re::render::{
    raster::Frag, render, shader::Shader, Model, ModelToProjective,
    NdcToScreen, View, ViewToProjective, World,
};
use re_front::minifb::Window;

mod cam;

type WorldMat = Mat4x4<RealToReal<3, Model, World>>;
type ViewMat = Mat4x4<RealToReal<3, World, View>>;
type ProjMat = Mat4x4<ViewToProjective>;
type ScreenMat = Mat4x4<NdcToScreen>;

type MvpMat = Mat4x4<ModelToProjective>;

const X: Vec3 = vec3(1.0, 0.0, 0.0);
const Y: Vec3 = vec3(0.0, 1.0, 0.0);
const Z: Vec3 = vec3(0.0, 0.0, 1.0);

fn main() {
    let verts: [Vertex<Vec3<Real<3, Model>>, _>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0).to(), rgb(1.0, 0.3, 0.1)),
        vertex(vec3(-1.0, 1.0, 0.0).to(), rgb(0.2, 0.8, 0.3)),
        vertex(vec3(1.0, -1.0, 0.0).to(), rgb(0.2, 0.5, 1.0)),
        vertex(vec3(1.0, 1.0, 0.0).to(), rgb(1.0, 0.3, 0.1)),
    ];

    let mut win = Window::builder()
        .title("minifb front demo")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &MvpMat| vertex(mvp.apply(&v.pos), v.pos),
        |frag: Frag<Vec3<_>>| {
            let [r, g, b] = frag.var.add(&splat(1.0)).mul(100.0).0;

            const ROTATE: u32 = 31;
            const SEED: u64 = 0x517cc1b727220a95;

            Some(rgba(r as u8, g as u8, b as u8, 0xFF))
        },
    );

    let world: WorldMat = scale(splat(4.0))
        //.then(&rotate_x(degs(90.0)))
        .then(&translate(vec3(0.0, -1.0, 0.0)))
        .to();

    let mut cam = FpsCamera::new(640, 480)
        .perspective(1.0, 0.1..10_000.0)
        .viewport(vec2(10, 470)..vec2(630, 10));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        // Camera

        let mut cam_vel = Vec3::zero();

        use minifb::Key::*;
        /*for k in frame.win.imp.get_keys() {
            match k {
                W => cam_vel[2] += 4.0,
                S => cam_vel[2] -= 2.0,
                D => cam_vel[0] += 3.0,
                A => cam_vel[0] -= 3.0,

                _ => {}
            }
        }

        let (mx, my) = frame
            .win
            .imp
            .get_mouse_pos(MouseMode::Pass)
            .unwrap();

        cam.rotate_to(degs(-0.4 * mx), degs(-0.4 * (my - 240.0)));*/
        cam.translate(cam_vel.mul(frame.dt.as_secs_f32()));

        // Transform

        let mvp = world.then(&cam.world_to_project());

        // Render

        *frame.stats += render(
            [Tri([0, 1, 2]), Tri([3, 2, 1])],
            verts,
            &shader,
            &mvp,
            cam.viewport,
            &mut frame.buf,
        );

        Continue(())
    });
}
