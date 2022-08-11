use std::cell::RefCell;
use std::ops::ControlFlow::*;

use sdl2::keyboard::Scancode;

use front::sdl::SdlRunner;
use geom::{
    Align,
    bbox::BoundingBox,
    mesh::{Face, Mesh},
    Sprite,
};
use geom::mesh::vertex_indices;
use math::{
    Angle::*,
    mat::Mat4,
    rand::{Distrib, Random, Uniform},
    spline::BezierCurve,
    transform::*,
    vec::{self, *},
};
use render::{
    fx::{anim, anim::*, particle::*},
    raster::Fragment,
    Render as _,
    scene::{Obj, Scene},
    shade::ShaderImpl,
    State,
};
use util::{
    color::*,
    tex::{TexCoord, Texture, uv},
};
use util::tex::{SamplerOnce, SamplerRepeatPot};

type VA = TexCoord;
type FA = usize;

fn main() {
    let margin = 10;
    let w = 960;
    let h = 540;

    const YELLOW: Color = rgb(255, 255, 0);

    let textures = [
        Texture::owned(2, 2, &[BLACK, WHITE, WHITE, BLACK]),
        Texture::owned(
            4, 4,
            &[
                //
                BLACK, RED, RED, BLACK, //
                RED, YELLOW, YELLOW, RED, //
                RED, YELLOW, YELLOW, RED, //
                BLACK, RED, RED, BLACK,
                //
                RED, RED, RED, RED,
                //
                RED,
            ],
        ),
    ];

    let camera = translate(pt(0., -1., 5.));

    let objects = vec![Obj {
        tf: Mat4::identity(),
        geom: checkers(),
    }];
    let mut scene = Scene::<Mesh<(VA, ), FA>> { objects, camera };

    let bez = bezier();

    let ptx = ParticleSys::new(
        10000,
        Sprite::new(
            ORIGIN,
            Align::Center,
            0.2, 0.2,
            [uv(0.0, 0.0), uv(1.0, 0.0), uv(1.0, 1.0), uv(0.0, 1.0)],
            1,
        ),
    );

    let mut st = state(w, h, margin);

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    let mut animation = Animation::default();

    let ptx = RefCell::new(ptx);
    let rand = &mut Random::new();
    let mut t = 0.0;
    let mut burst = |dt: f32| {
        t = (t + 0.01 * dt) % 1.0;
        ptx.borrow_mut().emit(1, || Particle {
            pos: 2.0 * Y + 2.0 * bez.eval(t),
            vel: 0.5 * InUnitBall.from(rand),
            life: 5.0,
        });
    };
    let mut burst = anim::repeat(0.01, &mut burst);
    animation.add(&mut burst);

    let mut update = |dt| {
        ptx.borrow_mut().update(dt, |p| {
            if p.pos.y < 0.055 {
                p.pos.y = 0.05;
                p.vel.y = 0.0;
                p.vel -= (10.0 * dt).min(1.0) * p.vel
            } else {
                p.vel.y -= 0.9 * dt;
                p.vel -= 0.1 * dt * p.vel;
            }
        });
    };
    animation.add(&mut update);

    let sample_checkers = SamplerRepeatPot::new(&textures[0]);
    let sample_ptx = SamplerOnce;

    runner
        .run(|mut frame| {
            // Update
            {
                animation.animate(frame.delta_t);
            }
            // Render
            {
                let shade_checkers = &mut ShaderImpl {
                    vs: |v| v,
                    fs: |frag: Fragment<(VA, ), FA>| {
                        Some(sample_checkers.sample(&textures[0], frag.varying.0))
                    },
                };
                let shade_ptx = &mut ShaderImpl {
                    vs: |v| v,
                    fs: |frag: Fragment<VA, FA>| {
                        sample_ptx.sample(&textures[1], frag.varying)
                    },
                };

                scene.render(&mut st, shade_checkers, &mut frame.buf);

                st.modelview = scene.camera.clone();
                ptx.borrow().render(&mut st, shade_ptx, &mut frame.buf);
            }
            // Input
            {
                let t = -4. * frame.delta_t;
                let r = -2. * Rad(frame.delta_t);
                let cam = &mut scene.camera;
                for scancode in &frame.pressed_keys {
                    use Scancode::*;
                    match scancode {
                        W => *cam *= &translate(t * vec::Z),
                        A => *cam *= &translate(-t * vec::X),
                        S => *cam *= &translate(-t * vec::Z),
                        D => *cam *= &translate(t * vec::X),

                        Left => *cam *= &rotate_y(-r),
                        Right => *cam *= &rotate_y(r),

                        P => animation.speed = 1.0,
                        O => animation.speed = -0.25,
                        //frame.screenshot("screenshot.ppm")?,
                        _ => {}
                    }
                }
            }

            st.stats.frames += 1;
            Continue(())
        })
        .unwrap();

    runner.print_stats(st.stats);
}

fn checkers() -> Mesh<(VA, ), FA> {
    let size = 40.0;
    let vcs = [-X - Z, -X + Z, X - Z, X + Z].map(|c| c * size + W);

    let tcs = [uv(0.0, 0.0), uv(0.0, size), uv(size, 0.0), uv(size, size)];

    let verts = vec![
        vertex_indices(0, 0),
        vertex_indices(1, 1),
        vertex_indices(2, 2),
        vertex_indices(3, 3),
    ];

    let faces = vec![
        Face { verts: [0, 1, 3], attr: 0 },
        Face { verts: [0, 3, 2], attr: 0 },
    ];

    let bbox = BoundingBox::of(&vcs);

    Mesh {
        verts,
        faces,
        bbox,
        vertex_coords: vcs.into(),
        vertex_attrs: tcs.into(),
        face_attrs: vec![0],
    }
}

fn state(w: i32, h: i32, margin: i32) -> State {
    let mut st = State::new();
    st.options.perspective_correct = true;
    st.projection = perspective(0.1, 50., w as f32 / h as f32, Deg(90.0));
    st.viewport = viewport(
        margin as f32,
        (h - margin) as f32,
        (w - margin) as f32,
        margin as f32,
    );
    st
}

fn bezier() -> BezierCurve<Vec4> {
    let mut rnd = Random::new();
    let dist = Uniform(pt(-1.0, -1.0, -1.0)..pt(1.0, 1.0, 1.0));
    let mut pts = rnd.iter(dist);
    let mut points = Vec::new();
    points.extend(&[pts.next().unwrap(), pts.next().unwrap()]);
    for _ in 0..32 {
        let cp = pts.next().unwrap();
        let p = pts.next().unwrap();
        points.extend(&[cp, p, p + (p - cp)]);
    }
    points.extend(pts.take(2));
    let bez = BezierCurve::new(&points);
    bez
}
