use std::cell::RefCell;

use sdl2::keyboard::Scancode;

use geom::{mesh::Mesh, Sprite};
use math::{
    Angle::*,
    rand::{Distrib, Random, Uniform},
    spline::BezierCurve,
    transform::*,
    vec::{self, *},
};
use render::{
    fx::{anim, anim::*, particle::*},
    Raster,
    raster::Fragment,
    Render,
    Renderer,
    scene::{Obj, Scene},
    tex::{TexCoord, Texture, uv}
};
use util::color::*;

use crate::runner::*;

mod runner;

type VA = TexCoord;
type FA = usize;

fn main() {
    let margin = 10;
    let w = 960;
    let h = 540;

    const YELLOW: Color = rgb(255, 255, 0);

    let textures = [
        Texture::new(2, &[BLACK, WHITE, WHITE, BLACK]),
        Texture::new(
            4,
            &[
                //
                BLACK, RED, RED, BLACK, //
                RED, YELLOW, YELLOW, RED, //
                RED, YELLOW, YELLOW, RED, //
                BLACK, RED, RED, BLACK,
            ],
        ),
    ];

    let camera = translate(pt(0., -1., 5.));

    let objects = vec![Obj {
        geom: checkers(),
        ..Obj::default()
    }];
    let mut scene = Scene::<Mesh<VA, FA>> { objects, camera };

    let bez = bezier();

    let ptx = ParticleSys::new(
        10000,
        Sprite {
            center: ORIGIN,
            width: 0.1,
            height: 0.1,
            vertex_attrs: [
                uv(0.0, 0.0),
                uv(1.0, 0.0),
                uv(1.0, 1.0),
                uv(0.0, 1.0),
            ],
            face_attr: 1,
        },
    );

    let mut rdr = renderer(w, h, margin);

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

    runner
        .run(|mut frame| {
            // Update
            {
                animation.animate(frame.delta_t);
            }
            // Render
            {
                let raster = &mut Raster {
                    shade: |frag: Fragment<VA>, idx: FA| {
                        textures[idx].sample(frag.varying)
                    },
                    test: |_| true,
                    output: |(x, y), c| frame.buf.plot(x, y, c)
                };

                rdr.render_scene(&scene, raster);

                rdr.modelview = scene.camera.clone();
                ptx.borrow().render(&mut rdr, raster);
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

            rdr.stats.frames += 1;
            Ok(Run::Continue)
        })
        .unwrap();

    runner.print_stats(rdr.stats);
}

fn checkers() -> Mesh<VA, FA> {
    let size: isize = 10;

    let mut vs = vec![];
    let mut uvs = vec![];
    for j in -size..=size {
        for i in -size..=size {
            vs.push(pt(i as f32, 0.0, j as f32));
            uvs.push(uv(i as f32, j as f32));
        }
    }
    let mut fs = vec![];
    let size = size as usize;
    for j in 0..2 * size {
        for i in 0..2 * size {
            let w = 2 * size + 1;
            let idx = w * j + i;
            fs.push([idx, idx + w + 1, idx + 1]);
            fs.push([idx, idx + w, idx + w + 1]);
        }
    }
    Mesh::builder()
        .verts(vs.clone())
        .faces(fs)
        .vertex_attrs(uvs)
        .face_attrs(std::iter::repeat(0))
        .build()
}

fn renderer(w: i32, h: i32, margin: i32) -> Renderer {
    let mut rdr = Renderer::new();
    rdr.options.perspective_correct = true;
    rdr.projection = perspective(0.1, 50., w as f32 / h as f32, Deg(90.0));
    rdr.viewport = viewport(
        margin as f32,
        (h - margin) as f32,
        (w - margin) as f32,
        margin as f32,
    );
    rdr
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
