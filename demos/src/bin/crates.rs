use core::ops::ControlFlow::*;

use re::prelude::*;

use re::core::math::color::gray;
use re::core::render::{
    cam::{FirstPerson, Fov},
    clip::Status::*,
    light::Kind,
    scene::Obj,
    tex::SamplerClamp,
};
// Try also Rgb565 or Rgba4444
use re::core::util::{pixfmt::Rgba8888, pnm::read_pnm};

use re::front::sdl2::Window;
use re::geom::solids::{Build, Cube};

struct Uniform {
    pub mv: Mat4<Model, View>,
    pub proj: ProjMat3<View>,
    pub light: Light<View>,
}

type Varying = (Normal3, TexCoord);

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let tex_data = *include_bytes!("../../assets/crate.ppm");
    let tex = Texture::from(read_pnm(&tex_data[..]).expect("data exists"));

    let light_dir = vec3(-2.0, 1.0, -4.0).normalize();

    let floor_shader = shader::new(
        |v: Vertex3<_>, u: &Uniform| {
            vertex(u.mv.then(&u.proj).apply(&v.pos), v.attrib)
        },
        |frag: Frag<Vec2>, _: &Uniform| {
            let even_odd = (frag.var.x() > 0.5) ^ (frag.var.y() > 0.5);
            gray(if even_odd { 0.8 } else { 0.1 }).to_color4()
        },
    );
    let crate_shader = shader::new(
        |v: Vertex3<Varying>, u: &Uniform| {
            vertex(u.mv.then(&u.proj).apply(&v.pos), v.attrib)
        },
        |frag: Frag<Varying>, _: &Uniform| {
            let (n, uv) = frag.var;
            let kd = lerp(n.dot(&light_dir).max(0.0), 0.4, 1.0);
            let col = SamplerClamp.sample(&tex, uv);
            (col.to_color3f() * kd).to_color4()
        },
    );

    let shader3 = shader::new(
        |v: Vertex3<Color3f>, u: &Uniform| {
            let pos = u.mv.apply(&v.pos);
            let (light_col, light_dir) = u.light.eval(pos);
            let light_col = light_col.add(&gray(0.05));
            let lam = -light_dir.y();
            // TODO light_col * surface_col should be compwise multiplication
            vertex(u.proj.apply(&pos), light_col.mul(lam).mul(v.attrib.r()))
        },
        |f: Frag<Color3f>, _: &_| f.var.to_color4(),
    );

    let crate_shader_light = shader::new(
        // TODO
        |v: Vertex3<Normal3>, u: &Uniform| {
            let n_modl = v.attrib.to();
            let n_view = u.mv.apply(&n_modl);
            let pos_view = u.mv.apply(&v.pos);
            let (light_col, light_dir) = u.light.eval(pos_view);

            let refl = light_dir.reflect(n_view);

            let specular = (-pos_view.to_vec())
                .normalize()
                .dot(&refl)
                .max(0.0)
                .powi(20);

            let lam = n_view.dot(&light_dir).max(0.0);

            let pos_proj = u.proj.apply(&pos_view);
            let color = light_col.mul(lam + specular);

            vertex(pos_proj, color)
        },
        |frag: Frag<Color3f>, _: &_| {
            //let [x, y, z] = ((f.var + splat(1.0)) / 2.0).0;
            //rgb(x, y, z).to_color4()
            frag.var.to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        .transform(FirstPerson::default())
        .viewport((0..w, h..0))
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let light = Light::<Model> {
        color: rgb(1.0, 0.5, 0.1) * 1.1,
        kind: Kind::Spot {
            pos: pt3(0.0, -8.0, 0.0),
            dir: vec3(0.0, 1.0, 0.0),
            radii: (0.1, 0.3),
        },
        falloff: 0,
    };

    let floor = floor();
    let crates = crates();

    win.run(|frame| {
        //
        // Camera
        //

        let mut cam_vel = Vec3::zero();

        let ep = &frame.win.ev_pump;

        for key in ep.keyboard_state().pressed_scancodes() {
            use sdl2::keyboard::Scancode as Sc;
            match key {
                Sc::W => cam_vel[2] += 4.0,
                Sc::S => cam_vel[2] -= 2.0,
                Sc::D => cam_vel[0] += 3.0,
                Sc::A => cam_vel[0] -= 3.0,
                _ => {}
            }
        }

        let ms = ep.relative_mouse_state();
        cam.transform.rotate(
            turns(ms.x() as f32) * -0.001,
            turns(ms.y() as f32) * -0.001,
        );
        cam.transform
            .translate(cam_vel.mul(frame.dt.as_secs_f32()));

        let light_to_view = translate(4.0 * Vec3::X)
            .then(&rotate_y(turns(frame.t.as_secs_f32() * 0.1)))
            .to()
            .then(&cam.world_to_view());

        let light = light.transform(&light_to_view);

        //
        // Render
        //

        let batch = Batch::new()
            .viewport(cam.viewport)
            .target(frame.buf)
            .context(frame.ctx);

        // Floor
        {
            let Obj { geom, bbox, tf } = &floor;
            let mv = tf.then(&cam.world_to_view());
            let proj = cam.project;
            let mvp = mv.then(&proj);
            let light = Default::default();
            let uni = Uniform { mv, proj, light };

            if bbox.visibility(&mvp) != Hidden {
                batch
                    .clone()
                    .mesh(geom)
                    .uniform(&uni)
                    .shader(floor_shader)
                    .render();
            }
        }

        // Crates

        for Obj { geom, bbox, tf } in &crates {
            frame.ctx.stats.borrow_mut().objs.i += 1;

            let mv = tf.then(&cam.world_to_view());
            let proj = cam.project;
            let mvp = mv.then(&proj);

            // TODO Also if `Visible`, no further clipping or culling needed
            if bbox.visibility(&mvp) == Hidden {
                continue;
            }

            batch
                // TODO Try to get rid of clone
                .clone()
                .mesh(geom)
                .uniform(&Uniform { mv, proj, light })
                .shader(crate_shader)
                .render();

            frame.ctx.stats.borrow_mut().objs.o += 1;
        }

        Continue(())
    })
    .expect("should run");
}

fn crates() -> Vec<Obj<(Normal3, TexCoord)>> {
    let obj = Obj::new(Cube { side_len: 2.0 }.build());

    // let obj = Sphere {
    //     sectors: 80,
    //     segments: 50,
    //     radius: 1.0,
    // }
    // .build();

    // let obj = parse_obj(*include_bytes!("../../assets/teapot.obj"))
    //     .unwrap()
    //     .transform(
    //         &scale(splat(0.4))
    //             .then(&translate(-0.5 * Vec3::Y))
    //             .to(),
    //     )
    //     .with_vertex_normals()
    //     .build();

    let mut res = vec![];
    let n = 30;
    for i in (-n..=n).step_by(5) {
        for j in (-n..=n).step_by(5) {
            res.push(Obj {
                tf: translate3(i as f32, 0.0, j as f32).to(),
                // TODO Same geometry cloned many times
                ..obj.clone()
            });
        }
    }
    res
}
fn floor() -> Obj<Vec2> {
    let mut bld = Mesh::builder();

    let size = 50;
    for j in -size..=size {
        for i in -size..=size {
            let i_odd = i & 1;
            let j_odd = j & 1;

            let pos = pt3(i as f32, -1.0, j as f32);
            let attrib = vec2(i_odd as f32, j_odd as f32);
            bld.push_vert(pos, attrib);

            if j > -size && i > -size {
                let w = size * 2 + 1;
                let j = size + j;
                let i = size + i;
                let [a, b, c, d] = [
                    w * (j - 1) + (i - 1),
                    w * (j - 1) + i,
                    w * j + (i - 1),
                    w * j + i,
                ]
                .map(|i| i as usize);

                if i_odd ^ j_odd != 0 {
                    bld.push_face(a, c, d);
                    bld.push_face(a, d, b);
                } else {
                    bld.push_face(b, c, d);
                    bld.push_face(b, a, c)
                }
            }
        }
    }
    Obj::new(bld.build())
}
