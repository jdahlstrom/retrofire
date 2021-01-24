use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use geom::solids::unit_cube;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::color::*;
use render::raster::Fragment;
use render::tex::*;
use Run::*;

use crate::runner::*;
use math::Angle::{Deg, Rad};

mod runner;

fn checkers() -> Mesh<TexCoord, Color> {
    let size: usize = 40;
    let isize = size as i32;

    let mut vs = vec![];
    let mut texcoords = vec![];
    for j in -isize..=isize {
        for i in -isize..=isize {
            vs.push(pt(i as f32, 0.0, j as f32));
            texcoords.push(uv(0.0, 0.0));
        }
    }
    let mut fs = vec![];
    let mut colors = vec![];
    for j in 0..2 * size {
        for i in 0..2 * size {
            let w = 2 * size + 1;
            fs.push([w * j + i, w * (j + 1) + i + 1, w * j + i + 1]);
            fs.push([w * j + i, w * (j + 1) + i, w * (j + 1) + i + 1]);

            let c = gray(255 * (j & 1 ^ i & 1) as u8);
            colors.extend(&[c, c]);
        }
    }
    Mesh::from_verts_and_faces(vs.clone(), fs)
        .with_vertex_attrs(texcoords)
        .with_face_attrs(colors)
        .validate().unwrap()
}

fn main() {
    let margin = 50;
    let w = 800;
    let h = 600;

    let camera = Mat4::identity();
    let mut objects = vec![];
    objects.push(Obj { tf: translate(0., -1., 0.), mesh: checkers() });

    for j in -10..=10 {
        for i in -10..=10 {
            let mesh = unit_cube(); //unit_sphere(9, 9);
            let flen = mesh.faces.len();
            let mesh = mesh.with_face_attrs(
                [0xBB6611, 0xBB6611, 0x3399FF, 0x3399FF, 0x229944, 0x229944]
                    .iter().copied().map(Color).cycle().take(flen));

            // tex coords
            let mesh = mesh.with_vertex_attrs([
                uv(1.0, 1.0), uv(0.0, 1.0), uv(1.0, 0.0), uv(0.0, 0.0),
                uv(0.0, 1.0), uv(1.0, 1.0), uv(0.0, 0.0), uv(1.0, 0.0),
            ].iter().copied());

            let tf = translate(4. * i as f32, 0., 4. * j as f32);
            objects.push(Obj { tf, mesh });
        }
    }

    let mut scene = Scene { objects, camera };

    let mut rdr = Renderer::new();
    rdr.options.perspective_correct = true;
    rdr.projection = perspective(0.1, 50., w as f32 / h as f32, Deg(90.0));
    rdr.viewport = viewport(margin as f32, (h - margin) as f32,
                            (w - margin) as f32, margin as f32);

    let tex = {
        let (b, w) = (gray(0x33), gray(0xFF));
        Texture::new(8, &[
            w, b, b, b, b, b, w, w,
            w, b, b, w, w, b, b, w,
            w, b, b, w, w, b, b, w,
            w, b, b, w, w, b, b, w,
            w, b, b, b, b, b, w, w,
            w, b, b, w, w, b, b, w,
            w, b, b, w, w, b, b, w,
            w, b, b, w, w, w, b, b,
        ])
    };

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    runner.run(|Frame { mut buf, zbuf, pressed_keys, delta_t, .. }| {

        rdr.render_scene(&scene, &mut Raster {
            shade: |frag: Fragment<_>, c| c * tex.sample(frag.varying),
            test: |frag| zbuf.test(frag),
            output: |(x, y), c| buf.plot(x, y, c),
        });

        for scancode in pressed_keys {
            let t = -8. * delta_t;
            let r = -2. * Rad(delta_t);
            let cam = &mut scene.camera;
            use Scancode::*;
            match scancode {
                W => *cam *= &translate(0.0, 0.0, t),
                A => *cam *= &translate(-t, 0.0, 0.0),
                S => *cam *= &translate(0.0, 0.0, -t),
                D => *cam *= &translate(t, 0.0, 0.0),

                Left => *cam *= &rotate_y(r),
                Right => *cam *= &rotate_y(-r),

                _ => {},
            }
        }
        rdr.stats.frames += 1;
        Ok(Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}
