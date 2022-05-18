use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use criterion::*;

use geom::mesh::{Mesh, GenVertex, VertexIndices};
use geom::solids::{Torus, UnitCube, UnitSphere};
use math::Angle::Rad;
use math::transform::*;
use math::vec::{dir, Y, Z};
use render::{Raster, Render as _, State};
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use render::shade::ShaderImpl;
use util::buf::Buffer;
use util::color::*;
use util::io::{load_pnm, save_ppm};
use util::tex::{SamplerOnce, TexCoord, Texture};

const W: usize = 128;

fn state() -> State {
    let mut st = State::new();
    st.projection = perspective(1.0, 10.0, 1.0, Rad(1.0));
    st.viewport = viewport(0.0, 0.0, W as f32, W as f32);
    st
}

fn check_hash(buf: &[Color], expected: u64) {
    let h = &mut DefaultHasher::new();
    buf.hash(h);
    let actual = h.finish();
    if actual != expected {
        eprintln!("Hashes differ: actual={} vs expected={}", actual, expected);
    }
}

fn save_screenshot(name: &str, buf: &mut [Color]) {
    let path = PathBuf::from(name);
    if path.exists() {
        let mut prev = path.clone();
        prev.set_extension("prev.ppm");
        std::fs::rename(path, prev).unwrap();
    }
    save_ppm(name, &Buffer::borrow(W, buf)).unwrap();
}

fn torus(c: &mut Criterion) {
    let mut st = state();
    st.modelview = scale(4.0);

    let mesh = Torus(0.2, 9, 9).build();

    let mut buf = vec![BLACK; W * W];
    c.bench_function("torus", |b| {
        b.iter(|| mesh.render(
            &mut st,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<(f32, Color)>| {
                    let (x, y) = frag.coord;
                    buf[W * y + x] = frag.varying.1;
                },
            },
        ));
    });
    check_hash(&buf, 5667555972845400088);
    save_screenshot("target/torus.ppm", &mut buf);
    eprintln!("Stats/s: {}\n", st.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut st = state();

    let mut objects = vec![];
    let camera = translate(4.0 * Y) * &rotate_x(Rad(0.5));
    for j in -4..=4 {
        for i in -4..=4 {
            objects.push(Obj {
                tf: translate(dir(4. * i as f32, 0., 4. * j as f32)),
                geom: UnitSphere(9, 9).build(),
            });
        }
    }
    let scene = Scene { objects, camera };

    let mut buf = [BLACK; W * W];
    c.bench_function("scene", |b| {
        b.iter(|| scene.render(
            &mut st,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<(f32, Color)>| {
                    let (x, y) = frag.coord;
                    buf[W * y + x] = frag.varying.1
                },
            },
        ));
    });
    check_hash(&buf, 17516720479059830591);
    save_screenshot("target/scene.ppm", &mut buf);
    eprintln!("Stats/s: {}\n", st.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut st = state();
    st.modelview = scale(2.0) * &translate(6.0 * Z);

    let mesh = UnitCube.build();
    let mesh = Mesh::<(Color, )> {
        verts: mesh.verts.iter()
            .zip((0..3).cycle())
            .map(|(v, ci)| v.attr(ci))
            .collect(),
        vertex_attrs:  vec![RED, GREEN, BLUE],

        faces: mesh.faces,
        face_attrs: mesh.face_attrs,
        vertex_coords: mesh.vertex_coords,
        bbox: mesh.bbox,
    };

    let mut buf = [BLACK; W * W];
    c.bench_function("gouraud", |b| {
        b.iter(|| mesh.render(
            &mut st,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |frag: Fragment<(Color, )>| Some(frag.varying.0),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<(f32, Color)>| {
                    let (x, y) = frag.coord;
                    buf[W * y + x] = frag.varying.1;
                },
            },
        ));
    });
    check_hash(&buf, 3313217684519750548);
    save_screenshot("target/gouraud.ppm", &mut buf);
    eprintln!("Stats/frame: {}\n", st.stats.avg_per_frame());
}

fn texture_fillrate(c: &mut Criterion) {
    let mut st = state();

    st.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = UnitCube.with_texcoords();

    //let tex = Texture::from(Buffer::from_vec(2, vec![RED, BLUE, BLUE, GREEN]));
    let tex = Texture::from(load_pnm("../examples/sdl/crate.ppm").unwrap());

    let mut buf = [BLACK; W * W];
    c.bench_function("texture", |b| {
        b.iter(|| {
            let clock = Instant::now();
            mesh.render(
                &mut st,
                &mut ShaderImpl {
                    vs: |v| v,
                    fs: |f: Fragment<(TexCoord, )>| {
                        SamplerOnce.sample(&tex, f.varying.0)
                    }
                },
                &mut Raster {
                    test: |_| true,
                    output: |frag: Fragment<(f32, Color)>| {
                        let (x, y) = frag.coord;
                        buf[W * y + x] = frag.varying.1;
                    },
                },
            );
            st.stats.time_used += clock.elapsed();
            st.stats.frames += 1;
        });
    });
    check_hash(&buf, 12904971692947500411);
    save_screenshot("target/texture.ppm", &mut buf);
    eprintln!("Stats:     {}", st.stats);
    eprintln!("Stats/sec: {}\n", st.stats.avg_per_sec());
}

criterion_group!(benches,
    torus,
    scene,
    gouraud_fillrate,
    texture_fillrate
);

criterion_main!(benches);
