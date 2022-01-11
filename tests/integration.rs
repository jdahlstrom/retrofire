use std::convert::identity;
use std::fmt::Debug;

use geom::mesh::{Mesh, Soa};
use geom::solids::{UnitCube, UnitSphere};
use math::Angle::{Deg, Rad};
use math::Linear;
use math::transform::*;
use math::vec::{dir, Y, Z};
use render::*;
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use render::shade::ShaderImpl;
use util::color::BLACK;

static EXPECTED_CUBE: &str =
    "..........\
     ..........\
     ..######..\
     ..######..\
     ..######..\
     ..######..\
     ..######..\
     ..######..\
     ..........\
     ..........";

fn render<VA, FA>(mesh: Mesh<VA, FA>) -> String
where
    VA: Soa + Linear<f32> + Copy + Debug,
    FA: Copy + Debug
{
    let mut rdr = Renderer::new();
    rdr.modelview = translate(4.0 * Z);
    rdr.projection = perspective(1.0, 10.0, 1., Rad(1.0));
    rdr.viewport = viewport(0.0, 0.0, 10.0, 10.0);

    let mesh = mesh.validate().expect("Invalid mesh!");

    let mut buf = [['.'; 10]; 10];

    mesh.render(
        &mut rdr,
        &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(BLACK),
        },
        &mut Raster {
            test: |_| true,
            output: &mut |frag: Fragment<_>| {
                buf[frag.coord.1][frag.coord.0] = '#';
            }
        },
    );
    eprintln!("Stats: {}", rdr.stats);
    buf.into_iter().flatten().collect()
}

fn render_scene<VA, FA>(scene: Scene<Mesh<VA, FA>>) -> Stats
where
    VA: Copy + Debug + Linear<f32> + Soa,
    FA: Copy
{
    const W: usize = 50;
    const H: usize = 20;
    let mut buf = [b'.'; W * H];

    let mut rdr = Renderer::new();
    rdr.projection = perspective(1.0, 100.0, 1.0, Deg(90.0));
    rdr.viewport = viewport(0.0, 0.0, W as f32, H as f32);

    scene.render(
        &mut rdr,
        &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(BLACK),
        },
        &mut Raster {
            test: |_| true,
            output: |f: Fragment<_>| buf[W * f.coord.1 + f.coord.0] = b'#',
        }
    );
    for row in buf.chunks(W) {
        eprintln!("{}", std::str::from_utf8(&row).unwrap());
    }

    eprintln!("Stats: {}", rdr.stats);
    rdr.stats
}

#[test]
fn render_cube_with_unit_attrs() {
    let actual = render(UnitCube.build());

    assert_eq!(EXPECTED_CUBE, &actual);
}

#[test]
fn render_cube_with_vector_attrs() {
    let actual = render(UnitCube.build());

    assert_eq!(EXPECTED_CUBE, &actual);
}

#[test]
fn render_sphere_field() {
    let mut objects = vec![];
    let camera = translate(4.0 * Y) * &rotate_x(Rad(0.5));
    let sph = UnitSphere(9, 9).build().validate().unwrap();

    for j in -10..=10 {
        for i in -10..=10 {
            objects.push(Obj {
                tf: translate(dir(4. * i as f32, 0., 4. * j as f32)),
                geom: sph.clone()
            });
        }
    }

    let stats = render_scene(Scene { objects, camera });

    assert_eq!(126 * 21 * 21, stats.faces_in);
    assert_eq!(6626, stats.faces_out);
    assert_eq!(302, stats.pixels);
}
