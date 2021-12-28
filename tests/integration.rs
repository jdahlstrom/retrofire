use std::convert::identity;
use std::fmt::Debug;

use geom::mesh::*;
use geom::mesh2;
use geom::mesh2::Soa;
use geom::solids::{unit_sphere, UnitCube};
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
     ..#######.\
     ..#######.\
     ..#######.\
     ..#######.\
     ..#######.\
     ..#######.\
     ..........\
     ..........";

fn render<VA, FA>(mesh: mesh2::Mesh<VA, FA>) -> String
where
    VA: Soa + Linear<f32> + Copy + Debug,
    FA: Copy + Debug
{
    let mut rdr = Renderer::new();
    rdr.modelview = translate(4.0 * Z);
    rdr.projection = perspective(1.0, 10.0, 1., Rad(1.0));
    rdr.viewport = viewport(0.0, 0.0, 10.0, 10.0);

    let mesh = mesh.validate().expect("Invalid mesh!");

    let mut buf = ['.'; 100];

    mesh.render(
        &mut rdr,
        &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(BLACK),
        },
        &mut Raster {
            test: |_| true,
            output: &mut |frag: Fragment<_>| buf[10 * frag.coord.1 + frag.coord.0] = '#'
        }
    );
    eprintln!("Stats: {}", rdr.stats);
    buf.iter().collect()
}

fn render_scene<VA, FA>(scene: Scene<Mesh<VA, FA>>) -> Stats
where VA: Copy + Linear<f32>, FA: Copy
{
    const W: usize = 50;
    const H: usize = 20;
    let mut rdr = Renderer::new();
    rdr.projection = perspective(1.0, 100.0, 1.0, Deg(90.0));
    rdr.viewport = viewport(0.0, 0.0, W as f32, H as f32);

    let stats = rdr.render_scene(
        &scene,
        &mut ShaderImpl {
            vs: identity,
            fs: |_| Some(BLACK),
        },
        &mut Raster {
            test: |_| true,
            output: |_| {}
        }
    );
    eprintln!("Stats: {}", rdr.stats);
    stats
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
    let sph = unit_sphere(9, 9).build().validate().unwrap();

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
    assert_eq!(1248, stats.pixels);
}
