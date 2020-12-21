use std::f32::consts::PI;
use std::fmt::Debug;

use geom::mesh::*;
use geom::solids::{unit_cube, unit_sphere};
use math::Linear;
use math::transform::*;
use render::*;
use render::color::BLACK;

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

fn render<VA, FA>(mesh: Mesh<VA, FA>) -> String
where VA: Copy + Linear<f32>, FA: Copy + Debug
{
    let mut rdr = Renderer::new();
    rdr.set_transform(translate(0.0, 0.0, 4.0));
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, 10.0, 10.0));

    let mesh = mesh.validate().expect("Invalid mesh!");

    let mut buf = ['.'; 100];

    let stats = rdr.render(
        &mesh,
        &|_, _| BLACK,
        &mut |x, y, _| buf[10 * y + x] = '#'
    );
    eprintln!("Stats: {}", stats);
    buf.iter().collect()
}

fn render_scene<VA, FA>(scene: Scene<VA, FA>) -> Stats
where VA: Copy + Linear<f32>, FA: Copy
{
    const W: usize = 50;
    const H: usize = 20;
    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1.0, 100.0, 1.0, 0.5 * PI));
    rdr.set_viewport(viewport(0.0, 0.0, W as f32, H as f32));

    let stats = rdr.render_scene(
        &scene,
        &|_, _| BLACK,
        &mut |_, _, _| ()
    );
    eprintln!("Stats: {}", stats);
    stats
}

#[test]
fn render_cube_with_unit_attrs() {
    let actual = render(unit_cube());

    assert_eq!(EXPECTED_CUBE, &actual);
}

#[test]
fn render_cube_with_vector_attrs() {
    let actual = render(unit_cube().gen_normals());

    assert_eq!(EXPECTED_CUBE, &actual);
}

#[test]
fn render_sphere_field() {
    let mut objects = vec![];
    let camera = translate(0., 4., 0.) * &rotate_x(0.5);
    for j in -10..=10 {
        for i in -10..=10 {
            objects.push(Obj {
                tf: translate(4. * i as f32, 0., 4. * j as f32),
                mesh: unit_sphere(9, 9)
            });
        }
    }

    let stats = render_scene(Scene { objects, camera });

    assert_eq!(126 * 21 * 21, stats.faces_in);
    assert_eq!(6550, stats.faces_out);
    assert_eq!(1216, stats.pixels);
}
