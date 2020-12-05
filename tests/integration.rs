use std::f32::INFINITY;
use std::fmt::Debug;

use geom::mesh::*;
use geom::solids::{unit_cube, unit_sphere};
use math::transform::*;
use math::vec::*;
use render::*;
use std::f32::consts::PI;

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
where VA: VertexAttr, FA: Copy + Debug
{
    let mut rdr = Renderer::new();
    rdr.set_transform(translate(0.0, 0.0, 4.0));
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, 10.0, 10.0));
    rdr.set_z_buffer(vec![INFINITY; 100], 10);

    let mesh = mesh.validate().expect("Invalid mesh!");

    let mut buf = ['.'; 100];

    let stats = rdr.render(
        mesh,
        &|_, _| ZERO,
        &mut |x, y, _| buf[10 * y + x] = '#'
    );
    eprintln!("Stats: {}", stats);
    buf.iter().collect()
}

fn render_scene<VA, FA>(scene: Scene<VA, FA>) -> Stats
where VA: VertexAttr, FA: Copy
{
    const W: usize = 50;
    const H: usize = 20;
    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1.0, 100.0, 1.0, 0.5 * PI));
    rdr.set_viewport(viewport(0.0, 0.0, W as f32, H as f32));
    rdr.set_z_buffer(vec![INFINITY; W * H], W);

    let stats = rdr.render_scene(
        scene,
        &|_, _| ZERO,
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

    assert_eq!(55566, stats.faces_in);
    assert_eq!(6550, stats.faces_out);
    assert_eq!(428, stats.pixels);
}