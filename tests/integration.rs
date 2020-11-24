use std::f32::INFINITY;

use geom::mesh::*;
use geom::solids::unit_cube;
use math::transform::*;
use math::vec::*;
use render::*;
use std::fmt::Debug;

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

fn render<VA: VertexAttr, FA: Copy + Debug>(mesh: Mesh<VA, FA>) -> String {

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

#[test]
fn renderer_works_with_unit_attrs() {
    let actual = render(unit_cube());

    assert_eq!(EXPECTED_CUBE, &actual);
}

#[test]
fn renderer_works_with_vector_attrs() {
    let actual = render(unit_cube().gen_normals());

    assert_eq!(EXPECTED_CUBE, &actual);
}