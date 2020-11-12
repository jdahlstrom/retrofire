use geom::mesh::*;
use geom::solids::unit_cube;
use math::transform::*;
use math::vec::*;
use render::*;
use render::raster::Fragment;

const EXPECTED_CUBE: &[u8; 100] =
    b"..........\
      ..........\
      ..#######.\
      ..#######.\
      ..#######.\
      ..#######.\
      ..#######.\
      ..#######.\
      ..........\
      ..........";

fn render<VA: VertexAttr, FA: Copy>(mesh: Mesh<VA, FA>) -> Vec<u8> {
    let mut rdr = Renderer::new();
    rdr.set_transform(translate(0.0, 0.0, 4.0));
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, 10.0, 10.0));

    let mut buf = b".".repeat(100);

    rdr.render(
        mesh,
        &|_: Fragment<(Vec4, VA)>, _: FA| {
            ZERO
        },
        &mut |x, y, _col| {
            buf[10 * y + x] = '#' as u8;
        }
    );
    buf
}

#[test]
fn renderer_works_with_unit_attrs() {
    let actual = render(unit_cube());

    assert_eq!(EXPECTED_CUBE, actual.as_slice());
}

#[test]
fn renderer_works_with_vector_attrs() {
    let actual = render(unit_cube().gen_normals());

    assert_eq!(EXPECTED_CUBE, actual.as_slice());
}