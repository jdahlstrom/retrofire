#![allow(unused)]

use retrofire_core::{prelude::*, render::tex::SamplerClamp, util};

const VERTS: [Vertex3<TexCoord>; 4] = [
    vertex(pt3(-1.0, -1.0, 0.0), uv(0.0, 0.0)),
    vertex(pt3(-1.0, 1.0, 0.0), uv(0.0, 1.0)),
    vertex(pt3(1.0, -1.0, 0.0), uv(1.0, 0.0)),
    vertex(pt3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
];
const FACES: [Tri<usize>; 2] = [Tri([0, 1, 2]), Tri([3, 2, 1])];

#[cfg(feature = "std")]
#[test]
fn textured_quad() {
    let checker = Texture::from(Buf2::new_with((8, 8), |x, y| {
        let xor = (x ^ y) & 1;
        // Blue if x == y, dark red otherwise.
        rgba(0x7F * xor as u8, 0, 0xFF * (1 - xor) as u8, 0)
    }));

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<_>| SamplerClamp.sample(&checker, frag.var),
    );

    let (w, h) = (256, 256);
    let project = perspective(1.0, 1.0, 0.1..1000.0);
    let viewport = viewport(pt2(0, 0)..pt2(w, h));
    let mvp = translate3(0.0, 0.0, 1.0).to().then(&project);

    let mut buf = Buf2::new((w, h));
    let ctx = Context::default();

    render(FACES, VERTS, &shader, &mvp, viewport, &mut buf, &ctx);

    assert_eq!(buf[0][0], 0x00_00_00_FF);
    assert_eq!(buf[255][0], 0x00_7F_00_00);
    assert_eq!(buf[0][255], 0x00_7F_00_00);

    let buf = Buf2::new_from(
        (w, h),
        buf.data().iter().map(|u| {
            // Current pixel format is hardcoded to ARGB
            let [_, rgb @ ..] = u.to_be_bytes();
            Color3::from(rgb)
        }),
    );

    use util::pnm::*;

    let comp = load_pnm("tests/textured_quad.ppm") //
        .expect("should load comparison image");

    assert_eq!(buf, comp);

    // Uncomment to (re)generate the comparison image
    //save_ppm("tests/triangle.ppm", &buf).expect("should save image");
}
