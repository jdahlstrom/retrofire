#![allow(unused)]

use retrofire_core::{
    math::color::Rgb,
    prelude::*,
    render::{target::Colorbuf, tex::SamplerClamp},
    util::{self, pixfmt::Xrgb8888, pnm::parse_pnm},
};

const VERTS: [Vertex3<TexCoord>; 4] = [
    vertex(pt3(-1.0, -1.0, 0.0), uv(0.0, 0.0)),
    vertex(pt3(-1.0, 1.0, 0.0), uv(0.0, 1.0)),
    vertex(pt3(1.0, -1.0, 0.0), uv(1.0, 0.0)),
    vertex(pt3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
];
const FACES: [Tri<usize>; 2] = [Tri([0, 1, 2]), Tri([3, 2, 1])];

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

    let mut color_buf = Colorbuf {
        buf: Buf2::new((w, h)),
        fmt: Xrgb8888,
    };
    let ctx = Context::default();

    render(FACES, VERTS, &shader, &mvp, viewport, &mut color_buf, &ctx);

    let buf = color_buf.buf;
    assert_eq!(buf[0][0], 0x00_00_00_FF);
    assert_eq!(buf[255][0], 0x00_7F_00_00);
    assert_eq!(buf[0][255], 0x00_7F_00_00);

    let buf = Buf2::new_from(
        (w, h),
        buf.data().iter().map(|u| {
            // Xrgb8888: 0x00_RR_GG_BB
            let [_, rgb @ ..] = u.to_be_bytes();
            Color3::<Rgb>::from(rgb)
        }),
    );

    let comp = *include_bytes!("textured_quad.ppm");
    let comp = parse_pnm(comp).expect("should be a valid ppm");

    assert_eq!(buf, comp);

    #[cfg(feature = "std")]
    {
        use util::pnm::*;
        // Uncomment to save generated image to compare visually
        //save_ppm("tests/textured_quad_actual.ppm", &buf);
        // Uncomment to (re)generate the comparison image
        //save_ppm("tests/textured_quad.ppm", &buf).expect("should save image");
    }
}
