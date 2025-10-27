use retrofire_core::geom::tri;
use retrofire_core::{prelude::*, util::*};

fn main() {
    let verts = [
        vertex(pt3(-1.0, 1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, 1.0, 0.0), rgb(0.0, 0.8, 0.0)),
        vertex(pt3(0.0, -1.0, 0.0), rgb(0.4, 0.4, 1.0)),
    ];

    #[cfg(feature = "fp")]
    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &Mat4x4<ModelToProj>| {
            // Transform vertex position from model to projection space
            // Interpolate vertex colors in linear color space
            vertex(mvp.apply(&v.pos), v.attrib.to_linear())
        },
        |frag: Frag<Color3f<_>>| frag.var.to_srgb().to_color4(),
    );
    #[cfg(not(feature = "fp"))]
    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &Mat4x4<ModelToProj>| {
            // Transform vertex position from model to projection space
            // Interpolate vertex colors in normal sRGB color space
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>| frag.var.to_color4(),
    );

    let dims @ (w, h) = (640, 480);
    let modelview = translate3(0.0, 0.0, 2.0).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, 0)..pt2(w, h));

    let mut framebuf = Buf2::<Color4>::new(dims);

    render(
        [tri(0, 1, 2)],
        verts,
        &shader,
        &modelview.then(&project),
        viewport,
        &mut framebuf,
        &Context::default(),
    );

    let center_pixel = framebuf[[w / 2, h / 2]];

    if cfg!(feature = "fp") {
        assert_eq!(center_pixel, rgba(150, 128, 185, 255));
    } else {
        assert_eq!(center_pixel, rgba(114, 102, 127, 255));
    }
    #[cfg(feature = "std")]
    {
        pnm::save_ppm("triangle.ppm", framebuf).unwrap();
    }
}
