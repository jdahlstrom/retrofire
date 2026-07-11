use retrofire_core::prelude::*;
use retrofire_core::render::{Model, TriFan, render, shader};
use retrofire_core::util::dims;

fn main() {
    let tri = TriFan(
        [
            vertex(pt3(1.0, 1.0, 0.0), rgb(1.0, 0.2, 0.0)),
            vertex(pt3(-1.0, 1.0, 0.0), rgb(0.0, 0.8, 0.2)),
            vertex(pt3(0.0, -1.0, 0.0), rgb(0.4, 0.4, 1.0)),
            vertex(pt3(2.0, -0.8, 0.0), rgb(0.9, 0.4, 1.0)),
            vertex(pt3(1.5, 1.0, 0.0), rgb(0.9, 0.4, 0.0)),
        ]
        .to_vec(),
    );

    #[cfg(feature = "fp")]
    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            // Transform vertex position from model to projection space
            // Interpolate vertex colors in linear color space
            vertex(mvp.apply(&v.pos), v.attrib.to_linear())
        },
        |frag: Frag<Color3f<_>>, _| frag.var.to_srgb().to_color4(),
    );
    #[cfg(not(feature = "fp"))]
    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            // Transform vertex position from model to projection space
            // Interpolate vertex colors in normal sRGB color space
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>, _| frag.var.to_color4(),
    );

    let dims = dims::VGA_640_480;
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, dims.aspect(), 0.1..1000.0);
    let viewport = viewport(dims.into());

    let mut framebuf = Buf2::<Color4>::new(dims);

    render(
        tri,
        &shader,
        &modelview.then(&project),
        viewport,
        &mut framebuf,
        &Context::default(),
    );
    #[cfg(feature = "std")]
    {
        use retrofire_core::util::pnm;
        pnm::save_ppm("triangle.ppm", &framebuf).unwrap();
    }

    let center_pixel = framebuf[[dims.0 / 2, dims.1 / 2]];
    if cfg!(feature = "fp") {
        assert_eq!(center_pixel, rgba(152, 130, 187, 255));
    } else {
        assert_eq!(center_pixel, rgba(115, 114, 140, 255));
    }
}
