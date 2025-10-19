use core::ops::ControlFlow::*;

use wasm_bindgen::prelude::*;

use re::prelude::*;

use re::render::{ModelToView, render, shader::Shader};
use re::util::Dims;

use re_front::{dims::SVGA_800_600, wasm::Window};

// Entry point from JS
#[wasm_bindgen(start)]
pub fn start() {
    const DIMS: Dims = SVGA_800_600;

    console_error_panic_hook::set_once();

    let mut win = Window::new(DIMS).expect("could not create window");
    win.ctx.color_clear = Some(rgba(0, 0, 0, 0x80));

    let vs = [
        vertex(pt3(-2.0, 1.0, 0.0), rgba(1.0, 0.2, 0.1, 0.9)),
        vertex(pt3(2.0, 2.0, 0.0), rgba(0.2, 0.9, 0.1, 0.8)),
        vertex(pt3(0.0, -2.0, 0.0), rgba(0.3, 0.4, 1.0, 1.0)),
    ];

    let proj = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let vp = viewport(pt2(8, 8)..pt2(DIMS.0 - 8, DIMS.1 - 8));

    win.run(move |frame| {
        let t = frame.t.as_secs_f32();

        let mv = rotate_z(rads(t))
            .then(&translate((3.0 + 2.0 * t.sin()) * Vec3::Z))
            .to::<ModelToView>();
        let mvp = mv.then(&proj);

        let sh = Shader::new(
            |v: Vertex3<Color4f>, _| vertex(mvp.apply(&v.pos), v.attrib),
            |f: Frag<Color4f>| f.var.to_color4(),
        );

        render(
            [tri(0, 1, 2)], //
            vs,
            &sh,
            (),
            vp,
            &mut frame.buf,
            frame.ctx,
        );
        Continue(())
    });
}
