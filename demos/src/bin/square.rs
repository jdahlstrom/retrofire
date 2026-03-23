#![feature(portable_simd)]

use core::ops::ControlFlow::*;
use core::simd::f32x2;

use re::prelude::*;

use re::core::math::vary::ZDiv;
use re::core::render::{
    render, shader,
    tex::{SamplerClamp, Tex, uv as uv_},
};

use re::front::minifb::Window;

#[derive(Default, Copy, Clone, Debug)]
struct Simd2(f32x2);

type TexCoord = Simd2; //Vector<Simd2, Tex>;

impl Affine for TexCoord {
    type Space = Tex;
    type Diff = Self;
    const DIM: usize = 2;

    fn add(&self, diff: &Self::Diff) -> Self {
        Self(self.0 + diff.0)
    }

    fn sub(&self, other: &Self) -> Self::Diff {
        Self(self.0 - other.0)
    }
}
impl Linear for TexCoord {
    type Scalar = f32;

    fn zero() -> Self {
        Self(f32x2::default())
    }

    fn mul(&self, s: Self::Scalar) -> Self {
        Self(self.0 * f32x2::splat(s))
    }
}
impl ZDiv for TexCoord {}

fn uv(x: f32, y: f32) -> TexCoord {
    Simd2(f32x2::from_array([x, y]))
}

fn main() {
    // Vertices of a square
    let verts: [Vertex3<TexCoord>; 4] = [
        vertex(pt3(-1.0, -1.0, 0.0), uv(0.0, 0.0)),
        vertex(pt3(-1.0, 1.0, 0.0), uv(0.0, 1.0)),
        vertex(pt3(1.0, -1.0, 0.0), uv(1.0, 0.0)),
        vertex(pt3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
    ];

    let mut win = Window::builder()
        .title("retrofire//square")
        .build()
        .expect("should create window");

    // Disable backface culling and depth buffering
    win.ctx = Context {
        face_cull: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,
        ..win.ctx
    };

    // Texture with a check pattern
    let checker = Texture::from(Buf2::new_with((8, 8), |x, y| {
        let xor = (x ^ y) & 1;
        rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)
    }));

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &ProjMat3<_>| vertex(mvp.apply(&v.pos), v.attrib),
        |frag: Frag<TexCoord>| {
            //gray(255).to_rgba();
            let &[x, y] = frag.var.0.as_array();
            SamplerClamp.sample(&checker, uv_(x, y))
        },
    );

    let (w, h) = win.dims;
    let projection = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(10, 10)..pt2(w - 10, h - 10));

    win.run(|frame| {
        let time = frame.t.as_secs_f32();

        let model_view_project = rotate_y(rads(time))
            .then(&translate((0.0, 0.0, 3.0 + time.sin())))
            .to()
            .then(&projection);

        render(
            [tri(0, 1, 2), tri(3, 2, 1)],
            verts,
            &shader,
            &model_view_project,
            viewport,
            &mut frame.buf,
            frame.ctx,
        );

        Continue(())
    });
}
