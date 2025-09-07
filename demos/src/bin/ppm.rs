use re::{prelude::*, util::pnm::save_ppm};

fn main() {
    let verts = [
        vertex(pt3(-1.0, -1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, -1.0, 0.0), rgb(0.0, 0.8, 0.0)),
        vertex(pt3(0.0, 1.0, 0.0), rgb(0.4, 0.4, 1.0)),
    ];

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );

    let (w, h) = (640, 480);
    let modelview = &translate3(0.0, 0.0, 2.0).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, 0)..pt2(w, h));

    let mut framebuf = Buf2::<Color4>::new((w, h));

    render(
        [Tri([0, 2, 1])],
        verts,
        &shader,
        &modelview.then(&project),
        viewport,
        &mut framebuf,
        &Context::default(),
    );

    save_ppm("triangle.ppm", framebuf).unwrap();
}
