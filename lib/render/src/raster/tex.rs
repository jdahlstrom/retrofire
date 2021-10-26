use std::ops::DerefMut;

use geom::mesh::Vertex;
use math::Linear;
use util::Buffer;
use util::color::Color;

use crate::Framebuf;
use crate::tex::TexCoord;
use crate::vary::Varying;

const CHUNK_SIZE: usize = 32;
const INV_CHUNK_SIZE: f32 = 1.0 / CHUNK_SIZE as f32;

#[inline]
fn half_tri(
    y: usize,
    y_end: usize,
    left: &mut Varying<((f32, f32), TexCoord)>,
    right: &mut Varying<((f32, f32), TexCoord)>,
    tex: &Buffer<Color>,
    fb: &mut Framebuf<impl DerefMut<Target=[u8]>>,
) -> usize {

    let tw = tex.width() as f32;
    let th = tex.height() as f32;

    let width = fb.color.width();
    let cb = fb.color.data_mut();
    let zb = fb.depth.data_mut();

    for y in (width * y..width * y_end).step_by(width) {

        let mut l = left.next().unwrap();
        let mut r = right.next().unwrap();

        l.1.u *= tw;
        l.1.v *= th;
        r.1.u *= tw;
        r.1.v *= th;

        let left = y + l.0.0 as usize;
        let right = y + r.0.0 as usize;
        let right = left.max(right);
        let xline = &mut cb[4 * left..4 * right];
        let zline = &mut zb[left..right];

        scanline(&l, &r, tex, xline, zline);
    }
    y_end
}

fn scanline(
    &((x_left, z_left), uv_left): &((f32, f32), TexCoord),
    &((x_right, z_right), uv_right): &((f32, f32), TexCoord),
    tex: &Buffer<Color>,
    xline: &mut [u8],
    zline: &mut [f32],
) {
    assert_eq!(xline.len(), 4 * zline.len());

    let mut uv = uv_left;
    let mut uv_pc = uv.w_div();
    let uv_step = uv_right.sub(uv_left)
        .mul(CHUNK_SIZE as f32 / (x_right - x_left));

    let mut z = z_left;
    let z_step = (z_right - z_left) / (x_right - x_left);
    let mut zi = 0;

    for chunk in xline.chunks_mut(4 * CHUNK_SIZE) {
        uv = uv.add(uv_step);
        let uv_pc1 = uv.w_div();
        let uv_pc_step = uv_pc1.sub(uv_pc).mul(INV_CHUNK_SIZE);

        for pix in chunk.chunks_exact_mut(4) {

            let z_curr = unsafe { zline.get_unchecked_mut(zi) };
            if z >= *z_curr {
                continue;
            }

            let tex_i = tex.width()
                * (uv_pc.v as isize as usize & 0xFF)
                + (uv_pc.u as isize as usize & 0xFF);
            let [_, r, g, b] = unsafe { tex.data().get_unchecked(tex_i) }.to_argb();

            pix[0] = b;
            pix[1] = g;
            pix[2] = r;
            *z_curr = z;

            zi += 1;
            z += z_step;
            uv_pc = uv_pc.add(uv_pc_step);
        }
    }
}

pub fn tex_fill(
    mut verts: [Vertex<TexCoord>; 3],
    tex: &Buffer<Color>,
    fb: &mut Framebuf<impl DerefMut<Target=[u8]>>
) {
    super::ysort(&mut verts);

    let [a, b, c] = verts;
    let y = a.coord.y.round() as usize;

    let (ay, av) = (a.coord.y, ((a.coord.x, a.coord.z), a.attr));
    let (by, bv) = (b.coord.y, ((b.coord.x, b.coord.z), b.attr));
    let (cy, cv) = (c.coord.y, ((c.coord.x, c.coord.z), c.attr));

    let ab = &mut Varying::between(av, bv, by - ay);
    let ac = &mut Varying::between(av, cv, cy - ay);
    let bc = &mut Varying::between(bv, cv, cy - by);

    if ab.step.0 < ac.step.0 {
        let y = half_tri(y, by.round() as usize, ab, ac, tex, fb);
        half_tri(y, cy.round() as usize, bc, ac, tex, fb);
    } else {
        let y = half_tri(y, by.round() as usize, ac, ab, tex, fb);
        half_tri(y, cy.round() as usize, ac, bc, tex, fb);
    }
}
