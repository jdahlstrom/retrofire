use geom::mesh::Vertex;
use math::Linear;
use util::buf::Buffer;
use util::color::Color;
use util::tex::TexCoord;

use crate::raster::Span;
use crate::vary::Varying;

pub fn tri_fill<V, F>(mut verts: [Vertex<V>; 3], ref mut span_fn: F)
where
    V: Linear<f32> + Copy,
    F: FnMut(Span<V>),
{
    super::ysort(&mut verts);

    let [a, b, c] = verts;
    let y = a.coord.y.round() as usize;

    let (ay, av) = (a.coord.y, (a.coord.x, a.attr));
    let (by, bv) = (b.coord.y, (b.coord.x, b.attr));
    let (cy, cv) = (c.coord.y, (c.coord.x, c.attr));

    let ab = &mut Varying::between(av, bv, by - ay);
    let ac = &mut Varying::between(av, cv, cy - ay);
    let bc = &mut Varying::between(bv, cv, cy - by);

    if ab.step.0 < ac.step.0 {
        let y = half_tri(y, by.round() as usize, ab, ac, span_fn);
        half_tri(y, cy.round() as usize, bc, ac, span_fn);
    } else {
        let y = half_tri(y, by.round() as usize, ac, ab, span_fn);
        half_tri(y, cy.round() as usize, ac, bc, span_fn);
    }
}

#[inline]
fn half_tri<V, F>(
    y: usize,
    y_end: usize,
    left: &mut Varying<(f32, V)>,
    right: &mut Varying<(f32, V)>,
    span_fn: &mut F,
) -> usize
where
    V: Linear<f32> + Copy,
    F: FnMut(Span<V>),
{
    for y in y..y_end {
        let (xl, vl) = left.next().unwrap();
        let (xr, vr) = right.next().unwrap();
        span_fn(Span {
            y,
            // TODO use integer math
            xs: (xl.round() as usize, xr.round() as usize),
            vs: (vl, vr)
        });
    }
    y_end
}


const CHUNK_SIZE: usize = 32;
const INV_CHUNK_SIZE: f32 = 1.0 / CHUNK_SIZE as f32;

fn span_tex_pc(
    &((x_left, z_left), uv_left): &((f32, f32), TexCoord),
    &((x_right, z_right), uv_right): &((f32, f32), TexCoord),
    tex: &Buffer<Color>,
    xline: &mut [u8],
    zb: &mut [f32],
) {
    assert_eq!(xline.len(), 4 * zb.len());

    let mut uv = uv_left;
    let mut uv_pc = uv.w_div();
    let uv_step = uv_right
        .sub(uv_left)
        .mul(CHUNK_SIZE as f32 / (x_right - x_left));

    let mut z = z_left;
    let z_step = (z_right - z_left) / (x_right - x_left);
    let mut zi = 0;

    for chunk in xline.chunks_mut(4 * CHUNK_SIZE) {
        uv = uv.add(uv_step);
        let uv_pc1 = util::tex::uv(uv.u / uv.w, uv.v / uv.w);
        let uv_pc_step = uv_pc1.sub(uv_pc).mul(INV_CHUNK_SIZE);

        for pix in chunk.chunks_exact_mut(4) {
            let z_curr = unsafe { zb.get_unchecked_mut(zi) };
            if z >= *z_curr {
                continue;
            }

            let tex_i = tex.width() * (uv_pc.v as isize as usize)
                + (uv_pc.u as isize as usize);
            let [_, r, g, b] =
                unsafe { tex.data().get_unchecked(tex_i) }.to_argb();

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
