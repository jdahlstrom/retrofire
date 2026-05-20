//! Render impls for primitives and related items.

use alloc::vec::Vec;

use crate::{
    geom::{Edge, Tri, Vertex, Winding},
    math::{Mat4, Vary, pt3, vary::ZDiv},
};

use super::{
    Clip, Ndc, Render, Screen,
    clip::{ClipPlane, ClipVert},
    raster::{Scanline, ScreenPt, line, tri_fill},
};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Point<Vtx>(pub Vtx);

impl<V: Clone> Clip for [Point<ClipVert<V>>] {
    type Item = Point<ClipVert<V>>;

    fn clip(&self, _: &[ClipPlane], out: &mut Vec<Self::Item>) {
        for pt in self {
            if pt.0.outcode == 0 {
                out.push(pt.clone());
            }
        }
    }
}

impl<V: Vary> Render<V> for Point<usize> {
    type Clip = Point<ClipVert<V>>;
    type Clips = [Point<ClipVert<V>>];
    type Screen = Point<Vertex<ScreenPt, V>>;

    fn inline(ixd: Self, vs: &[ClipVert<V>]) -> Self::Clip {
        Point(vs[ixd.0].clone())
    }

    fn to_screen(clip: Self::Clip, tf: &Mat4<Ndc, Screen>) -> Self::Screen {
        Point(to_screen([clip.0], tf)[0].clone())
    }

    fn rasterize<F: FnMut(Scanline<V>)>(pt: Self::Screen, mut scanline_fn: F) {
        let [x, y, _] = pt.0.pos.0;

        let vs = (pt.0.pos, pt.0.attrib.clone());
        let step = vs.dv_dt(&vs, 1.0);
        let vs = vs.clone().vary(step, None);

        let y = y as usize;

        for y in y - 4..=y + 4 {
            scanline_fn(Scanline {
                y,
                xs: x as usize - 4..(x as usize + 5),
                vs: vs.clone(),
            });
        }
    }
}

impl<V: Vary> Render<V> for Tri<usize> {
    type Clip = Tri<ClipVert<V>>;
    type Clips = [Tri<ClipVert<V>>];
    type Screen = Tri<Vertex<ScreenPt, V>>;

    #[inline]
    fn inline(tri: Self, vs: &[ClipVert<V>]) -> Tri<ClipVert<V>> {
        tri.map(|i| vs[i].clone())
    }

    #[inline]
    fn depth(Tri([a, b, c]): &Self::Clip) -> f32 {
        (a.pos.z() + b.pos.z() + c.pos.z()) / 3.0
    }

    #[inline]
    fn is_backface(tri: &Self::Screen) -> bool {
        tri.winding() == Winding::Cw
    }

    #[inline]
    fn to_screen(
        clip: Tri<ClipVert<V>>,
        tf: &Mat4<Ndc, Screen>,
    ) -> Self::Screen {
        Tri(to_screen(clip.0, tf))
    }

    #[inline]
    fn rasterize<F: FnMut(Scanline<V>)>(scr: Self::Screen, scanline_fn: F) {
        tri_fill(scr.0, scanline_fn);
    }
}

impl<V: Vary> Render<V> for Edge<usize> {
    type Clip = Edge<ClipVert<V>>;

    type Clips = [Self::Clip];

    type Screen = Edge<Vertex<ScreenPt, V>>;

    #[inline]
    fn inline(e: Edge<usize>, vs: &[ClipVert<V>]) -> Self::Clip {
        Edge(vs[e.0].clone(), vs[e.1].clone())
    }

    #[inline]
    fn to_screen(e: Self::Clip, tf: &Mat4<Ndc, Screen>) -> Self::Screen {
        to_screen([e.0, e.1], tf).into()
    }

    #[inline]
    fn rasterize<F: FnMut(Scanline<V>)>(e: Self::Screen, scanline_fn: F) {
        line([e.0, e.1], scanline_fn);
    }
}

#[inline]
pub fn to_screen<V: ZDiv, const N: usize>(
    vs: [ClipVert<V>; N],
    tf: &Mat4<Ndc, Screen>,
) -> [Vertex<ScreenPt, V>; N] {
    vs.map(|v| {
        let [x, y, _, w] = v.pos.0;
        // Perspective division (projection to the real plane)
        //
        // We use the screen-space z coordinate to store the reciprocal
        // of the original view-space depth. The interpolated reciprocal
        // is used in fragment processing for depth testing (larger values
        // are closer) and for perspective correction of the varyings.
        //
        // TODO Ad-hoc conversion from clip space vector to screen space point.
        //      This should be a typed conversion from projective to real space.
        //      Vec3 was used here, which only worked because apply used to use
        //      w=1 for vectors. Fixing it made viewport transform incorrect.
        //      The z-div concept and trait likely need clarification.
        let pos = pt3(x, y, 1.0).z_div(w);
        Vertex {
            // Viewport transform
            pos: tf.apply(&pos),
            // Perspective correction
            attrib: v.attrib.z_div(w),
        }
    })
}
