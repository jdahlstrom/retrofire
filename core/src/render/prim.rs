//! Render impls for primitives and related items.

use crate::geom::{Tri, Vertex};
use crate::math::{Mat4x4, Vary, vary::ZDiv, vec3};

use super::clip::ClipVert;
use super::raster::{Scanline, ScreenPt, line, tri_fill};
use super::{NdcToScreen, Render};

impl<V: Vary> Render<V> for Tri<usize> {
    type Clip = Tri<ClipVert<V>>;
    type Clips = [Tri<ClipVert<V>>];
    type Screen = Tri<Vertex<ScreenPt, V>>;

    fn inline(ixd: Tri<usize>, vs: &[ClipVert<V>]) -> Tri<ClipVert<V>> {
        Tri(ixd.0.map(|i| vs[i].clone()))
    }

    fn depth(Tri([a, b, c]): &Self::Clip) -> f32 {
        (a.pos.z() + b.pos.z() + c.pos.z()) / 3.0
    }

    fn is_backface(Tri(vs): &Self::Screen) -> bool {
        let v = vs[1].pos - vs[0].pos;
        let u = vs[2].pos - vs[0].pos;
        v[0] * u[1] - v[1] * u[0] > 0.0
    }

    fn to_screen(
        clip: Tri<ClipVert<V>>,
        tf: &Mat4x4<NdcToScreen>,
    ) -> Self::Screen {
        Tri(to_screen(clip.0, tf))
    }

    fn rasterize<F: FnMut(Scanline<V>)>(scr: Self::Screen, scanline_fn: F) {
        tri_fill(scr.0, scanline_fn);
    }
}

impl<V: Vary> Render<V> for [usize; 2] {
    type Clip = [ClipVert<V>; 2];

    type Clips = [Self::Clip];

    type Screen = [Vertex<ScreenPt, V>; 2];

    fn inline([i, j]: [usize; 2], vs: &[ClipVert<V>]) -> Self::Clip {
        [vs[i].clone(), vs[j].clone()]
    }

    fn to_screen(clip: Self::Clip, tf: &Mat4x4<NdcToScreen>) -> Self::Screen {
        to_screen(clip, tf)
    }

    fn rasterize<F: FnMut(Scanline<V>)>(scr: Self::Screen, scanline_fn: F) {
        line(scr, scanline_fn);
    }
}

pub fn to_screen<V: ZDiv, const N: usize>(
    vs: [ClipVert<V>; N],
    tf: &Mat4x4<NdcToScreen>,
) -> [Vertex<ScreenPt, V>; N] {
    vs.map(|v| {
        let [x, y, _, w] = v.pos.0;
        // Perspective division (projection to the real plane)
        //
        // We use the screen-space z coordinate to store the reciprocal
        // of the original view-space depth. The interpolated reciprocal
        // is used in fragment processing for depth testing (larger values
        // are closer) and for perspective correction of the varyings.
        // TODO z_div could be space-aware
        let pos = vec3(x, y, 1.0).z_div(w);
        Vertex {
            // Viewport transform
            pos: tf.apply(&pos).to_pt(),
            // Perspective correction
            attrib: v.attrib.z_div(w),
        }
    })
}
