//! Render impls for primitives and related items.

use crate::geom::{Edge, Tri, Vertex, Winding};
use crate::math::{Apply, Mat4x4, Vary, vary::ZDiv, vec3};

use super::{
    NdcToScreen, Render,
    clip::ClipVert,
    raster::{Scanline, ScreenPt, line, tri_fill},
};

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

    fn is_backface(tri: &Self::Screen) -> bool {
        tri.winding() == Winding::Cw
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

impl<V: Vary> Render<V> for Edge<usize> {
    type Clip = Edge<ClipVert<V>>;

    type Clips = [Self::Clip];

    type Screen = Edge<Vertex<ScreenPt, V>>;

    fn inline(Edge(i, j): Edge<usize>, vs: &[ClipVert<V>]) -> Self::Clip {
        Edge(vs[i].clone(), vs[j].clone())
    }

    fn to_screen(e: Self::Clip, tf: &Mat4x4<NdcToScreen>) -> Self::Screen {
        let [a, b] = to_screen([e.0, e.1], tf);
        Edge(a, b)
    }

    fn rasterize<F: FnMut(Scanline<V>)>(e: Self::Screen, scanline_fn: F) {
        line([e.0, e.1], scanline_fn);
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
