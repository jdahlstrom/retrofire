//! Render impls for primitives and related items.

use crate::geom::{Edge, Tri, Vertex, Winding};
use crate::math::{Mat4, Vary, pt3, vary::ZDiv};

use super::{
    Clip, Ndc, Screen,
    clip::ClipVert,
    raster::{Scanline, ScreenPt, line, tri_fill},
};

/// Renderable geometric primitive.
pub trait Primitive<V: Vary> {
    /// The type of this primitive in clip space
    type Clip: Clip;

    /// The type of this primitive in screen space.
    type Screen;

    /// Maps the indices of the argument to vertices.
    fn inline(ixd: Self, vs: &[ClipVert<V>]) -> Self::Clip;

    /// Returns the (average) depth of the argument.
    fn depth(_clip: &Self::Clip) -> f32 {
        f32::INFINITY
    }

    /// Returns whether the argument is facing away from the camera.
    fn is_backface(_: &Self::Screen) -> bool {
        false
    }

    /// Transforms the argument from NDC to screen space.
    fn to_screen(clip: Self::Clip, tf: &Mat4<Ndc, Screen>) -> Self::Screen;

    /// Rasterizes the argument by calling the function for each scanline.
    fn rasterize<F: FnMut(Scanline<V>)>(scr: Self::Screen, scanline_fn: F);
}

impl<V: Vary + 'static> Primitive<V> for Tri<usize> {
    type Clip = Tri<ClipVert<V>>;
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
        tri.winding_xy() == Winding::Cw
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

impl<V: Vary> Primitive<V> for Edge<usize> {
    type Clip = Edge<ClipVert<V>>;
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
