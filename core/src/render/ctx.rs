use core::cell::RefCell;
use core::cmp::Ordering;

use crate::math::color::{rgba, Color4};

use super::stats::Stats;

/// Rendering context.
#[derive(Clone, Debug)]
pub struct Context {
    /// The color with which to fill the color buffer to clear it, if any.
    /// If rendered geometry always fills the entire frame, `color_clear`
    /// can be set to `None`.
    pub color_clear: Option<Color4>,

    /// The value with which to fill the depth buffer to clear it, if any.
    pub depth_clear: Option<f32>,

    /// Whether to cull (discard) faces pointing either away from or towards
    /// the camera. If all geometry drawn is "solid" meshes without holes,
    /// backfaces can usually be culled because they are always occluded by
    /// frontfaces and drawing them would be useless.
    pub face_cull: Option<FaceCull>,

    /// Whether to sort visible faces by their depth. This is important when
    /// rendering overlapping transparent faces, which *have* to be drawn
    /// back-to-front to get correct results. On the other hand, rendering
    /// nontransparent geometry in front-to-back order can improve performance
    /// by reducing overdraw.
    pub depth_sort: Option<DepthSort>,

    /// Whether to do depth testing and which predicate to use. If set to
    /// `Some(Ordering::Less)`, a fragment passes the depth test *iff*
    /// `new_z < old_z` (the default). If set to `None`, depth test is not
    /// performed. This setting has no effect if the render target does not
    /// support z-buffering.
    pub depth_test: Option<Ordering>,

    /// Whether to write color values. If `false`, other fragment processing
    /// is done but there is no color output. This setting has no effect if
    /// the render target does not support color writes.
    pub color_write: bool,

    /// Whether to write depth values. If `false`, other fragment processing
    /// is done but there is no depth output. This setting has no effect if
    /// the render target does not support depth writes.
    pub depth_write: bool,

    /// Collecting rendering statistics.
    pub stats: RefCell<Stats>,
}

/// Whether to sort faces front to back or back to front.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DepthSort {
    FrontToBack,
    BackToFront,
}

/// Whether to cull frontfaces or backfaces.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FaceCull {
    Front,
    Back,
}

impl Context {
    /// Compares the reciprocal depth value `new` to `curr` and returns
    /// whether `new` passes the depth test specified by `self.depth_test`.
    /// If `self.depth_test` is `None`, always returns `true`.
    #[inline]
    pub fn depth_test(&self, new: f32, curr: f32) -> bool {
        // Reverse comparison because we're comparing reciprocals
        self.depth_test
            .map_or(true, |ord| curr.partial_cmp(&new) == Some(ord))
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            color_clear: Some(rgba(0, 0, 0, 0xFF)),
            depth_clear: Some(f32::INFINITY),
            face_cull: Some(FaceCull::Back),
            depth_sort: None,
            color_write: true,
            depth_test: Some(Ordering::Less),
            depth_write: true,
            stats: Default::default(),
        }
    }
}
