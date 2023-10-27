//! Render targets.
//!
//! The typical render target is a framebuffer, comprising a color buffer,
//! depth buffer, and possible auxiliary buffers. Special render targets can
//! be used, for example, for visibility or occlusion computations.

use core::cmp::Ordering;

use crate::math::vary::Vary;
use crate::util::buf::AsMutSlice2;

use super::raster::{Frag, Scanline};
use super::shader::FragmentShader;

/// Trait for types that can be used as render targets.
pub trait Target {
    /// Writes a single scanline into `self`.
    fn rasterize<V, Fs>(
        &mut self,
        scanline: Scanline<V>,
        frag_shader: &Fs,
        config: Config,
    ) where
        V: Vary,
        Fs: FragmentShader<Frag<V>>;
}

/// Configuration for fragment processing.
#[derive(Copy, Clone, Debug)]
pub struct Config {
    /// Whether to do depth testing and which predicate to use.
    ///
    /// If set to `Some(Ordering::Less)`, a fragment passes the depth test
    /// *iff* `new_z < old_z` (the default). If set to `None`, depth test is
    /// not performed. This setting has no effect if the render target does
    /// not support z-buffering.
    pub depth_test: Option<Ordering>,

    /// Whether to write color values. If `false`, other fragment processing
    /// is done but there is no color output. This setting has no effect if
    /// the render target does not support color writes.
    pub color_write: bool,

    /// Whether to write depth values. If `false`, other fragment processing
    /// is done but there is no depth output. This setting has no effect if
    /// the render target does not support depth writes.
    pub depth_write: bool,
}

/// Framebuffer, combining a color (pixel) buffer and a depth buffer.
#[derive(Clone)]
pub struct Framebuf<Col, Dep> {
    pub color_buf: Col,
    pub depth_buf: Dep,
}

impl Config {
    /// Compares the depth value `new` to `curr` and returns whether
    /// `new` passes the depth test specified by `self.depth_test`.
    /// If `self.depth_test` is `None`, always returns `true`.
    pub fn depth_test(&self, new: f32, curr: f32) -> bool {
        self.depth_test
            .map_or(true, |ord| new.partial_cmp(&curr) == Some(ord))
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            color_write: true,
            depth_test: Some(Ordering::Less),
            depth_write: true,
        }
    }
}

impl<Col, Dep> Target for Framebuf<Col, Dep>
where
    Col: AsMutSlice2<u32>,
    Dep: AsMutSlice2<f32>,
{
    /// Rasterizes `scanline` into this framebuffer.
    fn rasterize<V, Fs>(&mut self, sl: Scanline<V>, fs: &Fs, cfg: Config)
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        let Scanline { y, xs, frags } = sl;

        let x0 = xs.start;
        let x1 = xs.end.max(xs.start);
        let cbuf_span = &mut self.color_buf.as_mut_slice2()[y][x0..x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[y][x0..x1];

        frags
            .zip(cbuf_span)
            .zip(zbuf_span)
            .for_each(|(((pos, var), c), z)| {
                let new_z = pos.z();
                if cfg.depth_test(new_z, *z) {
                    let frag = Frag { pos, var };
                    if let Some(new_c) = fs.shade_fragment(frag) {
                        if cfg.color_write {
                            // TODO Blending should happen here
                            *c = new_c.to_argb_u32();
                        }
                        if cfg.depth_write {
                            *z = new_z;
                        }
                    }
                }
            });
    }
}

impl<Buf: AsMutSlice2<u32>> Target for Buf {
    /// Rasterizes `scanline` into this `u32` color buffer.
    /// Does no z-buffering.
    fn rasterize<V, Fs>(&mut self, sl: Scanline<V>, fs: &Fs, cfg: Config)
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        let cbuf_span = &mut self.as_mut_slice2()[sl.y][sl.xs];
        sl.frags
            .zip(cbuf_span)
            .for_each(|((pos, var), pix)| {
                let frag = Frag { pos, var };
                if let Some(color) = fs.shade_fragment(frag) {
                    if cfg.color_write {
                        *pix = color.to_argb_u32();
                    }
                }
            });
    }
}
