//! Translation of vector shapes into discrete pixels in the framebuffer.
//!
//! Rasterization proceeds by turning a primitive such as a triagle into
//! a sequence of *scanlines*, each corresponding to a horizontal span of
//! pixels covered by the primitive on a given line. The scanlines, in turn,
//! are converted into a series of *fragments* that represent potentially
//! drawn pixels.
//!
//! If depth testing (z-buffering) is enabled, the fragments are then tested
//! against the current depth value in their position. For each fragment that
//! passes the depth test, a color is computed by the fragment shader and
//! written into the framebuffer. Fragments that fail the test are discarded.

use core::fmt::Debug;
use core::ops::Range;

use crate::geom::Vertex;
use crate::math::{space::Real, vary::Vary, Vec3};

use super::Screen;

/// A fragment, or a single "pixel" in a rasterized primitive.
#[derive(Clone, Debug)]
pub struct Frag<V> {
    pub pos: ScreenVec,
    pub var: V,
}

/// A horizontal, 1-pixel-thick "slice" of a primitive being rasterized.
pub struct Scanline<V: Vary> {
    /// The y coordinate of the line.
    pub y: usize,
    /// The range of x coordinates spanned by the line.
    pub xs: Range<usize>,
    /// Iterator emitting the fragments on the line.
    pub frags: <Varyings<V> as Vary>::Iter,
}

/// Iterator emitting scanlines, linearly interpolating values between the
/// left and right endpoints as it goes.
pub struct ScanlineIter<V: Vary> {
    y: f32,
    left: <Varyings<V> as Vary>::Iter,
    right: <f32 as Vary>::Iter,
    df_dx: <Varyings<V> as Vary>::Diff,
    n: u32,
}

/// Vector in screen space.
/// `x` and `y` are viewport pixel coordinates, `z` is depth.
pub type ScreenVec = Vec3<Real<3, Screen>>;

/// Values to interpolate across a rasterized primitive.
pub type Varyings<V> = (ScreenVec, V);

impl<V: Vary> Iterator for ScanlineIter<V> {
    type Item = Scanline<V>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            return None;
        }
        let v0 = self.left.next()?;
        let x1 = self.right.next()?;
        let y = self.y;

        // Find the next pixel centers to the right
        //
        // If left.pos.x().fract() < 0.5, the pixel is covered and thus drawn;
        // otherwise it's not and we skip to the next pixel.
        //
        // Similarly, if x_right.fract() < 0.5 that's the "one-past-the-end"
        // pixel, otherwise it's the last covered pixel and the next one is
        // the actual one-past-the-end pixel.
        let (x0, x1) = (round_up_to_half(v0.0.x()), round_up_to_half(x1));

        // Adjust v0 to match the rounded x0
        let v0 = v0.step(&scale::<V>(&self.df_dx, x0 - v0.0.x()));

        let frags = v0.vary(self.df_dx.clone(), Some((x1 - x0) as u32));

        self.y += 1.0;
        self.n -= 1;

        Some(Scanline {
            y: y as usize,
            xs: x0 as usize..x1 as usize,
            frags,
        })
    }
}

/// Converts a triangle defined by vertices `verts` into scanlines and calls
/// `scanline_fn` for each scanline. The scanlines are guaranteed to cover
/// exactly those pixels whose center point lies inside the triangle. For more
/// information on the scanline conversion, see [`scan`].
pub fn tri_fill<V, F>(mut verts: [Vertex<ScreenVec, V>; 3], mut scanline_fn: F)
where
    V: Vary,
    F: FnMut(Scanline<V>),
{
    // Sort by y coordinate, start from the top
    verts.sort_by(|a, b| a.pos.y().partial_cmp(&b.pos.y()).unwrap());
    let [top, mid0, bot] = verts.map(|v| (v.pos, v.attrib));

    let [top_y, mid_y, bot_y] = [top.0.y(), mid0.0.y(), bot.0.y()];

    // Interpolate a point on the "long" edge at the same y as `mid0`
    let mid1 = {
        let t = (mid_y - top_y) / (bot_y - top_y);
        top.step(&scale::<V>(&bot.diff(&top), t))
    };

    let (left, right) = if mid0.0.x() < mid1.0.x() {
        (mid0, mid1)
    } else {
        (mid1, mid0)
    };

    //                       X <--top
    //                     ***
    //                   ******
    //                 ********
    //               ** upper **
    // mid0/left--> X**********X <--right/mid1
    //                ** lower **
    //                   ********
    //                      ******
    //                         ***
    //                            X <--bot

    // Rasterize the upper half triangle...
    scan(top_y..mid_y, &top..&left, &top..&right).for_each(&mut scanline_fn);

    // ...and the lower half triangle
    scan(mid_y..bot_y, &left..&bot, &right..&bot).for_each(&mut scanline_fn);
}

/// Returns an iterator that emits a scanline for each line from `y0` to `y1`,
/// interpolating varyings from `l0` to `l1` on the left and from `r0` to `r1`
/// on the right side.
///
/// The three input ranges define a *trapezoid* with horizontal bases, or, in
/// the special case where `l0 == r0` or `l1 == r1`, a triangle:
/// ```text
///            l0___________ r0
/// y0        _|____________|     .next()
///         _|_______________|    .next()
///       _|__________________|     ...
///      |_____________________|    ...
/// y1   l1                     r1
/// ```
/// Any convex polygon can be converted into scanlines by dividing it into
/// trapezoidal segments and calling this function for each segment.
///
/// The exact pixels that are drawn are determined by whether the vector shape
/// *covers* a pixel or not. A pixel is covered, and drawn, if and only if its
/// center point lies inside the shape. This ensures that if two polygons
/// share an edge, or several share a vertex, each pixel at the boundary will
/// be drawn by exactly one of the polygons, with no gaps or overdrawn pixels.
pub fn scan<V: Vary>(
    Range { start: y0, end: y1 }: Range<f32>,
    Range { start: l0, end: l1 }: Range<&Varyings<V>>,
    Range { start: r0, end: r1 }: Range<&Varyings<V>>,
) -> ScanlineIter<V> {
    let inv_dy = (y1 - y0).recip();

    // df/dy for the left edge
    let dl_dy = scale::<V>(&l1.diff(l0), inv_dy);
    // df/dy for the right edge
    let dr_dy = scale::<V>(&r1.diff(r0), inv_dy);

    // df/dx is constant for the whole polygon; precompute it
    let df_dx = {
        let df = r0.step(&dr_dy).diff(&l0.step(&dl_dy));
        scale::<V>(&df, df.0.x().recip())
    };

    // Find the y value of the next pixel center (.5) vertically
    //
    // We want to draw exactly those pixels whose center is *covered* by this
    // polygon. Thus if y_range.start.fract() > 0.5, we skip to the next line.
    // We align the y values with the pixel grid so that on each line, if
    // x_range.start.fract() <= 0.5, the pixel is covered, otherwise it is not.
    //
    // This ensures that whenever two polygons share an edge, every pixel at
    // the edge belongs to exactly one of the polygons, avoiding both gaps and
    // overdrawn pixels. For example, on the left edge:
    //
    //      COVERED               NOT COVERED             NOT COVERED
    //   +-----/-----+           +---------/-+           +-----------+
    //   |    /·     |           |     ·  /  |           |     ·     |
    //   | · p +  ·  | p.y=0.5   |  ·  + p · | p.y=0.5   |  ·  +  ·  |
    //   |  /  ·     |           |     ·/    |           |   p-------- p.y>0.5
    //   +-/---------+           +-----/-----+           +--/--------+
    //    p.x<0.5                    p.x>0.5              p.x<0.5
    //
    let y0_rounded = round_up_to_half(y0);
    let y1_rounded = round_up_to_half(y1);

    let y_tweak = y0_rounded - y0;

    // Adjust varyings to correspond to the aligned y value
    let l0 = l0.step(&scale::<V>(&dl_dy, y_tweak));
    let r0 = r0.0.x() + dr_dy.0.x() * y_tweak;

    ScanlineIter {
        y: y0_rounded,
        left: l0.vary(dl_dy, None),
        right: r0.vary(dr_dy.0.x(), None),
        df_dx,
        n: (y1_rounded - y0_rounded) as u32, // saturates to 0
    }
}

fn scale<V: Vary>(
    d: &<Varyings<V> as Vary>::Diff,
    s: f32,
) -> <Varyings<V> as Vary>::Diff {
    <Varyings<V> as Vary>::scale(d, s)
}

#[cfg(feature = "fp")]
#[inline]
fn round_up_to_half(x: f32) -> f32 {
    use crate::math::float::f32;
    f32::floor(x + 0.5) + 0.5
}
#[cfg(not(feature = "fp"))]
#[inline]
fn round_up_to_half(x: f32) -> f32 {
    (x + 0.5) as i32 as f32 + 0.5
}

#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};
    use core::iter::{once, repeat};

    use crate::geom::vertex;
    use crate::math::vec3;
    use crate::render::raster::tri_fill;
    use crate::util::buf::Buf2;

    // TODO Test different orientations and various edge cases

    #[test]
    fn shared_edge_should_not_have_gaps_or_overdraw() {
        let mut buf = Buf2::new(20, 10, repeat(0));

        let verts = [
            vec3(8.0, 0.0, 0.0),
            vec3(0.0, 6.0, 0.0),
            vec3(14.0, 10.0, 0.0),
            vec3(20.0, 3.0, 0.0),
        ]
        .map(|pos| vertex(pos.to(), 0.0));

        let expected = r"
00000001110000000000
00000011111111000000
00000111111111111100
00011111111111111111
00111111111111111110
01111111111111111100
00111111111111111000
00000111111111110000
00000000011111100000
00000000000011000000";

        tri_fill([verts[0], verts[1], verts[2]], |sl| {
            for x in sl.xs {
                buf[[x as i32, sl.y as i32]] += 1;
            }
        });
        tri_fill([verts[0], verts[2], verts[3]], |sl| {
            for x in sl.xs {
                buf[[x as i32, sl.y as i32]] += 1;
            }
        });

        let s: String = buf
            .rows()
            .flat_map(|r| {
                once("\n".to_string()).chain(r.iter().map(i32::to_string))
            })
            .collect();

        assert_eq!(s, expected);
    }

    #[test]
    fn gradient() {
        use core::fmt::Write;
        let verts = [
            vec3(15.0, 2.0, 0.0),
            vec3(2.0, 8.0, 1.0),
            vec3(26.0, 14.0, 0.5),
        ]
        .map(|pos| vertex(pos.to(), 0.0));

        let expected = r"
              0
            2110
          3322211
       55444332221
     76665544433222
   88877666554443322
    98887766655444332
        88776665544433
            77666554443
                66655444
                    65544
                        54
";
        let mut s = "\n".to_string();

        super::tri_fill(verts, |scanline| {
            write!(s, "{:w$}", " ", w = scanline.xs.start).ok();

            for c in scanline.frags.map(|f| ((10.0 * f.0.z()) as u8)) {
                write!(s, "{c}").ok();
            }
            writeln!(s).ok();
        });
        assert_eq!(s, expected);
    }
}
