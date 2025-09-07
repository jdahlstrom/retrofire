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

use core::{
    fmt::{Debug, Formatter},
    mem::swap,
    ops::Range,
};

use crate::{
    geom::Vertex,
    math::{Lerp, Vary, point::Point3},
    render::Screen,
};

/// A fragment, or a single "pixel" in a rasterized primitive.
#[derive(Clone, Debug)]
pub struct Frag<V> {
    pub pos: ScreenPt,
    pub var: V,
}

/// A horizontal, 1-pixel-thick "slice" of a primitive being rasterized.
pub struct Scanline<V: Vary> {
    /// The y coordinate of the line.
    pub y: usize,
    /// The range of x coordinates spanned by the line.
    pub xs: Range<usize>,
    /// Iterator emitting the varyings on the line.
    pub vs: <Varyings<V> as Vary>::Iter,
}

/// Iterator emitting scanlines, linearly interpolating values between the
/// left and right endpoints as it goes.
pub struct ScanlineIter<V: Vary> {
    y: f32,
    left: <Varyings<V> as Vary>::Iter,
    right: <f32 as Vary>::Iter,
    dv_dx: <Varyings<V> as Vary>::Diff,
    n: u32,
}

/// Point in screen space.
/// `x` and `y` are viewport pixel coordinates, `z` is depth.
pub type ScreenPt = Point3<Screen>;

/// Values to interpolate across a rasterized primitive.
pub type Varyings<V> = (ScreenPt, V);

impl<V: Vary> Scanline<V> {
    pub fn fragments(&mut self) -> impl Iterator<Item = Frag<V>> + '_ {
        self.vs.by_ref().map(|(pos, var)| {
            // Perspective correct varyings
            // TODO optimization: only every 16 or so pixels
            let var = var.z_div(pos.z());
            Frag { pos, var }
        })
    }
}

impl<V: Vary> Debug for Scanline<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scanline")
            .field("y", &self.y)
            .field("xs", &self.xs)
            .finish_non_exhaustive()
    }
}

impl<V: Vary> Iterator for ScanlineIter<V> {
    type Item = Scanline<V>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            return None;
        }
        let v0 = self.left.next()?;
        let x1 = self.right.next()?;

        // Find the next pixel centers to the right
        //
        // If left.pos.x().fract() < 0.5, the pixel is covered and thus drawn;
        // otherwise it's not, and we skip to the next pixel.
        //
        // Similarly, if x_right.fract() < 0.5 that's the "one-past-the-end"
        // pixel, otherwise it's the last covered pixel and the next one is
        // the actual one-past-the-end pixel.
        let (x0, x1) = (round_up_to_half(v0.0.x()), round_up_to_half(x1));

        // Adjust v0 to match the rounded x0
        let v0 = v0.lerp(&v0.step(&self.dv_dx), x0 - v0.0.x());

        let vs = v0.vary(self.dv_dx.clone(), Some((x1 - x0) as u32));

        let y = self.y as usize;
        let xs = x0 as usize..x1 as usize;

        self.y += 1.0;
        self.n -= 1;

        Some(Scanline { y, xs, vs })
    }
}

/// Rasterizes a one-pixel-thick line between two vertices.
///
/// Invokes `scan_fn` for each pixel drawn.
///
// TODO Optimize for cases where >1 pixels are drawn for each line
// TODO Guarantee subpixel precision
pub fn line<V, F>([mut v0, mut v1]: [Vertex<ScreenPt, V>; 2], mut scan_fn: F)
where
    V: Vary,
    F: FnMut(Scanline<V>),
{
    if v0.pos.y() > v1.pos.y() {
        swap(&mut v0, &mut v1);
    }
    let [dx, dy, _] = (v1.pos - v0.pos).0;

    if dx.abs() > dy {
        // More wide than tall
        if dx < 0.0 {
            // Always draw from left to right
            swap(&mut v0, &mut v1);
        }
        let x0 = round_up_to_half(v0.pos.x());
        let x1 = round_up_to_half(v1.pos.x());

        let dy_dx = dy / dx;
        // Adjust y0 to match the rounded x0
        let y0 = v0.pos.y() + dy_dx * (x0 - v0.pos.x());

        let (xs, mut y) = (x0 as usize..x1 as usize, y0);
        for x in xs {
            let vs = (v0.pos, v0.attrib.clone());
            let vs = vs.clone().vary_to(vs, 1); // TODO a bit silly
            scan_fn(Scanline {
                y: y as usize,
                xs: x..x + 1,
                vs,
            });
            y += dy_dx;
        }
    } else {
        // More tall than wide
        let y0 = round_up_to_half(v0.pos.y());
        let y1 = round_up_to_half(v1.pos.y());

        let dx_dy = dx / dy;
        // Adjust x0 to match the rounded y0
        let x0 = v0.pos.x() + dx_dy * (y0 - v0.pos.y());

        let mut x = x0;
        for y in y0 as usize..y1 as usize {
            let vs = (v0.pos, v0.attrib.clone());
            let vs = vs.clone().vary_to(vs.clone(), 1);
            scan_fn(Scanline {
                y,
                xs: x as usize..x as usize + 1,
                vs,
            });
            x += dx_dy;
        }
    }
}

/// Rasterizes a filled triangle defined by three vertices.
///
/// Converts the triangle into [scanlines][Scanline] and invokes `scanline_fn`
/// for each scanline. The scanlines are guaranteed to cover exactly those
/// pixels whose center point lies inside the triangle. For more information
/// on the scanline conversion, see [`scan`].
pub fn tri_fill<V, F>(mut verts: [Vertex<ScreenPt, V>; 3], mut scanline_fn: F)
where
    V: Vary,
    F: FnMut(Scanline<V>),
{
    // Sort by y coordinate, start from the top
    verts.sort_by(|a, b| a.pos.y().total_cmp(&b.pos.y()));
    let [top, mid0, bot] = verts.map(|v| (v.pos, v.attrib));

    let [top_y, mid_y, bot_y] = [top.0.y(), mid0.0.y(), bot.0.y()];

    // Interpolate a point on the "long" edge at the same y as `mid0`
    let mid1 = top.lerp(&bot, (mid_y - top_y) / (bot_y - top_y));

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
    let recip_dy = (y1 - y0).recip();

    // dv/dy for the left edge
    let dl_dy = l0.dv_dt(l1, recip_dy);
    // dv/dy for the right edge
    let dr_dy = r0.dv_dt(r1, recip_dy);

    // dv/dx is constant for the whole polygon; precompute it
    let dv_dx = {
        let (l0, r0) = (l0.step(&dl_dy), r0.step(&dr_dy));
        let dx = r0.0.x() - l0.0.x();
        l0.dv_dt(&r0, dx.recip())
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
    //   |    /······|           |        /··|           |     ·     |
    //   |   p·+·····| p.y=0.5   |     + p···| p.y=0.5   |  ·  +  ·  |
    //   |  /········|           |      /····|           |   p-------- p.y>0.5
    //   +-/---------+           +-----/-----+           +--/--------+
    //    p.x<0.5                    p.x>0.5              p.x<0.5
    //
    let y0_rounded = round_up_to_half(y0);
    let y1_rounded = round_up_to_half(y1);

    let y_tweak = y0_rounded - y0;

    // Adjust varyings to correspond to the aligned y value
    let l0 = l0.lerp(&l0.step(&dl_dy), y_tweak);
    let r0 = r0.0.x() + dr_dy.0.x() * y_tweak;

    ScanlineIter {
        y: y0_rounded,
        left: l0.vary(dl_dy, None),
        right: r0.vary(dr_dy.0.x(), None),
        dv_dx,
        n: (y1_rounded - y0_rounded) as u32, // saturates to 0
    }
}

#[inline]
fn round_up_to_half(x: f32) -> f32 {
    crate::math::float::f32::floor(x + 0.5) + 0.5
}

#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};
    use core::iter::once;

    use crate::{
        assert_approx_eq,
        geom::vertex,
        math::{point::pt3, vary::Vary, vary::ZDiv},
        util::buf::Buf2,
    };

    use super::{Scanline, tri_fill};

    // TODO Test different orientations and various edge cases

    #[test]
    fn shared_edge_should_not_have_gaps_or_overdraw() {
        let mut buf = Buf2::new((20, 10));

        let verts = [
            pt3(8.0, 0.0, 0.0),
            pt3(0.0, 6.0, 0.0),
            pt3(14.0, 10.0, 0.0),
            pt3(20.0, 3.0, 0.0),
        ]
        .map(|pos| vertex(pos, 0.0));

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
                buf[[x as u32, sl.y as u32]] += 1;
            }
        });
        tri_fill([verts[0], verts[2], verts[3]], |sl| {
            for x in sl.xs {
                buf[[x as u32, sl.y as u32]] += 1;
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
        let verts = [(15.0, 2.0, 0.0), (2.0, 8.0, 1.0), (26.0, 14.0, 0.5)]
            .map(|(x, y, val)| vertex(pt3(x, y, 1.0), val));

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

        super::tri_fill(verts, |mut sl| {
            write!(s, "{:w$}", " ", w = sl.xs.start).ok();

            for c in sl.fragments().map(|f| (10.0 * f.var) as u8) {
                write!(s, "{c}").ok();
            }
            writeln!(s).ok();
        });
        assert_eq!(s, expected);
    }

    #[test]
    fn scanline_fragments_iter() {
        let w0 = 2.0;
        let w1 = 4.0;
        let mut sl = Scanline {
            y: 42,
            xs: 8..16,
            vs: Vary::vary_to(
                (pt3(8.0, 42.0, 1.0 / w0), 3.0f32.z_div(w0)),
                (pt3(16.0, 42.0, 1.0 / w1), 5.0f32.z_div(w1)),
                9,
            ),
        };

        // Perspective correct values
        let zs = [
            2.0f32, 2.1333334, 2.2857144, 2.4615386, 2.6666667, 2.909091, 3.2,
            3.5555556, 4.0,
        ];
        let vars = [
            3.0f32, 3.1333334, 3.2857144, 3.4615386, 3.6666667, 3.909091,
            4.2000003, 4.555556, 5.0,
        ];
        let mut x = 8.0;

        for ((frag, z), v) in sl.fragments().zip(zs).zip(vars) {
            assert_approx_eq!(frag.pos, pt3(x, 42.0, z.recip()));
            assert_approx_eq!(frag.var, v);

            x += 1.0;
        }
        // vary_to is inclusive
        assert_eq!(x, 17.0);
    }
}
