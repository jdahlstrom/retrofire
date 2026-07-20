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

use core::{fmt::Debug, mem::swap, ops::Range};

use crate::{
    geom::Vertex3,
    math::{Lerp, Vary, inv_lerp, point::Point3},
    render::Screen,
};

/// A fragment, or a single "pixel" in a rasterized primitive.
#[derive(Clone, Debug)]
pub struct Frag<V> {
    pub pos: ScreenPt,
    pub var: V,
}

/// A horizontal, 1-pixel-thick "slice" of a primitive being rasterized.
#[derive(Clone, Debug)]
pub struct Scanline<V: Vary> {
    /// The y coordinate of the line.
    pub y: usize,
    /// The range of x coordinates spanned by the line.
    pub xs: Range<usize>,
    /// The leftmost varying value on the line.
    pub left: ScreenVert<V>,
    /// The varying step per x increment.
    pub d_dx: <ScreenVert<V> as Vary>::Diff,
}

/// Point in screen space.
/// `x` and `y` are viewport pixel coordinates, `z` is depth.
pub type ScreenPt = Point3<Screen>;

/// Vertex in screen space.
pub type ScreenVert<A> = Vertex3<A, Screen>;

impl<V: Vary> Scanline<V> {
    #[inline]
    pub fn fragments(&mut self) -> impl Iterator<Item = Frag<V>> + '_ {
        self.left
            .clone()
            .vary(self.d_dx.clone(), None)
            // TODO Optimize eg. only one div per 16 pixels
            .map(|v| Frag {
                pos: v.pos,
                var: v.attrib, //.z_div(v.pos.z()),
            })
            .take(self.xs.len())
    }
}

/// Rasterizes a one-pixel-thick line between two vertices.
///
/// Invokes `scan_fn` for each pixel drawn.
///
// TODO Optimize for cases where >1 pixels are drawn for each line
// TODO Guarantee subpixel precision
pub fn line<V, F>([mut v0, mut v1]: [ScreenVert<V>; 2], mut scan_fn: F)
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
        let dv_dx = v0.dv_dt(&v1, 1.0 / dx.abs());

        // Adjust y to match the rounded x0
        let mut y = v0.pos.y() + dy_dx * (x0 - v0.pos.x());
        let mut v = v0;
        for x in x0 as usize..x1 as usize {
            scan_fn(Scanline {
                y: y as usize,
                xs: x..x + 1,
                left: v.clone(),
                d_dx: dv_dx.clone(),
            });
            y += dy_dx;
            v = v.step(&dv_dx);
        }
    } else {
        // More tall than wide
        let y0 = round_up_to_half(v0.pos.y());
        let y1 = round_up_to_half(v1.pos.y());

        let dx_dy = dx / dy;
        let dv_dy = v0.dv_dt(&v1, 1.0 / dy);

        // Adjust x0 to match the rounded y0
        let mut x = v0.pos.x() + dx_dy * (y0 - v0.pos.y());
        let mut v = v0;
        for y in y0 as usize..y1 as usize {
            scan_fn(Scanline {
                y,
                xs: x as usize..x as usize + 1,
                left: v.clone(),
                // arbitrary, just one pixel drawn at time
                d_dx: dv_dy.clone(),
            });
            x += dx_dy;
            v = v.step(&dv_dy)
        }
    }
}

/// Rasterizes a filled triangle defined by three vertices.
///
/// Converts the triangle into [scanlines][Scanline] and invokes `scanline_fn`
/// for each scanline. The scanlines are guaranteed to cover exactly those
/// pixels whose center point lies inside the triangle. For more information
/// on the scanline conversion, see [`scan`].
pub fn tri_fill<V, F>(verts: [ScreenVert<V>; 3], mut scanline_fn: F)
where
    V: Vary,
    F: FnMut(Scanline<V>),
{
    let [mut a, mut b, mut c] = verts;

    // Sort by y coordinate, start from the top
    let sort2 = |a: &mut ScreenVert<V>, b: &mut ScreenVert<V>| {
        if a.pos.y() > b.pos.y() {
            swap(a, b);
        }
    };
    sort2(&mut a, &mut b);
    sort2(&mut b, &mut c);
    sort2(&mut a, &mut b);

    let [top, mut mid0, bot] = [a, b, c];

    // Find the point on the "long" edge at the same y as `mid0`
    let t = inv_lerp(mid0.pos.y(), top.pos.y(), bot.pos.y());
    let mut mid1 = top.lerp(&bot, t);

    if mid0.pos.x() > mid1.pos.x() {
        swap(&mut mid0, &mut mid1);
    };

    //                       X <--top
    //                     ***
    //                   ******
    //                 ********
    //               ** upper **
    // mid0/left--> X==========X <--right/mid1
    //                ** lower **
    //                   ********
    //                      ******
    //                         ***
    //                            X <--bot

    // dv/dx is constant over the whole triangle; precompute it
    let dv_dx = mid0.dv_dt(&mid1, 1.0 / (mid1.pos.x() - mid0.pos.x()));

    let top = {
        let x = top.pos.x();
        (top, x)
    };
    let mid = (mid0, mid1.pos.x());
    let bot = {
        let x = bot.pos.x();
        (bot, x)
    };

    // Rasterize the upper half triangle...
    //                       X
    //                     ***
    //                   ******
    //                 ********
    //               ** upper **
    scan(top, mid.clone(), &dv_dx, &mut scanline_fn);

    // ...and the lower half triangle
    //               X*********X
    //                ** lower **
    //                   ********
    //                      ******
    //                         ***
    //                            X
    scan(mid, bot, &dv_dx, &mut scanline_fn);
}

/// Calls a function for every *scanline* in a range.
///
/// The inputs define a *trapezoid* with horizontal bases, or, in
/// the special case where TODO, a triangle:
/// ```text
///             l0___________ r0
/// y0         _|____________|
///          _|_______________|
///        _|__________________|
///       |_____________________|
/// y1   l1                     r1
/// ```
/// Any y-monotonous polygon can be converted into scanlines by dividing it
/// into trapezoidal segments and calling this function for each segment.
///
/// The exact pixels that are drawn are determined by whether the vector shape
/// *covers* a pixel or not. A pixel is covered, and drawn, if and only if its
/// center point lies inside the shape. This ensures that if two polygons
/// share an edge, or several share a vertex, each pixel at the boundary will
/// be drawn by exactly one of the polygons, with no gaps or overdrawn pixels.
pub fn scan<V: Vary, F: FnMut(Scanline<V>)>(
    from: (ScreenVert<V>, f32),
    to: (ScreenVert<V>, f32),
    d_dx: &<ScreenVert<V> as Vary>::Diff,
    scanline_fn: &mut F,
) {
    let y0 = from.0.pos.y();
    let y1 = to.0.pos.y();

    // (delta-v, delta-x) per y-step
    let d_dy = from.dv_dt(&to, (y1 - y0).recip());

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
    //   |  /········|           |      /····|           |  p--------- p.y>0.5
    //   +-/---------+           +-----/-----+           +-/---------+
    //    p.x<0.5                    p.x>0.5              p.x<0.5
    //
    let y0_rounded = round_up_to_half(y0);
    let y1_rounded = round_up_to_half(y1);

    // Adjust varyings to correspond to the aligned y value
    let y_tweak = y0_rounded - y0;
    let (mut left, mut x_right) = from.lerp(&from.step(&d_dy), y_tweak);

    for y in y0_rounded as usize..y1_rounded as usize {
        let x_left = left.pos.x();

        let x_left_rounded = round_up_to_half(x_left);
        let x_right_rounded = round_up_to_half(x_right);
        // Adjust v0 to match the rounded x0
        let left_tweaked =
            left.lerp(&left.step(&d_dx), x_left_rounded - x_left);

        scanline_fn(Scanline {
            y,
            xs: x_left_rounded as usize..x_right_rounded as usize,
            left: left_tweaked,
            d_dx: d_dx.clone(),
        });

        (left, x_right) = (left, x_right).step(&d_dy);
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
        util::{Buf2, Dims},
    };

    use super::{Scanline, tri_fill};

    // TODO Test different orientations and various edge cases

    #[test]
    fn shared_edge_should_not_have_gaps_or_overdraw() {
        let mut buf = Buf2::new(Dims(20, 10));

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
        let w_left = 2.0;
        let w_right = 4.0;

        let left = vertex(pt3(8.0, 42.0, 1.0 / w_left), 3.0f32.z_div(w_left));
        let right =
            vertex(pt3(16.0, 42.0, 1.0 / w_right), 5.0f32.z_div(w_right));

        let d_dx = left.dv_dt(&right, 1.0 / 8.0);

        let mut sl = Scanline { y: 42, xs: 8..16, left, d_dx };

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
        assert_eq!(x, 16.0);
    }
}
