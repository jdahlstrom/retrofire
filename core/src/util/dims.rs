// Source for the names:
// https://commons.wikimedia.org/wiki/File:Vector_Video_Standards8.svg

#![allow(non_upper_case_globals)]

use core::ops::Range;

use crate::math::{Point2u, pt2};
use crate::util::Rect;

/// A width, height tuple for representing 2D buffer or window dimensions,
/// screen resolutions, and similar.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Dims(pub u32, pub u32);

// 5:4
pub const qSXGA_640_512: Dims = Dims(640, 512);
pub const SXGA_1280_1024: Dims = Dims(1280, 1024);

// 4:3
pub const qVGA_320_240: Dims = Dims(320, 240);
pub const qSVGA_400_300: Dims = Dims(400, 300);
pub const qXGA_512_384: Dims = Dims(512, 384);
pub const VGA_640_480: Dims = Dims(640, 480);
pub const SVGA_800_600: Dims = Dims(800, 600);
pub const XGA_1024_768: Dims = Dims(1024, 768);
pub const QVGA_1280_960: Dims = Dims(1280, 960);
pub const UXGA_1600_1200: Dims = Dims(1600, 1200);
pub const QXGA_2048_1536: Dims = Dims(2048, 1536);

// 16:10
pub const CGA_320_200: Dims = Dims(320, 200);
pub const MODE_13H: Dims = CGA_320_200;
pub const QCGA_640_400: Dims = Dims(640, 400);
pub const qWXGA_640_400: Dims = Dims(640, 400);
pub const WXGA_1280_800: Dims = Dims(1280, 800);
pub const WXGAP_1440_900: Dims = Dims(1440, 900);
pub const WSXGAP_1680_1050: Dims = Dims(1680, 1050);
pub const WUXGA_1920_1200: Dims = Dims(1920, 1200);
pub const WQXGA_2560_1600: Dims = Dims(2560, 1600);

// 16:9
// 640x360 = "qHD"?
// 800x450 = qWSXGA?
// 960x540 = qFHD
pub const HD_1280_720: Dims = Dims(1280, 720);
pub const WSXGA_1600_900: Dims = Dims(1600, 900);
pub const FHD_1920_1080: Dims = Dims(1920, 1080);
pub const QHD_2560_1440: Dims = Dims(2560, 1440);
pub const UHD_4K_3840_2160: Dims = Dims(3840, 2160);

// DCI ~17:9
pub const DCI_2K_2048_1080: Dims = Dims(2048, 1080);
pub const DCI_4K_4096_2160: Dims = Dims(4096, 2160);

// ~21:9
pub const qUWFHD_1280_540: Dims = Dims(1280, 540);
pub const UWFHD_2560_1080: Dims = Dims(2560, 1080);
pub const UWQHD_3440_1440: Dims = Dims(3440, 1440);

impl Dims {
    /// Returns the number of elements in a buffer of this size.
    pub fn count(&self) -> usize {
        (self.0 as u64 * self.1 as u64)
            .try_into()
            .expect("count should fit in usize")
    }

    /// Returns the width-to-height aspect ratio of `self`.
    pub fn aspect(&self) -> f32 {
        self.0 as f32 / self.1 as f32
    }
}

impl From<Dims> for Rect {
    fn from(Dims(w, h): Dims) -> Self {
        Rect {
            left: Some(0),
            top: Some(0),
            right: Some(w),
            bottom: Some(h),
        }
    }
}

impl<B> From<Dims> for Range<Point2u<B>> {
    fn from(Dims(w, h): Dims) -> Self {
        pt2(0, 0)..pt2(w, h)
    }
}
