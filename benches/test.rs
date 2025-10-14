use divan::Bencher;
use re_core::{
    geom::vertex,
    math::{Color3f, Lerp, Vary, pt3, rgb, vary::ZDiv},
    render::raster::{Scanline, tri_fill},
    util::pixfmt::IntoPixel,
};
use std::iter::{Repeat, repeat};
use std::{
    hint::black_box,
    iter::{Empty, empty, zip},
    marker::PhantomData,
};

fn main() {
    divan::main();
}

trait Fill<T: Vary + ZDiv> {
    fn fill(&self, sl: Scanline<T>, buf: &mut [u32], size: usize);
}
#[derive(Copy, Clone, Default)]
struct Filler<T>(PhantomData<T>);
impl Fill<()> for Filler<()> {
    fn fill(&self, sl: Scanline<()>, _: &mut [u32], _: usize) {
        black_box(sl);
    }
}
#[derive(Copy, Clone, Default)]
struct Flat;
impl Vary for Flat {
    type Iter = Repeat<Self>;
    type Diff = Self;

    fn vary(self, _: Self::Diff, _: Option<u32>) -> Self::Iter {
        repeat(Self)
    }

    fn dv_dt(&self, _: &Self, _: f32) -> Self::Diff {
        Self
    }

    fn step(&self, _: &Self::Diff) -> Self {
        Self
    }
}
impl ZDiv for Flat {}
impl Lerp for Flat {
    fn lerp(&self, _: &Self, _: f32) -> Self {
        Self
    }
}

impl Fill<Flat> for Filler<Flat> {
    fn fill(&self, sl: Scanline<Flat>, buf: &mut [u32], size: usize) {
        black_box(&mut buf[size * sl.y..][sl.xs]).fill(black_box(0xFF_FF_FF));
    }
}
impl Fill<Color3f> for Filler<Color3f> {
    fn fill(&self, mut sl: Scanline<Color3f>, buf: &mut [u32], size: usize) {
        let y = size * sl.y;
        let xs = sl.xs.clone();
        for (x, frag) in zip(xs, sl.fragments()) {
            let pix = black_box(&mut buf[y + x]);
            *pix = black_box(frag.var.to_color3().into_pixel())
        }
    }
}

#[divan::bench(args = [256, 512, 1024], types = [(), Flat, Color3f], min_time=0.1)]
fn fill_rate<T: Vary + ZDiv + Copy + Default>(b: Bencher, size: usize)
where
    Filler<T>: Fill<T>,
{
    let mut buf = vec![0u32; (size * size) as usize];

    let filler = Filler::<T>::default();

    let verts = [
        vertex(pt3(0.0, 0.0, 0.0), T::default()),
        vertex(pt3(size as f32, size as f32 / 2.0, 0.0), T::default()),
        vertex(pt3(0.0, size as f32, 0.0), T::default()),
    ];

    b.bench_local(|| {
        tri_fill(verts, |sl| filler.fill(sl, &mut buf, size));
    });
}

fn gradient_fill_xs(b: Bencher, size: f32) {
    let mut buf = vec![0u32; (size * size) as usize];

    b.bench_local(|| {
        //let mut count = 0;
        tri_fill(
            [
                vertex(pt3(0.0, 0.0, 0.0), rgb(1.0, 0.0, 0.0)),
                vertex(pt3(size, size / 2.0, 0.0), rgb(0.0, 1.0, 0.0)),
                vertex(pt3(0.0, size, 0.0), rgb(0.0, 0.0, 1.0)),
            ],
            |mut sl| {
                let y = size as usize * sl.y;
                let xs = sl.xs.clone();
                for (x, frag) in zip(xs, sl.fragments()) {
                    buf[y + x] = frag.var.to_color3().into_pixel()
                }
            },
        );
    });
}

fn gradient_fill_span(b: Bencher, size: f32) {
    let mut buf = vec![0u32; (size * size) as usize];

    b.bench_local(|| {
        //let mut count = 0;
        tri_fill(
            [
                vertex(pt3(0.0, 0.0, 0.0), rgb(1.0, 0.0, 0.0)),
                vertex(pt3(size, size / 2.0, 0.0), rgb(0.0, 1.0, 0.0)),
                vertex(pt3(0.0, size, 0.0), rgb(0.0, 0.0, 1.0)),
            ],
            |mut sl| {
                let y = size as usize * sl.y;
                let span = &mut buf[y..][sl.xs.clone()];
                for (pix, frag) in zip(span, sl.fragments()) {
                    *pix = frag.var.to_color3().into_pixel()
                }
            },
        );
        //assert_eq!(count, (size * size) as usize / 2);
    });

    //dbg!(buf.iter().filter(|&x| *x == 0xFF_FF_FF).count());
}
