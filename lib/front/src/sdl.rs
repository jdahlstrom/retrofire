use render::{Framebuf, Stats};
use sdl2::{EventPump, Sdl};
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color as SdlColor;
use sdl2::video::Window;
use std::ops::ControlFlow::{self, *};
use std::time::Instant;
use util::buf::Buffer;
use util::color::rgb;
use util::io::save_ppm;

pub struct SdlRunner {
    #[allow(unused)]
    sdl: Sdl,
    opts: Options,
    event_pump: EventPump,
    window: Window,
    zbuf: Buffer<f32>,
    start: Instant,
}

pub struct Options {
    pub resolution: Resolution,
    pub title: String,
    pub fill_color: Option<SdlColor>,
    pub fullscreen: bool,
    pub rel_mouse: bool,
    pub depth_buf: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            resolution: Resolution::SVGA,
            title: "=r/e/t/r/o/f/i/r/e=".into(),
            fill_color: Some(SdlColor::BLACK),
            fullscreen: false,
            rel_mouse: false,
            depth_buf: true
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Resolution(pub u32, pub u32);

#[allow(non_upper_case_globals)]
impl Resolution {
    pub const MODE_13H: Self = Self(320, 200);
    pub const qVGA: Self = Self(320, 240);
    pub const qSVGA: Self = Self(400, 300);
    pub const qXGA: Self = Self(512, 384);
    pub const VGA: Self = Self(640, 480);
    pub const SVGA: Self = Self(800, 600);
    pub const XGA: Self = Self(1024, 768);
    pub const HD: Self = Self(1280, 720);
    pub const WXGA: Self = Self(1366, 768);
    pub const WSXGA: Self = Self(1400, 900);
    pub const SXGA: Self = Self(1280, 1024);
    pub const UXGA: Self = Self(1600, 1200);
    pub const FHD: Self = Self(1920, 1080);
    pub const WUXGA: Self = Self(1920, 1200);
    pub const QHD: Self = Self(2560, 1440);
    pub const WQXGA: Self = Self(2560, 1600);
    pub const UHD: Self = Self(3840, 2160);

    pub fn aspect_ratio(&self) -> f32 {
        self.0 as f32 / self.1 as f32
    }
}

pub struct Frame<'a> {
    pub buf: Framebuf<'a>,
    pub delta_t: f32,
    pub events: Vec<Event>,
    pub pressed_keys: Vec<Scancode>,
}

impl<'a> Frame<'a> {
    pub fn screenshot(&self, filename: &str) -> Result<(), String> {
        let buf = &self.buf.color;
        let data: Vec<_> = buf.data().chunks(4)
            .map(|bgra| rgb(bgra[2], bgra[1], bgra[0]))
            .collect();

        // TODO Better way to map Buf<u8> to Buf<Color>
        save_ppm(filename, &Buffer::from_vec(buf.width(), data))
            .map_err(|e| e.to_string())
    }
}

impl SdlRunner {
    pub fn new(res_w: u32, res_h: u32) -> Result<SdlRunner, String> {
        Self::with_options(Options {
            resolution: Resolution(res_w, res_h),
            ..Options::default()
        })
    }

    pub fn with_options(opts: Options) -> Result<SdlRunner, String> {
        let Options {
            resolution: Resolution(res_w, res_h),
            ref title,
            fullscreen,
            rel_mouse,
            depth_buf,
            ..
        } = opts;

        let sdl = sdl2::init()?;
        let mut window = sdl.video()?
            .window(title, res_w, res_h);
        if fullscreen {
            window.fullscreen();
        }
        sdl.mouse().set_relative_mouse_mode(rel_mouse);

        let (z_w, z_h) = if depth_buf {
            (res_w as usize, res_h as usize)
        } else {
            (0, 0)
        };

        let event_pump = sdl.event_pump()?;
        Ok(SdlRunner {
            sdl, opts, event_pump,
            window: window.build().map_err(|e| e.to_string())?,
            zbuf: Buffer::new(z_w, z_h, f32::INFINITY),
            start: Instant::now(),
        })
    }

    pub fn run<F>(&mut self, mut frame_fn: F) -> Result<(), String>
    where
        F: FnMut(Frame) -> ControlFlow<Result<(), String>>
    {
        let mut clock = Instant::now();
        loop {
            let ep = &mut self.event_pump;

            let events = ep.poll_iter()
                .collect::<Vec<_>>();
            let pressed_keys = ep.keyboard_state()
                .pressed_scancodes().collect();

            if events.iter().any(is_quit) {
                return Ok(());
            }

            let mut surf = self.window.surface(ep)?;

            if let Some(c) = self.opts.fill_color {
                surf.fill_rect(None, c)?;
            }

            let zbuf = &mut self.zbuf;
            zbuf.fill(f32::INFINITY);

            let delta_t = clock.elapsed().as_secs_f32();
            clock = Instant::now();

            let w = surf.width() as usize;
            let frame = Frame {
                buf: Framebuf {
                    color: Buffer::borrow(w, surf.without_lock_mut().unwrap()),
                    depth: zbuf,
                },
                delta_t,
                events,
                pressed_keys,
            };

            let res = frame_fn(frame);
            surf.update_window()?;

            if let Break(res) = res {
                return res;
            }
        }
    }

    pub fn pause(&mut self) {
        self.event_pump.wait_iter().find(is_quit);
    }

    pub fn print_stats(self, stats: Stats) {
        let Stats { frames, pix_out, prims_in, prims_out, .. } = stats;
        let elapsed = self.start.elapsed().as_secs_f32();
        println!();
        println!(" S  T  A  T  S  │ frames │   objs i/o    │   prims i/o   │   verts i/o   │    pix i/o    │   time  ");
        println!("════════════════╪════════╪═══════════════╪═══════════════╪═══════════════╪═══════════════╪═════════");
        println!(" Total          │ {:#}", stats);
        println!(" Per sec        │ {:#}", stats.avg_per_sec());
        println!(" Per frame      │ {:#}", stats.avg_per_frame());
        println!("────────────────┼────────┴───────────────┴───────────────┴───────────────┴───────────────┴─────────");
        println!(" Avg pix/prim   │ {}", pix_out / prims_out.max(1));
        println!(" Avg vis prims  │ {}%", 100 * prims_out / prims_in.max(1));
        println!(" Elapsed time   │ {:.2}s", elapsed);
        println!(" Average fps    │ {:.2}\n", frames as f32 / elapsed);
    }
}

fn is_quit(e: &Event) -> bool {
    matches!(e,
        | Event::Quit { .. }
        | Event::KeyDown { keycode: Some(Keycode::Escape), .. })
}
