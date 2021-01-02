use std::time::Instant;

use sdl2::{EventPump, Sdl};
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color as SdlColor;
use sdl2::rect::Rect;
use sdl2::video::{Window, WindowSurfaceRef};

use render::{color::Color, Plot, Stats};
use Run::*;

pub struct SdlRunner {
    #[allow(unused)]
    sdl: Sdl,
    window: Window,
    event_pump: EventPump,
    start: Instant,
}

pub struct Frame<'a> {
    pub buf: Buf<'a>,
    pub delta_t: f32,
    pub events: Vec<Event>,
    pub pressed_keys: Vec<Scancode>,
}

pub struct Buf<'a> {
    width: usize,
    #[allow(unused)]
    height: usize,
    data: &'a mut[u8]
}

impl<'a> Plot for Buf<'a> {
    fn plot(&mut self, x: usize, y: usize, c: Color) {
        let idx = 4 * (self.width * y + x);
        let [_, r, g, b] = c.to_argb();
        self.data[idx + 0] = b;
        self.data[idx + 1] = g;
        self.data[idx + 2] = r;
    }
}

#[derive(Eq, PartialEq)]
pub enum Run { Continue, Quit }

impl SdlRunner {
    pub fn new(win_w: u32, win_h: u32) -> Result<SdlRunner, String> {
        let sdl = sdl2::init()?;
        let window = sdl.video()?
            .window("=r/e/t/r/o/f/i/r/e=", win_w, win_h)
            .build()
            .map_err(|e| e.to_string())?;
        let event_pump = sdl.event_pump()?;

        Ok(SdlRunner { sdl, window, event_pump, start: Instant::now() })
    }

    pub fn run<F>(&mut self, mut frame_fn: F) -> Result<(), String>
    where F: FnMut(Frame) -> Result<Run, String>
    {
        let mut clock = Instant::now();
        loop {
            let events = self.event_pump.poll_iter()
                             .collect::<Vec<_>>();
            let pressed_keys = self.event_pump.keyboard_state()
                                   .pressed_scancodes().collect();

            if events.iter().any(Self::is_quit) {
                return Ok(())
            }

            let mut surf = self.surface()?;
            let rect = Rect::new(0, 0, surf.width(), surf.height());
            surf.fill_rect(rect, SdlColor::GREY)?;

            let now = Instant::now();
            let delta_t = (now - clock).as_secs_f32();
            clock = now;

            let frame = Frame {
                buf: Buf {
                    width: surf.width() as usize,
                    height: surf.height() as usize,
                    data: surf.without_lock_mut().unwrap(),
                },
                delta_t, events, pressed_keys,
            };

            let res = frame_fn(frame)?;
            self.surface()?.update_window()?;

            if res == Quit {
                return Ok(())
            }
        }
    }

    #[allow(unused)]
    pub fn pause(&mut self) {
        self.event_pump.wait_iter().find(Self::is_quit);
    }

    pub fn print_stats(self, stats: Stats) {
        let Stats { frames, pixels, faces_in, faces_out, .. } = stats;
        let elapsed = self.start.elapsed().as_secs_f32();
        println!("\n S  T  A  T  S");
        println!("═══════════════╕");
        println!(" Total         │ {}", stats);
        println!(" Per sec       │ {}", stats.avg_per_sec());
        println!(" Per frame     │ {}", stats.avg_per_frame());
        println!("───────────────┤");
        println!(" Avg pix/face  │ {}", pixels / faces_out.max(1));
        println!(" Avg vis faces │ {}%", 100 * faces_out / faces_in.max(1));
        println!(" Elapsed time  │ {:.2}s", elapsed);
        println!(" Average fps   │ {:.2}\n", frames as f32 / elapsed);
    }

    fn surface(&self) -> Result<WindowSurfaceRef, String> {
        self.window.surface(&self.event_pump)
    }

    fn is_quit(e: &Event) -> bool {
        matches!(e, Event::Quit { .. }
            | Event::KeyDown { keycode: Some(Keycode::Escape), .. })
    }
}
