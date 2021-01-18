use std::time::Instant;

use sdl2::{EventPump, Sdl};
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color as SdlColor;
use sdl2::rect::Rect;
use sdl2::video::Window;

use render::{Buffer, DepthBuf, Stats};
use Run::*;

pub struct SdlRunner {
    #[allow(unused)]
    sdl: Sdl,
    window: Window,
    zbuf: DepthBuf,
    event_pump: EventPump,
    start: Instant,
}

pub struct Frame<'a> {
    pub buf: Buffer<u8, &'a mut [u8]>,
    pub zbuf: &'a mut DepthBuf,
    pub delta_t: f32,
    pub events: Vec<Event>,
    pub pressed_keys: Vec<Scancode>,
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

        Ok(SdlRunner {
            sdl,
            window,
            event_pump,
            zbuf: DepthBuf::new(win_w as usize, win_h as usize),
            start: Instant::now(),
        })
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
                return Ok(());
            }

            let mut surf = self.window.surface(&self.event_pump)?;
            let rect = Rect::new(0, 0, surf.width(), surf.height());
            surf.fill_rect(rect, SdlColor::GREY)?;

            let zbuf = &mut self.zbuf;
            zbuf.clear();

            let delta_t = clock.elapsed().as_secs_f32();
            clock = Instant::now();

            let w = surf.width() as usize;
            let frame = Frame {
                buf: Buffer::borrow(w, surf.without_lock_mut().unwrap()),
                zbuf,
                delta_t,
                events,
                pressed_keys,
            };

            let res = frame_fn(frame)?;
            surf.update_window()?;

            if res == Quit {
                return Ok(());
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

    fn is_quit(e: &Event) -> bool {
        matches!(e, Event::Quit { .. }
           | Event::KeyDown { keycode: Some(Keycode::Escape), .. })
    }
}
