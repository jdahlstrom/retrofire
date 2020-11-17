#![allow(dead_code)]
#![allow(unused)]

use core::mem;

use sdl2::{EventPump, Sdl};
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::video::{Window, WindowSurfaceRef};

use math::vec::Vec4;
use Run::*;

pub struct SdlRunner {
    #[allow(unused)]
    sdl: Sdl,
    window: Window,
    event_pump: EventPump,
    frame_evts: Vec<Event>,
}

pub enum Run { Continue, Quit }

impl SdlRunner {
    pub fn new(win_w: u32, win_h: u32) -> Result<SdlRunner, String> {
        let sdl = sdl2::init()?;
        let window = sdl.video()?
            .window("=r/e/t/r/o/f/i/r/e=", win_w, win_h)
            .build()
            .map_err(|e| e.to_string())?;
        let event_pump = sdl.event_pump()?;

        Ok(SdlRunner { sdl, window, event_pump, frame_evts: vec![] })
    }

    pub fn surface(&self) -> Result<WindowSurfaceRef, String> {
        self.window.surface(&self.event_pump)
    }

    pub fn events(&mut self) -> impl Iterator<Item=Event> + '_ {
        let evts = mem::take(&mut self.frame_evts);
        evts.into_iter()
    }

    pub fn keystate(&self) -> impl Iterator<Item=Scancode> + '_ {
        self.event_pump.keyboard_state().pressed_scancodes()
            .collect::<Vec<_>>().into_iter()
    }

    pub fn run<F>(mut self, mut frame_fn: F) -> Result<(), String>
    where F: FnMut(&mut SdlRunner) -> Result<Run, String>
    {
        loop {
            self.frame_evts.clear();
            self.frame_evts = self.event_pump.poll_iter()
                                  .collect::<Vec<_>>();

            for e in &self.frame_evts {
                match e {
                    | Event::Quit { .. }
                    | Event::KeyDown { keycode: Some(Keycode::Escape), .. }
                    => return Ok(()),
                    | _
                    => {}
                }
            }

            let mut surf = self.surface()?;
            let rect = Rect::new(0, 0, surf.width(), surf.height());
            surf.fill_rect(rect, Color::GREY)?;
            if matches!(frame_fn(&mut self)?, Quit) {
                return Ok(())
            }

            self.surface()?.update_window()?
        }
    }

    pub fn plot(&mut self, x: usize, y: usize, col: Vec4) {
        let mut surf = self.surface().unwrap();
        let idx = 4 * (surf.width() as usize * y + x);
        let col = 255.0 * col;

        let buf = surf.without_lock_mut().unwrap();
        buf[idx + 0] = col.z as u8;
        buf[idx + 1] = col.y as u8;
        buf[idx + 2] = col.x as u8;
    }
}

