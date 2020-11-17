use std::time::Duration;
use std::fmt::{Display, Formatter, Result};

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    pub frames: usize,
    pub faces_in: usize,
    pub faces_out: usize,
    pub pixels: usize,
    pub time_used: Duration,
}

impl Stats {
    pub fn avg_per_frame(&self) -> Stats {
        let frames = self.frames.max(1);
        Stats {
            frames,
            faces_in: self.faces_in / frames,
            faces_out: self.faces_out / frames,
            pixels: self.pixels / frames,
            time_used: self.time_used / frames as u32,
        }
    }

    pub fn avg_per_sec(&self) -> Stats {
        let secs = self.time_used.as_secs_f32();
        Stats {
            frames: (self.frames as f32 / secs) as usize,
            faces_in: (self.faces_in as f32 / secs) as usize,
            faces_out: (self.faces_out as f32 / secs) as usize,
            pixels: (self.pixels as f32 / secs) as usize,
            time_used: Duration::from_secs(1),
        }
    }
}

fn human_num(n: usize) -> String {
    if n < 1_000 { format!("{:5}", n) }
    else if n < 100_000 { format!("{:4.1}k", n as f32 / 1_000.) }
    else if n < 1_000_000 { format!("{:4}k", n / 1_000) }
    else if n < 100_000_000 { format!("{:4.1}M", n as f32 / 1_000_000.) }
    else if n < 1_000_000_000 { format!("{:4}M", n / 1_000_000) }
    else if n < 1_000_000_000_000 { format!("{:4.1}M", n as f32 / 1_000_000.) }
    else { format!("{:5.1e}", n) }
}

fn human_time(d: Duration) -> String {
    let s = d.as_secs_f32();
    if s < 1.0 { format!("{:4.1}ms", s * 1000.) }
    else { format!("{:.1}s ", s) }
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "frames: {} │ faces in: {} │ \
                   faces out: {} │ pixels: {} │ \
                   time used: {:>9}",
               human_num(self.frames),
               human_num(self.faces_in), human_num(self.faces_out),
               human_num(self.pixels), human_time(self.time_used))
    }
}