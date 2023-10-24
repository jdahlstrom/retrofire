//! Rendering statistics.

use alloc::{format, string::String};
use core::fmt::{self, Display, Formatter};
use core::ops::AddAssign;
use core::time::Duration;

use Stat::*;

//
// Types
//

#[derive(Copy, Clone, Debug)]
enum Stat {
    /// Objects input/output.
    Objs,
    /// Primitives input/output.
    Prims,
    /// Vertices input/output.
    Verts,
    /// Fragments input/output.
    Frags,
}

/// Collects and accumulates performance statistics.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Time spent rendering.
    pub time: Duration,
    /// Number of render calls issued.
    pub calls: f32,
    /// Number of frames rendered.
    pub frames: f32,

    /// Objects, primitives, vertices, and fragments input/output.
    throughput: [Throughput; 4],
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Throughput {
    // Count of items submitted for rendering.
    pub i: usize,
    // Count of items output to the render target.
    pub o: usize,
}

//
// Impls
//

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn objs(&mut self) -> &mut Throughput {
        &mut self.throughput[Objs as usize]
    }
    pub fn prims(&mut self) -> &mut Throughput {
        &mut self.throughput[Prims as usize]
    }
    pub fn verts(&mut self) -> &mut Throughput {
        &mut self.throughput[Verts as usize]
    }
    pub fn frags(&mut self) -> &mut Throughput {
        &mut self.throughput[Frags as usize]
    }

    /// Returns average throughput in items per second.
    pub fn per_sec(&self) -> Self {
        let secs = if self.time.is_zero() {
            1.0
        } else {
            self.time.as_secs_f32()
        };
        Self {
            frames: self.frames / secs,
            calls: self.calls / secs,
            time: Duration::from_secs(1),
            throughput: self.throughput.map(|stat| stat.per_sec(secs)),
        }
    }
    /// Returns average throughput in items per frame.
    pub fn per_frame(&self) -> Self {
        let frames = self.frames.max(1.0);
        Self {
            frames: 1.0,
            calls: self.calls / frames,
            time: self.time.div_f32(frames),
            throughput: self.throughput.map(|stat| stat.per_frame(frames)),
        }
    }
}

impl Throughput {
    fn per_sec(&self, secs: f32) -> Self {
        Self {
            i: (self.i as f32 / secs) as usize,
            o: (self.o as f32 / secs) as usize,
        }
    }
    fn per_frame(&self, frames: f32) -> Self {
        Self {
            i: self.i / frames as usize,
            o: self.o / frames as usize,
        }
    }
}

impl Display for Stat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut str = format!("{:?}", self);
        str.make_ascii_lowercase();
        write!(f, "{:6}", str)
    }
}

impl Display for Stats {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let w = f.width().unwrap_or(16);
        let ps = self.per_sec();
        let pf = self.per_frame();
        write!(f,
            " STATS  {:>w$} │ {:>w$} │ {:>w$}\n\
             ────────{empty:─>w$}─┼─{empty:─>w$}─┼─{empty:─>w$}─\n \
              time   {:>w$} │ {empty:w$} │ {:>w$}\n \
              calls  {:>w$} │ {:>w$.1} │ {:>w$.1}\n \
              frames {:>w$} │ {:>w$.1} │\n\
             ────────{empty:─>w$}─┼─{empty:─>w$}─┼─{empty:─>w$}─\n", 
            "TOTAL", "PER SEC", "PER FRAME",
            human_time(self.time), human_time(pf.time),
            self.calls, ps.calls, pf.calls,
            self.frames, ps.frames,
            empty = ""
        )?;

        use Stat::*;
        for stat in [Objs, Prims, Verts, Frags] {
            let i = stat as usize;
            let (s, ps, pf) = (self.throughput[i], ps.throughput[i], pf.throughput[i]);
            if f.alternate() {
                write!(f, " {} {:#w$} │ {:#w$} │ {:#w$}\n", stat, s, ps, pf)?;
            } else {
                write!(f, " {} {:w$} │ {:w$} │ {:w$}\n", stat, s, ps, pf)?;
            }
        }
        Ok(())
    }
}

impl Display for Throughput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let &Self { i, o } = self;
        let w = f.width().unwrap_or(10);
        if f.alternate() {
            if i != 0 {
                write!(f, "{:>w$.1}%", 100.0 * o as f32 / i as f32, w = w - 1)
            } else {
                write!(f, "{:>w$}", "--")
            }
        } else {
            write!(f, "{:>w$}", format!("{} / {}", human_num(i), human_num(o)),)
        }
    }
}

impl AddAssign for Stats {
    fn add_assign(&mut self, rhs: Self) {
        self.time += rhs.time;
        self.calls += rhs.calls;
        self.frames += rhs.frames;
        for i in 0..4 {
            self.throughput[i] += rhs.throughput[i];
        }
    }
}

impl AddAssign for Throughput {
    fn add_assign(&mut self, rhs: Self) {
        self.i += rhs.i;
        self.o += rhs.o;
    }
}

fn human_num(n: usize) -> String {
    if n < 1_000 {
        format!("{n:5}")
    } else if n < 100_000 {
        format!("{:4.1}k", n as f32 / 1_000.)
    } else if n < 1_000_000 {
        format!("{:4}k", n / 1_000)
    } else if n < 100_000_000 {
        format!("{:4.1}M", n as f32 / 1_000_000.)
    } else if n < 1_000_000_000 {
        format!("{:4}M", n / 1_000_000)
    } else if n < 100_000_000_000 {
        format!("{:4.1}G", n as f32 / 1_000_000_000.)
    } else {
        format!("{n:5.1e}")
    }
}

fn human_time(d: Duration) -> String {
    let secs = d.as_secs_f32();
    if secs < 1e-3 {
        format!("{:4.1}μs", secs * 1_000_000.)
    } else if secs < 1.0 {
        format!("{:4.1}ms", secs * 1_000.)
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        format!("{:.0}min {:02.0}s", secs / 60.0, secs % 60.0)
    }
}

#[cfg(test)]
mod tests {
    use core::array::from_fn;
    use core::time::Duration;

    use super::*;

    #[test]
    fn stats_display() {
        let stats = Stats {
            frames: 1234.0,
            calls: 5678.0,
            time: Duration::from_millis(4321),
            throughput: from_fn(|i| Throughput {
                i: 12345 * (i + 1) as usize,
                o: 4321 * (i + 1) as usize,
            }),
        };

        assert_eq!(
            format!("{}", stats),
            " \
 STATS             TOTAL │          PER SEC │        PER FRAME
─────────────────────────┼──────────────────┼──────────────────
 time               4.3s │                  │            3.5ms
 calls              5678 │           1314.0 │              4.6
 frames             1234 │            285.6 │
─────────────────────────┼──────────────────┼──────────────────
 objs      12.3k /  4.3k │     2.9k /  1.0k │       10 /     3
 prims     24.7k /  8.6k │     5.7k /  2.0k │       20 /     7
 verts     37.0k / 13.0k │     8.6k /  3.0k │       30 /    10
 frags     49.4k / 17.3k │    11.4k /  4.0k │       40 /    14
"
        );

        assert_eq!(
            format!("{:#}", stats),
            " \
 STATS             TOTAL │          PER SEC │        PER FRAME
─────────────────────────┼──────────────────┼──────────────────
 time               4.3s │                  │            3.5ms
 calls              5678 │           1314.0 │              4.6
 frames             1234 │            285.6 │
─────────────────────────┼──────────────────┼──────────────────
 objs              35.0% │            35.0% │            30.0%
 prims             35.0% │            35.0% │            35.0%
 verts             35.0% │            35.0% │            33.3%
 frags             35.0% │            35.0% │            35.0%
"
        );
    }

    #[test]
    fn human_nums() {
        assert_eq!(human_num(10), "   10");
        assert_eq!(human_num(123), "  123");
        assert_eq!(human_num(1234), " 1.2k");
        assert_eq!(human_num(123456), " 123k");
        assert_eq!(human_num(1234567), " 1.2M");
        assert_eq!(human_num(123456789), " 123M");
        assert_eq!(human_num(1234567890), " 1.2G");
        assert_eq!(human_num(123456789000), "1.2e11");
    }

    #[test]
    fn human_times() {
        assert_eq!(human_time(Duration::from_micros(123)), "123.0μs");
        assert_eq!(human_time(Duration::from_millis(123)), "123.0ms");
        assert_eq!(human_time(Duration::from_millis(1234)), "1.2s");
        assert_eq!(human_time(Duration::from_secs(1234)), "21min 34s");
    }
}
