//! Rendering statistics.

use alloc::{format, string::String};
use core::{
    fmt::{self, Display, Formatter},
    ops::AddAssign,
    time::Duration,
};

#[cfg(feature = "std")]
use std::time::Instant;

//
// Types
//

/// Collects and accumulates rendering statistics and performance data.
#[derive(Clone, Debug)]
pub struct Stats {
    #[cfg(feature = "std")]
    pub start: Instant,

    /// Wall clock time elapsed.
    pub wall_time: Duration,
    /// Time spent rendering.
    pub render_time: Duration,
    /// Number of render calls issued.
    pub calls: f32,
    /// Number of frames rendered.
    pub frames: f32,

    /// Objects, primitives, vertices, and fragments input/output.
    pub objs: Throughput,
    pub prims: Throughput,
    pub verts: Throughput,
    pub frags: Throughput,
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
    /// Creates a new zeroed `Stats` instance.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "std")]
            start: Instant::now(),
            wall_time: Default::default(),
            render_time: Default::default(),
            calls: 0.0,
            frames: 0.0,
            objs: Default::default(),
            prims: Default::default(),
            verts: Default::default(),
            frags: Default::default(),
        }
    }

    pub fn start_call(n_prims: usize, n_verts: usize) -> Self {
        Self {
            calls: 1.0,
            prims: Throughput { i: n_prims, o: 0 },
            verts: Throughput { i: n_verts, o: 0 },
            ..Self::new()
        }
    }

    pub fn finish(self) -> Self {
        Self {
            #[cfg(feature = "std")]
            wall_time: self.start.elapsed(),
            ..self
        }
    }

    /// Stops the timer and records the elapsed time to `self.time`.
    ///
    /// No-op if the timer was not running. This method is also no-op unless
    /// the `std` feature is enabled.
    #[must_use]
    pub fn finish_call(mut self, prims_out: usize, verts_out: usize) -> Self {
        self.prims.o += prims_out;
        self.verts.o += verts_out;
        Self {
            #[cfg(feature = "std")]
            render_time: self.start.elapsed(),
            ..self
        }
    }

    /// Returns the average throughput in items per second.
    pub fn per_render_sec(&self) -> Self {
        let secs = if self.render_time.is_zero() {
            1.0
        } else {
            self.render_time.as_secs_f32()
        };
        let [objs, prims, verts, frags] =
            self.throughput().map(|stat| stat.per_sec(secs));
        Self {
            #[cfg(feature = "std")]
            start: self.start,
            render_time: Duration::from_secs(1),
            wall_time: self.wall_time.div_f32(secs),
            frames: self.frames / secs,
            calls: self.calls / secs,
            objs,
            prims,
            verts,
            frags,
        }
    }

    pub fn per_wall_sec(&self) -> Self {
        let secs = if self.wall_time.is_zero() {
            1.0
        } else {
            self.wall_time.as_secs_f32()
        };
        let [objs, prims, verts, frags] =
            self.throughput().map(|stat| stat.per_sec(secs));
        Self {
            #[cfg(feature = "std")]
            start: self.start,
            wall_time: Duration::from_secs(1),
            render_time: self.render_time.div_f32(secs),
            frames: self.frames / secs,
            calls: self.calls / secs,
            objs,
            prims,
            verts,
            frags,
        }
    }
    /// Returns the average throughput in items per frame.
    pub fn per_frame(&self) -> Self {
        let frames = self.frames.max(1.0) as u32;
        let [objs, prims, verts, frags] = self
            .throughput()
            .map(|stat| stat.per_frame(frames));
        Self {
            #[cfg(feature = "std")]
            start: self.start,
            frames: 1.0,
            render_time: self.render_time / frames,
            wall_time: self.wall_time / frames,
            calls: self.calls / frames as f32,
            objs,
            prims,
            verts,
            frags,
        }
    }

    fn throughput(&self) -> [Throughput; 4] {
        [self.objs, self.prims, self.verts, self.frags]
    }

    fn throughput_mut(&mut self) -> [&mut Throughput; 4] {
        let Self { objs, prims, verts, frags, .. } = self;
        [objs, prims, verts, frags]
    }
}

impl Throughput {
    fn per_sec(&self, secs: f32) -> Self {
        Self {
            i: (self.i as f32 / secs) as usize,
            o: (self.o as f32 / secs) as usize,
        }
    }
    fn per_frame(&self, frames: u32) -> Self {
        Self {
            i: self.i / frames as usize,
            o: self.o / frames as usize,
        }
    }
}

impl Display for Stats {
    #[rustfmt::skip]
    #[inline(never)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let per_ws = self.per_wall_sec();
        let per_rs = self.per_render_sec();
        let per_f = self.per_frame();

        let ws_per_rs = self.wall_time.div_duration_f32(self.render_time);

        let w = f.width().unwrap_or(14);
        let e = "";

        writeln!(f,
            " STATS     {:>w$} │ {:>w$} │ {:>w$} │ {:>w$}\n\
             ───────────{e:─>w$}─┼─{e:─>w$}─┼─{e:─>w$}─┼─{e:─>w$}─",
            "TOTAL", "PER WALL-SEC", "PER REND-SEC", "PER FRAME"
        )?;
        writeln!(f,
            " wall-time {:>w$} │ {e:w$} │ {:>w$.2} │ {:>w$}",
            human_time(self.wall_time), ws_per_rs,
            human_time(per_f.wall_time)
        )?;


        let  rs_per_ws = 1.0 / ws_per_rs;

        writeln!(f,
            " rend-time {:>w$} │ {:>w$.2} │ {e:w$} │ {:>w$}\n \
              calls     {:>w$} │ {:>w$.1} │ {:>w$.1} │ {:>w$.1}\n \
              frames    {:>w$} │ {:>w$.1} │ {:>w$.1} │\n\
             ───────────{e:─>w$}─┼─{e:─>w$}─┼─{e:─>w$}─┼─{e:─>w$}─",

            human_time(self.render_time), rs_per_ws, human_time(per_f.render_time),
            self.calls, per_ws.calls, per_rs.calls, per_f.calls,
            self.frames, per_ws.frames, per_rs.frames,
            e = ""
        )?;

        let labels = ["objs", "prims", "verts", "frags"];
        for (i, lbl) in (0..4).zip(labels) {
            let [tot, per_ws, per_rs, per_f] =
                [self, &per_ws, &per_rs, &per_f].map(|s| s.throughput()[i]);

            if f.alternate() {
                writeln!(f, " {lbl:9} {tot:#w$} │ {per_ws:#w$} │ {per_rs:#w$} │ {per_f:#w$}")?;
            } else {
                writeln!(f, " {lbl:9} {tot:w$} │ {per_ws:w$} │ {per_rs:w$} │ {per_f:w$}")?;
            }
        }

        Ok(())
    }
}

impl Display for Throughput {
    #[inline(never)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let &Self { i, o } = self;
        let w = f.width().unwrap_or(10);
        if f.alternate() {
            if i == 0 {
                write!(f, "{:>w$}", "--")
            } else {
                let pct = 100.0 * o as f32 / i as f32;
                write!(f, "{pct:>w$.1}%", w = w - 1)
            }
        } else {
            let io = format!("{} / {}", human_num(i), human_num(o));
            write!(f, "{io:>w$}")
        }
    }
}

impl AddAssign for Stats {
    /// Appends the stats of `other` to `self`.
    fn add_assign(&mut self, other: Self) {
        self.wall_time += other.wall_time;
        self.render_time += other.render_time;
        self.calls += other.calls;
        self.frames += other.frames;
        for i in 0..4 {
            *self.throughput_mut()[i] += other.throughput()[i];
        }
    }
}

impl AddAssign for Throughput {
    fn add_assign(&mut self, rhs: Self) {
        self.i += rhs.i;
        self.o += rhs.o;
    }
}

#[inline(never)]
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
    } else if (n as u64) < 100_000_000_000 {
        format!("{:4.1}G", n as f32 / 1_000_000_000.)
    } else {
        format!("{n:5.1e}")
    }
}

#[inline(never)]
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
    use core::{array::from_fn, time::Duration};

    use super::*;

    #[test]
    fn stats_display() {
        let [objs, prims, verts, frags] = from_fn(|i| Throughput {
            i: 12345 * (i + 1),
            o: 4321 * (i + 1),
        });
        let stats = Stats {
            frames: 1234.0,
            calls: 5678.0,
            wall_time: Duration::from_millis(5432),
            render_time: Duration::from_millis(4321),
            objs,
            prims,
            verts,
            frags,
            ..Stats::new()
        };

        assert_eq!(
            format!("{stats}"),
            " \
 STATS              TOTAL │   PER WALL-SEC │   PER REND-SEC │      PER FRAME
──────────────────────────┼────────────────┼────────────────┼────────────────
 wall-time           5.4s │                │           1.26 │          4.4ms
 rend-time           4.3s │           0.80 │                │          3.5ms
 calls               5678 │         1045.3 │         1314.0 │            4.6
 frames              1234 │          227.2 │          285.6 │
──────────────────────────┼────────────────┼────────────────┼────────────────
 objs       12.3k /  4.3k │   2.3k /   795 │   2.9k /  1.0k │     10 /     3
 prims      24.7k /  8.6k │   4.5k /  1.6k │   5.7k /  2.0k │     20 /     7
 verts      37.0k / 13.0k │   6.8k /  2.4k │   8.6k /  3.0k │     30 /    10
 frags      49.4k / 17.3k │   9.1k /  3.2k │  11.4k /  4.0k │     40 /    14
"
        );

        assert_eq!(
            format!("{stats:#}"),
            " \
 STATS              TOTAL │   PER WALL-SEC │   PER REND-SEC │      PER FRAME
──────────────────────────┼────────────────┼────────────────┼────────────────
 wall-time           5.4s │                │           1.26 │          4.4ms
 rend-time           4.3s │           0.80 │                │          3.5ms
 calls               5678 │         1045.3 │         1314.0 │            4.6
 frames              1234 │          227.2 │          285.6 │
──────────────────────────┼────────────────┼────────────────┼────────────────
 objs               35.0% │          35.0% │          35.0% │          30.0%
 prims              35.0% │          35.0% │          35.0% │          35.0%
 verts              35.0% │          35.0% │          35.0% │          33.3%
 frags              35.0% │          35.0% │          35.0% │          35.0%
"
        );
    }

    #[test]
    fn human_nums() {
        assert_eq!(human_num(10), "   10");
        assert_eq!(human_num(123), "  123");
        assert_eq!(human_num(1_234), " 1.2k");
        assert_eq!(human_num(12_3456), " 123k");
        assert_eq!(human_num(1_234_567), " 1.2M");
        assert_eq!(human_num(123_456_789), " 123M");
        assert_eq!(human_num(1_234_567_890), " 1.2G");
        assert_eq!(human_num(123_456_789_000), "1.2e11");
    }

    #[test]
    fn human_times() {
        assert_eq!(human_time(Duration::from_micros(123)), "123.0μs");
        assert_eq!(human_time(Duration::from_millis(123)), "123.0ms");
        assert_eq!(human_time(Duration::from_millis(1234)), "1.2s");
        assert_eq!(human_time(Duration::from_secs(1234)), "21min 34s");
    }
}
