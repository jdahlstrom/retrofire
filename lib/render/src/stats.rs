use std::time::Duration;
use std::fmt::{Display, Formatter, Result};

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    pub frames: usize,
    pub objs_in: usize,
    pub objs_out: usize,
    pub prims_in: usize,
    pub prims_out: usize,
    pub verts_in: usize,
    pub verts_out: usize,
    pub pix_in: usize,
    pub pix_out: usize,
    pub time_used: Duration,
}

impl Stats {
    pub fn avg_per_frame(&self) -> Stats {
        let frames = self.frames.max(1);
        Stats {
            frames: 1,
            objs_in: self.objs_in / frames,
            objs_out: self.objs_out / frames,
            prims_in: self.prims_in / frames,
            prims_out: self.prims_out / frames,
            verts_in: self.verts_in / frames,
            verts_out: self.verts_out / frames,
            pix_in: self.pix_in / frames,
            pix_out: self.pix_out / frames,
            time_used: self.time_used / frames as u32,
        }
    }

    pub fn avg_per_sec(&self) -> Stats {
        let secs = self.time_used.as_secs_f32();
        Stats {
            frames: (self.frames as f32 / secs) as usize,
            objs_in: (self.objs_in as f32 / secs) as usize,
            objs_out: (self.objs_out as f32 / secs) as usize,
            prims_in: (self.prims_in as f32 / secs) as usize,
            prims_out: (self.prims_out as f32 / secs) as usize,
            verts_in: (self.verts_in as f32 / secs) as usize,
            verts_out: (self.verts_out as f32 / secs) as usize,
            pix_in: (self.pix_in as f32 / secs) as usize,
            pix_out: (self.pix_out as f32 / secs) as usize,
            time_used: Duration::from_secs(1),
        }
    }

    pub fn diff(&self, other: &Stats) -> Stats {
        Stats {
            frames: self.frames - other.frames,
            objs_in: self.objs_in - other.objs_in,
            objs_out: self.objs_out - other.objs_out,
            prims_in: self.prims_in - other.prims_in,
            prims_out: self.prims_out - other.prims_out,
            verts_in: self.verts_in - other.verts_in,
            verts_out: self.verts_out - other.verts_out,
            pix_in: self.pix_in - other.pix_in,
            pix_out: self.pix_out - other.pix_out,
            time_used: self.time_used - other.time_used,
        }
    }
}

fn human_num(n: usize) -> String {
    if n < 1_000 { format!("{:5}", n) }
    else if n < 100_000 { format!("{:4.1}k", n as f32 / 1_000.) }
    else if n < 1_000_000 { format!("{:4}k", n / 1_000) }
    else if n < 100_000_000 { format!("{:4.1}M", n as f32 / 1_000_000.) }
    else if n < 1_000_000_000 { format!("{:4}M", n / 1_000_000) }
    else if n < 100_000_000_000 { format!("{:4.1}G", n as f32 / 1_000_000_000.) }
    else { format!("{:5.1e}", n) }
}

fn human_time(d: Duration) -> String {
    let s = d.as_secs_f32();
    if s < 0.001 { format!("{:4.1}us", s * 1_000_000.) }
    else if s < 1.0 { format!("{:4.1}ms", s * 1_000.) }
    else if s < 60.0 { format!("{:.1}s ", s) }
    else { format!("{:.0}min{:02.0}s ", s / 60.0, s % 60.0)}
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let frames = human_num(self.frames);
        let objs_in = human_num(self.objs_in);
        let objs_out = human_num(self.objs_out);
        let prims_in = human_num(self.prims_in);
        let prims_out = human_num(self.prims_out);
        let verts_in = human_num(self.verts_in);
        let verts_out = human_num(self.verts_out);
        let pix_in = human_num(self.pix_in);
        let pix_out = human_num(self.pix_out);
        let time_used = human_time(self.time_used);

        if f.alternate() {
            write!(f, "{frames:>6} │ \
                       {objs_in} / {objs_out} │ \
                       {prims_in} / {prims_out} │ \
                       {verts_in} / {verts_out} │ \
                       {pix_in} / {pix_out} │ \
                       {time_used:>8}")
        } else {
            write!(f, "frames: {frames} │ \
                   objs i/o: {objs_in} / {objs_out} │ \
                   faces i/o: {prims_in} / {prims_out} │ \
                   verts i/o: {verts_in} / {verts_out} │ \
                   pix i/o: {pix_in} / {pix_out} │ \
                   time used: {time_used:>8}")
        }

    }
}
