use std::time::Instant;

use pancurses::*;
use re::math::mat::ProjMat3;
use re::prelude::*;

use re::render::{
    ctx::DepthSort::BackToFront, raster::Scanline, stats::Throughput,
};

use re_geom::solids::{Build, Torus};

struct Win(Window);

impl Win {
    fn new() -> Self {
        let w = initscr();
        w.nodelay(true);
        curs_set(0);
        start_color();

        // Create an RGB 332 palette but keep the eight standard colors
        for i in 8..256 {
            // Range from 0 to 1000
            let r = (i & 0b111_000_00) * 4;
            let g = (i & 0b000_111_00) * 35;
            let b = (i & 0b000_000_11) * 330;

            init_color(i, r, g, b);
            init_pair(i, i, i);
        }
        Self(w)
    }
}
impl Drop for Win {
    fn drop(&mut self) {
        endwin();
    }
}

fn main() {
    let mut win = Win::new();

    let ctx = Context {
        depth_sort: Some(BackToFront),
        ..Context::default()
    };

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let [x, y, z] = (frag.var / 2.0 + splat(0.5)).0;
            rgb(x, y, z).to_color4()
        },
    );

    let torus = Torus {
        major_radius: 1.0,
        minor_radius: 0.3,
        major_sectors: 32,
        minor_sectors: 16,
    }
    .build();

    let (wh, ww) = win.0.get_max_yx();
    let aspect = ww as f32 / wh as f32 / 2.0;
    let project = perspective(1.0, aspect, 1.0..10.0);
    let viewport = viewport(pt2(4, 2)..pt2(ww as u32 - 4, wh as u32 - 2));

    let start = Instant::now();
    loop {
        win.0.clear();
        win.0.attrset(COLOR_PAIR(0));
        win.0.mvprintw(0, 0, "Q to quit");

        ctx.stats.borrow_mut().frames += 1.0;
        let t_secs = start.elapsed().as_secs_f32();

        let mvp = rotate_x(rads(t_secs))
            .then(&rotate_y(rads(t_secs / 1.7)))
            .then(&translate3(0.0, 0.0, 3.0 + t_secs.sin()))
            .to()
            .then(&project);

        render(
            &torus.faces,
            &torus.verts,
            &shader,
            &mvp,
            viewport,
            &mut win,
            &ctx,
        );

        win.0.refresh();
        napms(10);

        if let Some(Input::Character('q')) = win.0.getch() {
            break;
        }
    }

    println!("{}", ctx.stats.borrow());
}

impl Target for Win {
    fn rasterize<V, Fs>(
        &mut self,
        mut sc: Scanline<V>,
        fs: &Fs,
        _ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<V>,
    {
        let w = sc.xs.len();
        let y = sc.y;

        self.0.mv(y as i32, sc.xs.start as i32);

        for frag in sc.fragments() {
            let Some(col) = fs.shade_fragment(frag) else {
                continue;
            };
            let [r, g, b, _] = col.0.map(|c| c as u32);

            let col = (r & 0b111_000_00)
                | (g / 9 & 0b000_111_00)
                | (b / 85 & 0b000_000_11);

            // Avoid the eight standard colors
            self.0.addch(COLOR_PAIR(col.max(8)));
        }
        Throughput { i: w, o: w }
    }
}
