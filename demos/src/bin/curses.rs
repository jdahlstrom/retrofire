use std::time::Instant;

use pancurses::*;

use re::prelude::*;

use re::core::render::{Model, render, shader};
use re::core::render::{
    ctx::DepthSort::BackToFront, debug::dir_to_rgb, raster::Scanline,
};
use re::geom::solids::{Build, Torus};

struct Win(Window);

impl Win {
    fn new() -> Self {
        let w = initscr();
        w.nodelay(true);
        curs_set(0);
        start_color();

        // Use the standard xterm 8-bit palette:
        // 0..8: basic dark
        // 8..16: basic bright
        // 16..232: 6x6x6 RGB cube
        // 232..256: sixteen shades of gray
        for i in 16..232 {
            init_pair(i, 15, i);
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
        |v: Vertex3<_>, mvp: &ProjMat3<Model>| v.transform_pos(mvp),
        |frag: Frag<Normal3>, _: &_| dir_to_rgb(frag.var).to_color4(),
    );

    let torus = Torus {
        major_radius: 1.0,
        minor_radius: 0.3,
        major_sectors: 19,
        minor_sectors: 13,
    }
    .build();

    let (wh, ww) = win.0.get_max_yx();
    let aspect = ww as f32 / wh as f32 / 2.0;
    let project = perspective(1.0, aspect, 0.1..10.0);
    let viewport = viewport(pt2(4, wh as u32 - 2)..pt2(ww as u32 - 4, 2));

    let start = Instant::now();
    loop {
        win.0.clear();
        win.0.attrset(COLOR_PAIR(0));
        win.0.mvprintw(0, 0, "Q to quit");

        #[cfg(feature = "stats")]
        {
            ctx.stats.borrow_mut().frames += 1.0;
        }

        let t_secs = start.elapsed().as_secs_f32();

        let mvp = rotate_x(rads(t_secs))
            .then(&rotate_y(rads(t_secs / 1.7)))
            .then(&translate((0.0, 0.0, 3.0 + t_secs.sin())))
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

    // Return to normal terminal mode
    drop(win);

    #[cfg(feature = "stats")]
    {
        println!("{}", ctx.stats.into_inner().finish());
    }
}

impl Target for Win {
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        mut sc: Scanline<V>,
        fs: &Fs,
        uni: U,
        _ctx: &Context,
    ) {
        self.0.mv(sc.y as i32, sc.xs.start as i32);

        for frag in sc.fragments() {
            let Some(col) = fs.shade_fragment(frag, uni) else {
                continue;
            };
            // Map the RGB to the closest color in the 6x6x6 xterm RGB cube
            let [r, g, b, _] = col.0.map(|c| c / 43);
            let col = 16 + 36 * r + 6 * g + b;
            self.0
                .addch(COLOR_PAIR(col as chtype) | ' ' as chtype);
        }
        //Throughput { i: sc.xs.len(), o: sc.xs.len() }
    }
}
