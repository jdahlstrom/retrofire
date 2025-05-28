use std::time::Instant;

use pancurses::*;

use re::prelude::ctx::DepthSort::BackToFront;
use re::prelude::raster::Scanline;
use re::prelude::stats::Throughput;
use re::prelude::*;
use re_geom::solids::Torus;

struct Win(Window);

fn main() {
    let mut win = Win(initscr());
    let w = &win.0;
    curs_set(0);
    start_color();

    for i in 0..256 {
        let r = (i & 0b111_000_00) * 4;
        let g = (i & 0b000_111_00) * 35;
        let b = (i & 0b000_000_11) * 330;

        init_color(i, r, g, b);
        init_pair(i, i, i);
    }

    /*let verts: [Vertex3<Color3f>; 4] = [
        vertex(pt3(-1.0, -1.0, 0.0), rgb(0.25, 0.0, 0.0)),
        vertex(pt3(-1.0, 1.0, 0.0), rgb(0.0, 1.0, 0.0)),
        vertex(pt3(1.0, -1.0, 0.0), rgb(0.5, 0.0, 1.0)),
        vertex(pt3(1.0, 1.0, 0.0), rgb(1.0, 1.0, 1.0)),
    ];*/

    let ctx = Context {
        /*face_cull: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,*/
        depth_sort: Some(BackToFront),
        ..Context::default()
    };

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
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
        major_sectors: 19,
        minor_sectors: 9,
    }
    .build();

    let (wh, ww) = w.get_max_yx();
    let aspect = ww as f32 / wh as f32 / 2.0;
    let project = perspective(1.0, aspect, 0.1..1000.0);
    let viewport = viewport(pt2(4, 2)..pt2(ww as u32 - 4, wh as u32 - 2));

    let start = Instant::now();
    loop {
        win.0.clear();

        let t_secs = start.elapsed().as_secs_f32();

        let mvp = rotate_x(rads(t_secs))
            .then(&rotate_y(rads(t_secs / 1.7)))
            .then(&translate3(0.0, 0.0, 3.0 + t_secs.sin()))
            .to()
            .then(&project);

        render(
            //[Tri([0, 1, 2]), Tri([3, 2, 1])],
            //verts,
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
    }

    w.getch();
    w.clear();

    endwin();

    println!("{}", ctx.stats.borrow());
}

impl Target for Win {
    fn rasterize<V, Fs>(
        &mut self,
        mut sc: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
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

            //let lum = (r / 2 + g + b / 2) / 40;

            /*let rg = r - g;
            let rb = r - b;
            let gb = g - b;

            let d = 32;
            let col = if rg.abs() < d && rb.abs() < d {
                COLOR_WHITE
            } else if rg > d && rb > d {
                COLOR_RED
            } else if rg < -d && gb > d {
                COLOR_GREEN
            } else if rb < -d && gb < -d {
                COLOR_BLUE
            } else if rb > d {
                COLOR_YELLOW
            } else if rg > d {
                COLOR_MAGENTA
            } else if rg < -d {
                COLOR_CYAN
            } else {
                COLOR_BLACK
            } as u32;*/

            let ch = ' '; //b" .-,:=+*oO#%@"[lum.clamp(0, 12) as usize] as char;

            self.0.attrset(COLOR_PAIR(col));
            self.0.addch(ch);
        }

        Throughput { i: w, o: w }
    }
}
