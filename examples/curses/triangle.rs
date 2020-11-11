use pancurses as nc;
use pancurses::Input::*;

use math::vec::Vec4;
use render::raster::flat;

fn frag(x: f32, y: f32) -> flat::Fragment {
    flat::Fragment { coord: Vec4 { x, y, z: 0.0, w: 0.0 }, varying: () }
}

fn main() {
    let win = nc::initscr();
    nc::noecho();
    nc::curs_set(0);
    win.nodelay(true);

    let mut xs = [(10.0, 0.1), (50.0, 0.2), (30.0, -0.25)];
    let mut ys = [(30.0, -0.2), (10.0, -0.15), (40.0, 0.4)];
    let mut i = 0;

    loop {
        win.mvprintw(0, 0, "Q or ^C to quit, WSAD to change speed of X, space to switch X");

        flat::flat_fill(frag(xs[0].0, ys[0].0), frag(xs[1].0, ys[1].0), frag(xs[2].0, ys[2].0),
                        |frag| {
                            win.mv(frag.coord.y as i32, frag.coord.x as i32);
                            win.addch('*');
                        });

        for j in 0..3 {
            win.mvaddch(ys[j].0.round() as i32,
                        xs[j].0.round() as i32,
                        if i == j { 'X' } else { 'O' });
        }

        for (x, dx) in &mut xs {
            if *x < 0.0 || *x as i32 >= win.get_max_x() - 1 { *dx = -*dx }
            *x += *dx;
        }
        for (y, dy) in &mut ys {
            if *y < 0.0 || *y as i32 >= win.get_max_y() - 1 { *dy = -*dy }
            *y += *dy;
        }

        if let Some(c) = win.getch() {
            match c {
                Character('q') => break,
                Character('w') => ys[i].1 -= 0.05,
                Character('s') => ys[i].1 += 0.05,
                Character('a') => xs[i].1 -= 0.05,
                Character('d') => xs[i].1 += 0.05,
                Character(' ') => i = (i + 1) % 3,
                _ => ()
            }
        }
        nc::flushinp();
        nc::napms(17);
        win.clear();
    }

    nc::endwin();
}
