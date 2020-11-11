use geom::solids::*;
use math::transform::*;
use math::vec::Vec4;
use pancurses as nc;
use pancurses::Input::*;
use render::raster::gouraud::*;

fn frag(v: Vec4) -> Fragment {
    Fragment { coord: v, varying: 127. * (v.x * 1.).sin() * (v.y * 1.).cos() + 128. }
}

struct Nc;

impl Drop for Nc {
    fn drop(&mut self) {
        nc::endwin();
    }
}

fn main() {
    let mesh = torus(0.35, 9, 13)
        .gen_normals()
        .validate().unwrap();

    let _scr = Nc;
    let win = nc::initscr();
    nc::noecho();
    nc::curs_set(0);
    win.nodelay(true);

    let mut theta = 0.0;
    loop {
        win.mvprintw(0, 0, "Q or ^C to quit");

        let w = win.get_max_x() as f32;
        let h = win.get_max_y() as f32;

        let mut tf_mesh = mesh.clone();

        let tf = scale(h / 4.0, h / 4.0, h / 4.0)
            * &rotate_x(theta)
            * &rotate_z(theta * 0.37)
            * &translate(w / 2.0, h / 2.0, 0.0);

        tf_mesh.verts.iter_mut().for_each(|v| *v = &tf * *v);

        let mut verts = tf_mesh.face_verts()
            .collect::<Vec<_>>();

        // z sort
        verts.sort_unstable_by(|a, b| b[0].z.partial_cmp(&a[0].z).unwrap());

        for (_i, v) in verts.into_iter().enumerate() {
            gouraud_fill(frag(v[0]), frag(v[1]), frag(v[2]),
                      |Fragment { coord, varying }| {
                          win.mvaddch(coord.y as i32,
                                      coord.x as i32,
                                      b"..-:;=+<ox*XO@MW"[varying as usize / 0x10 & 0xF] as char);
                      });
        }

        theta += 0.01;

        if let Some(c) = win.getch() {
            match c {
                Character('q') => break,
                _ => ()
            }
        }
        nc::flushinp();
        nc::napms(33);
        win.clear();
    }
}
