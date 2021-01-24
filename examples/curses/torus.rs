use pancurses as nc;
use pancurses::Input::*;

use geom::mesh::Vertex;
use geom::solids::*;
use math::transform::*;
use math::vec::Vec4;
use render::raster::gouraud::*;
use math::Angle::Rad;

fn vert(v: Vec4) -> Vertex<f32> {
    Vertex { coord: v, attr: 127. * (v.x * 1.).sin() * (v.y * 1.).cos() + 128. }
}

struct Nc;

impl Drop for Nc {
    fn drop(&mut self) {
        nc::endwin();
    }
}

fn main() {
    let mesh = torus(0.35, 9, 13);

    let _scr = Nc;
    let win = nc::initscr();
    nc::noecho();
    nc::curs_set(0);
    win.nodelay(true);

    let mut theta = Rad(0.0);
    loop {
        win.mvprintw(0, 0, "Q or ^C to quit");

        let w = win.get_max_x() as f32;
        let h = win.get_max_y() as f32;

        let mut tf_mesh = mesh.clone();

        let tf = scale(h / 4.0, h / 4.0, h / 4.0)
            * &rotate_x(theta)
            * &rotate_z(0.37 * theta)
            * &translate(w / 2.0, h / 2.0, 0.0);

        tf_mesh.verts.iter_mut().for_each(|v| *v = &tf * *v);

        let mut faces = tf_mesh.faces()
                               .collect::<Vec<_>>();

        // z sort
        faces.sort_unstable_by(|a, b| {
            b.verts[0].coord.z.partial_cmp(&a.verts[0].coord.z).unwrap()
        });

        for [a, b, c] in faces.into_iter().map(|f| f.verts) {
            gouraud_fill([vert(a.coord), vert(b.coord), vert(c.coord)],
                         |Fragment { coord, varying }| {
                             win.mvaddch(coord.1 as i32, coord.0 as i32,
                                         b"..-:;=+<ox*XO@MW"[varying as usize / 0x10 & 0xF] as char);
                         });
        }

        theta = theta + Rad(0.01);

        if let Some(Character('q')) = win.getch() {
            break;
        }
        nc::flushinp();
        nc::napms(33);
        win.clear();
    }
}
