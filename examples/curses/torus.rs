use pancurses as nc;
use pancurses::Input::*;

use geom::mesh::Face;
use geom::mesh::Vertex;
use geom::solids::*;
use math::Angle::Rad;
use math::transform::*;
use math::vec::{dir, Vec4};
use render::raster::gouraud::*;

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
    let mesh = Torus(0.35, 9, 13).build();

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

        let tf = scale(h / 2.0)
            * rotate_x(theta)
            * rotate_z(0.37 * theta)
            * scale_axes(1.0, 0.5, 1.0)
            * translate(dir(w / 2.0, h / 2.0, 0.0));

        let tf_mesh = mesh.clone().transform(&tf);

        let mut faces: Vec<_> = tf_mesh.faces.into_iter()
            .map(|Face { verts, attr }| {
                Face { verts: verts.map(|i| tf_mesh.vertex_coords[i]), attr }
            })
            .collect();

        // z sort
        faces.sort_unstable_by(|a, b| {
            b.verts[0].z.partial_cmp(&a.verts[0].z).unwrap()
        });

        for [a, b, c] in faces.into_iter().map(|f| f.verts) {
            gouraud_fill(
                [vert(a), vert(b), vert(c)],
                |Fragment { coord: (x, y), varying: v, .. }| {
                    win.mvaddch(
                        y as i32, x as i32,
                        b"..-:;=+<ox*XO@MW"[v as usize / 0x10 & 0xF] as char,
                    );
                },
            );
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
