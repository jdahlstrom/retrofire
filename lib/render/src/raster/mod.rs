use std::mem::swap;

use geom::mesh::Vertex;
use math::Linear;

use crate::vary::{Bresenham, Varying};

pub mod flat;
pub mod gouraud;

#[derive(Copy, Clone, Debug)]
pub struct Fragment<V: Copy, C: Copy = (usize, usize)> {
    pub coord: C,
    pub varying: V,
}

fn ysort<V>([a, b, c]: &mut [Vertex<V>; 3]) {
    if a.coord.y > b.coord.y { swap(a, b); }
    if a.coord.y > c.coord.y { swap(a, c); }
    if b.coord.y > c.coord.y { swap(b, c); }
}

pub fn tri_fill<V, P>(mut verts: [Vertex<V>; 3], mut plot: P)
where
    V: Linear<f32> + Copy,
    P: FnMut(Fragment<V>)
{
    ysort(&mut verts);

    let mut y = verts[0].coord.y.round() as usize;

    let mut half_tri = |y_end,
                        left: &mut Varying<(f32, _)>,
                        right: &mut Varying<(f32, _)>| {

        for (y, (left, right)) in (y..y_end).zip(left.zip(right)) {

            let v = Varying::between(left.1, right.1, right.0 - left.0);

            let x_left = left.0.round() as usize;
            let x_right = right.0.round() as usize;

            for (x, v) in (x_left..=x_right).zip(v) {
                plot(Fragment { coord: (x, y), varying: v });
            }
        }
        y = y_end;
    };

    let [a, b, c] = verts;

    let (ay, av) = (a.coord.y, (a.coord.x, a.attr));
    let (by, bv) = (b.coord.y, (b.coord.x, b.attr));
    let (cy, cv) = (c.coord.y, (c.coord.x, c.attr));

    let ab = &mut Varying::between(av, bv, by - ay);
    let ac = &mut Varying::between(av, cv, cy - ay);
    let bc = &mut Varying::between(bv, cv, cy - by);

    if ab.step.0 < ac.step.0 {
        half_tri(by.round() as usize, ab, ac);
        half_tri(cy.round() as usize, bc, ac);
    } else {
        half_tri(by.round() as usize, ac, ab);
        half_tri(cy.round() as usize, ac, bc);
    }
}

pub fn line<V, P>(a: Vertex<V>, b: Vertex<V>, mut plot: P)
where
    V: Copy,
    P: FnMut(Fragment<()>)
{
    let (mut a, mut b) = (a.coord, b.coord);
    let (mut dx, mut dy) = (b.x - a.x, b.y - a.y);

    if dx.abs() >= dy.abs() {
        // Angle <= diagonal
        if a.x > b.x { swap(&mut a, &mut b); dx = -dx; }

        let xs = a.x as usize ..= b.x as usize;
        let ys = Bresenham::between(a.y as usize, b.y as usize, dx as usize);

        for coord in xs.zip(ys) {
            plot(Fragment { coord, varying: () });
        }
    } else {
        // Angle > diagonal
        if a.y > b.y { swap(&mut a, &mut b); dy = -dy; }

        let xs = Bresenham::between(a.x as usize, b.x as usize, dy as usize);
        let ys = a.y as usize ..= b.y as usize;

        for coord in xs.zip(ys) {
            plot(Fragment { coord, varying: () });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::*;

    use math::vec::{pt, vec4};

    use super::*;

    fn v(x: f32, y: f32) -> Vertex<()> {
        Vertex { coord: pt(x, y, 0.0), attr: () }
    }

    pub fn vert<V: Copy>(x: f32, y: f32, attr: V) -> Vertex<V> {
        Vertex { coord: vec4(x, y, 0.0, 1.0), attr }
    }

    #[test]
    fn line_all_octants() {
        let endpoints = [
            (10, 0), (10, 3), (10, 10), (7, 10),
            (0, 10), (-4, 10), (-10, 10), (-10, 2),
            (-10, 0), (-10, -9), (-10, -10), (-1, -10),
            (0, -10), (3, -10), (10, -10), (10, -8),
        ];

        for &(ex, ey) in &endpoints {
            let o = 20.0;
            let mut pts = vec![];
            line(v(o, o), v(o + ex as f32, o + ey as f32), |frag| {
                let (x, y) = frag.coord;
                pts.push((x as i32 - o as i32, y as i32 - o as i32));
            });
            if *pts.first().unwrap() != (0, 0) {
                pts.reverse();
            }
            assert_eq!((0, 0), *pts.first().unwrap(), "unexpected first {:?}", pts);
            assert_eq!((ex, ey), *pts.last().unwrap(), "unexpected last {:?}", pts);
        }
    }

    pub struct Buf(pub Vec<u8>, usize, usize);

    const _MENLO_GRADIENT: &[u8; 16] = b" -.,:;=*o%$O&@MW";
    const IBMPC_GRADIENT: &[u8; 16] = b"..-:;=+<*xoXO@MW";

    impl Buf {
        pub fn new(w: usize, h: usize) -> Buf {
            Buf(vec![0; w * h], w, h)
        }

        pub fn put<V: Copy>(&mut self, frag: Fragment<V>, val: f32) {
            self.0[(frag.coord.1 as usize) * self.1 + (frag.coord.0 as usize)] += val as u8;
        }
    }

    impl Display for Buf {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            for i in 0..self.2 {
                f.write_char('\n')?;
                for j in 0..self.1 {
                    let mut c = self.0[i * self.1 + j];

                    c = c.saturating_add((5 * ((i + j) & 3)) as u8); // dither

                    let q = c >> 4;

                    write!(f, "{}", IBMPC_GRADIENT[q as usize] as char)?
                }
            }
            f.write_char('\n')
        }
    }

    #[test]
    fn test_fill() {
        let mut buf = Buf::new(100, 60);
        tri_fill([vert(80.0, 5.0, 0.0), vert(20.0, 15.0, 128.0), vert(50.0, 50.0, 255.0)], |frag| {
            buf.put(frag, 128.0)
        });

        eprintln!("{}", buf);
    }
}
