use std::mem::swap;

use geom::mesh2::GenVertex;
use math::Linear;
use math::vec::Vec4;

use crate::vary::{Bresenham, Varying};

pub mod flat;
pub mod gouraud;

#[derive(Copy, Clone, Debug)]
pub struct Fragment<V, U = ()> {
    pub coord: (usize, usize),
    pub varying: V,
    pub uniform: U,
}

impl<V, U> Fragment<V, U> {
    pub fn varying<W>(&self, v: W) -> Fragment<W, U>
    where U: Copy {
        Fragment {
            varying: v,
            uniform: self.uniform,
            coord: self.coord,
        }
    }
    pub fn uniform<T>(&self, u: T) -> Fragment<V, T>
    where V: Copy {
        Fragment {
            uniform: u,
            varying: self.varying,
            coord: self.coord,
        }
    }
}

fn ysort<V>([a, b, c]: &mut [GenVertex<Vec4, V>; 3]) {
    if a.coord.y > b.coord.y { swap(a, b); }
    if a.coord.y > c.coord.y { swap(a, c); }
    if b.coord.y > c.coord.y { swap(b, c); }
}

pub fn tri_fill<V, P>(mut verts: [GenVertex<Vec4, V>; 3], mut plot: P)
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

            for (x, v) in (x_left..x_right).zip(v) {
                plot(Fragment { coord: (x, y), varying: v, uniform: () });
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

pub fn line<V, P>([mut a, mut b]: [GenVertex<Vec4, V>; 2], mut plot: P)
where
    V: Copy + Linear<f32>,
    P: FnMut(Fragment<V>)
{
    let mut d = b.coord - a.coord;

    if d.x.abs() >= d.y.abs() {
        // Angle <= diagonal
        if d.x < 0.0 { swap(&mut a, &mut b); d.x = -d.x; }

        let xs = a.coord.x as usize ..= b.coord.x as usize;
        let ys = Bresenham::between(a.coord.y as usize, b.coord.y as usize, d.x as usize);
        let vs = Varying::between(a.attr, b.attr, d.x);

        for (coord, varying) in xs.zip(ys).zip(vs) {
            plot(Fragment { coord, varying, uniform: () });
        }
    } else {
        // Angle > diagonal
        if d.y < 0.0 { swap(&mut a, &mut b); d.y = -d.y; }

        let xs = Bresenham::between(a.coord.x as usize, b.coord.x as usize, d.y as usize);
        let ys = a.coord.y as usize ..= b.coord.y as usize;
        let vs = Varying::between(a.attr, b.attr, d.y);

        for (coord, varying) in xs.zip(ys).zip(vs) {
            plot(Fragment { coord, varying, uniform: () });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::*;

    use math::vec::{pt, vec4};

    use super::*;

    type Vert<A> = GenVertex<Vec4, A>;

    fn v(x: f32, y: f32) -> Vert<()> {
        GenVertex { coord: pt(x, y, 0.0), attr: () }
    }

    pub fn vert<V: Copy>(x: f32, y: f32, attr: V) -> Vert<V> {
        GenVertex { coord: vec4(x, y, 0.0, 1.0), attr }
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
            line([v(o, o), v(o + ex as f32, o + ey as f32)], |frag| {
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
