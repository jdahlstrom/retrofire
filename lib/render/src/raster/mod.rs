use std::mem::swap;

use geom::mesh::Vertex;
use math::Linear;

use crate::vary::{Bresenham, Varying};

mod tests;

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

#[derive(Copy, Clone)]
pub struct Span<V, U> {
    pub y: usize,
    pub xs: (usize, usize),
    pub vs: (V, V),
    pub uni: U,
}

impl<V, U> Span<V, U>
where
    V: Linear<f32> + Copy,
    U: Copy,
{
    pub fn fragments(&self) -> impl Iterator<Item=Fragment<V, U>> + '_ {
        let Self { y, xs: (x0, x1), vs: (v0, v1), uni } = *self;
        let v = Varying::between(v0, v1, (x1 - x0) as _);
        (x0..x1).zip(v)
            .map(move |(x, v)| Fragment {
                coord: (x, y),
                varying: v,
                uniform: uni
            })
    }
}

fn ysort<V>([a, b, c]: &mut [Vertex<V>; 3]) {
    if a.coord.y > b.coord.y { swap(a, b); }
    if a.coord.y > c.coord.y { swap(a, c); }
    if b.coord.y > c.coord.y { swap(b, c); }
}

pub fn tri_fill<V, U, F>(mut verts: [Vertex<V>; 3], uni: U, mut span_fn: F)
where
    V: Linear<f32> + Copy,
    U: Copy,
    F: FnMut(Span<V, U>)
{
    ysort(&mut verts);

    let mut y = verts[0].coord.y.round() as usize;

    let mut half_tri = |y_end,
                        left: &mut Varying<(f32, _)>,
                        right: &mut Varying<(f32, _)>| {

        for (y, (left, right)) in (y..y_end).zip(left.zip(right)) {

            // let v = Varying::between(left.1, right.1, right.0 - left.0);

            let x_left = left.0.round() as usize;
            let x_right = right.0.round() as usize;

            span_fn(Span {
                y,
                xs: (x_left, x_right),
                vs: (left.1, right.1),
                uni,
            });

            /*for (x, v) in (x_left..x_right).zip(v) {
                plot(Fragment { coord: (x, y), varying: v, uniform: () });
            }*/
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

pub fn line<V, U, P>([mut a, mut b]: [Vertex<V>; 2], uniform: U, mut plot: P)
where
    V: Copy + Linear<f32>,
    U: Copy,
    P: FnMut(Fragment<V, U>)
{
    let mut d = b.coord - a.coord;

    if d.x.abs() >= d.y.abs() {
        // Angle <= diagonal
        if d.x < 0.0 { swap(&mut a, &mut b); d.x = -d.x; }

        let xs = a.coord.x as usize ..= b.coord.x as usize;
        let ys = Bresenham::between(a.coord.y as usize, b.coord.y as usize, d.x as usize);
        let vs = Varying::between(a.attr, b.attr, d.x);

        for (coord, varying) in xs.zip(ys).zip(vs) {
            plot(Fragment { coord, varying, uniform });
        }
    } else {
        // Angle > diagonal
        if d.y < 0.0 { swap(&mut a, &mut b); d.y = -d.y; }

        let xs = Bresenham::between(a.coord.x as usize, b.coord.x as usize, d.y as usize);
        let ys = a.coord.y as usize ..= b.coord.y as usize;
        let vs = Varying::between(a.attr, b.attr, d.y);

        for (coord, varying) in xs.zip(ys).zip(vs) {
            plot(Fragment { coord, varying, uniform });
        }
    }
}
