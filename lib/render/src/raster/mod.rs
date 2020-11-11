use std::mem::swap;

use math::Linear;
use math::vec::{vec4, Vec4};

use crate::vary::Varying;

pub mod flat;
pub mod gouraud;

#[derive(Copy, Clone, Debug)]
pub struct Fragment<V: Copy> {
    pub coord: Vec4,
    pub varying: V,
}

fn ysort<V: Copy>(a: &mut Fragment<V>, b: &mut Fragment<V>, c: &mut Fragment<V>) {
    if a.coord.y > b.coord.y { swap(a, b); }
    if a.coord.y > c.coord.y { swap(a, c); }
    if b.coord.y > c.coord.y { swap(b, c); }
}

pub fn tri_fill<V>(mut a: Fragment<V>, mut b: Fragment<V>, mut c: Fragment<V>,
                   mut plot: impl FnMut(Fragment<V>))
where
    V: Linear<f32> + Copy,
{
    ysort(&mut a, &mut b, &mut c);

    let (a, av, b, bv, c, cv) = (a.coord, a.varying, b.coord, b.varying, c.coord, c.varying);

    let mut y = a.y.round();

    let mut half_tri = |y_end,
                        x_left: &mut Varying<f32>, v_left: &mut Varying<_>,
                        x_right: &mut Varying<f32>, v_right: &mut Varying<_>| {
        while y < y_end {
            let x_end = x_right.step().round();

            let mut x = x_left.step().round();
            let mut v = Varying::between(v_left.step(), v_right.step(), x_end - x);

            while x <= x_end {
                let (z, v) = v.step();
                plot(Fragment {
                    coord: vec4(x, y, z, 1.0), // TODO w?
                    varying: v,
                });
                x += 1.0;
            }
            y += 1.0;
        }
    };

    let mut x_ab = Varying::between(a.x, b.x, b.y - a.y);
    let mut v_ab = Varying::between((a.z, av), (b.z, bv), b.y - a.y);

    let mut x_ac = Varying::between(a.x, c.x, c.y - a.y);
    let mut v_ac = Varying::between((a.z, av), (c.z, cv), c.y - a.y);

    let mut x_bc = Varying::between(b.x, c.x, c.y - b.y);
    let mut v_bc = Varying::between((b.z, bv), (c.z, cv), c.y - b.y);

    if x_ab.step < x_ac.step {
        half_tri(b.y.round(), &mut x_ab, &mut v_ab, &mut x_ac, &mut v_ac);
        half_tri(c.y.round(), &mut x_bc, &mut v_bc, &mut x_ac, &mut v_ac);
    } else {
        half_tri(b.y.round(), &mut x_ac, &mut v_ac, &mut x_ab, &mut v_ab);
        half_tri(c.y.round(), &mut x_ac, &mut v_ac, &mut x_bc, &mut v_bc);
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::*;

    use super::*;

    pub struct Buf(pub Vec<u8>, usize, usize);

    const _MENLO_GRADIENT: &[u8; 16] = b" -.,:;=*o%$O&@MW";
    const IBMPC_GRADIENT: &[u8; 16] = b"..-:;=+<*xoXO@MW";

    pub fn frag<V: Copy>(x: f32, y: f32, varying: V) -> Fragment<V> {
        Fragment { coord: vec4(x, y, 0.0, 1.0), varying }
    }

    impl Buf {
        pub fn new(w: usize, h: usize) -> Buf {
            Buf(vec![0; w * h], w, h)
        }

        pub fn put<V: Copy>(&mut self, frag: Fragment<V>, val: f32) {
            self.0[(frag.coord.y as usize) * self.1 + (frag.coord.x as usize)] += val as u8;
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
        tri_fill(frag(80.0, 5.0, 0.0), frag(20.0, 15.0, 128.0), frag(50.0, 50.0, 255.0), |frag| {
            buf.put(frag, 128.0)
        });

        eprintln!("{}", buf);
    }
}
