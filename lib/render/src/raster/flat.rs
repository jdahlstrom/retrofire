use geom::mesh::Vertex;

use crate::Scanline;
use crate::Varying;

#[inline]
fn half_tri(
    y: usize,
    y_end: usize,
    left: &mut Varying<f32>,
    right: &mut Varying<f32>,
    sc: &mut impl FnMut(Scanline<()>),
) {
    /*let width = 4 * fb.color.width();
    let cb = fb.color.data_mut();
    let [_, r, g, b] = col.to_argb();


    let left = y + 4 * x_left as usize;
    let right = y + 4 * x_right as usize;
    let xline = &mut cb[left..right];

    for pix in xline.chunks_exact_mut(4) {
        pix[0] = b;
        pix[1] = g;
        pix[2] = r;
    }*/

    for y in y..y_end {

        let xs = left.next().unwrap() as usize..right.next().unwrap() as usize;

        sc(Scanline { y, xs, vs: ()..() });
    }
}

pub fn flat_fill(mut verts: [Vertex<()>; 3],
                 ref mut sc: impl FnMut(Scanline<()>)) {
    super::ysort(&mut verts);

    let verts @ [(ax, ay), (bx, by), (cx, cy)]
        = verts.map(|v| (v.coord.x, v.coord.y));

    let ab = &mut Varying::between(ax, bx, by - ay);
    let ac = &mut Varying::between(ax, cx, cy - ay);
    let bc = &mut Varying::between(bx, cx, cy - by);

    let [ay, by, cy] = verts.map(|v| v.1.round() as usize);

    if ab.step < ac.step {
        half_tri(ay, by, ab, ac, sc);
        half_tri(ay, cy, bc, ac, sc);
    } else {
        half_tri(ay, by, ac, ab, sc);
        half_tri(by, cy, ac, bc, sc);
    }
}

#[cfg(test)]
mod tests {
    use crate::raster::tests::Buf;

    use super::*;

    fn vert(x: f32, y: f32) -> Vertex<()> {
        crate::raster::tests::vert(x, y, ())
    }

    fn plotter(buf: &mut Buf) -> Box<dyn FnMut(Scanline<()>) + '_> {
        Box::new(move |sc| for frag in sc.fragments() { buf.put(frag, 255.0) })
    }

    #[test]
    fn test_flat_fill_zero() {
        flat_fill([vert(0.0, 0.0), vert(0.0, 0.0), vert(0.0, 0.0)], |x| {
            assert!(false, "plot called for {:?}", x)
        });
    }

    #[test]
    fn test_flat_fill_1x1() {
        let mut buf = Buf::new(2, 2);
        flat_fill([vert(0.0, 0.0), vert(0.0, 1.0), vert(1.0, 0.0)], plotter(&mut buf));

        assert_eq!(buf.0[0], 255);
    }

    #[test]
    fn test_flat_fill_2x2() {
        let mut buf = Buf::new(3, 3);
        flat_fill([vert(0.0, 0.0), vert(2.0, 0.0), vert(0.0, 2.0)], plotter(&mut buf));

        assert_eq!(buf.to_string(), "\nWWW\nWW.\n...\n");
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_flat_fill_5x5() {
        let mut buf = Buf::new(5, 5);

        flat_fill([vert(2.0, 1.0), vert(1.0, 3.0), vert(4.0, 3.0)], plotter(&mut buf));

        assert_eq!(
            buf.to_string(),
            "\n\
             .....\n\
             ..W..\n\
             .WW..\n\
             .WWW.\n\
             .....\n"
        );
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_flat_fill_7x7() {
        let (a, b, c) = (vert(2.0, 1.0), vert(6.0, 3.0), vert(1.0, 5.0));
        let perms = [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        for vs in &perms {
            let mut buf = Buf::new(7, 7);
            flat_fill(*vs, plotter(&mut buf));

            assert_eq!(
                buf.to_string(),
                "\n\
                 .......\n\
                 ..W....\n\
                 ..WWW..\n\
                 .WWWWW.\n\
                 .WWW...\n\
                 .WW....\n\
                 .......\n"
            );
        }
    }

    #[test]
    fn test_big() {
        let mut buf = Buf::new(80, 40);
        let (a, b, c) = (vert(50.0, 35.0), vert(75.0, 5.0), vert(5.0, 10.0));
        let perms = [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        flat_fill([a, b, c], plotter(&mut buf));

        for vs in &perms {
            let mut buf2 = Buf::new(80, 40);
            flat_fill(*vs, plotter(&mut buf2));

            assert_eq!(buf.to_string(), buf2.to_string());
        }
        eprintln!("{}", buf.to_string().replace("0", " ").replace("1", "#"));
    }
}
