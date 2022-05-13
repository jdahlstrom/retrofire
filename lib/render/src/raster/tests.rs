#[cfg(test)]
mod tests {
    use std::fmt::*;

    use math::vec::pt;

    use crate::raster::*;

    pub fn v(x: f32, y: f32) -> Vertex<()> {
        va(x, y, ())
    }
    pub fn va<A>(x: f32, y: f32, attr: A) -> Vertex<A> {
        Vertex { coord: pt(x, y, 0.0), attr }
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
            let (x, y) = frag.coord;
            self.0[(y as usize) * self.1 + (x as usize)] += val as u8;
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
        let verts = [
            va(80.0, 5.0, 0.0),
            va(20.0, 15.0, 128.0),
            va(50.0, 50.0, 255.0)
        ];
        tri_fill(verts, |span| {
            for frag in span.fragments() {
                buf.put(frag, frag.varying)
            }
        });

        eprintln!("{}", buf);
    }


    type FlatFrag = Fragment<()>;

    fn flat_fill(verts: [Vertex<()>; 3], mut plot: impl FnMut(FlatFrag)) {
        tri_fill(verts, |span| {
            span.fragments().for_each(&mut plot);
        });
    }

    fn flat_plot(buf: &mut Buf) -> Box<dyn FnMut(FlatFrag) + '_> {
        Box::new(move |frag| buf.put(frag, 255.0))
    }


    #[test]
    fn test_flat_fill_zero() {
        flat_fill([v(0.0, 0.0), v(0.0, 0.0), v(0.0, 0.0)], |frag| {
            assert!(false, "plot called for {:?}", frag)
        });
    }

    #[test]
    fn test_flat_fill_1x1() {
        let mut buf = Buf::new(2, 2);
        flat_fill([v(0.0, 0.0), v(0.0, 1.0), v(1.0, 0.0)], flat_plot(&mut buf));

        assert_eq!(buf.0[0], 255);
    }

    #[test]
    fn test_flat_fill_2x2() {
        let mut buf = Buf::new(3, 3);
        flat_fill([v(0.0, 0.0), v(2.0, 0.0), v(0.0, 2.0)], flat_plot(&mut buf));

        assert_eq!(buf.to_string(), "\nWW.\nW..\n...\n");
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_flat_fill_5x5() {
        let mut buf = Buf::new(5, 5);

        flat_fill([v(2.0, 1.0), v(1.0, 3.0), v(4.0, 3.0)], flat_plot(&mut buf));

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
        let (a, b, c) = (v(2.0, 1.0), v(6.0, 3.0), v(1.0, 5.0));
        let perms = [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        for vs in &perms {
            let mut buf = Buf::new(7, 7);
            flat_fill(*vs, flat_plot(&mut buf));

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
    fn test_flat_fill_big() {
        let mut buf = Buf::new(80, 40);
        let (a, b, c) = (v(50.0, 35.0), v(75.0, 5.0), v(5.0, 10.0));
        let perms = [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        flat_fill([a, b, c], flat_plot(&mut buf));

        for vs in &perms {
            let mut buf2 = Buf::new(80, 40);
            flat_fill(*vs, flat_plot(&mut buf2));

            assert_eq!(buf.to_string(), buf2.to_string());
        }
        eprintln!("{}", buf.to_string().replace("0", " ").replace("1", "#"));
    }


    type GourFrag = Fragment<f32>;

    fn gouraud_fill(vs: [Vertex<f32>; 3], mut plot: impl FnMut(GourFrag)) {
        tri_fill(vs, |span| {
            span.fragments().for_each(&mut plot);
        })
    }


    fn gour_plot(buf: &mut Buf) -> Box<dyn FnMut(GourFrag) + '_> {
        Box::new(move |frag| buf.put(frag, frag.varying))
    }

    #[test]
    fn test_gouraud_fill_zero() {
        gouraud_fill([va(0.0, 0.0, 0.0), va(0.0, 0.0, 0.0), va(0.0, 0.0, 0.0)], |frag| {
            assert!(false, "plot called for {:?}", frag)
        });
    }

    #[test]
    fn test_gouraud_fill_1x1() {
        let mut buf = Buf::new(2, 2);
        gouraud_fill(
            [va(0.0, 0.0, 1.0), va(0.0, 1.0, 1.0), va(1.0, 0.0, 1.0)],
            gour_plot(&mut buf),
        );

        assert_eq!(buf.0[0], 1);
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_gouraud_fill_2x2() {
        let mut buf = Buf::new(3, 3);
        gouraud_fill(
            [va(0.0, 0.0, 255.0), va(2.0, 0.0, 255.0), va(0.0, 2.0, 255.0)],
            gour_plot(&mut buf),
        );

        assert_eq!(buf.to_string(), "\nWW.\nW..\nW..\n");
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_gouraud_fill_5x5() {
        let mut buf = Buf::new(5, 5);

        gouraud_fill(
            [va(2.0, 1.0, 255.0), va(1.0, 3.0, 255.0), va(4.0, 3.0, 255.0)],
            gour_plot(&mut buf),
        );

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
    fn test_gouraud_fill_7x7() {
        let (a, b, c) = (va(2.0, 1.0, 200.0), va(6.0, 3.0, 250.0), va(1.0, 5.0, 220.0));
        let perms = [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        for vs in &perms {
            let mut buf = Buf::new(7, 7);
            gouraud_fill(*vs, gour_plot(&mut buf));

            assert_eq!(
                buf.to_string(),
                "\n\
                 .......\n\
                 ..M....\n\
                 ..@MW..\n\
                 .@MWWW.\n\
                 .MWW...\n\
                 .MW....\n\
                 .......\n"
            );
        }
    }

    #[test]
    fn gouraud_big() {
        let mut buf = Buf::new(80, 40);
        let (a, b, c) = (va(50.0, 35.0, 16.0), va(75.0, 5.0, 255.0), va(5.0, 10.0, 128.0));
        let perms = [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        gouraud_fill([a, b, c], gour_plot(&mut buf));

        for vs in &perms {
            let mut buf2 = Buf::new(80, 40);
            gouraud_fill(*vs, gour_plot(&mut buf2));

            assert_eq!(buf.to_string(), buf2.to_string());
        }
        eprintln!("{}", buf.to_string());
    }
}
