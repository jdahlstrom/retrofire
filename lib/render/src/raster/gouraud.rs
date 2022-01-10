use geom::mesh::Vertex;

pub type Fragment = super::Fragment<f32>;

pub fn gouraud_fill(vs: [Vertex<f32>; 3], plot: impl FnMut(Fragment)) {
    // TODO super::tri_fill(vs, plot)
}

#[cfg(test)]
mod tests {
    use math::vec::vec4;

    use super::*;
    use super::super::tests::Buf;

    fn plotter(buf: &mut Buf) -> Box<dyn FnMut(Fragment) + '_> {
        Box::new(move |frag| buf.put(frag, frag.varying))
    }

    pub fn v<V: Copy>(x: f32, y: f32, attr: V) -> Vertex<V> {
        Vertex { coord: vec4(x, y, 0.0, 1.0), attr }
    }

    #[test]
    fn test_gouraud_fill_zero() {
        gouraud_fill([v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0)], |frag| {
            assert!(false, "plot called for {:?}", frag)
        });
    }

    #[test]
    fn test_gouraud_fill_1x1() {
        let mut buf = Buf::new(2, 2);
        gouraud_fill(
            [v(0.0, 0.0, 1.0),
            v(0.0, 1.0, 1.0),
            v(1.0, 0.0, 1.0)],
            plotter(&mut buf),
        );

        assert_eq!(buf.0[0], 1);
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_gouraud_fill_2x2() {
        let mut buf = Buf::new(3, 3);
        gouraud_fill(
            [v(0.0, 0.0, 255.0),
            v(2.0, 0.0, 255.0),
            v(0.0, 2.0, 255.0)],
            plotter(&mut buf),
        );

        assert_eq!(buf.to_string(), "\nWW.\nW..\nW..\n");
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_gouraud_fill_5x5() {
        let mut buf = Buf::new(5, 5);

        gouraud_fill(
            [v(2.0, 1.0, 255.0),
            v(1.0, 3.0, 255.0),
            v(4.0, 3.0, 255.0)],
            plotter(&mut buf),
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
        let (a, b, c) = (v(2.0, 1.0, 200.0), v(6.0, 3.0, 250.0), v(1.0, 5.0, 220.0));
        let perms = [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        for vs in &perms {
            let mut buf = Buf::new(7, 7);
            gouraud_fill(*vs, plotter(&mut buf));

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
    fn test_big() {
        let mut buf = Buf::new(80, 40);
        let (a, b, c) = (v(50.0, 35.0, 16.0), v(75.0, 5.0, 255.0), v(5.0, 10.0, 128.0));
        let perms = [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];

        gouraud_fill([a, b, c], plotter(&mut buf));

        for vs in &perms {
            let mut buf2 = Buf::new(80, 40);
            gouraud_fill(*vs, plotter(&mut buf2));

            assert_eq!(buf.to_string(), buf2.to_string());
        }
        eprintln!("{}", buf.to_string());
    }
}
