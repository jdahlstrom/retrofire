pub type Fragment = super::Fragment<()>;

pub fn flat_fill(a: Fragment, b: Fragment, c: Fragment, plot: impl FnMut(Fragment)) {
    super::tri_fill(a, b, c, plot)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::tests::Buf;

    fn frag(x: f32, y: f32) -> Fragment {
        crate::raster::tests::frag(x, y, ())
    }

    fn plotter(buf: &mut Buf) -> Box<dyn FnMut(Fragment) + '_> {
        Box::new(move |frag| buf.put(frag, 255.0))
    }

    #[test]
    fn test_flat_fill_zero() {
        flat_fill(frag(0.0, 0.0), frag(0.0, 0.0), frag(0.0, 0.0), |frag| {
            assert!(false, "plot called for {:?}", frag)
        });
    }

    #[test]
    fn test_flat_fill_1x1() {
        let mut buf = Buf::new(2, 2);
        flat_fill(frag(0.0, 0.0), frag(0.0, 1.0), frag(1.0, 0.0), plotter(&mut buf));

        assert_eq!(buf.0[0], 255);
    }

    #[test]
    fn test_flat_fill_2x2() {
        let mut buf = Buf::new(3, 3);
        flat_fill(frag(0.0, 0.0), frag(2.0, 0.0), frag(0.0, 2.0), plotter(&mut buf));

        assert_eq!(buf.to_string(), "\nWWW\nWW.\n...\n");
    }

    #[test]
    #[ignore] // TODO FIX TEST
    fn test_flat_fill_5x5() {
        let mut buf = Buf::new(5, 5);

        flat_fill(frag(2.0, 1.0), frag(1.0, 3.0), frag(4.0, 3.0), plotter(&mut buf));

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
        let (a, b, c) = (frag(2.0, 1.0), frag(6.0, 3.0), frag(1.0, 5.0));
        let perms = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)];

        for (a, b, c) in &perms {
            let mut buf = Buf::new(7, 7);
            flat_fill(*a, *b, *c, plotter(&mut buf));

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
        let (a, b, c) = (frag(50.0, 35.0), frag(75.0, 5.0), frag(5.0, 10.0));
        let perms = [(a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)];

        flat_fill(a, b, c, plotter(&mut buf));

        for &(a, b, c) in &perms {
            let mut buf2 = Buf::new(80, 40);
            flat_fill(a, b, c, plotter(&mut buf2));

            assert_eq!(buf.to_string(), buf2.to_string());
        }
        eprintln!("{}", buf.to_string().replace("0", " ").replace("1", "#"));
    }
}
