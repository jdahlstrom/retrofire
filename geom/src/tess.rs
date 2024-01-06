pub fn tessellate(pts: &[Vec2]) -> Vec<Tri<Vec2>> {
    if pts.is_empty() {
        return vec![];
    }
    if pts.len() == 1 || pts.len() == 2 {
        panic!("unexpected len {}", pts.len());
    }
    if pts.len() == 3 {
        return vec![Tri([pts[0], pts[1], pts[2]])];
    }

    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let c = pts[(i + 2) % pts.len()];

        if winding(a, b, c) > 0.0 {
            let mut rest = pts[i + 2..].to_vec();
            rest.extend(&pts[0..=i]);

            let mut res = vec![Tri([a, b, c])];
            res.extend(tessellate(&rest));
            return res;
        }
    }
    unreachable!()
}

fn winding(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    let ab = b - a;
    let bc = c - b;
    ab.x() * bc.y() - ab.y() * bc.x()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winding_test() {
        assert!(winding(vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 2.0)) == 0.0);
        assert!(winding(vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 2.0)) < 0.0);
        assert!(winding(vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 3.0)) > 0.0);
    }

    #[test]
    fn cap_test() {
        dbg!(Prism::tess(&[]));

        let res = Prism::tess(&[
            vec2(0.0, 0.0),
            vec2(1.0, 0.5),
            vec2(2.0, 0.0),
            vec2(1.5, 1.0),
            vec2(2.0, 2.0),
            vec2(1.0, 1.5),
            vec2(0.0, 2.0),
            vec2(0.5, 1.0),
        ]);

        for tri in &res {
            eprintln!("{tri:?}");
        }
        eprintln!("count: {}", res.len())
    }
}
