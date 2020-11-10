use std::f32::EPSILON;
use std::mem::swap;

use geom::mesh::Mesh;
use math::{lerp, Linear};
use math::vec::Vec4;

pub fn hidden_surface_removal<VA, FA>(mesh: &mut Mesh<VA, FA>)
where VA: Copy + Linear<f32>, FA: Copy, {

    let Mesh { verts, faces, vertex_attrs, face_attrs, .. } = mesh;

    let vertex_attrs = vertex_attrs.as_mut().unwrap(); // FIXME

    let mut visible_faces = Vec::with_capacity(faces.len() / 2);
    let mut visible_attrs = Vec::with_capacity(faces.len() / 2);

    for (&[a, b, c], &fa) in faces.iter().zip(face_attrs.as_ref().unwrap()) {
        match face_visibility(&[verts[a], verts[b], verts[c]]) {
            FaceVis::Hidden => {
                continue
            },
            FaceVis::Unclipped => {
                visible_faces.push([a, b, c]);
                visible_attrs.push(fa);
            },
            FaceVis::Clipped => {
                let face_verts = [(verts[a], vertex_attrs[a]),
                    (verts[b], vertex_attrs[b]),
                    (verts[c], vertex_attrs[c])];

                let clipped_verts = clip(&face_verts);
                if clipped_verts.is_empty() {
                    continue;
                }
                let cn = clipped_verts.len();
                let vn = verts.len();
                verts.extend(clipped_verts.iter().map(|v| v.0));
                vertex_attrs.extend(clipped_verts.into_iter().map(|v| v.1));

                for i in 1..cn - 1 {
                    visible_faces.push([vn, vn + i, vn + i + 1]);
                    visible_attrs.push(fa);
                }
            }
        }
    }
    mesh.faces = visible_faces;
    mesh.face_attrs = Some(visible_attrs);
}


enum FaceVis {
    Unclipped,
    Clipped,
    Hidden
}

fn face_visibility(face: &[Vec4; 3]) -> FaceVis {
    if !frontface(face) {
        FaceVis::Hidden
    } else if face.iter().all(vertex_in_frustum) {
        FaceVis::Unclipped
    } else {
        FaceVis::Clipped
    }
}

fn clip<VA>(verts: &[(Vec4, VA)]) -> Vec<(Vec4, VA)>
where VA: Linear<f32> + Copy {
    let mut verts = verts.to_vec();
    let mut verts2 = Vec::with_capacity(8);

    for i in 0..3 {
        for &o in &[-1.0, 1.0] {
            verts2.clear();
            for (&a, &b) in edges(&verts) {
                let vs = intersect(a, b, a.0[i], b.0[i], o);
                verts2.extend(vs.iter().flatten());
            }
            swap(&mut verts, &mut verts2);
        }
    }

    verts
}

fn intersect<V>(a: V, b: V, ac: f32, bc: f32, oc: f32) -> [Option<V>; 2]
where V: Copy + Linear<f32> {
    //eprint!("Intersecting {} = {} .. {} with {}: ", c, ac, bc, oc);
    let mut res = [None, None];
    if inside(ac, oc) {
        //eprint!("a = {:?} ", a);
        res[0] = Some(a);
    }
    if inside(ac, oc) != inside(bc, oc) {
        let t = (oc - ac) / (bc - ac);
        let o = lerp(t, a, b);
        //eprint!("o = {:?}", o);
        res[1] = Some(o);
    }
    //eprintln!();
    res
}

fn edges<T>(ts: &[T]) -> impl Iterator<Item=(&T, &T)> {
    (0..ts.len()).map(move |i| (&ts[i], &ts[(i + 1) % ts.len()]))
}

fn frontface(&[a, b, c]: &[Vec4; 3]) -> bool {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) < 0.0
}

fn vertex_in_frustum(v: &Vec4) -> bool {
    inside(v.x.abs(), 1.0)
        && inside(v.y.abs(), 1.0)
        && inside(v.z.abs(), 1.0)
}

fn inside(a: f32, o: f32) -> bool {
    if o > 0.0 {
        a <= o + EPSILON
    } else {
        a >= o - EPSILON
    }
}


#[cfg(test)]
mod tests {
    use math::ApproxEq;

    use super::*;
    use math::vec::*;

    // TODO Test interpolation of vertex attributes

    fn assert_approx_eq(expected: Vec<Vtx>, actual: Vec<Vtx>) {
        assert_eq!(expected.len(), actual.len(), "expected: {:#?}\nactual: {:#?}", expected, actual);
        for (&e, &a) in expected.iter().zip(&actual) {
            assert!(e.0.approx_eq(a.0), "expected: {:?}, actual: {:?}", e, a);
            // TODO assert!(e.1.approx_eq(a.1), "expected: {:?}, actual: {:?}", e, a);
        }
    }

    type Vtx = (Vec4, ());
    fn v(v: Vec4) -> Vtx { (v, ()) }

    fn vs(vs: &[Vec4]) -> Vec<Vtx> {
        vs.iter().copied().map(v).collect()
    }

    #[test]
    fn clip_fully_outside_triangle() {
        let expected = Vec::<Vtx>::new();
        let actual = clip(&vs(&[2.0 * Y, -X + 3.0 * Y, X + 3.0 * Y]));

        assert_eq!(expected, actual);
    }

    #[test]
    fn clip_all_vertices_inside() {
        let expected = vs(&[Y, -X, X]);
        let actual = clip(&expected);

        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn clip_vertices_on_bounds() {
        let expected = vs(&[-X, Y, X - Y]);
        let actual = clip(&expected);
        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn clip_all_vertices_outside() {
        let expected = vs(&[
            0.25 * X + Y,
            X - 0.5 * Y,
            X - Y,
            0.5 * X - Y,
            -X - 0.25 * Y,
            -X + 0.5 * Y,
            -0.5 * X + Y]
        );
        let actual = clip(&vs(&[
            1.5 * Y,
            1.5 * (X - Y),
            -1.5 * X]
        ));

        assert_approx_eq(expected, actual)
    }

    #[test]
    fn clip_screen_filling_triangle() {
        let expected = vs(&[X + Y, X - Y, -X - Y, -X + Y]);
        let actual = clip(&vs(&[-20.0 * (X + Y), 20.0 * Y, 20.0 * (X - Y)]));

        assert_approx_eq(expected, actual)
    }
}
