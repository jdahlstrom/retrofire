use std::f32::EPSILON;
use std::mem;

use geom::mesh::Mesh;
use math::{lerp, Linear};
use math::vec::Vec4;

pub fn hidden_surface_removal<VA, FA>(mesh: &mut Mesh<VA, FA>)
where VA: Copy + Linear<f32>, FA: Copy, {
    let Mesh { verts, faces, vertex_attrs, face_attrs, .. } = mesh;

    // Assume roughly 50% of faces visible
    let mut visible_faces = Vec::with_capacity(faces.len() / 2);
    let mut visible_attrs = Vec::with_capacity(faces.len() / 2);

    for (&[a, b, c], &mut fa) in faces.iter().zip(face_attrs) {
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
                if !frontface(&[clipped_verts[0].0, clipped_verts[1].0, clipped_verts[2].0]) {
                    // New faces are coplanar, if any is a backface then all are
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
    mesh.face_attrs = visible_attrs;
}

#[derive(Debug, Eq, PartialEq)]
enum FaceVis {
    Unclipped,
    Clipped,
    Hidden
}

fn face_visibility(face: &[Vec4; 3]) -> FaceVis {
    // TODO Still should improve handling faces that span w=0
    if face.iter().all(|v| v.w <= 0.0) {
        FaceVis::Hidden
    } else if face.iter().all(|v| v.w > 0.0) && !frontface(face) {
        FaceVis::Hidden
    } else if face.iter().all(vertex_in_frustum) {
        FaceVis::Unclipped
    } else {
        FaceVis::Clipped
    }
}

struct ClipPlane(f32, f32);

impl ClipPlane {
    fn inside(&self, c: f32, w: f32) -> bool {
        self.0 * c + self.1 * w > -EPSILON
    }

    fn intersect(&self, (x1, w1): (f32, f32), (x2, w2): (f32, f32)) -> f32 {
        let Self(x, w) = self;
        (w * w1 + x * x1) / ((w * w1 + x * x1) - (w * w2 + x * x2))
    }
}

fn clip<VA>(verts: &[(Vec4, VA)]) -> Vec<(Vec4, VA)>
where VA: Linear<f32> + Copy {
    let mut verts = verts.to_vec();
    let mut verts2 = Vec::with_capacity(8);

    let planes = &[ClipPlane(1.0, 1.0), ClipPlane(-1.0, 1.0)];

    for idx in 0..3 {
        for plane in planes {
            for (&a, &b) in edges(&verts) {
                let vs = intersect(a, b, idx, plane);
                verts2.extend(vs.iter().flatten());
            }
            verts = mem::take(&mut verts2);
        }
    }

    verts
}


fn intersect<VA>(a: (Vec4, VA), b: (Vec4, VA), ci: usize, plane: &ClipPlane)
                 -> [Option<(Vec4, VA)>; 2]
where VA: Copy + Linear<f32>,
{
    let mut res = [None, None];
    if plane.inside(a.0[ci], a.0.w) {
        res[0] = Some(a);
    }
    if plane.inside(a.0[ci], a.0.w) != plane.inside(b.0[ci], b.0.w) {
        // If edge intersects clipping plane,
        // add intersection point as a new vertex
        let t = plane.intersect((a.0[ci], a.0.w), (b.0[ci], b.0.w));
        let o = lerp(t, a, b);
        res[1] = Some(o);
    }
    res
}

fn edges<T>(ts: &[T]) -> impl Iterator<Item=(&T, &T)> {
    (0..ts.len()).map(move |i| (&ts[i], &ts[(i + 1) % ts.len()]))
}

fn frontface(&[a, b, c]: &[Vec4; 3]) -> bool {

    // Compute z component of faces's normal in screen space
    let nz = (b.x/b.w - a.x/a.w) * (c.y/c.w - a.y/a.w)
        - (b.y/b.w - a.y/a.w) * (c.x/c.w - a.x/a.w);

    return nz < 0.0;
}

fn vertex_in_frustum(v: &Vec4) -> bool {
    inside(v.x, v.w)
        && inside(v.y, v.w)
        && inside(v.z, v.w)
}

fn inside(a: f32, w: f32) -> bool {
    w + a > 0.0 && w - a > 0.0
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use FaceVis::*;
    use math::ApproxEq;
    use math::transform::{perspective, translate};
    use math::vec::*;

    use super::*;

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
    fn clip_plane_inside() {
        let p = ClipPlane(1.0, 1.0);

        dbg!(p.inside(2.0, 3.0));
        dbg!(p.inside(-2.0, 3.0));
        dbg!(p.inside(3.0, 2.0));
        dbg!(p.inside(-3.0, 2.0));
        dbg!(p.inside(2.0, 2.0));
        dbg!(p.inside(-2.0, 2.0));

        dbg!(p.inside(1.0, -2.0));
        dbg!(p.inside(-1.0, -2.0));
    }

    #[test]
    fn clip_plane_intersect() {
        let p = ClipPlane(1.0, 1.0);

        dbg!(p.intersect((-2.0, 1.0), (0.0, 1.0)));
        dbg!(p.intersect((-3.0, 0.0), (0.0, 3.0)));
    }

    #[test]
    fn test_asdf() {
        let square = [pt(-2., -1., -2.), pt(-2., -1., 2.), pt(2., -1., 2.), pt(2., -1., -2.)];

        let proj = translate(0.0, 0.0, 2.0) * &perspective(1., 1000.0, 1.0, PI / 2.);

        let proj_square = square.iter().map(|&v| &proj * v).collect::<Vec<_>>();

        dbg!(&proj_square);

        let clipped_square = clip(&proj_square.into_iter().map(|v| (v, ())).collect::<Vec<_>>());

        dbg!(&clipped_square);

        let pdiv_square = clipped_square.into_iter().map(|(v, ())| v / v.w).collect::<Vec<_>>();

        dbg!(&pdiv_square);
    }

    #[test]
    fn test_inside() {
        assert!(inside(0.0, 1.0));
        assert!(inside(0.0, -1.0));

        assert!(inside(10.0, -1.0));
        assert!(inside(-10.0, 1.0));

        assert!(!inside(10.0, 1.0));
        assert!(!inside(-10.0, -1.0));
    }


    #[test]
    fn test_vertex_in_frustum() {
        assert!(vertex_in_frustum(&vec4(0.0, 0.0, 0.0, 0.0)));

        assert!(vertex_in_frustum(&vec4(1.0, 0.0, 0.0, 1.0)));
        assert!(vertex_in_frustum(&vec4(-2.0, 0.0, 0.0, 3.0)));
        assert!(vertex_in_frustum(&vec4(0.0, 1.0, 0.0, 1.0)));
        assert!(vertex_in_frustum(&vec4(0.0, -2.0, 0.0, 3.0)));
        assert!(vertex_in_frustum(&vec4(0.0, 0.0, 1.0, 1.0)));
        assert!(vertex_in_frustum(&vec4(0.0, 0.0, -2.0, 3.0)));

        assert!(!vertex_in_frustum(&vec4(2.0, 0.0, 0.0, 1.0)));
        assert!(!vertex_in_frustum(&vec4(-3.0, 0.0, 0.0, 2.0)));
        assert!(!vertex_in_frustum(&vec4(0.0, 2.0, 0.0, 1.0)));
        assert!(!vertex_in_frustum(&vec4(0.0, -3.0, 0.0, 2.0)));
        assert!(!vertex_in_frustum(&vec4(0.0, 0.0, 2.0, 1.0)));
        assert!(!vertex_in_frustum(&vec4(0.0, 0.0, -3.0, 2.0)));
    }

    #[test]
    fn backface_visibility_hidden() {
        assert_eq!(Hidden, face_visibility(&[X, Y, Z]));
        assert_eq!(Hidden, face_visibility(&[2. * X, X + Y, X]));
        assert_eq!(Hidden, face_visibility(&[X + Z, Y + Z, 2. * Z]));
    }

    #[test]
    fn fully_inside_frontface_unclipped() {
        assert_eq!(Unclipped, face_visibility(&[X + W, Z + W, Y + W]));
        assert_eq!(Unclipped, face_visibility(&[-X + W, -Z + W, -Y + W]));
    }

    #[test]
    fn partially_outside_face_clipped() {
        assert_eq!(Clipped, face_visibility(&[2. * X, Z, Y]));
        assert_eq!(Clipped, face_visibility(&[X, -2. * Z, Y]));
        assert_eq!(Clipped, face_visibility(&[X, Z, 2. * Y]));
    }

    #[test]
    fn fully_outside_face_clipped() {
        assert_eq!(Clipped, face_visibility(&[2. * X, X, Y]));
        assert_eq!(Clipped, face_visibility(&[X - Z, -2. * Z, Y - Z]));
        assert_eq!(Clipped, face_visibility(&[X + Z, 2. * Z, Y + Z]));
    }

    #[test]
    fn clip_triangle_fully_outside() {
        for &(a, b) in &[(X, Y), (Y, Z), (X, Z)] {
            let expected = Vec::<Vtx>::new();
            let actual = clip(&vs(&[2.0 * b, -a + 3.0 * b, a + 3.0 * b]));
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn clip_all_vertices_inside() {
        for &(a, b) in &[(X, Y), (Y, Z), (X, Z)] {
            let expected = vs(&[b + W, -a + W, a + W]);
            let actual = clip(&expected);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn clip_vertices_on_bounds() {
        for &(a, b) in &[(X, Y), (Y, Z), (X, Z)] {
            let expected = vs(&[-a, b, a - b]);
            let actual = clip(&expected);
            assert_eq!(expected, actual);
        }
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
        for &(a, b) in &[(X, Y), (Y, Z), (X, Z)] {
            let expected = vs(&[a + b, a - b, -a - b, -a + b]);
            let actual = clip(&vs(&[-20.0 * (a + b), 20.0 * b, 20.0 * (a - b)]));
            assert_approx_eq(expected, actual)
        }
    }
}
