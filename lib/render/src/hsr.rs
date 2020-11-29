use std::f32::EPSILON;
use std::mem;
use std::option::Option::Some;

use geom::mesh::Mesh;
use math::{lerp, Linear};
use math::vec::Vec4;

pub fn hidden_surface_removal<VA, FA>(mesh: &mut Mesh<VA, FA>)
where VA: Copy + Linear<f32>, FA: Copy, {
    let Mesh { verts, faces, vertex_attrs, face_attrs, .. } = mesh;

    let clip_masks = verts.iter().map(clip_vertex).collect::<Vec<_>>();

    // Assume roughly 50% of faces visible
    let mut visible_faces = Vec::with_capacity(faces.len() / 2);
    let mut visible_attrs = Vec::with_capacity(faces.len() / 2);

    for (&[a, b, c], &mut fa) in faces.iter().zip(face_attrs) {

        let masks = [clip_masks[a], clip_masks[b], clip_masks[c]];

        match face_visibility(&[verts[a], verts[b], verts[c]], &masks) {
            FaceVis::Hidden | FaceVis::Backface => {
                continue
            },
            FaceVis::Unclipped => {
                visible_faces.push([a, b, c]);
                visible_attrs.push(fa);
            },
            FaceVis::Clipped => {
                let face_verts = [
                    (verts[a], vertex_attrs[a]),
                    (verts[b], vertex_attrs[b]),
                    (verts[c], vertex_attrs[c])
                ];
                let clipped_verts = clip(&face_verts);
                if clipped_verts.is_empty() {
                    continue;
                }
                if !frontface(&[clipped_verts[0].0,
                                clipped_verts[1].0,
                                clipped_verts[2].0]) {
                    // New faces are coplanar, if one is a backface then all are
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

pub fn bbox_test(verts: &[Vec4]) -> u8 {
   // return 1;
    let bb_any = verts.iter().map(clip_vertex).fold(0, |a, b| a | b);
    let bb_all = verts.iter().map(clip_vertex).fold(!0, |a, b| a & b);

    //eprintln!("an {:06b} al {:06b}", bb_any, bb_all);

    if bb_all == clip_mask::ALL {
        return 2; // fully inside
    }

    if bb_any != clip_mask::ALL {
        //eprintln!("Mesh culled! {:?}", bbox);
        return 0;
    }
    return 1;
}

struct ClipPlane(f32, f32);

impl ClipPlane {
    fn inside(&self, x: f32, w: f32) -> bool {
        let Self(x1, w1) = self;
        w * w1 + x * x1 > -EPSILON
    }
    fn intersect(&self, (x1, w1): (f32, f32), (x2, w2): (f32, f32)) -> Option<f32> {
        let Self(x, w) = self;
        let d1 = w * w1 + x * x1;
        let d2 = w * w2 + x * x2;

        if d1.signum() != d2.signum() {
            Some(d1 / (d1 - d2))
        } else {
            None
        }
    }
}

const CLIP_PLANES: [ClipPlane; 2] = [ClipPlane(1.0, 1.0), ClipPlane(-1.0, 1.0)];

mod clip_mask {
    pub const LEFT: u8 = 0b1;
    pub const RIGHT: u8 = 0b10;
    pub const BOTTOM: u8 = 0b100;
    pub const TOP: u8 = 0b1000;
    pub const NEAR: u8 = 0b10000;
    pub const FAR: u8 = 0b100000;

    pub const ALL: u8 = 0b111111;
}

#[derive(Debug, Eq, PartialEq)]
enum FaceVis {
    Unclipped,
    Clipped,
    Hidden,
    Backface,
}

fn clip_vertex(v: &Vec4) -> u8 {
    use clip_mask::*;
    CLIP_PLANES[0].inside(v.x, v.w) as u8 * LEFT
        | CLIP_PLANES[1].inside(v.x, v.w) as u8 * RIGHT
        | CLIP_PLANES[0].inside(v.y, v.w) as u8 * BOTTOM
        | CLIP_PLANES[1].inside(v.y, v.w) as u8 * TOP
        | CLIP_PLANES[0].inside(v.z, v.w) as u8 * NEAR
        | CLIP_PLANES[1].inside(v.z, v.w) as u8 * FAR
}

fn face_visibility(verts: &[Vec4; 3], masks: &[u8; 3]) -> FaceVis {

    // Mask of planes that all verts are inside of
    let all_verts_inside = masks[0] & masks[1] & masks[2];
    // Mask of planes that at least one vert is inside of
    let any_vert_inside = masks[0] | masks[1] | masks[2];

    if any_vert_inside != clip_mask::ALL {
        // Face hidden if at least one plane that no vert is inside of
        FaceVis::Hidden
    } else if (all_verts_inside & clip_mask::NEAR != 0) && !frontface(verts) {
        // Backfaces hidden even if inside frustum
        FaceVis::Backface
    } else if all_verts_inside == clip_mask::ALL {
        // If all vertices inside the frustum, the face is unclipped
        FaceVis::Unclipped
    } else {
        FaceVis::Clipped
    }
}

fn clip<VA>(verts: &[(Vec4, VA)]) -> Vec<(Vec4, VA)>
where VA: Linear<f32> + Copy {
    let mut verts = verts.to_vec();
    let mut verts2 = Vec::with_capacity(8);

    for idx in 0..3 {
        for plane in &CLIP_PLANES {
            for (&a, &b) in edges(&verts) {
                let vs = intersect(a, b, idx, plane);
                verts2.extend(vs.iter().flatten());
            }
            verts = mem::take(&mut verts2);
        }
    }
    verts
}


fn intersect<VA>(v1: (Vec4, VA), v2: (Vec4, VA), ci: usize, plane: &ClipPlane)
                 -> [Option<(Vec4, VA)>; 2]
where VA: Copy + Linear<f32>,
{
    let c1 = v1.0;
    let c2 = v2.0;
    let mut res = [None, None];
    // TODO Use clip mask information already computed
    if plane.inside(c1[ci], c1.w) {
        res[0] = Some(v1);
    }
    if let Some(t) = plane.intersect((c1[ci], c1.w), (c2[ci], c2.w)) {
        // If edge intersects clipping plane,
        // add intersection point as a new vertex
        let o = lerp(t, v1, v2);
        res[1] = Some(o);
    }
    res
}

fn edges<T>(ts: &[T]) -> impl Iterator<Item=(&T, &T)> {
    (0..ts.len()).map(move |i| (&ts[i], &ts[(i + 1) % ts.len()]))
}

pub fn frontface(&[a, b, c]: &[Vec4; 3]) -> bool {
    debug_assert!(a.w != 0.0 && b.w != 0.0 && c.w != 0.0, "{:?}", (a,b,c));

    // Compute z component of faces's normal in screen space
    let nz = (b.x / b.w - a.x / a.w) * (c.y / c.w - a.y / a.w)
        - (b.y / b.w - a.y / a.w) * (c.x / c.w - a.x / a.w);

    // Count degenerate faces (nz==0) as front, at least for now
    return nz <= 0.0;
}

#[cfg(test)]
mod tests {
    use FaceVis::*;
    use math::ApproxEq;
    use math::vec::*;

    use super::*;
    use super::clip_mask::*;

// TODO Test interpolation of vertex attributes

    fn assert_approx_eq(expected: Vec<Vtx>, actual: Vec<Vtx>) {
        assert_eq!(expected.len(), actual.len(), "expected: {:#?}\nactual: {:#?}", expected, actual);
        for (&e, &a) in expected.iter().zip(&actual) {
            assert!(e.0.approx_eq(a.0), "expected: {:?}, actual: {:?}", e, a);
            // TODO assert!(e.1.approx_eq(a.1), "expected: {:?}, actual: {:?}", e, a);
        }
    }

    type Vtx = (Vec4, ());

    fn v(v: Vec4) -> Vtx { (v+W, ()) }

    fn vs(vs: &[Vec4]) -> Vec<Vtx> {
        vs.iter().copied().map(v).collect()
    }


    #[test]
    fn clip_plane_inside() {
        let p = ClipPlane(1.0, 1.0);

        assert!(p.inside(2.0, 3.0));
        assert!(p.inside(-2.0, 3.0));
        assert!(p.inside(3.0, 2.0));
        assert!(!p.inside(-3.0, 2.0));
        assert!(p.inside(2.0, 2.0));
        assert!(p.inside(-2.0, 2.0));

        assert!(!p.inside(2.0, -3.0));
        assert!(!p.inside(-2.0, -3.0));
        assert!(p.inside(3.0, -2.0));
    }

    #[test]
    fn clip_plane_intersect() {
        let p = ClipPlane(1.0, 1.0);

        assert_eq!(Some(0.5), p.intersect((-2.0, 1.0), (0.0, 1.0)));
        assert_eq!(Some(0.5), p.intersect((-3.0, 0.0), (0.0, 3.0)));

        assert_eq!(None, p.intersect((1.0, 2.0), (2.0, 3.0)));
        assert_eq!(None, p.intersect((-3.0, 2.0), (-2.0, 1.0)));
    }

    #[test]
    fn clip_vertex_inside_frustum() {
        assert_eq!(ALL, clip_vertex(&vec4(0.0, 0.0, 0.0, 0.0)));

        assert_eq!(ALL, clip_vertex(&vec4(1.0, 0.0, 0.0, 1.0)));
        assert_eq!(ALL, clip_vertex(&vec4(-2.0, 0.0, 0.0, 3.0)));
        assert_eq!(ALL, clip_vertex(&vec4(0.0, 1.0, 0.0, 1.0)));
        assert_eq!(ALL, clip_vertex(&vec4(0.0, -2.0, 0.0, 3.0)));
        assert_eq!(ALL, clip_vertex(&vec4(0.0, 0.0, 1.0, 1.0)));
        assert_eq!(ALL, clip_vertex(&vec4(0.0, 0.0, -2.0, 3.0)));
    }

    #[test]
    fn clip_vertex_outside_single_plane() {
        assert_eq!(ALL & !RIGHT, clip_vertex(&vec4(2.0, 0.0, 0.0, 1.0)));
        assert_eq!(ALL & !LEFT, clip_vertex(&vec4(-3.0, 0.0, 0.0, 2.0)));
        assert_eq!(ALL & !TOP, clip_vertex(&vec4(0.0, 2.0, 0.0, 1.0)));
        assert_eq!(ALL & !BOTTOM, clip_vertex(&vec4(0.0, -3.0, 0.0, 2.0)));
        assert_eq!(ALL & !FAR, clip_vertex(&vec4(0.0, 0.0, 2.0, 1.0)));
        assert_eq!(ALL & !NEAR, clip_vertex(&vec4(0.0, 0.0, -3.0, 2.0)));
    }

    #[test]
    fn clip_vertex_outside_several_planes() {
        assert_eq!(ALL & !(RIGHT|NEAR), clip_vertex(&vec4(2.0, 0.0, -3.0, 1.0)));
        assert_eq!(ALL & !(LEFT|FAR|TOP), clip_vertex(&vec4(-3.0, 4.0, 5.0, 2.0)));
    }


    fn assert_vis(vis: FaceVis, [a,b,c]: [Vec4; 3]) {
        let verts = [a+W, b+W, c+W];
        let [a,b,c] = verts;
        let masks = [
            clip_vertex(&a),
            clip_vertex(&b),
            clip_vertex(&c)];
        assert_eq!(vis, face_visibility(&verts, &masks), "verts: {:?}", verts)
    }

    #[test]
    fn backface_visibility_hidden() {
        assert_vis(Backface, [X, Y, Z]);
        assert_vis(Backface, [2. * X, X + Y, X]);
        assert_vis(Backface, [X + Z, Y + Z, 2. * Z]);
    }

    #[test]
    fn fully_inside_frontface_unclipped() {
        assert_vis(Unclipped, [X, Z, Y]);
        assert_vis(Unclipped, [X, Z, Y]);
        assert_vis(Unclipped, [-X, -Z, -Y]);

    }

    #[test]
    fn partially_outside_face_clipped() {
        assert_vis(Clipped, [2. * X, Z, Y]);
        assert_vis(Clipped, [X, -2. * Z, Y]);
        assert_vis(Clipped, [X, Z, 2. * Y]);
    }

    #[test]
    fn fully_outside_face_clipped() {
        assert_vis(Clipped, [2. * X, X, Y]);
        assert_vis(Clipped, [X - Z, -2. * Z, Y - Z]);
        assert_vis(Clipped, [X + Z, 2. * Z, Y + Z]);
    }

    #[test]
    fn fully_outside_single_plane_hidden() {
        assert_vis(Hidden, [-2. * X, -3. * X, -1.1 * X + Y]);
        assert_vis(Hidden, [2. * X, 1.1 * X, 1.1 * X + Y]);
        assert_vis(Hidden, [1.1 * Y, 2. * Y, X + 2. * Y]);
        assert_vis(Hidden, [1.1 * Z, 1.1 * Z + Y, 1.1 * Z + X]);
        assert_vis(Hidden, [-1.1 * Z, -1.1 * Z + Y, -1.1 * Z + X]);
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
            let expected = vs(&[b, -a, a]);
            let actual = clip(&expected);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn clip_vertices_on_bounds() {
        for &(a, b) in &[(X, Y), (Y, Z), (X, Z)] {
            let expected = vs(&[-a, b, a - b]);
            let actual = clip(&expected);
            assert_approx_eq(expected, actual);
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
