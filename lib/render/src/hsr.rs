use std::f32::EPSILON;
use std::mem;
use std::option::Option::Some;

use geom::mesh::{Mesh, Vertex};
use math::{lerp, Linear};
use math::vec::Vec4;

pub fn hidden_surface_removal<VA, FA>(mesh: &mut Mesh<VA, FA>, bbox_vis: Visibility)
where VA: Copy + Linear<f32>, FA: Copy, {

    let clip_masks = mesh.verts.iter().map(vertex_mask).collect::<Vec<_>>();

    let mut mesh2 = Mesh {
        verts: Vec::with_capacity(16),
        vertex_attrs: Vec::with_capacity(16),
        // Assume roughly 50% of faces visible
        faces: Vec::with_capacity(mesh.faces.len() / 2),
        face_attrs: Vec::with_capacity(mesh.faces.len() / 2),
        ..*mesh
    };

    for face in mesh.faces() {
        if bbox_vis == Unclipped {
            if frontface(&face.verts) {
                mesh2.faces.push(face.indices);
                mesh2.face_attrs.push(face.attr);
            }
            continue;
        }

        let [a, b, c] = face.indices;
        let masks = [clip_masks[a], clip_masks[b], clip_masks[c]];

        match visibility(masks.iter().copied()) {
            Hidden => {
                continue
            },
            Unclipped => {
                if frontface(&face.verts) {
                    mesh2.faces.push(face.indices);
                    mesh2.face_attrs.push(face.attr);
                }
            },
            Clipped => {
                let clipped_verts = clip(&face.verts);
                if clipped_verts.is_empty() {
                    continue;
                }
                if !frontface(&[clipped_verts[0], clipped_verts[1], clipped_verts[2]]) {
                    // New faces are coplanar, if one is a backface then all are
                    continue;
                }
                let cn = clipped_verts.len();
                let vn = mesh.verts.len() + mesh2.verts.len();
                mesh2.verts.extend(clipped_verts.iter().map(|v| v.coord));
                mesh2.vertex_attrs.extend(clipped_verts.into_iter().map(|v| v.attr));

                for i in 1..cn - 1 {
                    mesh2.faces.push([vn, vn + i, vn + i + 1]);
                    mesh2.face_attrs.push(face.attr);
                }
            }
        }
    }
    mesh.faces = mesh2.faces;
    mesh.face_attrs = mesh2.face_attrs;
    mesh.verts.extend(&mesh2.verts);
    mesh.vertex_attrs.extend(&mesh2.vertex_attrs);
}

struct ClipPlane { sign: f32, coord: u8, bit: u8 }

impl ClipPlane {
    fn test(&self, v: &Vec4) -> u8 {
        (self.signed_dist(v) > -EPSILON) as u8 * self.bit
    }
    fn clip<VA>(&self, a: &Vertex<VA>, b: &Vertex<VA>)
        -> [Option<Vertex<VA>>; 2]
    where VA: Copy + Linear<f32>,
    {
        let mut res = [None, None];
        let da = self.signed_dist(&a.coord);
        let db = self.signed_dist(&b.coord);
        if da > -EPSILON {
            res[0] = Some(*a);
        }
        if da.signum() != db.signum() {
            // If edge intersects clipping plane,
            // add intersection point as a new vertex
            let t = da / (da - db);
            res[1] = Some(Vertex {
                coord: lerp(t, a.coord, b.coord),
                attr: lerp(t, a.attr, b.attr),
            });
        }
        res
    }
    fn signed_dist(&self, v: &Vec4) -> f32 {
        self.sign * v[self.coord as usize] + v[3]
    }
}

mod clip_mask {
    pub const LEFT: u8 = 0b1;
    pub const RIGHT: u8 = 0b10;
    pub const BOTTOM: u8 = 0b100;
    pub const TOP: u8 = 0b1000;
    pub const NEAR: u8 = 0b10000;
    pub const FAR: u8 = 0b100000;

    pub const ALL: u8 = 0b111111;
}

const CLIP_PLANES: [ClipPlane; 6] = [
    ClipPlane { sign: 1.0, coord: 0, bit: clip_mask::LEFT, },
    ClipPlane { sign: -1.0, coord: 0, bit: clip_mask::RIGHT, },
    ClipPlane { sign: 1.0, coord: 1, bit: clip_mask::BOTTOM, },
    ClipPlane { sign: -1.0, coord: 1, bit: clip_mask::TOP, },
    ClipPlane { sign: 1.0, coord: 2, bit: clip_mask::NEAR, },
    ClipPlane { sign: -1.0, coord: 2, bit: clip_mask::FAR, },
];

#[derive(Debug, Eq, PartialEq)]
pub enum Visibility {
    Unclipped,
    Clipped,
    Hidden
}
use Visibility::*;

fn vertex_mask(v: &Vec4) -> u8 {
      CLIP_PLANES.iter().fold(0, |mask, p| mask | p.test(v))
}

fn visibility(masks: impl IntoIterator<Item=u8>) -> Visibility {
    let (all_inside, any_inside) = masks.into_iter()
        .fold((!0, 0), |(a, b), v| (a & v, b | v));

    if all_inside == clip_mask::ALL {
        Unclipped
    } else if any_inside != clip_mask::ALL {
        Hidden
    } else {
        Clipped
    }
}

pub fn vertex_visibility<'a>(verts: impl IntoIterator<Item=&'a Vec4>) -> Visibility {
    visibility(verts.into_iter().map(vertex_mask))
}

pub fn clip<VA>(verts: &[Vertex<VA>]) -> Vec<Vertex<VA>>
where
    VA: Linear<f32> + Copy
{
    let mut verts = verts.to_vec();
    let mut verts2 = Vec::with_capacity(8);

    for plane in &CLIP_PLANES {
        for (a, b) in edges(&verts) {
            let vs = plane.clip(a, b);
            verts2.extend(vs.iter().flatten());
        }
        verts = mem::take(&mut verts2);
    }
    verts
}

fn edges<T>(ts: &[T]) -> impl Iterator<Item=(&T, &T)> {
    (0..ts.len()).map(move |i| (&ts[i], &ts[(i + 1) % ts.len()]))
}

pub fn frontface<A>(verts: &[Vertex<A>; 3]) -> bool {
    let (a, b, c) = (verts[0].coord, verts[1].coord, verts[2].coord);
    debug_assert!(a.w != 0.0 && b.w != 0.0 && c.w != 0.0, "{:?}", (a,b,c));

    // Compute z component of faces's normal in screen space
    let nz = (b.x / b.w - a.x / a.w) * (c.y / c.w - a.y / a.w)
        - (b.y / b.w - a.y / a.w) * (c.x / c.w - a.x / a.w);

    // Count degenerate faces (nz==0) as front, at least for now
    return nz <= 0.0;
}

#[cfg(test)]
mod tests {
    use math::ApproxEq;
    use math::vec::*;

    use super::*;
    use clip_mask::*;

    // TODO Test interpolation of vertex attributes

    fn assert_approx_eq(expected: Vec<Vertex<()>>, actual: Vec<Vertex<()>>) {
        assert_eq!(expected.len(), actual.len(), "expected: {:#?}\nactual: {:#?}", expected, actual);
        for (&e, &a) in expected.iter().zip(&actual) {
            assert!(e.coord.approx_eq(a.coord), "expected: {:?}, actual: {:?}", e, a);
            // TODO assert!(e.1.approx_eq(a.1), "expected: {:?}, actual: {:?}", e, a);
        }
    }

    fn vs(vs: &[Vec4]) -> Vec<Vertex<()>> {
        vs.iter().map(|&v| Vertex { coord: v + W, attr: () }).collect()
    }

    fn vec(x: f32, w: f32) -> Vec4 { vec4(x, 0.0, 0.0, w) }

    #[test]
    fn clip_plane_inside() {
        let p = &CLIP_PLANES[0];

        assert_eq!(1, p.test(&vec(2.0, 3.0)));
        assert_eq!(1, p.test(&vec(-2.0, 3.0)));
        assert_eq!(1, p.test(&vec(3.0, 2.0)));
        assert_eq!(0, p.test(&vec(-3.0, 2.0)));
        assert_eq!(1, p.test(&vec(2.0, 2.0)));
        assert_eq!(1, p.test(&vec(-2.0, 2.0)));

        assert_eq!(0, p.test(&vec(2.0, -3.0)));
        assert_eq!(0, p.test(&vec(-2.0, -3.0)));
        assert_eq!(1, p.test(&vec(3.0, -2.0)));
    }

    #[test]
    fn clip_plane_intersect() {
        let p = &CLIP_PLANES[0];

        fn v(c: Vec4) -> Vertex<()> { Vertex { coord: c, attr: () } }

        // out -> in
        assert_eq!([None, Some(v(-1.5*X+1.5*W))], p.clip(&v(-2.0*X+W), &v(3.0*W)));
        // in -> out
        assert_eq!([Some(v(3.0*W)), Some(v(-1.5*X+1.5*W))], p.clip(&v(3.0*W), &v(-2.0*X+W)));

        // in -> in
        assert_eq!([Some(v(X+2.0*W)), None], p.clip(&v(X+2.0*W), &v(2.0*X+3.0*W)));
        // out -> out
        assert_eq!([None, None], p.clip(&v(-3.0*X+2.0*W), &v(-2.0*X+W)));
    }

    #[test]
    fn clip_vertex_inside_frustum() {
        assert_eq!(ALL, vertex_mask(&vec4(0.0, 0.0, 0.0, 0.0)));

        assert_eq!(ALL, vertex_mask(&vec4(1.0, 0.0, 0.0, 1.0)));
        assert_eq!(ALL, vertex_mask(&vec4(-2.0, 0.0, 0.0, 3.0)));
        assert_eq!(ALL, vertex_mask(&vec4(0.0, 1.0, 0.0, 1.0)));
        assert_eq!(ALL, vertex_mask(&vec4(0.0, -2.0, 0.0, 3.0)));
        assert_eq!(ALL, vertex_mask(&vec4(0.0, 0.0, 1.0, 1.0)));
        assert_eq!(ALL, vertex_mask(&vec4(0.0, 0.0, -2.0, 3.0)));
    }

    #[test]
    fn clip_vertex_outside_single_plane() {
        assert_eq!(ALL & !RIGHT, vertex_mask(&vec4(2.0, 0.0, 0.0, 1.0)));
        assert_eq!(ALL & !LEFT, vertex_mask(&vec4(-3.0, 0.0, 0.0, 2.0)));
        assert_eq!(ALL & !TOP, vertex_mask(&vec4(0.0, 2.0, 0.0, 1.0)));
        assert_eq!(ALL & !BOTTOM, vertex_mask(&vec4(0.0, -3.0, 0.0, 2.0)));
        assert_eq!(ALL & !FAR, vertex_mask(&vec4(0.0, 0.0, 2.0, 1.0)));
        assert_eq!(ALL & !NEAR, vertex_mask(&vec4(0.0, 0.0, -3.0, 2.0)));
    }

    #[test]
    fn clip_vertex_outside_several_planes() {
        assert_eq!(ALL & !(RIGHT|NEAR), vertex_mask(&vec4(2.0, 0.0, -3.0, 1.0)));
        assert_eq!(ALL & !(LEFT|FAR|TOP), vertex_mask(&vec4(-3.0, 4.0, 5.0, 2.0)));
    }


    fn assert_vis(vis: Visibility, [a,b,c]: [Vec4; 3]) {
        let verts = [a+W, b+W, c+W];
        let [a,b,c] = verts;
        let masks = [
            vertex_mask(&a),
            vertex_mask(&b),
            vertex_mask(&c)
        ];
        assert_eq!(vis, visibility(masks.iter().copied()), "verts: {:?}", verts)
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
            let expected = Vec::<Vertex<()>>::new();
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
