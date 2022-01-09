use std::fmt::Debug;
use math::mat::Mat4;
use math::transform::Transform;

use math::vec::{Vec4, ZERO};

use crate::bbox::BoundingBox;

pub trait Soa {
    type Vecs;
    type Indices: Copy + Debug;

    fn get(vecs: &Self::Vecs, idcs: &Self::Indices) -> Self;
}

impl Soa for () {
    type Vecs = ();
    type Indices = [usize; 0];

    fn get(_: &Self::Vecs, _: &Self::Indices) -> Self {
        ()
    }
}

impl<T0: Copy> Soa for (T0, ) {
    type Vecs = Vec<T0>;
    type Indices = usize;

    fn get(vec: &Self::Vecs, &idx: &Self::Indices) -> Self {
        (vec[idx], )
    }
}

impl<T0: Copy, T1: Copy> Soa for (T0, T1) {
    type Vecs = (Vec<T0>, Vec<T1>);
    type Indices = [usize; 2];

    fn get(vecs: &Self::Vecs, &[i0, i1]: &Self::Indices) -> Self {
        (vecs.0[i0], vecs.1[i1])
    }
}

impl<T0: Copy, T1: Copy, T2: Copy> Soa for (T0, T1, T2) {
    type Vecs = (Vec<T0>, Vec<T1>, Vec<T2>);
    type Indices = [usize; 3];

    fn get(vecs: &Self::Vecs, &[i0, i1, i2]: &Self::Indices) -> Self {
        (vecs.0[i0], vecs.1[i1], vecs.2[i2])
    }
}

impl<T0: Copy, T1: Copy, T2: Copy, T3: Copy> Soa for (T0, T1, T2, T3) {
    type Vecs = (Vec<T0>, Vec<T1>, Vec<T2>, Vec<T3>);
    type Indices = [usize; 4];

    fn get(vecs: &Self::Vecs, &[i0, i1, i2, i3]: &Self::Indices) -> Self {
        (vecs.0[i0], vecs.1[i1], vecs.2[i2], vecs.3[i3])
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GenVertex<C, A> {
    pub coord: C,
    pub attr: A,
}

pub type Vertex<A> = GenVertex<Vec4, A>;
pub type VertexIndices<A> = GenVertex<usize, A>;

pub fn vertex<A>(coord: Vec4, attr: A) -> Vertex<A> {
    Vertex { coord, attr }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Face<V, A> {
    pub verts: [V; 3],
    pub attr: A,
}

#[derive(Clone, Debug)]
pub struct Mesh<VA: Soa, FA = ()> {
    pub verts: Vec<VertexIndices<VA::Indices>>,
    pub vertex_coords: Vec<Vec4>,
    pub vertex_attrs: VA::Vecs,

    pub faces: Vec<Face<usize, usize>>,
    pub face_attrs: Vec<FA>,

    pub bbox: BoundingBox,
}

impl<VA: Soa, FA> Default for Mesh<VA, FA>
where
    VA::Vecs: Default
{
    fn default() -> Self {
        Self {
            verts: vec![],
            vertex_coords: vec![],
            vertex_attrs: VA::Vecs::default(),
            faces: vec![],
            face_attrs: vec![],
            bbox: BoundingBox::default(),
        }
    }
}

impl<VA: Soa, FA> Transform for Mesh<VA, FA> {
    fn transform_mut(&mut self, tf: &Mat4) {
        self.bbox.transform_mut(tf);
        self.vertex_coords.transform_mut(tf);
    }
}

impl Mesh<(), ()> {
    pub fn gen_normals(self) -> Mesh<(Vec4, ), Vec4> {
        self.gen_face_normals().gen_vert_normals()
    }
}

impl<VA: Soa> Mesh<VA, ()> {
    pub fn gen_face_normals(self) -> Mesh<VA, Vec4> {
        let Self {
            verts, vertex_coords, vertex_attrs, faces, bbox, ..
        } = self;

        let face_ns: Vec<_> = faces.iter()
            .map(|f| {
                let [a, b, c] = f.verts.map(|i| vertex_coords[verts[i].coord]);
                (b - a).cross(c - a).normalize()
            })
            .collect();

        let faces = faces.into_iter().enumerate()
            .map(|(i, f)| Face { verts: f.verts, attr: i })
            .collect();

        Mesh {
            faces,
            face_attrs: face_ns,
            verts,
            vertex_coords,
            vertex_attrs,
            bbox,
        }
    }
}

impl<FA> Mesh<(), FA> {
    pub fn gen_vert_normals(self) -> Mesh<(Vec4, ), FA> {
        let Self {
            verts, vertex_coords, faces, face_attrs, bbox, ..
        } = self;

        let face_ns: Vec<_> = faces.iter()
            .map(|f| {
                let [a, b, c] = f.verts.map(|i| vertex_coords[verts[i].coord]);
                (b - a).cross(c - a)
            })
            .collect();

        let mut vert_ns = vec![ZERO; verts.len()];

        for (&Face { verts: [a, b, c], .. }, &n) in faces.iter().zip(&face_ns) {
            vert_ns[a] += n;
            vert_ns[b] += n;
            vert_ns[c] += n;
        }
        for v in &mut vert_ns {
            *v = v.normalize();
        }

        let verts = verts.into_iter().zip(0..vert_ns.len())
            .map(|(v, attr)| VertexIndices { coord: v.coord, attr })
            .collect();

        Mesh {
            faces,
            face_attrs,
            verts,
            vertex_coords,
            vertex_attrs: vert_ns,
            bbox,
        }
    }
}

impl<VA: Soa, FA> Mesh<VA, FA> {

    pub fn validate(self) -> Result<Self, String> {

        fn check_indices<I>(name: &str, indices: I, max: usize) -> Result<(), String>
        where I: IntoIterator<Item=usize>
        {
            let mut used = vec![false; max];
            for i in indices {
                let u = used.get_mut(i)
                    .ok_or_else(|| format!("{} index out of bounds: {}", name, i))?;
                *u = true;
            }
            used.iter().position(|b| !b)
                .map_or(Ok(()), |i| Err(format!("unused {} at {}", name, i)))
        }

        let Mesh { verts, vertex_coords, faces, face_attrs, .. } = &self;

        check_indices(
            "vertex coord",
            verts.iter().map(|v| v.coord),
            vertex_coords.len()
        )?;
        check_indices(
            "vertex",
            faces.iter().flat_map(|f| f.verts),
            verts.len()
        )?;
        check_indices(
            "face attr",
            faces.iter().map(|f| f.attr),
            face_attrs.len()
        )?;

        // TODO validate vertex attr indices

        Ok(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Builder(Mesh<()>);

impl Builder {
    pub fn new() -> Self {
        Self(Mesh::default())
    }

    pub fn add_vert(&mut self, c: Vec4) -> isize {
        let idx = self.0.vertex_coords.len();
        self.0.vertex_coords.push(c.to_pt());
        self.0.verts.push(VertexIndices { coord: idx, attr: [] });
        idx as isize
    }

    pub fn add_face(&mut self, a: isize, b: isize, c: isize) {
        assert!(a != b && a != c && b != c,
                "degenerate face {:?}", (a, b, c));
        let idcs = [a, b, c].map(|i| {
            i.rem_euclid(self.0.vertex_coords.len() as isize) as usize
        });
        self.0.faces.push(Face { verts: idcs, attr: 0 })
    }

    pub fn build(self) -> Mesh<()> {
        let mut face_attrs = vec![];
        if !self.0.faces.is_empty() {
            // TODO hack
            face_attrs.push(())
        }
        Mesh {
            face_attrs,
            bbox: BoundingBox::of(&self.0.vertex_coords),
            ..self.0
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        // TODO
    }
}
