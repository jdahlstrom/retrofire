use std::fmt::Debug;
use math::mat::Mat4;
use math::transform::Transform;

use math::vec::{Vec4, ZERO};

use crate::bbox::BoundingBox;

pub trait Soa: Sized {
    type Vecs: SoaVecs<Self, Self::Indices>;
    type Indices: Copy + Debug;
}
pub trait SoaVecs<T, I> {
    fn get(&self, ixs: &I) -> T;
}
macro_rules! impl_soa_for_tuple {
    ($N:literal; $($T:ident $I:tt),*) => {
        #[allow(unused)]
        impl<$($T: Copy),*> Soa for ($($T, )*) {
            type Vecs = ( $(Vec<$T>, )* );
            type Indices = [usize; $N];
        }
        #[allow(unused)]
        impl<$($T: Copy, )*> SoaVecs<($($T, )*), <( $($T, )* ) as Soa>::Indices> for ( $(Vec<$T>, )* ) {
            #[inline]
            fn get(&self, ixs: &<( $($T, )* ) as Soa>::Indices) -> ($($T, )*) {
                ( $(self.$I[ixs[$I]], )* )
            }
        }
    };
}
impl_soa_for_tuple!(0; );
impl_soa_for_tuple!(2; T0 0, T1 1);
impl_soa_for_tuple!(3; T0 0, T1 1, T2 2);
impl_soa_for_tuple!(4; T0 0, T1 1, T2 2, T3 3);

impl<T0: Copy> Soa for (T0, ) {
    type Vecs = Vec<T0>;
    type Indices = usize;
}
impl<T0: Copy> SoaVecs<(T0, ), usize> for Vec<T0> {

    fn get(&self, &idx: &usize) -> (T0,) {
        (self[idx], )
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GenVertex<C, A> {
    pub coord: C,
    pub attr: A,
}

impl<C: Copy, A> GenVertex<C, A> {
    pub fn attr<B>(self, attr: B) -> GenVertex<C, B> {
        GenVertex { coord: self.coord, attr }
    }
    pub fn attr_with<F, B>(self, mut attr_fn: F) -> GenVertex<C, B>
    where
        F: FnMut(GenVertex<C, A>) -> B
    {
        GenVertex {
            coord: self.coord,
            attr: attr_fn(self)
        }
    }
}

pub type Vertex<A> = GenVertex<Vec4, A>;
pub type VertexIndices<A> = GenVertex<usize, A>;

pub fn vertex<A>(coord: Vec4, attr: A) -> Vertex<A> {
    Vertex { coord, attr }
}

pub fn vertex_indices<A>(coord: usize, attr: A) -> VertexIndices<A> {
    VertexIndices { coord, attr }
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
            .map(|(v, a)| v.attr(a))
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

    pub fn sub_mesh(&self, face_indices: Vec<usize>) -> SubMesh<VA, FA> {

        let coords = face_indices.iter()
            .flat_map(|&i| self.faces[i].verts)
            .map(|vi| self.verts[vi].coord)
            .map(|ci| &self.vertex_coords[ci]);

        let bbox = BoundingBox::of(coords);

        SubMesh {
            mesh: self,
            face_indices,
            bbox,
        }
    }

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

#[derive(Clone)]
pub struct SubMesh<'a, VA: Soa, FA> {
    pub mesh: &'a Mesh<VA, FA>,
    pub face_indices: Vec<usize>,
    pub bbox: BoundingBox,
}

impl<'a, VA: Soa, FA> SubMesh<'a, VA, FA> {

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
        self.0.verts.push(vertex_indices(idx, []));
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
