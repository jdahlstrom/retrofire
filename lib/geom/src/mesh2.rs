use std::fmt::Debug;
use math::vec::Vec4;

use crate::bbox::BoundingBox;

pub trait Soa {
    type Vecs;
    type Indices;

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

#[derive(Debug, Clone)]
pub struct Mesh<VA: Soa, FA = ()> {
    pub verts: Vec<(usize, VA::Indices)>,
    pub vertex_coords: Vec<Vec4>,
    pub vertex_attrs: VA::Vecs,

    pub faces: Vec<([usize; 3], usize)>,
    pub face_attrs: Vec<FA>,

    pub bbox: BoundingBox,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex<A> {
    pub coord: Vec4,
    pub attr: A,
}

pub fn vertex<A>(coord: Vec4, attr: A) -> Vertex<A> {
    Vertex { coord, attr }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Face<VA: Soa, FA> {
    pub indices: ([usize; 3], usize),
    pub verts: [Vertex<VA>; 3],
    pub attr: FA,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // TODO
    }
}
