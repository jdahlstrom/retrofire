use std::collections::HashSet;
use std::fmt::Debug;

use math::vec::Vec4;

use crate::bbox::BoundingBox;

#[deprecated]
#[derive(Debug, Clone)]
pub struct Mesh<VA = (), FA = ()> {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_attrs: Vec<VA>,
    pub face_attrs: Vec<FA>,
    pub bbox: BoundingBox,
}

#[deprecated]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex<A> {
    pub coord: Vec4,
    pub attr: A,
}

#[deprecated]
pub fn vertex<A>(coord: Vec4, attr: A) -> Vertex<A> {
    Vertex { coord, attr }
}

#[deprecated]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Face<VA, FA> {
    pub indices: [usize; 3],
    pub verts: [Vertex<VA>; 3],
    pub attr: FA,
}

#[deprecated]
#[derive(Debug, Clone)]
pub struct Builder<VA = (), FA = ()> {
    pub mesh: Mesh<VA, FA>
}

impl Mesh {
    #[deprecated]
    pub fn builder() -> Builder {
        Builder {
            mesh: Mesh {
                verts: vec![],
                faces: vec![],
                vertex_attrs: vec![],
                face_attrs: vec![],
                bbox: Default::default()
            }
        }
    }
}

impl Builder {

    pub fn verts(self, verts: impl IntoIterator<Item=Vec4>) -> Self {
        let verts: Vec<_> = verts.into_iter().collect();
        let vertex_attrs = vec![(); verts.len()];
        Builder {
            mesh: Mesh {
                verts,
                vertex_attrs,
                faces: self.mesh.faces,
                face_attrs: self.mesh.face_attrs,
                bbox: BoundingBox::default(),
            }
        }
    }

    pub fn faces(self, faces: impl IntoIterator<Item=[usize; 3]>) -> Self {
        let faces: Vec<_> = faces.into_iter().collect();
        let face_attrs = vec![(); faces.len()];
        Builder {
            mesh: Mesh {
                faces,
                face_attrs,
                verts: self.mesh.verts,
                vertex_attrs: self.mesh.vertex_attrs,
                bbox: self.mesh.bbox,
            }
        }
    }
}

impl<VA, FA> Builder<VA, FA> {
    pub fn vertex_attrs<A>(self, attrs: impl IntoIterator<Item=A>)
        -> Builder<A, FA> where A: Clone
    {
        let Mesh { verts, faces, face_attrs, bbox, .. } = self.mesh;
        let vertex_attrs = attrs.into_iter().take(verts.len()).collect();
        Builder {
            mesh: Mesh {
                verts,
                vertex_attrs,
                faces,
                face_attrs,
                bbox,
            }
        }
    }

    pub fn face_attrs<A>(self, attrs: impl IntoIterator<Item=A>)
        -> Builder<VA, A> where A: Clone
    {
        let Mesh { verts, vertex_attrs, faces, bbox, .. } = self.mesh;
        let face_attrs = attrs.into_iter().take(faces.len()).collect();
        Builder {
            mesh: Mesh {
                verts,
                vertex_attrs,
                faces,
                face_attrs,
                bbox,
            }
        }
    }
}

impl<VA, FA> Builder<VA, FA> {
    pub fn build(self) -> Mesh<VA, FA> {
        let mut mesh = self.mesh;
        mesh.bbox = BoundingBox::of(&mesh.verts);
        mesh
    }
}

impl<VA: Copy, FA: Copy> Mesh<VA, FA> {

    pub fn verts(&self) -> impl Iterator<Item=Vertex<VA>> + '_ {
        self.verts.iter().zip(&self.vertex_attrs)
            .map(|(&coord, &attr)| Vertex { coord, attr })
    }

    pub fn faces(&self) -> impl Iterator<Item=Face<VA, FA>> + '_ {
        self.faces.iter().zip(&self.face_attrs)
            .map(move |(&indices, &attr)| {
                Face {
                    indices,
                    verts: indices.map(|i|
                        vertex(self.verts[i], self.vertex_attrs[i])),
                    attr,
                }
            })
    }

    #[deprecated]
    pub fn edges(&self) -> Vec<[Vec4; 2]> {
        let mut edges = HashSet::new();

        for &[a, b, c] in &self.faces {
            edges.insert([a, b]);
            edges.insert([b, c]);
            edges.insert([c, a]);
        }

        edges.into_iter()
            .map(|[a, b]| [self.verts[a], self.verts[b]])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
    #[test]
    fn test() {}
}
