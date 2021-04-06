use std::fmt::Debug;

use math::mat::Mat4;
use math::transform::Transform;
use math::vec::{Vec4, ZERO};

use crate::bbox::BoundingBox;

#[derive(Default, Debug, Clone)]
pub struct Mesh<VA = (), FA = ()> {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_attrs: Vec<VA>,
    pub face_attrs: Vec<FA>,
    pub bbox: BoundingBox,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex<A> {
    pub coord: Vec4,
    pub attr: A,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Face<VA, FA> {
    pub indices: [usize; 3],
    pub verts: [Vertex<VA>; 3],
    pub attr: FA,
}

#[derive(Default, Debug, Clone)]
pub struct Builder<VA = (), FA = ()> {
    pub mesh: Mesh<VA, FA>
}

pub enum FaceVert {
    New(Vec4),
    Existing(isize),
}

impl From<Vec4> for FaceVert {
    fn from(v: Vec4) -> Self { Self::New(v) }
}

impl From<isize> for FaceVert {
    fn from(i: isize) -> Self { Self::Existing(i) }
}

impl Mesh {
    pub fn builder() -> Builder {
        Builder { mesh: Self::default() }
    }
}

impl Builder {
    pub fn add_vert(&mut self, v: Vec4) -> &mut Self {
        self.mesh.verts.push(v.to_pt());
        self.mesh.vertex_attrs.push(());
        self
    }

    pub fn add_face<T, U, V>(&mut self, a: T, b: U, c: V) -> &mut Self
    where
        T: Into<FaceVert>, U: Into<FaceVert>, V: Into<FaceVert>,
    {
        let Mesh { ref mut verts, ref mut vertex_attrs, .. } = self.mesh;
        let len = verts.len() as isize;
        let mut idx = |v| {
            let i = match v {
                FaceVert::New(v) => {
                    verts.push(v.to_pt());
                    vertex_attrs.push(());
                    verts.len() - 1
                }
                FaceVert::Existing(i) => i.rem_euclid(len) as usize
            };
            debug_assert!(i < verts.len(), "i={} len={}", i, verts.len());
            i
        };
        self.mesh.faces.push([idx(a.into()), idx(b.into()), idx(c.into())]);
        self.mesh.face_attrs.push(());
        self
    }

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

    pub fn vertex_attrs_with<A>(self, f: impl FnMut(Vertex<VA>) -> A)
        -> Builder<A, FA>
    where A: Clone, VA: Copy, FA: Copy
    {
        let attrs: Vec<_> = self.mesh.verts().map(f).collect();
        self.vertex_attrs(attrs)
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

    pub fn face_attrs_with<A>(self, f: impl FnMut(Face<VA, FA>) -> A)
        -> Builder<VA, A>
    where A: Clone, VA: Copy, FA: Copy
    {
        let attrs: Vec<_> = self.mesh.faces().map(f).collect();
        self.face_attrs(attrs)
    }
}

impl<VA: Debug, FA: Debug> Builder<VA, FA> {
    pub fn build(self) -> Mesh<VA, FA> {
        let mut mesh = self.mesh;
        mesh.bbox = BoundingBox::of(mesh.verts.iter().copied());
        mesh
    }
}

impl<VA, FA> Mesh<VA, FA> {
    pub fn verts(&self) -> impl Iterator<Item=Vertex<VA>> + '_
    where VA: Copy
    {
        self.verts.iter().zip(&self.vertex_attrs)
            .map(|(&coord, &attr)| Vertex { coord, attr })
    }

    pub fn faces(&self) -> impl Iterator<Item=Face<VA, FA>> + '_
    where VA: Copy, FA: Copy,
    {
        self.faces.iter().zip(&self.face_attrs)
            .map(move |(&indices, &attr)| {
                let [a, b, c] = indices;
                Face {
                    indices,
                    verts: [self.vertex(a), self.vertex(b), self.vertex(c)],
                    attr,
                }
            })
    }

    fn vertex(&self, i: usize) -> Vertex<VA>
    where VA: Copy
    {
        Vertex { coord: self.verts[i], attr: self.vertex_attrs[i] }
    }

    pub fn validate(self) -> Result<Self, String> {
        let Mesh { verts, faces, vertex_attrs, face_attrs, .. } = &self;

        if let Some(idx) = faces.iter().flatten().find(|&&idx| idx >= verts.len()) {
            return Err(format!("Vertex index out of bounds: {:?}", idx));
        }
        if let Some(face) = faces.iter().find(|&[a, b, c]| a == b || b == c || a == c) {
            return Err(format!("Degenerate face: {:?}", face));
        }

        let mut verts = vec![false; verts.len()];
        for idx in faces.iter().flatten() {
            verts[*idx] = true;
        }
        if let Some((idx, _)) = verts.iter().enumerate().find(|(_, &v)| !v) {
            return Err(format!("Unused vertex: {:?}", idx));
        }

        if face_attrs.len() != faces.len() {
            return Err("Missing or extra face attrs".into());
        }

        if vertex_attrs.len() != verts.len() {
            return Err("Missing or extra vertex attrs".into());
        }

        Ok(self)
    }

    pub fn gen_normals(self) -> Mesh<Vec4, Vec4>
    where VA: Copy, FA: Copy
    {
        let face_ns: Vec<_> = self.faces()
            .map(|Face { verts: [a, b, c], .. }| {
                (b.coord - a.coord).cross(c.coord - a.coord)
            })
            .collect();

        let mut vert_ns = vec![ZERO; self.verts.len()];

        for (&[a, b, c], &n) in self.faces.iter().zip(&face_ns) {
            vert_ns[a] = vert_ns[a] + n;
            vert_ns[b] = vert_ns[b] + n;
            vert_ns[c] = vert_ns[c] + n;
        }

        Mesh::builder()
            .verts(self.verts)
            .faces(self.faces)
            .vertex_attrs(vert_ns.into_iter().map(Vec4::normalize))
            .face_attrs(face_ns.into_iter().map(Vec4::normalize))
            .build()
    }
}

impl Transform for Mesh {
    fn transform(&mut self, tf: &Mat4) {
        self.bbox.transform(tf);
        self.verts.transform(tf);
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
    #[test]
    fn test() {}
}