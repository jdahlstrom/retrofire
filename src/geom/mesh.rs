use crate::geom::{Face, Vertex, bbox::BoundingBox};
use crate::math::vec::{Vec4, ZERO};

#[derive(Default, Debug, Clone)]
pub struct Mesh<VA = (), FA = ()> {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_attrs: Vec<VA>,
    pub face_attrs: Vec<FA>,
    pub bbox: BoundingBox,
}

impl Mesh {
    pub fn from_verts_and_faces(verts: impl IntoIterator<Item=Vec4>,
                                faces: impl IntoIterator<Item=[usize; 3]>)
                                -> Mesh
    {
        let verts = verts.into_iter().collect::<Vec<_>>();
        let faces = faces.into_iter().collect::<Vec<_>>();

        Mesh {
            bbox: BoundingBox::of(verts.iter().copied()),
            vertex_attrs: vec![(); verts.len()],
            face_attrs: vec![(); faces.len()],
            verts,
            faces,
        }
    }
}

impl<VA, FA> Mesh<VA, FA> {

    pub fn with_vertex_attrs<A>(self, attrs: impl IntoIterator<Item=A>)
                                -> Mesh<A, FA> where A: Clone
    {
        Mesh {
            verts: self.verts,
            faces: self.faces,
            bbox: self.bbox,
            face_attrs: self.face_attrs,

            vertex_attrs: attrs.into_iter().collect(),
        }
    }

    pub fn with_face_attrs<A>(self, attrs: impl IntoIterator<Item=A>)
                              -> Mesh<VA, A> where A: Clone
    {
        Mesh {
            verts: self.verts,
            faces: self.faces,
            bbox: self.bbox,
            vertex_attrs: self.vertex_attrs,

            face_attrs: attrs.into_iter().collect(),
        }
    }

    pub fn faces(&self) -> impl Iterator<Item=Face<VA, FA>> + '_
    where VA: Copy, FA: Copy,
    {
        self.faces.iter().zip(&self.face_attrs).map(move |(&[a, b, c], &fa)| {
            Face {
                indices: [a, b, c],
                verts: [
                    Vertex { coord: self.verts[a], attr: self.vertex_attrs[a] },
                    Vertex { coord: self.verts[b], attr: self.vertex_attrs[b] },
                    Vertex { coord: self.verts[c], attr: self.vertex_attrs[c] }
                ],
                attr: fa,
            }

        })
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
        let face_ns = self.faces()
                          .map(|Face { verts: [a, b, c], ..}| {
                              (b.coord - a.coord).cross(c.coord - a.coord)
                          })
                          .collect::<Vec<_>>();

        let mut vert_ns = vec![ZERO; self.verts.len()];

        for (&[a, b, c], &n) in self.faces.iter().zip(&face_ns) {
            vert_ns[a] = vert_ns[a] + n;
            vert_ns[b] = vert_ns[b] + n;
            vert_ns[c] = vert_ns[c] + n;
        }

        Mesh::from_verts_and_faces(self.verts, self.faces)
            .with_vertex_attrs(vert_ns.into_iter().map(Vec4::normalize))
            .with_face_attrs(face_ns.into_iter().map(Vec4::normalize))
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
    #[test]
    fn test() {}
}