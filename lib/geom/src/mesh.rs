use math::Linear;
use math::mat::Mat4;
use math::vec::{Vec4, ZERO};

pub trait VertexAttr: Linear<f32> + Copy {
    fn transform(&mut self, mat: &Mat4);
}

impl VertexAttr for Vec4 {
    fn transform(&mut self, mat: &Mat4) {
        *self = (mat * *self).normalize(); // TODO only for normals!
    }
}

impl VertexAttr for () {
    fn transform(&mut self, _: &Mat4) {}
}

#[derive(Default, Debug, Clone)]
pub struct Mesh<VA = (), FA = ()> {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_attrs: Vec<VA>,
    pub face_attrs: Vec<FA>,
}

impl Mesh {
    pub fn from_verts_and_faces(verts: impl IntoIterator<Item=Vec4>,
                                faces: impl IntoIterator<Item=[usize; 3]>)
                                -> Mesh
    {
        let verts = verts.into_iter().collect::<Vec<_>>();
        let faces = faces.into_iter().collect::<Vec<_>>();

        Mesh {
            vertex_attrs: vec![(); verts.len()],
            face_attrs: vec![(); faces.len()],
            verts,
            faces,
        }
    }
}

impl<VA, FA> Mesh<VA, FA> {
    pub fn new() -> Self {
        Mesh {
            verts: vec![],
            faces: vec![],
            vertex_attrs: vec![],
            face_attrs: vec![]
        }
    }

    pub fn with_vertex_attrs<A>(self, attrs: impl IntoIterator<Item=A>)
                                -> Mesh<A, FA> where A: Clone
    {
        Mesh {
            verts: self.verts,
            faces: self.faces,
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
            vertex_attrs: self.vertex_attrs,
            face_attrs: attrs.into_iter().collect(),
        }
    }


    pub fn faces(&self) -> impl Iterator<Item=&[usize; 3]> + '_ {
        self.faces.iter()
    }

    pub fn face_verts(&self) -> impl Iterator<Item=[Vec4; 3]> + '_ {
        self.faces().map(move |&[a, b, c]| {
            [self.verts[a], self.verts[b], self.verts[c]]
        })
    }

    pub fn validate(self) -> Result<Self, String> {
        let Mesh { verts, faces, vertex_attrs, face_attrs } = &self;

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
            return Err(format!("Missing or extra face attrs"));
        }

        if vertex_attrs.len() != verts.len() {
            return Err(format!("Missing or extra vertex attrs"));
        }

        Ok(self)
    }

    pub fn gen_normals(self) -> Mesh<Vec4, Vec4> {
        let face_ns = self.face_verts()
                          .map(|[a, b, c]| (b - a).cross(c - a))
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