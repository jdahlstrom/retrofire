use math::vec::{Vec4, ZERO};
use math::mat::Mat4;
use math::Linear;

pub trait VertexAttr : Linear<f32> + Copy + Default {
    fn transform(&mut self, mat: &Mat4);
}

impl VertexAttr for Vec4 {
    fn transform(&mut self, mat: &Mat4) {
        *self = (mat * *self).normalize(); // TODO only for normals!
    }
}

#[derive(Clone)]
pub struct Mesh<VA = (), FA = ()> {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_attrs: Option<Vec<VA>>,
    pub face_attrs: Option<Vec<FA>>,
}


impl<VA, FA> Mesh<VA, FA> {
    pub fn new() -> Self {
        Mesh { verts: vec![], faces: vec![], vertex_attrs: None, face_attrs: None }
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
        if let Some(idx) = self.faces().flatten().find(|&&idx| idx >= self.verts.len()) {
            return Err(format!("Vertex index out of bounds: {:?}", idx));
        }
        if let Some(face) = self.faces().find(|&[a, b, c]| a == b || b == c || a == c) {
            return Err(format!("Degenerate face: {:?}", face));
        }

        let mut verts = vec![false; self.verts.len()];
        for idx in self.faces().flatten() {
            verts[*idx] = true;
        }
        if let Some((idx, _)) = verts.iter().enumerate().find(|(_, &v)| !v) {
            return Err(format!("Unused vertex: {:?}", idx));
        }

        if let Some(attrs) = &self.face_attrs {
            if attrs.len() != self.faces.len() {
                return Err(format!("Missing or extra face attrs"));
            }
        }
        if let Some(attrs) = &self.vertex_attrs {
            if attrs.len() != self.verts.len() {
                return Err(format!("Missing or extra vertex attrs"));
            }
        }

        Ok(self)
    }

    pub fn gen_normals(self) -> Mesh<Vec4, Vec4> {
        let face_norms = self.face_verts()
                             .map(|[a, b, c]| (b - a).cross(c - a))
                             .collect::<Vec<_>>();

        let mut vertex_norms = vec![ZERO; self.verts.len()];

        for (&[a, b, c], &n) in self.faces.iter().zip(&face_norms) {
            vertex_norms[a] = vertex_norms[a] + n;
            vertex_norms[b] = vertex_norms[b] + n;
            vertex_norms[c] = vertex_norms[c] + n;
        }

        Mesh {
            verts: self.verts,
            faces: self.faces,
            vertex_attrs: Some(vertex_norms.into_iter().map(Vec4::normalize).collect()),
            face_attrs: Some(face_norms.into_iter().map(Vec4::normalize).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
}