use math::vec::{Vec4, ZERO};

#[derive(Clone)]
pub struct Mesh {
    pub verts: Vec<Vec4>,
    pub faces: Vec<[usize; 3]>,
    pub vertex_norms: Option<Vec<Vec4>>,
    pub face_norms: Option<Vec<Vec4>>,
}

pub struct Face {
    pub verts: [Vec4; 3],
    pub vertex_norms: [Vec4; 3],
    pub normal: Vec4,
}

impl Mesh {
    pub fn new() -> Self {
        Mesh { verts: vec![], faces: vec![], vertex_norms: None, face_norms: None }
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

        if let Some(norms) = &self.face_norms {
            if norms.len() != self.faces.len() {
                return Err(format!("Missing or extra face normals"));
            }
        }
        if let Some(norms) = &self.vertex_norms {
            if norms.len() != self.verts.len() {
                return Err(format!("Missing or extra vertex normals"));
            }
        }

        Ok(self)
    }

    pub fn gen_normals(self) -> Mesh {
        let face_norms = self.face_verts()
                             .map(|[a, b, c]| (a - b).cross(c - b))
                             .collect::<Vec<_>>();

        let mut vertex_norms = vec![ZERO; self.verts.len()];

        for (&[a, b, c], &n) in self.faces.iter().zip(&face_norms) {
            vertex_norms[a] = vertex_norms[a] + n;
            vertex_norms[b] = vertex_norms[b] + n;
            vertex_norms[c] = vertex_norms[c] + n;
        }

        Mesh {
            vertex_norms: Some(vertex_norms.into_iter().map(Vec4::normalize).collect()),
            face_norms: Some(face_norms.into_iter().map(Vec4::normalize).collect()),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // TODO tests
}