use std::ops::Deref;
use std::path::Path;

use gltf::{import, mesh::Mode};

use retrofire_core::geom::{Mesh, Normal3};

#[derive(Debug)]
pub enum Error {
    Gltf(gltf::Error),
    NoMeshesFound,
    NoPositionsFound,
    NoNormalsFound,
    NoIndicesFound,
}
use Error::*;

pub fn load_gltf(path: impl AsRef<Path>) -> Result<Mesh<Normal3>, Error> {
    let (doc, bufs, _imgs) = import(path).map_err(Gltf)?;

    let mesh = doc.meshes().next().ok_or(NoMeshesFound)?;

    let mut res = Mesh::builder();

    for prim in mesh.primitives() {
        if prim.mode() != Mode::Triangles {
            continue;
        }
        let reader = prim.reader(|a| Some(&bufs[a.index()]));

        let positions = reader.read_positions().ok_or(NoPositionsFound)?;
        let normals = reader.read_normals().ok_or(NoNormalsFound)?;

        for (pos, norm) in positions.zip(normals) {
            res.push_vert(pos.into(), norm.into());
        }
        let mut indices = reader
            .read_indices()
            .ok_or(NoIndicesFound)?
            .into_u32();

        for _ in 0..indices.len() / 3 {
            let i = indices.next().unwrap();
            let j = indices.next().unwrap();
            let k = indices.next().unwrap();
            res.push_face(i, j, k);
        }
    }

    Ok(res.build())
}
