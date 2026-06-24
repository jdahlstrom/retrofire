use std::ops::Deref;
use std::path::Path;

use gltf::{import, mesh::Mode};

use retrofire_core::geom::{Mesh, Normal3};

#[derive(Debug)]
pub enum Error {
    Gltf(gltf::Error),
}

pub fn load_gltf(path: impl AsRef<Path>) -> Result<Mesh<Normal3>, Error> {
    let (doc, bufs, _imgs) = import(path).map_err(Error::Gltf)?;

    let mesh = doc.meshes().next().expect("at least one mesh");

    let mut res = Mesh::builder();

    for prim in mesh.primitives() {
        if prim.mode() != Mode::Triangles {
            continue;
        }
        let reader = prim.reader(|a| Some(&bufs[a.index()]));

        let positions = reader.read_positions().unwrap();
        let normals = reader.read_normals().unwrap();

        for (pos, norm) in positions.zip(normals) {
            res.push_vert(pos.into(), norm.into());
        }
        let mut indices = reader.read_indices().unwrap().into_u32();

        for _ in 0..indices.len() / 3 {
            let i = indices.next().unwrap();
            let j = indices.next().unwrap();
            let k = indices.next().unwrap();
            res.push_face(i as usize, j as usize, k as usize);
        }
    }

    Ok(res.build())
}
