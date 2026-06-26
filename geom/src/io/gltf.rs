use alloc::vec::Vec;
use gltf::{
    Buffer, Document, Primitive, buffer, image, import, mesh, mesh::Mode,
};
#[cfg(feature = "std")]
use std::{env::current_dir, io, iter, path::Path};

use retrofire_core::{
    geom::{Builder, Mesh, Normal3, Tri, Vertex3, tri, vertex},
    render::TexCoord,
    util::Load,
};

/// Represents a glTF document being loaded.
pub struct Gltf {
    doc: Document,
    bufs: Vec<buffer::Data>,
    #[allow(unused)]
    imgs: Vec<image::Data>,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Gltf(gltf::Error),
    NoIndicesFound,
    NoAttributesFound(&'static str),
    UnsupportedPrimitive(Mode),
}
use Error::*;

#[cfg(feature = "std")]
pub fn load_gltf<A>(path: impl AsRef<Path>) -> Result<Vec<Builder<A>>, Error>
where
    Gltf: TryInto<Vec<Builder<A>>, Error = Error>,
{
    Gltf::from_path(path)?.try_into()
}

impl Load for Gltf {
    type Error = Error;

    fn from_path(path: impl AsRef<Path>) -> Result<Self, Self::Error> {
        let (doc, bufs, imgs) = import(path).map_err(Gltf)?;
        Ok(Self { doc, bufs, imgs })
    }

    fn from_reader(
        reader: impl io::Read + io::Seek,
    ) -> Result<Self, Self::Error> {
        let doc = gltf::Gltf::from_reader(reader)?.document;

        let cwd = current_dir().ok();
        let bufs = gltf::import_buffers(&doc, cwd.as_deref(), None)?;

        //let imgs = gltf::import_images(&doc, cwd.as_deref(), &bufs)?;
        let imgs = Vec::new();

        Ok(Self { doc, bufs, imgs })
    }

    fn from_slice(slice: &[u8]) -> Result<Self, Self::Error> {
        let doc = gltf::Gltf::from_slice(slice)?.document;

        let cwd = current_dir().ok();
        let bufs = gltf::import_buffers(&doc, cwd.as_deref(), None)?;

        //let imgs = gltf::import_images(&doc, cwd.as_deref(), &bufs)?;
        let imgs = Vec::new();

        Ok(Self { doc, bufs, imgs })
    }
}

impl Gltf {
    fn reader<'a>(
        &'a self,
        prim: &'a Primitive,
    ) -> mesh::Reader<'a, 'a, impl Fn(Buffer<'a>) -> Option<&'a [u8]> + Clone>
    {
        prim.reader(|a| Some(&self.bufs[a.index()]))
    }

    fn load_verts<A>(
        &self,
        prim: &Primitive,
        attribs: impl Iterator<Item = A>,
        out: &mut Vec<Vertex3<A>>,
    ) -> Result<(), Error> {
        let positions = self
            .reader(&prim)
            .read_positions()
            .ok_or(NoAttributesFound("position"))?;

        out.extend(
            positions
                .zip(attribs)
                .map(|(p, a)| vertex(p.into(), a)),
        );

        Ok(())
    }

    fn load_faces(
        &self,
        prim: &Primitive,
        out: &mut Vec<Tri<u32>>,
    ) -> Result<(), Error> {
        let reader = prim.reader(|a| Some(&self.bufs[a.index()]));

        let mut indices = reader
            .read_indices()
            .ok_or(NoIndicesFound)?
            .into_u32();

        match prim.mode() {
            Mode::Triangles => {
                for _ in 0..indices.len() / 3 {
                    let i = indices.next().expect("cannot fail");
                    let j = indices.next().expect("cannot fail");
                    let k = indices.next().expect("cannot fail");
                    out.push(tri(i, j, k));
                }
            }
            Mode::TriangleFan => {
                let Some(i) = indices.next() else {
                    return Ok(()); // empty triangle fan
                };
                let mut j = indices.next().unwrap();
                while let Some(k) = indices.next() {
                    out.push(tri(i, j, k));
                    j = k;
                }
            }
            Mode::TriangleStrip => {
                let Some(mut i) = indices.next() else {
                    return Ok(()); // empty triangle strip
                };
                let mut j = indices.next().unwrap();
                let mut inv = false;
                while let Some(k) = indices.next() {
                    out.push(tri(i, j, k));
                    inv = !inv;
                    (i, j) = if inv { (k, j) } else { (j, k) };
                }
            }
            m => return Err(UnsupportedPrimitive(m)),
        }

        Ok(())
    }
}

//
// Foreign trait impls
//

impl From<gltf::Error> for Error {
    fn from(e: gltf::Error) -> Self {
        Gltf(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Io(e)
    }
}

impl TryFrom<Gltf> for Vec<Builder<()>> {
    type Error = Error;

    fn try_from(gltf: Gltf) -> Result<Self, Self::Error> {
        let mut res = Vec::new();
        for mesh in gltf.doc.meshes() {
            let mut b = Mesh::builder();
            for prim in mesh.primitives() {
                gltf.load_verts(&prim, iter::repeat(()), &mut b.mesh.verts)?;
                gltf.load_faces(&prim, &mut b.mesh.faces)?;
            }
            res.push(b)
        }
        Ok(res)
    }
}

impl TryFrom<Gltf> for Vec<Builder<Normal3>> {
    type Error = Error;
    fn try_from(gltf: Gltf) -> Result<Self, Self::Error> {
        let mut res = Vec::new();
        for mesh in gltf.doc.meshes() {
            let mut b = Mesh::builder();
            for prim in mesh.primitives() {
                let norms = gltf
                    .reader(&prim)
                    .read_normals()
                    .ok_or(NoAttributesFound("normals"))?;
                gltf.load_verts(
                    &prim,
                    norms.map(Normal3::from),
                    &mut b.mesh.verts,
                )?;
                gltf.load_faces(&prim, &mut b.mesh.faces)?;
            }
            res.push(b)
        }
        Ok(res)
    }
}

impl TryFrom<Gltf> for Vec<Builder<(Normal3, TexCoord)>> {
    type Error = Error;
    fn try_from(gltf: Gltf) -> Result<Self, Self::Error> {
        let mut res = Vec::new();
        for mesh in gltf.doc.meshes() {
            let mut b = Mesh::builder();
            for prim in mesh.primitives() {
                if prim.mode() != Mode::Triangles {
                    continue; // TODO support TriangleStrip, TriangleFan
                }
                let rdr = gltf.reader(&prim);
                let norms = rdr
                    .read_normals()
                    .ok_or(NoAttributesFound("normals"))?;
                let uvs = rdr
                    .read_tex_coords(0)
                    .ok_or(NoAttributesFound("texcoords"))?
                    .into_f32();
                gltf.load_verts(
                    &prim,
                    norms
                        .zip(uvs)
                        .map(|(n, uv)| (n.into(), uv.into())),
                    &mut b.mesh.verts,
                )?;
                gltf.load_faces(&prim, &mut b.mesh.faces)?;
            }
            res.push(b)
        }
        Ok(res)
    }
}
