/// Reading 3D file formats.

#[cfg(feature = "gltf")]
pub mod gltf;

pub mod obj;

#[cfg(feature = "gltf")]
pub use gltf::Gltf;

pub use obj::Obj;
