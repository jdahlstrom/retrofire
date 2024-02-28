use crate::geom::Mesh;
use crate::math::mat::{Mat4x4, RealToReal};
use crate::math::vary::Vary;
use crate::render::shader::{FragmentShader, VertexShader};
use crate::render::target::Target;
use crate::render::{Model, View, World};
use alloc::vec::Vec;

pub mod cam;

pub type ModelToWorld = RealToReal<3, Model, World>;
pub type WorldToView = RealToReal<3, World, View>;

///
pub struct Scene<A> {
    pub objs: Vec<Obj<A>>,
}

pub struct Obj<A> {
    pub mesh: Mesh<A>,
    pub transform: Mat4x4<ModelToWorld>,
}

//
// Inherent impls
//

//
// Local trait impls
//
