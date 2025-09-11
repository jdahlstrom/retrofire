use crate::{geom::Mesh, math::Mat4x4, render::ModelToWorld};

#[derive(Clone, Debug, Default)]
pub struct Obj<A> {
    pub geom: Mesh<A>,
    pub tf: Mat4x4<ModelToWorld>,
}
