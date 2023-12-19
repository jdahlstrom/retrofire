#![cfg(feature = "fp")]

use alloc::vec::Vec;
use core::ops::Range;

use crate::geom::{Mesh, Tri, Vertex};
use crate::math::angle::{degs, spherical, Angle, SphericalVec};
use crate::math::mat::{
    orient_z, orthographic, perspective, rotate_y, translate, viewport, Mat4x4,
    RealToProj, RealToReal,
};
use crate::math::space::Real;
use crate::math::vary::Vary;
use crate::math::vec::{vec2, vec3, Vec3};
use crate::render::clip::ClipVec;
use crate::render::ctx::Context;
use crate::render::raster::Frag;
use crate::render::shader::{FragmentShader, VertexShader};
use crate::render::target::Target;
use crate::render::{Model, NdcToScreen, View, ViewToProj, World};
use crate::util::rect::Rect;

pub trait CameraMode {
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
}

pub type ModelToWorld = RealToReal<3, Model, World>;
pub type WorldToView = RealToReal<3, World, View>;

type WorldVert<B, A> = Vertex<Vec3<Real<3, B>>, A>;

pub struct Scene<A> {
    pub objects: Vec<Obj<A>>,
}

pub struct Obj<A> {
    pub mesh: Mesh<A>,
    pub transform: Mat4x4<ModelToWorld>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Camera<M> {
    pub mode: M,
    pub res: (u32, u32),
    pub project: Mat4x4<ViewToProj>,
    pub viewport: Mat4x4<NdcToScreen>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FirstPerson {
    pub pos: Vec3,
    pub heading: SphericalVec,
}

//
// Inherent impls
//

impl<M: CameraMode> Camera<M> {
    ///
    pub fn new(res_x: u32, res_y: u32) -> Self
    where
        M: Default,
    {
        Self::with_mode(res_x, res_y, M::default())
    }

    ///
    pub fn with_mode(res_x: u32, res_y: u32, mode: M) -> Self {
        Self {
            res: (res_x, res_y),
            mode,
            project: Default::default(),
            viewport: viewport(vec2(0, 0)..vec2(res_x, res_y)),
        }
    }

    ///
    pub fn viewport(self, bounds: impl Into<Rect<u32>>) -> Self {
        let (w, h) = self.res;
        let bounds = bounds.into().intersect(&(0..w, 0..h).into());

        let viewport = viewport(
            vec2(bounds.left.unwrap(), bounds.top.unwrap())
                ..vec2(bounds.right.unwrap(), bounds.bottom.unwrap()),
        );
        Self {
            res: (bounds.width().unwrap(), bounds.height().unwrap()),
            viewport,
            ..self
        }
    }

    /// Returns a perspective camera.
    pub fn perspective(self, focal_ratio: f32, near_far: Range<f32>) -> Self {
        let (w, h) = self.res;
        Self {
            project: perspective(focal_ratio, w as f32 / h as f32, near_far),
            ..self
        }
    }

    /// Returns an orthographic camera.
    pub fn orthographic(self, lbn: Vec3, rtf: Vec3) -> Self {
        Self {
            project: orthographic(lbn, rtf),
            ..self
        }
    }

    pub fn render<A: Clone, B: Copy + Clone, Var: Vary, Uni: Copy, Shd>(
        &self,
        tris: impl AsRef<[Tri<usize>]>,
        verts: impl AsRef<[WorldVert<B, A>]>,
        model_to_world: &Mat4x4<RealToReal<3, B, World>>,
        shader: &Shd,
        uniform: Uni,
        target: &mut impl Target,
        ctx: &Context,
    ) where
        Shd: for<'a> VertexShader<
                WorldVert<B, A>,
                (&'a Mat4x4<RealToProj<B>>, Uni),
                Output = Vertex<ClipVec, Var>,
            > + FragmentShader<Frag<Var>>,
    {
        let tf = model_to_world
            .then(&self.mode.world_to_view())
            .then(&self.project);

        super::render(
            tris,
            verts,
            shader,
            (&tf, uniform),
            self.viewport,
            target,
            ctx,
        );
    }
}

impl FirstPerson {
    pub fn look_at(&mut self, pt: Vec3) {
        self.heading = (pt - self.pos).into();
        self.heading[0] = 1.0;
    }

    pub fn rotate(&mut self, az: Angle, alt: Angle) {
        self.rotate_to(self.heading.az() + az, self.heading.alt() + alt);
    }

    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.heading = spherical(
            1.0,
            az.wrap(degs(-180.0), degs(180.0)),
            alt.clamp(degs(-90.0), degs(90.0)),
        );
    }

    pub fn translate(&mut self, delta: Vec3) {
        // Zero azimuth means parallel to the x-axis
        let fwd = rotate_y(self.heading.az()).apply(&vec3(1.0, 0.0, 0.0));
        let up = vec3(0.0, 1.0, 0.0);
        let right = up.cross(&fwd);

        // / rx ux fx \ / dx \     / rx ry rz \ T / dx \
        // | ry uy fy | | dy |  =  | ux uy uz |   | dy |
        // \ rz uz fz / \ dz /     \ fx fy fz /   \ dz /

        self.pos += Mat4x4::<RealToReal<3>>::from_basis(right, up, fwd)
            .transpose()
            .apply(&delta);
    }
}

//
// Local trait impls
//

impl CameraMode for FirstPerson {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let &Self { pos, heading: dir, .. } = self;
        let fwd_move = spherical(1.0, dir.az(), degs(0.0));
        let fwd = self.heading;
        let right = vec3(0.0, 1.0, 0.0).cross(&fwd_move.into());

        let transl = translate(-pos);
        let orient = orient_z(fwd.into(), right);

        transl.then(&orient).to()
    }
}
