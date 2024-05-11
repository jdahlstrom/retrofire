use core::ops::Range;

use crate::{
    geom::{Tri, Vertex},
    math::angle::{degs, spherical, Angle, SphericalVec},
    math::mat::{
        orient_z, orthographic, perspective, rotate_y, translate, viewport,
        Mat4x4, RealToReal,
    },
    math::vary::Vary,
    math::vec::{vec2, vec3, Vec3},
    util::rect::Rect,
};

use super::{
    clip::ClipVec,
    ctx::Context,
    raster::Frag,
    shader::{FragmentShader, VertexShader},
    target::Target,
    NdcToScreen, RealToProj, ViewToProj, World, WorldToView,
};

pub trait Mode {
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
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

impl<M: Mode> Camera<M> {
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
        let b @ Rect { left, top, right, bottom } =
            bounds.into().intersect(&(0..w, 0..h).into());

        // Intersection with a bounded rect always results in a bounded rect
        let viewport = viewport(
            vec2(left.unwrap(), top.unwrap())
                ..vec2(right.unwrap(), bottom.unwrap()),
        );
        Self {
            res: (b.width().unwrap(), b.height().unwrap()),
            viewport,
            ..self
        }
    }

    /// Returns a perspective camera.
    pub fn perspective(self, focal_ratio: f32, near_far: Range<f32>) -> Self {
        let aspect_ratio = self.res.0 as f32 / self.res.1 as f32;
        Self {
            project: perspective(focal_ratio, aspect_ratio, near_far),
            ..self
        }
    }

    /// Returns an orthographic camera.
    pub fn orthographic(self, bounds: Range<Vec3>) -> Self {
        Self {
            project: orthographic(bounds),
            ..self
        }
    }

    pub fn render<B, Vtx: Clone, Var: Vary, Uni: Copy, Shd>(
        &self,
        tris: impl AsRef<[Tri<usize>]>,
        verts: impl AsRef<[Vtx]>,
        to_world: &Mat4x4<RealToReal<3, B, World>>,
        shader: &Shd,
        uniform: Uni,
        target: &mut impl Target,
        ctx: &Context,
    ) where
        Shd: for<'a> VertexShader<
                Vtx,
                (&'a Mat4x4<RealToProj<B>>, Uni),
                Output = Vertex<ClipVec, Var>,
            > + FragmentShader<Frag<Var>>,
    {
        let tf = to_world
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

impl Mode for FirstPerson {
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
