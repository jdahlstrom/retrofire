use core::ops::Range;

use crate::{
    geom::{Tri, Vertex},
    math::mat::{
        orthographic, perspective, translate, viewport, Mat4x4, RealToReal,
    },
    math::vary::Vary,
    math::vec::{vec2, vec3, Vec3},
    util::rect::Rect,
};

#[cfg(feature = "fp")]
use crate::{
    math::angle::{degs, spherical, Angle, SphericalVec},
    math::mat::{orient_z, rotate_y},
};

use super::{
    clip::ClipVec,
    ctx::Context,
    raster::Frag,
    shader::{FragmentShader, VertexShader},
    target::Target,
    NdcToScreen, RealToProj, ViewToProj, World, WorldToView,
};

/// Camera movement mode.
pub trait Mode {
    /// Returns the current world-to-view matrix of this camera mode.
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
}

/// Encapsulates the world-to-screen transform sequence.
#[derive(Copy, Clone, Debug)]
pub struct Camera<M> {
    pub mode: M,
    pub res: Resolution,
    pub project: Mat4x4<ViewToProj>,
    pub viewport: Mat4x4<NdcToScreen>,
}

/// First-person camera mode.
#[cfg(feature = "fp")]
#[derive(Copy, Clone, Debug, Default)]
pub struct FirstPerson {
    pub pos: Vec3,
    pub heading: SphericalVec,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Resolution(pub u32, pub u32);

pub const VGA_640_480: Resolution = Resolution(640, 480);
pub const SVGA_800_600: Resolution = Resolution(800, 600);
pub const HD_1280_720: Resolution = Resolution(1280, 720);
pub const HD_1920_1080: Resolution = Resolution(1920, 1080);

//
// Inherent impls
//

impl<M: Mode> Camera<M> {
    /// Returns a new camera with the given resolution.
    pub fn new(res: Resolution) -> Self
    where
        M: Default,
    {
        Self::with_mode(res, M::default())
    }

    /// Returns a new camera with the given resolution and mode.
    pub fn with_mode(res: Resolution, mode: M) -> Self {
        Self {
            res,
            mode,
            project: Default::default(),
            viewport: viewport(vec2(0, 0)..vec2(res.0, res.1)),
        }
    }

    /// Sets the viewport bounds of this camera.
    pub fn viewport(self, bounds: impl Into<Rect<u32>>) -> Self {
        let Rect {
            left: Some(l),
            top: Some(t),
            right: Some(r),
            bottom: Some(b),
        } = bounds
            .into()
            .intersect(&(0..self.res.0, 0..self.res.1).into())
        else {
            unreachable!("intersect with bounded is always bounded")
        };

        // Intersection with a bounded rect always results in a bounded rect
        Self {
            viewport: viewport(vec2(l, t)..vec2(r, b)),
            ..self
        }
    }

    /// Sets the projection of this camera to perspective.
    pub fn perspective(self, focal_ratio: f32, near_far: Range<f32>) -> Self {
        Self {
            project: perspective(
                focal_ratio,
                self.res.aspect_ratio(),
                near_far,
            ),
            ..self
        }
    }

    /// Sets the projection of this camera to orthographic.
    pub fn orthographic(self, bounds: Range<Vec3>) -> Self {
        Self {
            project: orthographic(bounds),
            ..self
        }
    }

    /// Renders the given geometry from the viewpoint of this camera.
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

#[cfg(feature = "fp")]
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

impl Resolution {
    pub fn aspect_ratio(&self) -> f32 {
        self.0 as f32 / self.1 as f32
    }
}

//
// Local trait impls
//

#[cfg(feature = "fp")]
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
