use core::ops::Range;

use crate::geom::{Tri, Vertex};
use crate::math::{
    angle::{spherical, turns, SphericalVec},
    mat::{orthographic, perspective, viewport, Mat4x4, RealToReal},
    space::Linear,
    vary::Vary,
    vec::{vec2, Vec3},
};
use crate::util::rect::Rect;

#[cfg(feature = "fp")]
use crate::math::{
    angle::Angle,
    mat::{orient_z, rotate_x, rotate_y, translate},
    vec::vec3,
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
///
/// TODO Rename to something more specific (e.g. `Motion`?)
pub trait Mode {
    /// Returns the current world-to-view matrix of this camera mode.
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
}

/// Type to manage the world-to-viewport transformation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Camera<M> {
    /// The movement mode of the camera.
    pub mode: M,
    /// Viewport width and height.
    pub res: (u32, u32),
    /// Projection matrix.
    pub project: Mat4x4<ViewToProj>,
    /// Viewport matrix.
    pub viewport: Mat4x4<NdcToScreen>,
}

/// First-person camera mode.
///
/// This is the familiar "FPS" movement mode, based on camera
/// position and heading (look-at vector).
#[derive(Copy, Clone, Debug)]
pub struct FirstPerson {
    /// Current position of the camera in world space.
    pub pos: Vec3,
    /// Current heading of the camera in world space.
    pub heading: SphericalVec,
}

/// Orbiting camera mode.
///
/// This mode can rotate the camera around a fixed point, centered on the
/// point, and change the camera's distance to the target point.
#[derive(Copy, Clone, Debug)]
pub struct Orbit {
    pub target: Vec3,
    pub dir: SphericalVec,
}

//
// Inherent impls
//

impl<M: Mode> Camera<M> {
    /// Creates a camera with the given resolution.
    pub fn new(res_x: u32, res_y: u32) -> Self
    where
        M: Default,
    {
        Self::with_mode(res_x, res_y, M::default())
    }

    /// Creates a camera with the given resolution and mode.
    pub fn with_mode(res_x: u32, res_y: u32, mode: M) -> Self {
        Self {
            res: (res_x, res_y),
            mode,
            project: Default::default(),
            viewport: viewport(vec2(0, 0)..vec2(res_x, res_y)),
        }
    }

    /// Sets the viewport bounds of this camera.
    pub fn viewport(self, bounds: impl Into<Rect<u32>>) -> Self {
        let (w, h) = self.res;

        let Rect {
            left: Some(l),
            top: Some(t),
            right: Some(r),
            bottom: Some(b),
        } = bounds.into().intersect(&(0..w, 0..h).into())
        else {
            unreachable!("bounded ∩ bounded should be bounded")
        };

        Self {
            res: (r.abs_diff(l), b.abs_diff(t)),
            viewport: viewport(vec2(l, t)..vec2(r, b)),
            ..self
        }
    }

    /// Sets up perspective projection.
    pub fn perspective(self, focal_ratio: f32, near_far: Range<f32>) -> Self {
        let aspect_ratio = self.res.0 as f32 / self.res.1 as f32;
        Self {
            project: perspective(focal_ratio, aspect_ratio, near_far),
            ..self
        }
    }

    /// Sets up orthographic projection.
    pub fn orthographic(self, bounds: Range<Vec3>) -> Self {
        Self {
            project: orthographic(bounds.start, bounds.end),
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

impl FirstPerson {
    /// Creates a first-person mode with position in the origin and heading
    /// in the direction of the positive x-axis.
    pub fn new() -> Self {
        Self {
            pos: Vec3::zero(),
            heading: spherical(1.0, turns(0.0), turns(0.0)),
        }
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
            az.wrap(turns(-0.5), turns(0.5)),
            alt.clamp(turns(-0.25), turns(0.25)),
        );
    }

    /// Translates the camera *in the camera's frame*.
    ///
    /// The components of `delta` determine how much the camera moves
    /// left/right (x), up/down (y), and forward/back (z).
    pub fn translate(&mut self, delta: Vec3) {
        // Zero azimuth means parallel to the x-axis
        let fwd = spherical(1.0, self.heading.az(), turns(0.0)).to_cart();
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

#[cfg(feature = "fp")]
impl Orbit {
    /// Adds the azimuth and altitude given to the camera's current direction.
    pub fn rotate(&mut self, az_delta: Angle, alt_delta: Angle) {
        self.rotate_to(self.dir.az() + az_delta, self.dir.alt() + alt_delta);
    }

    /// Rotates the camera to the *world-space* azimuth and altitude given.
    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.dir = spherical(
            self.dir.r(),
            az.wrap(turns(-0.5), turns(0.5)),
            alt.clamp(turns(-0.25), turns(0.25)),
        );
    }

    /// Moves the camera towards or away from the target.
    ///
    /// Note that a *negative* delta means towards the target, because it is
    /// simply added to the current distance. The distance is clamped to 0.0.
    pub fn zoom(&mut self, r_delta: f32) {
        self.zoom_to(self.dir.r() + r_delta);
    }
    /// Moves the camera to the given distance from the target.
    ///
    /// The distance is clamped to 0.0.
    pub fn zoom_to(&mut self, r: f32) {
        self.dir[0] = r.max(0.0);
    }
}

//
// Local trait impls
//

#[cfg(feature = "fp")]
impl Mode for FirstPerson {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let &Self { pos, heading: dir, .. } = self;
        let fwd_move = spherical(1.0, dir.az(), turns(0.0));
        let fwd = self.heading;
        let right = vec3(0.0, 1.0, 0.0).cross(&fwd_move.to_cart());

        let transl = translate(-pos);
        let orient = orient_z(fwd.into(), right);

        transl.then(&orient).to()
    }
}

impl Mode for Orbit {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        translate(self.target)
            .then(&rotate_y(self.dir.az()))
            .then(&rotate_x(self.dir.alt()))
            .then(&translate(vec3(0.0, 0.0, self.dir.r())))
            .to()
    }
}

impl Mode for Mat4x4<WorldToView> {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        *self
    }
}

//
// Foreign trait impls
//

#[cfg(feature = "fp")]
impl Default for FirstPerson {
    /// Returns [`FirstPerson::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "fp")]
impl Default for Orbit {
    fn default() -> Self {
        Self {
            target: Vec3::zero(),
            dir: spherical(1.0, turns(0.0), turns(0.0)),
        }
    }
}
