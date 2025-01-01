//! Cameras and camera transforms.

use core::ops::Range;

use crate::geom::{Tri, Vertex};
use crate::math::{
    mat::RealToReal, orient_z, orthographic, perspective, pt2, translate, vec3,
    viewport, BezierSpline, Lerp, Mat4x4, Point3, SphericalVec, Vary, Vec3,
};
use crate::util::{rect::Rect, Dims};

use crate::math::mat::orient_z_y;
#[cfg(feature = "fp")]
use crate::math::{pt3, rotate_x, rotate_y, spherical, turns, Angle};

use super::{
    clip::ClipVec, Context, FragmentShader, NdcToScreen, RealToProj, Target,
    VertexShader, View, ViewToProj, World, WorldToView,
};

/// Trait for different modes of camera motion.
pub trait Transform {
    /// Returns the current world-to-view matrix.
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
}

/// Type to manage the world-to-viewport transformation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Camera<Tf> {
    /// World-to-view transform.
    pub transform: Tf,
    /// Viewport width and height.
    pub dims: Dims,
    /// Projection matrix.
    pub project: Mat4x4<ViewToProj>,
    /// Viewport matrix.
    pub viewport: Mat4x4<NdcToScreen>,
}

/// First-person camera transform.
///
/// This is the familiar "FPS" movement mode, based on camera
/// position and heading (look-at vector).
#[derive(Copy, Clone, Debug)]
pub struct FirstPerson {
    /// Current position of the camera in **world** space.
    pub pos: Point3<World>,
    /// Current heading of the camera in **world** space.
    pub heading: SphericalVec<World>,
}

pub type ViewToWorld = RealToReal<3, View, World>;

#[cfg(feature = "fp")]
fn az_alt<B>(az: Angle, alt: Angle) -> SphericalVec<B> {
    spherical(1.0, az, alt)
}
/// Orbiting camera transform.
///
/// Keeps the camera centered on a **world-space** point, and allows free
/// 360°/180° azimuth/altitude rotation around that point as well as setting
/// the distance from the point.
#[derive(Copy, Clone, Debug)]
pub struct Orbit {
    /// The camera's target point in **world** space.
    pub target: Point3<World>,
    /// The camera's direction in **world** space.
    pub dir: SphericalVec<World>,
}

pub struct Spline {
    pub spline: BezierSpline<Point3<World>>,
    pub t: f32,
}

//
// Inherent impls
//

impl Camera<()> {
    /// Creates a camera with the given resolution.
    pub fn new(dims @ (w, h): Dims) -> Self {
        Self {
            dims,
            viewport: viewport(pt2(0, 0)..pt2(w, h)),
            ..Default::default()
        }
    }

    pub fn transform<T: Transform>(self, tf: T) -> Camera<T> {
        let Self { dims, project, viewport, .. } = self;
        Camera {
            transform: tf,
            dims,
            project,
            viewport,
        }
    }
}

impl<T> Camera<T> {
    /// Sets the viewport bounds of this camera.
    pub fn viewport(self, bounds: impl Into<Rect<u32>>) -> Self {
        let (w, h) = self.dims;

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
            dims: (r.abs_diff(l), b.abs_diff(t)),
            viewport: viewport(pt2(l, t)..pt2(r, b)),
            ..self
        }
    }

    /// Sets up perspective projection.
    pub fn perspective(
        mut self,
        focal_ratio: f32,
        near_far: Range<f32>,
    ) -> Self {
        let aspect_ratio = self.dims.0 as f32 / self.dims.1 as f32;
        self.project = perspective(focal_ratio, aspect_ratio, near_far);
        self
    }

    /// Sets up orthographic projection.
    pub fn orthographic(mut self, bounds: Range<Point3>) -> Self {
        self.project = orthographic(bounds.start, bounds.end);
        self
    }
}

impl<T: Transform> Camera<T> {
    /// Returns the composed camera and projection matrix.
    pub fn world_to_project(&self) -> Mat4x4<RealToProj<World>> {
        self.transform.world_to_view().then(&self.project)
    }

    /// Renders the given geometry from the viewpoint of this camera.
    pub fn render<B, Vtx: Clone, Var: Lerp + Vary, Uni: Copy, Shd>(
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
            > + FragmentShader<Var>,
    {
        let tf = to_world.then(&self.world_to_project());

        super::render(
            tris.as_ref(),
            verts.as_ref(),
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
    /// Creates a first-person transform with position in the origin
    /// and heading in the direction of the positive x-axis.
    pub fn new() -> Self {
        Self {
            pos: pt3(0.0, 0.0, 0.0),
            heading: az_alt(turns(0.0), turns(0.0)),
        }
    }

    /// Rotates the camera to center the view on a **world-space** point.
    pub fn look_at(&mut self, pt: Point3<World>) {
        let head = (pt - self.pos).to_spherical();
        self.rotate_to(head.az(), head.alt());
    }

    /// Rotates the camera by relative azimuth and altitude.
    pub fn rotate(&mut self, delta_az: Angle, delta_alt: Angle) {
        let head = self.heading;
        self.rotate_to(head.az() + delta_az, head.alt() + delta_alt);
    }

    /// Rotates the camera to an absolute orientation in **world** space.
    // TODO may confuse camera and world space
    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.heading = az_alt(
            az.wrap(turns(-0.5), turns(0.5)),
            alt.clamp(turns(-0.25), turns(0.25)),
        );
    }

    /// Translates the camera by a relative offset in **view** space.
    // TODO Explain that up/down is actually in world space (dir of gravity)
    pub fn translate(&mut self, delta: Vec3<View>) {
        // Zero azimuth means parallel to the x-axis
        // TODO This does basically the same as world_to_view
        let fwd = az_alt(self.heading.az(), turns(0.0)).to_cart();
        let right = vec3(fwd.z(), 0.0, -fwd.x());

        let to_world = orient_z(fwd, right).to::<RealToReal<3, _, _>>();
        self.pos += to_world.apply(&delta);
    }
}

#[cfg(feature = "fp")]
impl Orbit {
    /// Adds the azimuth and altitude to the camera's current direction.
    ///
    /// Wraps the resulting azimuth to [-180°, 180°) and clamps the altitude to [-90°, 90°].
    pub fn rotate(&mut self, az_delta: Angle, alt_delta: Angle) {
        self.rotate_to(self.dir.az() + az_delta, self.dir.alt() + alt_delta);
    }

    /// Rotates the camera to the **world**-space azimuth and altitude given.
    ///
    /// Wraps the azimuth to [-180°, 180°) and clamps the altitude to [-90°, 90°].
    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.dir = spherical(
            self.dir.r(),
            az.wrap(turns(-0.5), turns(0.5)),
            alt.clamp(turns(-0.25), turns(0.25)),
        );
    }

    /// Translates the camera's target point in **world** space.
    pub fn translate(&mut self, delta: Vec3<World>) {
        self.target += delta;
    }

    /// Moves the camera towards or away from the target.
    ///
    /// Multiplies the current camera distance by `factor`. The distance is
    /// clamped to zero. Note that if the distance becomes zero, you cannot use
    /// this method to make it nonzero again!
    ///
    /// To set an absolute zoom distance, use [`zoom_to`][Self::zoom_to].
    ///
    /// # Panics
    /// If `factor < 0`.
    pub fn zoom(&mut self, factor: f32) {
        assert!(factor >= 0.0, "zoom factor cannot be negative");
        self.zoom_to(self.dir.r() * factor);
    }
    /// Moves the camera to the given distance from the target.
    ///
    /// # Panics
    /// If `r < 0`.
    pub fn zoom_to(&mut self, r: f32) {
        assert!(r >= 0.0, "camera distance cannot be negative");
        self.dir[0] = r.max(0.0);
    }
}

//
// Local trait impls
//

#[cfg(feature = "fp")]
impl Transform for FirstPerson {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let &Self { pos, heading, .. } = self;

        let fwd = heading.to_cart();
        let right = az_alt(heading.az() - turns(0.25), turns(0.0)).to_cart();

        // World-to-view is inverse of camera's world transform
        let transl = translate(-pos.to_vec().to());
        let orient = orient_z(fwd.to(), right).transpose();

        transl.then(&orient).to()
    }
}

#[cfg(feature = "fp")]
impl Transform for Orbit {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        // TODO Figure out how to do this with orient
        //let fwd = self.dir.to_cart().normalize();
        //let o = orient_z(fwd, Vec3::X - 0.1 * Vec3::Z);

        // TODO Work out how and whether this is the correct inverse
        //      of the view-to-world transform
        translate(self.target.to_vec().to()) // to world-space target
            .then(&rotate_y(self.dir.az())) // to world-space az
            .then(&rotate_x(self.dir.alt())) // to world-space alt
            .then(&translate(self.dir.r() * Vec3::Z)) // view space
            .to()
    }
}

impl Transform for Spline {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let pos = self.spline.eval(self.t);
        let fwd = self.spline.tangent(self.t).normalize();

        // Inverse of the view-to-world: (OT)^-1 = T^-1 O^-1
        translate(-pos.to_vec().to())
            .then(&orient_z_y(fwd.to(), Vec3::Y).transpose())
            .to()
    }
}

impl Transform for Mat4x4<WorldToView> {
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
            target: Point3::default(),
            dir: az_alt(turns(0.0), turns(0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn camera_tests_here() {
        // TODO
    }
}
