//! Cameras and camera transforms.

use core::ops::Range;

#[cfg(feature = "fp")]
use crate::math::{
    Angle, Vec3, orient_z, rotate_x, rotate_y, spherical, translate, turns,
};
use crate::math::{
    Lerp, Mat4, Point3, SphericalVec, Vary, mat::ProjMat3, mat::RealToReal,
    orthographic, perspective, pt2, viewport,
};
use crate::util::{Dims, rect::Rect};

use super::{Clip, Context, Ndc, Render, Screen, Shader, Target, View, World};

/// Trait for different modes of camera motion.
pub trait Transform {
    /// Returns the current world-to-view matrix.
    fn world_to_view(&self) -> Mat4<World, View>;
}

/// Camera field of view.
///
/// Specifies how wide or narrow the *angle of view* of the camera is.
/// The smaller the angle, the more "zoomed in" the image is.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Fov {
    /// Ratio of focal length to aperture size.
    ///
    /// This value is also called the 𝑓-number. The value of 1.0 corresponds
    /// to a horizontal angle of view of 90°. Values less than 1.0 correspond
    /// to wider and values greater than 1.0 to narrower angles of view.
    FocalRatio(f32),
    /// Focal length in [35mm-equivalent millimeters.][1]
    ///
    /// For instance, the value of 28.0 corresponds to the moderate wide-angle
    /// view of a 28mm "full-frame" lens.
    ///
    /// [1]: https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
    Equiv35mm(f32),
    /// Angle of view as measured from the left to the right edge of the image.
    #[cfg(feature = "fp")]
    Horizontal(Angle),
    /// Angle of view as measured from the top to the bottom edge of the image.
    #[cfg(feature = "fp")]
    Vertical(Angle),
    /// Angle of view as measured between two opposite corners of the image.
    #[cfg(feature = "fp")]
    Diagonal(Angle),
}

/// Type to manage the world-to-viewport transformation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Camera<Tf> {
    /// World-to-view transform.
    pub transform: Tf,
    /// Viewport width and height.
    pub dims: Dims,
    /// Projection matrix.
    pub project: ProjMat3<View>,
    /// Viewport matrix.
    pub viewport: Mat4<Ndc, Screen>,
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

/// Creates a unit `SphericalVec` from azimuth and altitude.
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

//
// Inherent impls
//

impl Fov {
    /// TODO
    pub fn focal_ratio(self, aspect_ratio: f32) -> f32 {
        use Fov::*;
        #[cfg(feature = "fp")]
        fn ratio(a: Angle) -> f32 {
            1.0 / (a / 2.0).tan()
        }
        match self {
            FocalRatio(r) => r,
            Equiv35mm(mm) => mm / (36.0 / 2.0), // half frame width

            #[cfg(feature = "fp")]
            Horizontal(a) => ratio(a),

            #[cfg(feature = "fp")]
            Vertical(a) => ratio(a) / aspect_ratio,

            #[cfg(feature = "fp")]
            Diagonal(a) => {
                use crate::math::float::f32;
                let diag = f32::sqrt(1.0 + 1.0 / aspect_ratio / aspect_ratio);
                ratio(a) * diag
            }
        }
    }
}

impl Camera<()> {
    /// Creates a camera with the given resolution.
    pub fn new(dims: Dims) -> Self {
        Self {
            dims,
            viewport: viewport(pt2(0, 0)..pt2(dims.0, dims.1)),
            ..Self::default()
        }
    }

    /// Sets the world-to-view transform of this camera.
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

    /// Sets up perspective projection with the given field of view
    /// and near–far range.
    ///
    /// The endpoints of `near_far` denote the distance of the near and far
    /// clipping planes.
    ///
    /// # Panics
    /// * If any parameter value is non-positive.
    /// * If `near_far` is an empty range.
    pub fn perspective(mut self, fov: Fov, near_far: Range<f32>) -> Self {
        let aspect = self.dims.0 as f32 / self.dims.1 as f32;

        self.project = perspective(fov.focal_ratio(aspect), aspect, near_far);
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
    pub fn world_to_project(&self) -> ProjMat3<World> {
        self.transform.world_to_view().then(&self.project)
    }

    /// Renders the given geometry from the viewpoint of this camera.
    pub fn render<B, Prim, Vtx: Clone, Var: Lerp + Vary, Uni: Copy, Shd>(
        &self,
        prims: impl AsRef<[Prim]>,
        verts: impl AsRef<[Vtx]>,
        to_world: &Mat4<B, World>,
        shader: &Shd,
        uniform: Uni,
        target: &mut impl Target,
        ctx: &Context,
    ) where
        Prim: Render<Var> + Clone,
        [<Prim>::Clip]: Clip<Item = Prim::Clip>,
        Shd: for<'a> Shader<Vtx, Var, (&'a ProjMat3<B>, Uni)>,
    {
        let tf = to_world.then(&self.world_to_project());

        super::render(
            prims.as_ref(),
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
            pos: Point3::origin(),
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
        let fwd = az_alt(self.heading.az(), turns(0.0)).to_cart();
        let up = Vec3::Y;
        let right = up.cross(&fwd);

        let to_world = Mat4::from_linear(right, up, fwd);
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
    fn world_to_view(&self) -> Mat4<World, View> {
        let &Self { pos, heading, .. } = self;
        let fwd_move = az_alt(heading.az(), turns(0.0)).to_cart();
        let fwd = heading.to_cart();
        let right = Vec3::Y.cross(&fwd_move);

        // World-to-view is inverse of camera's world transform
        let transl = translate(-pos.to_vec().to());
        let orient = orient_z(fwd.to(), right).transpose();

        transl.then(&orient).to()
    }
}

#[cfg(feature = "fp")]
impl Transform for Orbit {
    fn world_to_view(&self) -> Mat4<World, View> {
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

impl Transform for Mat4<World, View> {
    fn world_to_view(&self) -> Mat4<World, View> {
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
    use super::*;

    use Fov::*;

    #[test]
    fn camera_tests_here() {
        // TODO
    }

    #[test]
    fn fov_focal_ratio() {
        assert_eq!(FocalRatio(2.345).focal_ratio(1.0), 2.345);
        assert_eq!(FocalRatio(2.345).focal_ratio(2.0), 2.345);

        assert_eq!(Equiv35mm(18.0).focal_ratio(1.0), 1.0);
        assert_eq!(Equiv35mm(36.0).focal_ratio(1.5), 2.0);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn angle_of_view_focal_ratio_with_unit_aspect_ratio() {
        use crate::math::degs;
        use core::f32::consts::SQRT_2;
        const SQRT_3: f32 = 1.7320509;

        assert_eq!(Horizontal(degs(60.0)).focal_ratio(1.0), SQRT_3);
        assert_eq!(Vertical(degs(60.0)).focal_ratio(1.0), SQRT_3);
        assert_eq!(Diagonal(degs(60.0)).focal_ratio(1.0), SQRT_3 * SQRT_2);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn angle_of_view_focal_ratio_with_other_aspect_ratio() {
        use crate::math::degs;
        const SQRT_3: f32 = 1.7320509;

        assert_eq!(Horizontal(degs(60.0)).focal_ratio(SQRT_3), SQRT_3);
        assert_eq!(Vertical(degs(60.0)).focal_ratio(SQRT_3), 1.0);
        assert_eq!(Diagonal(degs(60.0)).focal_ratio(SQRT_3), 2.0);
    }
}
