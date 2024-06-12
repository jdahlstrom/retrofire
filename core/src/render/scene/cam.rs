use core::ops::Range;

use crate::geom::Vertex;
use crate::math::{
    angle::{degs, rads, spherical, Angle, SphericalVec},
    mat::{orthographic, perspective, scale, viewport, Mat4x4},
    space::Linear,
    vary::Vary,
    vec::{vec2, vec3, Vec3},
};
use crate::render::{
    clip::ClipVec,
    scene::{Obj, Scene, WorldToView},
    shader::{FragmentShader, VertexShader},
    stats::Stats,
    target::Target,
    {self, Frag, Model, ModelToProj, NdcToScreen, ViewToProj},
};
use crate::util::rect::Rect;

#[cfg(feature = "fp")]
use crate::math::mat::{rotate_x, rotate_y, translate, RealToReal};

/// TODO
pub trait Mode {
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
    //fn view_to_world(&self) -> Mat4x4<WorldToView>;
}

/// TODO
#[derive(Copy, Clone, Debug)]
pub struct Camera<M> {
    pub mode: M,
    res: Resolution,
    aspect: AspectRatio,
    project: Mat4x4<ViewToProj>,
    viewport: Mat4x4<NdcToScreen>,
}

/// Camera field of view.
#[derive(Copy, Clone, Debug)]
pub enum Fov {
    /// Angle of view measured horizontally from the left to the right edge.
    #[cfg(feature = "fp")]
    Horizontal(Angle),
    /// Angle of view measured diagonally from one corner to the opposite.
    #[cfg(feature = "fp")]
    Diagonal(Angle),
    /// Ratio of focal length to aperture size.
    ///
    /// A ratio of 1.0 corresponds to a horizontal angle of view of 90 degrees.
    /// The larger the ratio, the narrower the field of view.
    FocalRatio(f32),
    ///
    Equiv35mm(f32),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Resolution(pub u32, pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AspectRatio {
    Custom(u32, u32),
    Square,
    FourThirds,
    FiveThirds,
    FiveFourths,
    EightFifths,
    SixteenNinths,
}

/// Camera mode using an arbitrary world-to-view matrix.
#[derive(Copy, Clone, Debug, Default)]
pub struct Matrix(pub Mat4x4<WorldToView>);

/// Camera mode with a standard first-person view.
#[derive(Copy, Clone, Debug, Default)]
pub struct FirstPerson {
    pub pos: Vec3,
    pub heading: SphericalVec,
}

/// TODO
#[derive(Copy, Clone, Debug)]
pub struct Orbit {
    pub target_pos: Vec3,
    pub dir: SphericalVec,
}

impl<M: Mode> Camera<M> {
    /// Creates a new camera with the given resolution.
    pub fn new(res: Resolution) -> Self
    where
        M: Default,
    {
        Self::with_mode(res, M::default())
    }

    /// Creates a new camera with the given resolution and mode.
    pub fn with_mode(res: Resolution, mode: M) -> Self {
        let Resolution(w, h) = res;
        Self {
            res,
            mode,
            aspect: res.aspect_ratio(),
            project: Default::default(),
            viewport: viewport(vec2(0, 0)..vec2(w, h)),
        }
    }

    /// Sets the viewport bounds.
    ///
    /// The default viewport is
    pub fn viewport(self, bounds: impl Into<Rect<u32>>) -> Self {
        let fullscreen = (0..self.res.0, 0..self.res.1).into();

        let Rect {
            left: Some(l),
            top: Some(t),
            right: Some(r),
            bottom: Some(b),
        } = bounds.into().intersect(&fullscreen)
        else {
            unreachable!("intersection with bounded should be bounded")
        };
        Self {
            viewport: viewport(vec2(l, t)..vec2(r, b)),
            aspect: AspectRatio::Custom(r - l, b - t),
            ..self
        }
    }

    /// Configures a perspective projection.
    pub fn perspective(self, fov: Fov, near_far: Range<f32>) -> Self {
        #[cfg(feature = "fp")]
        use crate::math::float::f32;

        let fr = match fov {
            #[cfg(feature = "fp")]
            Fov::Horizontal(a) => a.tan(),
            #[cfg(feature = "fp")]
            Fov::Diagonal(a) => (a * f32::cos(self.aspect.as_f32())).tan(),
            Fov::FocalRatio(f) => f,
            Fov::Equiv35mm(f) => f / 21.6, // Half diagonal of 35mm film
        };
        Self {
            project: perspective(fr, self.aspect.as_f32(), near_far),
            ..self
        }
    }

    pub fn orthographic(self, bounds: Range<Vec3>) -> Self {
        Self {
            project: orthographic(bounds),
            ..self
        }
    }

    pub fn render<A: Clone, Var: Vary, Uni: Copy, Shd>(
        &self,
        scene: &Scene<A>,
        shader: &Shd,
        uniform: Uni,
        target: &mut impl Target,
    ) -> Stats
    where
        Shd: for<'a> VertexShader<
                Vertex<Vec3<Model>, A>,
                (&'a Mat4x4<ModelToProj>, Uni),
                Output = Vertex<ClipVec, Var>,
            > + FragmentShader<Frag<Var>>,
    {
        let mut stats = Stats::new();
        let world_to_proj = scale(vec3(1.0, -1.0, -1.0))
            .to()
            .then(&self.mode.world_to_view())
            .then(&self.project);

        for Obj { mesh, transform } in &scene.objs {
            let tf = transform.then(&world_to_proj);

            stats += render::render(
                &mesh.faces,
                &mesh.verts,
                shader,
                (&tf, uniform),
                self.viewport,
                target,
            )
        }
        stats
    }
}

#[rustfmt::skip]
impl Resolution {
    pub const CGA:      Self = Self(320, 200);   // 4:3
    pub const MODE_13H: Self = Self::CGA;
    pub const QVGA:     Self = Self(320, 240);   // 4:3
    pub const VGA:      Self = Self(640, 480);   // 4:3
    pub const WVGA:     Self = Self(800, 480);   // 15:9
    pub const SVGA:     Self = Self(800, 600);   // 4:3
    pub const XGA:      Self = Self(1024, 768);  // 4:3
    pub const WXGA:     Self = Self(1280, 768);  // 5:3
    pub const SXGA:     Self = Self(1280, 1024); // 5:4
    pub const UXGA:     Self = Self(1600, 1200); // 4:3
    pub const WUXGA:    Self = Self(1920, 1200); // 16:10
    pub const WQXGA:    Self = Self(2560, 1600); // 16:10

    pub const HD:       Self = Self(1280, 720);  // 16:9
    pub const FULL_HD:  Self = Self(1920, 1080); // 16:9
    pub const DCI_2K:   Self = Self(2048, 1080); // ~17:9
    pub const QHD:      Self = Self(2560, 1440); // 16:9
    pub const UHD_4K:   Self = Self(3840, 2160); // 16:9
    pub const DCI_4K:   Self = Self(4096, 2160); // ~17:9

    pub fn aspect_ratio(self) -> AspectRatio {
        AspectRatio::from_ratio(self.0, self.1)
    }
}

impl AspectRatio {
    fn as_f32(self) -> f32 {
        use AspectRatio::*;
        match self {
            Custom(w, h) => w as f32 / h as f32,
            Square => 1.0,
            FourThirds => 4.0 / 3.0,
            FiveThirds => 5.0 / 3.0,
            FiveFourths => 5.0 / 4.0,
            EightFifths => 8.0 / 5.0,
            SixteenNinths => 16.0 / 9.0,
        }
    }
    #[rustfmt::skip]
    fn from_ratio(w: u32, h: u32) -> Self {
        use AspectRatio::*;
        if w == h { Square }
        else if w * 3 == h * 4  { FourThirds }
        else if w * 3 == h * 5  { FiveThirds }
        else if w * 4 == h * 5  { FiveFourths }
        else if w * 5 == h * 8  { EightFifths }
        else if w * 9 == h * 16 { SixteenNinths }
        else { Custom(w, h) }
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

#[cfg(feature = "fp")]
impl Orbit {
    pub fn zoom(&mut self, factor: f32) {
        self.dir[0] *= factor;
    }

    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.dir = spherical(
            self.dir.r(),
            az.wrap(degs(-180.0), degs(180.0)),
            alt.clamp(degs(-90.0), degs(90.0)),
        );
    }
}

impl Mode for Matrix {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        self.0
    }
}

#[cfg(feature = "fp")]
impl Mode for FirstPerson {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let &Self { pos, heading: dir, .. } = self;
        let fwd_move = spherical(1.0, dir.az(), degs(0.0));
        let fwd = self.heading;
        let right = vec3(0.0, 1.0, 0.0).cross(&fwd_move.into());

        let transl = translate(-pos);
        let orient = crate::math::mat::orient_z(fwd.into(), right);

        transl.then(&orient).to()
    }
}

#[cfg(feature = "fp")]
impl Mode for Orbit {
    fn world_to_view(&self) -> Mat4x4<WorldToView> {
        let rot = rotate_y(self.dir.az()).then(&rotate_x(self.dir.alt()));
        let transl = translate(vec3(0.0, 0.0, self.dir.r()));

        rot.then(&transl).to()
    }
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            target_pos: Vec3::zero(),
            dir: spherical(1.0, rads(0.0), rads(0.0)),
        }
    }
}
