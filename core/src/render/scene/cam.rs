use core::ops::Range;

use crate::geom::Vertex;
use crate::math::angle::SphericalVec;
use crate::math::mat::{
    orthographic, rotate_x, rotate_y, scale, translate, viewport, RealToReal,
};
use crate::math::{
    degs, rads, spherical, vec2, vec3, Angle, Linear, Mat4x4, Vary, Vec3,
};
use crate::render;
use crate::render::clip::ClipVec;
use crate::render::scene::{Obj, Scene, WorldToView};
use crate::render::shader::{FragmentShader, VertexShader};
use crate::render::stats::Stats;
use crate::render::target::Target;
use crate::render::{Frag, Model, ModelToProj, NdcToScreen, ViewToProj};
use crate::util::rect::Rect;

/// TODO
pub trait Mode {
    fn world_to_view(&self) -> Mat4x4<WorldToView>;
}

/// TODO
#[derive(Copy, Clone, Debug, Default)]
pub struct Camera<M> {
    pub mode: M,
    res: (u32, u32),
    aspect: f32,
    project: Mat4x4<ViewToProj>,
    viewport: Mat4x4<NdcToScreen>,
}

/// TODO
#[derive(Copy, Clone, Debug)]
pub enum Fov {
    #[cfg(feature = "fp")]
    Horizontal(Angle),
    #[cfg(feature = "fp")]
    Diagonal(Angle),
    FocalRatio(f32),
    Equiv35mm(f32),
}

/// TODO
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
    /// TODO
    pub fn new(res_x: u32, res_y: u32) -> Self
    where
        M: Default,
    {
        Self::with_mode(res_x, res_y, M::default())
    }

    /// TODO
    pub fn with_mode(res_x: u32, res_y: u32, mode: M) -> Self {
        Self {
            res: (res_x, res_y),
            aspect: res_x as f32 / res_y as f32,
            mode,
            project: Default::default(),
            viewport: viewport(vec2(0, 0)..vec2(res_x, res_y)),
        }
    }

    /// TODO
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
            aspect: (r - l) as f32 / (b - t) as f32,
            ..self
        }
    }

    /// TODO
    pub fn perspective(self, fov: Fov, near_far: Range<f32>) -> Self {
        #[cfg(feature = "fp")]
        use crate::math::float::f32;
        use crate::prelude::perspective;

        let fr = match fov {
            #[cfg(feature = "fp")]
            Fov::Horizontal(a) => a.tan(),
            #[cfg(feature = "fp")]
            Fov::Diagonal(a) => (a * f32::cos(self.aspect)).tan(),
            Fov::FocalRatio(f) => f,
            Fov::Equiv35mm(f) => f / 18.0, // Half width of 35mm film
        };
        Self {
            project: perspective(fr, self.aspect, near_far),
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
        let fwd = crate::math::mat::rotate_y(self.heading.az())
            .apply(&vec3(1.0, 0.0, 0.0));
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
        // TODO !!!
        self.dir[2] *= factor;
    }

    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.dir = spherical(
            self.dir.r(),
            az.wrap(degs(-180.0), degs(180.0)),
            alt.clamp(degs(-90.0), degs(90.0)),
        );
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
