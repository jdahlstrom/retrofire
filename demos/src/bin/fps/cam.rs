use std::ops::Range;

use re::math::angle::SphericalVec;
use re::math::mat::{
    orient_z, perspective, rotate_y, translate, viewport, RealToProjective,
    RealToReal,
};
use re::math::space::{Affine, Linear};
use re::math::{degs, spherical, vec2, Angle, Mat4x4, Vec2i, Vec3};
use re::render::World;

use crate::{ProjMat, ScreenMat, ViewMat, X, Y};

#[derive(Clone, Debug, Default)]
pub struct FpsCamera {
    pub pos: Vec3,
    pub azimuth: Angle,
    pub altitude: Angle,

    pub project: ProjMat,
    pub viewport: ScreenMat,
}

impl FpsCamera {
    pub fn new(viewport_w: u32, viewport_h: u32) -> Self {
        let bounds = vec2(0, 0)..(vec2(viewport_w as i32, viewport_h as i32));
        Self {
            viewport: viewport(bounds),
            ..Self::default()
        }
    }

    pub fn perspective(self, focal_ratio: f32, near_far: Range<f32>) -> Self {
        let aspect_ratio = self.viewport.0[0][0] / self.viewport.0[1][1];
        Self {
            project: perspective(focal_ratio, aspect_ratio, near_far),
            ..self
        }
    }

    pub fn viewport(self, bounds: Range<Vec2i>) -> Self {
        Self {
            viewport: viewport(bounds),
            ..self
        }
    }

    pub fn look_at(&mut self, pt: Vec3) {
        let dir = SphericalVec::from(pt.sub(&self.pos));
        self.azimuth = dir.az();
        self.altitude = dir.alt();
    }

    pub fn translate(&mut self, delta: Vec3) {
        // Zero azimuth means parallel to the x axis
        let fwd = rotate_y(self.azimuth).apply(&X);
        let up = Y;
        let right = up.cross(&fwd);

        // / rx ux fx \ / dx \     / rx ry rz \ T / dx \
        // | ry uy fy | | dy |  =  | ux uy uz |   | dy |
        // \ rz uz fz / \ dz /     \ fx fy fz /   \ dz /

        let m: Mat4x4<RealToReal<3>> =
            Mat4x4::from_basis(right, up, fwd).transpose();

        self.pos = self.pos.add(&m.apply(&delta));
    }

    pub fn rotate(&mut self, az: Angle, alt: Angle) {
        self.rotate_to(self.azimuth + az, self.altitude + alt);
    }

    pub fn rotate_to(&mut self, az: Angle, alt: Angle) {
        self.azimuth = az.wrap(degs(-180.0), degs(180.0));
        self.altitude = alt.clamp(degs(-90.0), degs(90.0));
    }

    pub fn world_to_view(&self) -> ViewMat {
        let &Self { pos, azimuth, altitude, .. } = self;
        let fwd_move = spherical(1.0, azimuth, degs(0.0));
        let fwd = spherical(1.0, azimuth, altitude);
        let right = Y.cross(&fwd_move.into());

        let transl = translate(pos.neg());
        let orient = orient_z(fwd.into(), right);

        transl.then(&orient).to()
    }

    pub fn world_to_project(&self) -> Mat4x4<RealToProjective<World>> {
        self.world_to_view().then(&self.project)
    }
}
