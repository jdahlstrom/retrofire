use geom::mesh::Mesh;
use math::{Angle, Linear};
use math::mat::Mat4;
use math::transform::{orient_z, rotate_y, translate, Transform};
use math::vec::*;

#[derive(Default, Clone)]
pub struct Obj<VA, FA> {
    pub tf: Mat4,
    pub mesh: Mesh<VA, FA>,
}

impl<VA, FA> Transform for Obj<VA, FA> {
    fn transform(&mut self, tf: &Mat4) {
        self.tf *= tf;
    }
}

#[derive(Default, Clone)]
pub struct Scene<VA, FA> {
    pub objects: Vec<Obj<VA, FA>>,
    pub camera: Mat4,
}

#[derive(Clone, Debug, Default)]
pub struct FpsCamera {
    pub pos: Vec4,
    pub azimuth: Angle,
    pub altitude: Angle,
}

impl FpsCamera {
    pub fn new(pos: Vec4, azimuth: Angle) -> Self {
        Self { pos, azimuth, ..Self::default() }
    }

    pub fn translate(&mut self, dir: Vec4) {
        let fwd = &rotate_y(self.azimuth) * Z;
        let right = Y.cross(fwd);
        self.pos += Vec4::lincomb(fwd, dir.z, right, dir.x);
    }

    pub fn rotate(&mut self, az: Angle, alt: Angle) {
        self.azimuth = (self.azimuth + az)
            .wrap(-Angle::STRAIGHT, Angle::STRAIGHT);
        self.altitude = (self.altitude + alt)
            .clamp(-Angle::RIGHT, Angle::RIGHT);
    }

    pub fn world_to_view(&self) -> Mat4 {
        let fwd_move = polar(1.0, self.azimuth);
        let fwd = spherical(1.0, self.azimuth, -self.altitude);
        let right = Y.cross(fwd_move);

        let orient = orient_z(fwd, right).transpose();
        let transl = translate(-self.pos);

        transl * orient
    }
}
