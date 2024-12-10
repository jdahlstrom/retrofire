//! Light sources

use crate::math::{
    angle::Angle,
    color::Color3f,
    mat::{Mat4x4, RealToReal},
    point::Point3,
    space::Linear,
    vec::Vec3,
};

#[derive(Copy, Clone, Debug, Default)]
pub struct Light<B: Default> {
    pub pos: Point3<B>,
    pub color: Color3f,
    pub kind: Kind,
    pub falloff: u8,
}

#[derive(Copy, Clone, Debug, Default)]
pub enum Kind {
    #[default]
    Point,
    Directional,
    #[cfg(feature = "fp")]
    Spot {
        dir: Vec3,
        radius: Angle,
    },
}

impl<B: Default> Light<B> {
    pub fn eval(&self, pt: Point3<B>) -> Color3f {
        let color = match self.kind {
            Kind::Point => self.color,
            Kind::Directional => self.color,
            #[cfg(feature = "fp")]
            Kind::Spot { dir, radius } => {
                let pt_dir = (pt - self.pos).normalize();
                let dir = dir.normalize();
                let dot = pt_dir.dot(&dir.to());
                let cos = radius.cos();
                if dot > cos {
                    self.color
                } else {
                    let cos2 = (radius * 2.0).cos();
                    self.color
                        .mul(((dot - cos2) / (cos - cos2)).max(0.0))
                }
            }
        };
        if self.falloff > 0 {
            let dist = (pt - self.pos).len_sqr() * 0.5 + 1.0;
            color.mul(dist) //.powi(-(self.falloff as i32)))
        } else {
            color
        }
    }

    pub fn transform<C: Default>(
        &self,
        mat: &Mat4x4<RealToReal<3, B, C>>,
    ) -> Light<C> {
        Light {
            pos: mat.apply_pt(&self.pos),
            color: self.color,
            kind: match self.kind {
                #[cfg(feature = "fp")]
                Kind::Spot { dir, radius } => Kind::Spot {
                    dir: mat.to().apply3(&dir),
                    radius,
                },
                _ => self.kind,
            },
            falloff: self.falloff,
        }
    }
}
