//! Light sources

use core::fmt::Debug;

use crate::math::{Color3f, Linear, Mat4, Point3, Vec3, color::gray};

#[derive(Copy, Clone, Debug)]
pub struct Light<B: Default> {
    pub color: Color3f,
    pub kind: Kind<B>,
    pub falloff: u8,
}

impl<B: Default> Default for Light<B> {
    fn default() -> Self {
        Self {
            color: gray(1.0),
            kind: Kind::default(),
            falloff: 0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Kind<B: Default> {
    Directional {
        dir: Vec3<B>,
    },
    Point {
        pos: Point3<B>,
    },
    Spot {
        pos: Point3<B>,
        dir: Vec3<B>,
        radii: (f32, f32),
    },
}

impl<B: Default> Default for Kind<B> {
    fn default() -> Self {
        Self::Directional { dir: -Vec3::Y }
    }
}

impl<B: Debug + Default> Light<B> {
    pub fn new(color: Color3f, kind: Kind<B>) -> Self {
        Self { color, kind, ..Self::default() }
    }

    pub fn direction(&self, pt: Point3<B>) -> Vec3<B> {
        match self.kind {
            Kind::Point { pos, .. } => pos - pt,
            Kind::Directional { dir } => dir,
            Kind::Spot { pos, .. } => pos - pt,
        }
        .normalize()
    }

    pub fn eval(&self, pt: Point3<B>) -> (Color3f, Vec3<B>) {
        match self.kind {
            Kind::Point { pos } => (self.color, (pos - pt).normalize()),
            Kind::Directional { dir } => (self.color, dir),
            Kind::Spot { pos, dir, radii } => {
                // TODO pt.direction_to(pt2) method?
                let pt_dir = (pt - pos).normalize();
                let dir = dir.normalize();
                let dot = pt_dir.dot(&dir);
                let (r0, r1) = (1.0 - radii.0, 1.0 - radii.1);
                let color = if dot > r0 {
                    self.color
                } else if dot > r1 {
                    // TODO inv_lerp
                    let t = (dot - r1) / (r0 - r1); // ok: r0 - r1 != 0
                    self.color.mul(t)
                } else {
                    gray(0.0)
                };
                (color, -pt_dir)
            }
        }
        /*if self.falloff > 0 {
            let dist = (pt - self.pos).len_sqr() * 0.5 + 1.0;
            color.mul(dist) //.powi(-(self.falloff as i32)))
        } else {
            color
        }*/
    }

    pub fn transform<C: Default>(&self, mat: &Mat4<B, C>) -> Light<C> {
        let kind = match self.kind {
            Kind::Point { pos } => Kind::Point { pos: mat.apply(&pos) },
            Kind::Directional { dir } => {
                Kind::Directional { dir: mat.to().apply(&dir) }
            }
            Kind::Spot { pos, dir, radii } => Kind::Spot {
                pos: mat.apply(&pos),
                dir: mat.to().apply(&dir),
                radii,
            },
        };
        Light {
            kind,
            color: self.color,
            falloff: self.falloff,
        }
    }
}
