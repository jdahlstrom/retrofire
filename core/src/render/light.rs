//! Light sources

use core::fmt::{self, Debug, Formatter};

use crate::math::{Color3f, Mat4, Point3, Vec3, color::gray, inv_lerp};

/// A light source.
#[derive(Copy, Clone, PartialEq)]
pub struct Light<B> {
    pub color: Color3f,
    pub kind: Kind<B>,
    pub falloff: u8,
}

#[derive(Copy, Clone, PartialEq)]
pub enum Kind<B> {
    /// A light source "at infinity", so that the light rays arrive
    /// approximately parallel and the direction of the light source
    /// is the same for every point. For example the sun or the moon.
    Directional(Vec3<B>),
    /// A light source radiating omnidirectionally from a single point.
    Point(Point3<B>),
    /// A light source radiating from a point in a cone shape.
    Spot {
        pos: Point3<B>,
        dir: Vec3<B>,
        radii: (f32, f32),
    },
}

impl<B: Copy> Light<B> {
    /// Creates a new light source of the given color and kind.
    pub fn new(color: Color3f, mut kind: Kind<B>) -> Self {
        match &mut kind {
            Kind::Directional(dir) => *dir = dir.normalize(),
            Kind::Spot { dir, .. } => *dir = dir.normalize(),
            _ => {}
        };
        Self { color, kind, ..Self::default() }
    }

    /// Returns the normalized direction vector from a point to `self`.
    #[inline]
    pub fn direction(&self, pt: Point3<B>) -> Vec3<B> {
        match self.kind {
            Kind::Point(pos) => (pos - pt).normalize_approx(),
            Kind::Directional(dir) => dir,
            Kind::Spot { pos, .. } => (pos - pt).normalize_approx(),
        }
    }

    #[inline]
    pub fn eval(&self, pt: Point3<B>) -> (Color3f, Vec3<B>) {
        let pt_dir = self.direction(pt);
        let color = match self.kind {
            Kind::Point(_) => self.color,
            Kind::Directional(_) => self.color,
            Kind::Spot { dir, radii, .. } => {
                let dot = pt_dir.dot(&dir);
                let (r0, r1) = (1.0 - radii.0, 1.0 - radii.1);
                if dot > r0 {
                    self.color
                } else if dot > r1 {
                    let t = inv_lerp(dot, r1, r0); // ok: r0 != r1
                    self.color * t
                } else {
                    gray(0.0)
                }
            }
        };
        (color, pt_dir)
    }

    pub fn transform<C>(&self, mat: &Mat4<B, C>) -> Light<C> {
        let Self { color, kind, falloff } = *self;
        let kind = match kind {
            Kind::Point(pos) => Kind::Point(mat.apply(&pos)),
            Kind::Directional(dir) => Kind::Directional(mat.apply(&dir)),
            Kind::Spot { pos, dir, radii } => Kind::Spot {
                pos: mat.apply(&pos),
                dir: mat.apply(&dir),
                radii,
            },
        };
        Light { kind, color, falloff }
    }
}

// Ugh, manual impls to avoid B: Default bound on the types...

impl<B: Debug + Default> Debug for Light<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Light")
            .field("kind", &self.kind)
            .field("color", &self.color)
            .field("falloff", &self.falloff)
            .finish()
    }
}

impl<B> Default for Light<B> {
    fn default() -> Self {
        Self {
            color: gray(1.0),
            kind: Kind::default(),
            falloff: 0,
        }
    }
}

impl<B: Debug + Default> Debug for Kind<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Directional(dir) => {
                f.debug_tuple("Directional").field(&dir).finish()
            }
            Kind::Point(pt) => f.debug_tuple("Point").field(&pt).finish(),
            Kind::Spot { pos, dir, radii } => f
                .debug_struct("Spot")
                .field("pos", &pos)
                .field("dir", &dir)
                .field("radii", radii)
                .finish(),
        }
    }
}

impl<B> Default for Kind<B> {
    fn default() -> Self {
        Self::Directional(Vec3::Y)
    }
}
