//! Linear algebra and more.
//!
//! Includes [vectors][self::vec], [matrices][mat], [colors][color],
//! [angles][angle], and [Bezier splines][spline], as well as support for
//! custom affine and linear types usable as [varyings][vary], and utilities
//! such as approximate equality comparisons.
//! TODO: pseudorandom numbers.
//!
//! This library is more strongly typed than many other similar ones. It aims
//! to diagnose at compile time many errors that might otherwise only manifest
//! as graphical glitches, runtime panics, or even – particularly in languages
//! that are unsafe-by-default – undefined behavior.
//!
//! In particular, vectors and colors are tagged with a type that represents
//! the *space* they're embedded in, and values in different spaces cannot be
//! mixed without explicit conversion (transformation). Matrices, similarly,
//! are tagged by both source and destination space, and can only be applied
//! to matching vectors. Angles are strongly typed as well, to allow working
//! with different angular units without confusion.

pub use angle::{degs, rads, turns, Angle};
pub use mat::{Mat3x3, Mat4x4, Matrix};
pub use vec::{vec2, vec3, vec4};
pub use vec::{Affine, Linear};
pub use vec::{Vec2, Vec2i, Vec3, Vec3i, Vec4, Vec4i, Vector};

pub mod angle;
pub mod approx;
pub mod color;
pub mod mat;
pub mod spline;
pub mod vary;
pub mod vec;
