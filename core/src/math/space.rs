//! Types and traits for representing linear (vector) and affine spaces.

use core::marker::PhantomData;

use crate::math::vary::{Iter, Vary};

/// Trait for types representing elements of an affine space.
///
/// # TODO
/// * More documentation, definition of affine space
pub trait Affine: Sized {
    /// The space that `Self` is the element type of.
    type Space;
    /// The (signed) difference of two values of `Self`.
    /// `Diff` must have the same dimension as `Self`.
    type Diff: Linear;

    /// The dimension of `Self`.
    const DIM: usize;

    /// Adds `diff` to `self` component-wise.
    ///
    /// `add` is commutative and associative.
    fn add(&self, diff: &Self::Diff) -> Self;

    /// Subtracts `other` from `self`, returning the (signed) difference.
    ///
    /// `sub` is anti-commutative: `v.sub(w) == w.sub(v).neg()`.
    fn sub(&self, other: &Self) -> Self::Diff;
}

impl Affine for f32 {
    type Space = ();
    type Diff = Self;
    const DIM: usize = 1;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
}

/// Trait for types representing elements of a linear space (vector space).
///
/// A `Linear` type is a type that is `Affine` and
/// additionally satisfies the following conditions:
///
/// * The difference type [`Diff`][Affine::Diff] is equal to `Self`
/// * The type has an additive identity, returned by the [`zero`][Self::zero] method
/// * Every value has an additive inverse, returned by the [`neg`][Self::neg] method
///
/// # TODO
/// * More documentation
pub trait Linear: Affine<Diff = Self> {
    /// The scalar type associated with `Self`.
    type Scalar: Sized;

    /// Returns the additive identity of `Self`.
    fn zero() -> Self;

    /// Returns the additive inverse of `self`.
    fn neg(&self) -> Self;

    /// Multiplies all components of `self` by `scalar`.
    ///
    /// `mul` is commutative and associative, and distributes over
    /// `add` and `sub` (up to rounding errors):
    /// ```
    /// # use retrofire_core::math::space::{Affine, Linear};
    /// # let [v, w, x, a] = [1.0f32, 2.0, 3.0, 4.0];
    /// v.mul(w) == w.mul(v);
    /// v.mul(w).mul(x) == v.mul(w.mul(x));
    /// v.mul(a).add(&w.mul(a)) == v.add(&w).mul(a);
    /// v.mul(a).sub(&w.mul(a)) == v.add(&w).sub(&a);
    /// ```
    fn mul(&self, scalar: Self::Scalar) -> Self;
}

/// Tag type for real vector spaces (Euclidean spaces) of dimension `DIM`.
/// For example, the type `Real<3>` corresponds to ℝ³.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Real<const DIM: usize, Basis = ()>(PhantomData<Basis>);

/// Tag type for the projective 4-space over reals, 𝗣<sub>4</sub>(ℝ).
/// The properties of this space make it useful for implementing perspective
/// projection. Clipping is also done in the projective space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Proj4;

impl Linear for f32 {
    type Scalar = Self;

    fn zero() -> Self {
        0.0
    }
    fn neg(&self) -> Self {
        -*self
    }
    fn mul(&self, scalar: Self) -> Self {
        self * scalar
    }
}

impl<V: Clone> Vary for V
where
    Self: Linear<Scalar = f32>,
{
    type Iter = Iter<Self>;
    type Diff = <Self as Affine>::Diff;

    #[inline]
    fn vary(self, step: Self::Diff, n: Option<u32>) -> Self::Iter {
        Iter { val: self, step, n }
    }

    fn dv_dt(&self, other: &Self, recip_dt: f32) -> Self::Diff {
        other.sub(self).mul(recip_dt)
    }

    /// Adds `delta` to `self`.
    #[inline]
    fn step(&self, delta: &Self::Diff) -> Self {
        self.add(delta)
    }

    fn z_div(&self, z: f32) -> Self {
        self.mul(z.recip())
    }
}
