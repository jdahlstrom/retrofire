use core::{
    array,
    fmt::{Debug, Formatter},
    marker::PhantomData as Pd,
    ops::{Add, Index, Sub},
};

use crate::math::{
    space::{Affine, Linear, Real},
    vec::Vector,
    ApproxEq,
};

#[repr(transparent)]
pub struct Point<Repr, Space = ()>(pub Repr, Pd<Space>);

/// A 2-point with `f32` components.
pub type Point2<Basis = ()> = Point<[f32; 2], Real<2, Basis>>;
/// A 3-point with `f32` components.
pub type Point3<Basis = ()> = Point<[f32; 3], Real<3, Basis>>;

/// Returns a real 2-point with `x` and `y` components.
pub const fn pt2<Sc, B>(x: Sc, y: Sc) -> Point<[Sc; 2], Real<2, B>> {
    Point([x, y], Pd)
}
/// Returns a real 3-point with `x`, `y`, and `z` components.
pub const fn pt3<Sc, B>(x: Sc, y: Sc, z: Sc) -> Point<[Sc; 3], Real<3, B>> {
    Point([x, y, z], Pd)
}

impl<R, Sp> Point<R, Sp> {
    #[inline]
    pub fn new(repr: R) -> Self {
        Self(repr, Pd)
    }
}

impl<const N: usize, B> Point<[f32; N], Real<N, B>> {
    #[cfg(feature = "fp")]
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        Affine::sub(self, other).len()
    }
    #[inline]
    pub fn distance_sqr(&self, other: &Self) -> f32 {
        Affine::sub(self, other).len_sqr()
    }
}

impl<R, B, Sc> Point<R, Real<2, B>>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the x component of `self`.
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
}

impl<R, Sc, B> Point<R, Real<3, B>>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the x component of `self`.
    #[inline]
    pub fn x(&self) -> Sc {
        self.0[0]
    }
    /// Returns the y component of `self`.
    #[inline]
    pub fn y(&self) -> Sc {
        self.0[1]
    }
    /// Returns the z component of `self`.
    #[inline]
    pub fn z(&self) -> Sc {
        self.0[2]
    }
}

impl<Sc, Sp, const N: usize> Affine for Point<[Sc; N], Sp>
where
    Sc: Linear<Scalar = Sc> + Copy,
{
    type Space = Sp;
    type Diff = Vector<[Sc; N], Sp>;
    const DIM: usize = N;

    #[inline]
    fn add(&self, other: &Self::Diff) -> Self {
        // TODO Profile performance of array::from_fn
        Self(array::from_fn(|i| self.0[i].add(&other.0[i])), Pd)
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self::Diff {
        Vector::new(array::from_fn(|i| self.0[i].sub(&other.0[i])))
    }
}

impl<Sc: ApproxEq, Sp, const N: usize> ApproxEq<Self, Sc>
    for Point<[Sc; N], Sp>
{
    fn approx_eq_eps(&self, other: &Self, eps: &Sc) -> bool {
        self.0.approx_eq_eps(&other.0, eps)
    }
    fn relative_epsilon() -> Sc {
        Sc::relative_epsilon()
    }
}

//
// Foreign trait impls
//

// Manual impls of Copy, Clone, Eq, and PartialEq to avoid
// superfluous where S: Trait bound

impl<R: Copy, S> Copy for Point<R, S> {}

impl<R: Clone, S> Clone for Point<R, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), Pd)
    }
}

impl<R: Default, S> Default for Point<R, S> {
    fn default() -> Self {
        Self(R::default(), Pd)
    }
}

impl<R: Debug, Sp: Debug + Default> Debug for Point<R, Sp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Point<{:?}>", Sp::default())?;
        Debug::fmt(&self.0, f)
    }
}

impl<R: Eq, S> Eq for Point<R, S> {}

impl<R: PartialEq, S> PartialEq for Point<R, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<R, Sp> From<R> for Point<R, Sp> {
    #[inline]
    fn from(repr: R) -> Self {
        Self(repr, Pd)
    }
}

impl<R, Sp> Add<<Self as Affine>::Diff> for Point<R, Sp>
where
    Self: Affine,
{
    type Output = Self;

    fn add(self, other: <Self as Affine>::Diff) -> Self {
        Affine::add(&self, &other)
    }
}

impl<R, Sp> Sub for Point<R, Sp>
where
    Self: Affine,
{
    type Output = <Self as Affine>::Diff;

    fn sub(self, other: Self) -> Self::Output {
        Affine::sub(&self, &other)
    }
}
