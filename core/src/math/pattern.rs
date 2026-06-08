use crate::math::{
    Affine, Apply, Lerp, Point, Point2, Point3, Vector,
    grad::Gradient2,
    lerp,
    noise::{Perlin2, Perlin3},
    pt2,
};
use crate::prelude::AsMutSlice2;

pub trait Pattern: Sized {
    type Point: Affine;
    type Value;
    fn eval(&self, pt: Self::Point) -> Self::Value;

    /*fn gradient(&self, pt: P) -> impl Linear<Scalar = f32>
    where
        Self: Pattern<P, Value = f32>,
        P: Affine<Diff: Linear<Scalar = f32> + IndexMut<usize, Output = f32>>,
    {
        let eps = 0.001;

        let epsilons: [P::Diff; _] = from_fn(|i| {
            let mut v: P::Diff = Linear::zero();
            v[i] = eps;
            v
        });

        let val = self.eval(pt);

        let res: Vector<[f32; _]> = epsilons
            .map(|e| (self.eval(pt.add(&e)).sub(&val)) / eps)
            .into();
        res
    }*/

    fn transform<T: Apply<Self::Point>>(self, tf: T) -> Transform<T, Self> {
        Transform(tf, self)
    }

    fn repeat<V>(self, v: V) -> Repeat<V, Self>
    where
        Self::Point: Affine<Diff = V>,
    {
        Repeat(v, self)
    }

    fn bake(&self, mut buf: impl AsMutSlice2)
    where
        Self::Point: From<[f32; 2]>,
    {
        buf.as_mut_slice2()
            .fill_with(|i, j| self.eval([i as f32, j as f32].into()));
    }
}

pub struct Function<F>(F);

impl<F, R> Pattern for Function<F>
where
    F: Fn(Point2) -> R,
{
    type Point = Point2;
    type Value = R;

    fn eval(&self, pt: Self::Point) -> Self::Value {
        self.0(pt)
    }
}

pub struct Repeat<V, Pat>(pub V, pub Pat);
pub struct Warp<Off, Pat>(pub Off, pub Pat);

pub struct Mix<T, A, B>(pub T, pub A, pub B);

/// Transforms the input point before evaluating a pattern.
pub struct Transform<T, Pat>(pub T, pub Pat);

pub fn from_fn<F>(f: F) -> Function<F>
where
    Function<F>: Pattern,
{
    Function(f)
}

// Impls for module types

impl<T, P> Pattern for Transform<T, P>
where
    P: Pattern,
    T: Apply<P::Point, Output = P::Point>,
{
    type Point = P::Point;
    type Value = P::Value;

    fn eval(&self, pt: Self::Point) -> Self::Value {
        self.1.eval(self.0.apply(&pt))
    }
}

impl<Pat, Sp, const N: usize> Pattern for Repeat<Vector<[f32; N], Sp>, Pat>
where
    Pat: Pattern<Point = Point<[f32; N], Sp>>,
{
    type Point = Pat::Point;
    type Value = Pat::Value;

    fn eval(&self, pt: Self::Point) -> Self::Value {
        let Self(v, pat) = self;
        pat.eval(pt.zip_map(v.to_pt(), f32::rem_euclid))
    }
}

impl<Off, Pat> Pattern for Warp<Off, Pat>
where
    Pat: Pattern<Point: Copy>,
    Off: Pattern<Point = Pat::Point, Value = <Pat::Point as Affine>::Diff>,
{
    type Point = Pat::Point;
    type Value = Pat::Value;

    fn eval(&self, pt: Self::Point) -> Self::Value {
        let Self(off, pat) = self;
        pat.eval(pt.add(&off.eval(pt)))
    }
}

impl<T, A, B> Pattern for Mix<T, A, B>
where
    T: Pattern<Point: Copy, Value = f32>,
    A: Pattern<Point = T::Point, Value: Lerp>,
    B: Pattern<Point = T::Point, Value = A::Value>,
{
    type Point = A::Point;
    type Value = A::Value;

    fn eval(&self, pt: Self::Point) -> Self::Value {
        let Self(t, a, b) = self;
        lerp(t.eval(pt), a.eval(pt), b.eval(pt))
    }
}

// Impls for crate types

impl<C: Lerp> Pattern for Gradient2<C> {
    type Point = Point2;
    type Value = C;
    fn eval(&self, pt: Point2) -> C {
        self.eval(pt)
    }
}

impl Pattern for Perlin2 {
    type Point = Point2;
    type Value = f32;
    fn eval(&self, pt: Point2) -> f32 {
        self.eval(pt)
    }

    /*fn gradient(&self, pt: Point2) -> impl Linear<Scalar = f32>
    where
        Self: Pattern<Point2, Value = f32>,
    {
        self.gradient(pt.to())
    }*/
}

impl Pattern for Perlin3 {
    type Point = Point3;
    type Value = f32;
    fn eval(&self, pt: Point3) -> f32 {
        self.eval(pt)
    }

    /*fn gradient(&self, pt: Point3) -> Vec2
    where
        Self: Pattern<Point2, Value = f32>,
    {
        self.gradient(pt)
    }*/
}

// Impls for foreign types

impl Pattern for f32 {
    type Point = Point2;
    type Value = Self;

    fn eval(&self, _: Point2) -> Self {
        *self
    }
}

impl<R> Pattern for fn(Point2) -> R {
    type Point = Point2;
    type Value = R;

    fn eval(&self, pt: Point2) -> R {
        self(pt)
    }
}
impl<R> Pattern for fn(Point3) -> R {
    type Point = Point3;
    type Value = R;

    fn eval(&self, pt: Point3) -> R {
        self(pt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mat, math::splat};

    fn checker(pt: Point2) -> f32 {
        (pt.x().floor() as i32 & 1 ^ pt.y().floor() as i32 & 1) as f32
    }

    #[test]
    fn repeat() {
        let pat: fn(Point2) -> f32 = |p: Point2| p.x();

        let rep = Repeat(splat(1.0), pat);

        assert_eq!(rep.eval(pt2(123.5, 1.0)), 0.5);
    }

    #[test]
    fn mix() {
        let pat1 = from_fn(checker);
        let pat2 = from_fn(checker);

        let mix = Mix(0.5, pat1, pat2.transform(mat![2.0,0.0;0.0,2.0]));

        assert_eq!(mix.eval(pt2(1.5, 0.5)), 0.5);
    }
}
