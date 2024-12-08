//! Pseudo-random number generation and distributions.

use crate::math::point::{Point, Point2, Point3};
use core::{array, fmt::Debug, ops::Range};

use super::vec::{Vec2, Vec3, Vector};

//
// Traits and types
//

pub type DefaultRng = Xorshift64;

/// Trait for generating values sampled from a probability distribution.
pub trait Distrib<R = DefaultRng>: Clone {
    /// The type of the elements of the sample space of `Self`, also called
    /// "outcomes".
    type Sample;

    /// Returns a pseudo-random value sampled from `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::*;
    /// // Simulate rolling a six-sided die
    /// let mut rng = DefaultRng::default();
    /// let d6 = Uniform(1..7).sample(&mut rng);
    /// assert_eq!(d6, 3);
    /// ```
    fn sample(&self, rng: &mut R) -> Self::Sample;

    /// Returns an iterator that yields samples from `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::*;
    /// // Simulate rolling a six-sided die
    /// let rng = DefaultRng::default();
    /// let mut iter = Uniform(1..7).samples(rng);
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// ```
    fn samples(&self, rng: R) -> Iter<Self, R> {
        Iter(self.clone(), rng)
    }
}

/// A pseudo-random number generator (PRNG) that uses a [Xorshift algorithm][^1]
/// to generate 64 bits of randomness at a time, represented by a `u64`.
///
/// Xorshift is a type of linear-feedback shift register that uses only three
/// right-shifts and three xor operations per generated number, making it very
/// efficient. Xorshift64 has a period of 2<sup>64</sup>-1: it yields every
/// number in the interval [1, 2<sup>64</sup>) exactly once before repeating.
///
/// [^1]: Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software,
///     8(14), 1–6. <https://doi.org/10.18637/jss.v008.i14>
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Xorshift64(pub u64);

/// A uniform distribution of values in a range.
#[derive(Clone, Debug)]
pub struct Uniform<T>(pub Range<T>);

/// A uniform distribution of unit 2-vectors.
#[derive(Copy, Clone, Debug)]
pub struct UnitCircle;

/// A uniform distribution of unit 3-vectors.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitSphere;

/// A uniform distribution of 2-vectors inside the (closed) unit disk.
#[derive(Copy, Clone, Debug, Default)]
pub struct VectorsOnUnitDisk;

/// A uniform distribution of 3-vectors inside the (closed) unit ball.
#[derive(Copy, Clone, Debug, Default)]
pub struct VectorsInUnitBall;

/// A uniform distribution of 2-points inside the (closed) unit disk.
#[derive(Copy, Clone, Debug, Default)]
pub struct PointsOnUnitDisk;

/// A uniform distribution of 3-points inside the (closed) unit ball.
#[derive(Copy, Clone, Debug, Default)]
pub struct PointsInUnitBall;

/// A Bernoulli distribution.
///
/// Generates boolean values such that:
/// * P(true) = p
/// * P(false) = 1 - p.
///
/// given a parameter p ∈ [0.0, 1.0].
#[derive(Copy, Clone, Debug)]
pub struct Bernoulli(pub f32);

/// Iterator returned by the [`Distrib::samples()`] method.
#[derive(Copy, Clone, Debug)]
pub struct Iter<D, R>(D, R);

//
// Inherent impls
//

impl Xorshift64 {
    /// A random 64-bit prime, used to initialize the generator returned by
    /// [`Xorshift64::default()`].
    pub const DEFAULT_SEED: u64 = 378682147834061;

    /// Returns a new `Xorshift64` seeded by the given number.
    ///
    /// Two `Xorshift64` instances generate the same sequence of pseudo-random
    /// numbers if and only if they were created with the same seed.
    /// (Technically, every `Xorshift64` instance yields values from the same
    /// sequence; the seed determines the starting point in the sequence).
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::rand::Xorshift64;
    /// let mut g = Xorshift64::from_seed(123);
    /// assert_eq!(g.next_bits(), 133101616827);
    /// assert_eq!(g.next_bits(), 12690785413091508870);
    /// assert_eq!(g.next_bits(), 7516749944291143043);
    /// ```
    ///
    /// # Panics
    ///
    /// If `seed` equals 0.
    pub fn from_seed(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift seed cannot be zero");
        Self(seed)
    }

    /// Returns a new `Xorshift64` seeded by the current system time.
    ///
    /// Note that depending on the precision of the system clock, two or more
    /// calls to this function in quick succession *may* return instances seeded
    /// by the same number.
    ///
    /// #  Examples
    /// ```
    /// # use std::thread;
    /// # use retrofire_core::math::rand::Xorshift64;
    /// let mut g = Xorshift64::from_time();
    /// thread::sleep_ms(1); // Just to be sure
    /// let mut h = Xorshift64::from_time();
    /// assert_ne!(g.next_bits(), h.next_bits());
    /// ```
    #[cfg(feature = "std")]
    pub fn from_time() -> Self {
        let t = std::time::SystemTime::UNIX_EPOCH
            .elapsed()
            .unwrap();
        Self(t.as_micros() as u64)
    }

    /// Returns 64 bits of pseudo-randomness.
    ///
    /// Successive calls to this function (with the same `self`) will yield
    /// every value in the interval [1, 2<sup>64</sup>) exactly once before
    /// starting to repeat the sequence.
    pub fn next_bits(&mut self) -> u64 {
        let Self(x) = self;
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }
}

//
// Foreign trait impls
//

/// An infinite iterator of pseudorandom values sampled from a distribution.
///
/// This type is returned by [`Distrib::samples`].
impl<D: Distrib> Iterator for Iter<D, DefaultRng> {
    type Item = D::Sample;

    /// Returns the next pseudorandom sample from this iterator.
    ///
    /// This method never returns `None`.
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.sample(&mut self.1))
    }
}

impl Default for Xorshift64 {
    /// Returns a `Xorshift64` seeded with [`DEFAULT_SEED`](Self::DEFAULT_SEED).
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::Xorshift64;
    /// let mut g = Xorshift64::default();
    /// assert_eq!(g.next_bits(), 11039719294064252060);
    /// ```
    fn default() -> Self {
        // Random 64-bit prime
        Self::from_seed(Self::DEFAULT_SEED)
    }
}

//
// Local trait impls
//

/// Uniformly distributed integers.
impl Distrib for Uniform<i32> {
    type Sample = i32;

    /// Returns a uniformly distributed `i32` in the range.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::*;
    /// let rng = DefaultRng::default();
    ///
    /// // Simulate rolling a six-sided die
    /// let mut iter = Uniform(1..7).samples(rng);
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// ```
    fn sample(&self, rng: &mut DefaultRng) -> i32 {
        let bits = rng.next_bits() as i32;
        // TODO rem introduces slight bias
        bits.rem_euclid(self.0.end - self.0.start) + self.0.start
    }
}

/// Uniformly distributed floats.
impl Distrib for Uniform<f32> {
    type Sample = f32;

    /// Returns a uniformly distributed `f32` in the range.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::*;
    /// let rng = DefaultRng::default();
    ///
    /// // Floats in the interval [-1, 1)
    /// let mut iter = Uniform(-1.0..1.0).samples(rng);
    /// assert_eq!(iter.next(), Some(0.19692874));
    /// assert_eq!(iter.next(), Some(-0.7686298));
    /// assert_eq!(iter.next(), Some(0.91969657));
    /// ```
    fn sample(&self, rng: &mut DefaultRng) -> f32 {
        let Range { start, end } = self.0;
        // Bit repr of a random f32 in range 1.0..2.0
        // Leaves a lot of precision unused near zero, but it's okay.
        let (exp, mantissa) = (127 << 23, rng.next_bits() >> 41);
        let unit = f32::from_bits(exp | mantissa as u32) - 1.0;
        unit * (end - start) + start
    }
}

impl<T, const N: usize> Distrib for Uniform<[T; N]>
where
    T: Copy,
    Uniform<T>: Distrib<Sample = T>,
{
    type Sample = [T; N];

    /// Returns the coordinates of a uniformly distributed point within
    /// the N-dimensional rectangular volume bounded by the range `self.0`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::rand::*;
    /// let rng = DefaultRng::default();
    ///
    /// // Pairs of integers [X, Y] such that 0 <= X < 4 and -2 <= Y <= 3
    /// let mut iter = Uniform([0, -2]..[4, 3]).samples(rng);
    /// assert_eq!(iter.next(), Some([0, -1]));
    /// assert_eq!(iter.next(), Some([1, 0]));
    /// assert_eq!(iter.next(), Some([3, 1]));
    /// ```
    fn sample(&self, rng: &mut DefaultRng) -> [T; N] {
        let Range { start, end } = self.0;
        array::from_fn(|i| Uniform(start[i]..end[i]).sample(rng))
    }
}

/// Uniformly distributed vectors within a rectangular volume.
impl<Sc, Sp, const DIM: usize> Distrib for Uniform<Vector<[Sc; DIM], Sp>>
where
    Sc: Copy,
    Uniform<[Sc; DIM]>: Distrib<Sample = [Sc; DIM]>,
{
    type Sample = Vector<[Sc; DIM], Sp>;

    /// Returns a vector uniformly sampled from the rectangular volume
    /// bounded by `self.0`.
    fn sample(&self, rng: &mut DefaultRng) -> Self::Sample {
        Uniform(self.0.start.0..self.0.end.0)
            .sample(rng)
            .into()
    }
}

/// Uniformly distributed points within a rectangular volume.
impl<Sc, Sp, const DIM: usize> Distrib for Uniform<Point<[Sc; DIM], Sp>>
where
    Sc: Copy,
    Uniform<[Sc; DIM]>: Distrib<Sample = [Sc; DIM]>,
{
    type Sample = Point<[Sc; DIM], Sp>;

    /// Returns a point uniformly sampled from the rectangular volume
    /// bounded by `self.0`.
    fn sample(&self, rng: &mut DefaultRng) -> Self::Sample {
        Uniform(self.0.start.0..self.0.end.0)
            .sample(rng)
            .into()
    }
}

#[cfg(feature = "fp")]
impl Distrib for UnitCircle {
    type Sample = Vec2;

    /// Returns a 2-vector uniformly distributed on the unit circle.
    fn sample(&self, rng: &mut DefaultRng) -> Vec2 {
        let d = Uniform([-1.0; 2]..[1.0; 2]);
        Vec2::from(d.sample(rng)).normalize()
    }
}

impl Distrib for VectorsOnUnitDisk {
    type Sample = Vec2;

    /// Returns a 2-vector uniformly sampled from the unit disk.
    ///
    /// # Example
    /// ```
    /// use retrofire_core::math::rand::*;
    /// let rng = &mut DefaultRng::default();
    ///
    /// let vec = VectorsOnUnitDisk.sample(rng);
    /// assert!(vec.len() <= 1.0);
    /// ```
    fn sample(&self, rng: &mut DefaultRng) -> Vec2 {
        let d = Uniform([-1.0f32; 2]..[1.0; 2]);
        loop {
            let v = Vec2::from(d.sample(rng));
            if v.len_sqr() <= 1.0 {
                return v;
            }
        }
    }
}

#[cfg(feature = "fp")]
impl Distrib for UnitSphere {
    type Sample = Vec3;

    /// Returns a vector uniformly distributed on the unit sphere.
    fn sample(&self, rng: &mut DefaultRng) -> Vec3 {
        let d = Uniform([-1.0f32; 3]..[1.0; 3]);
        Vec3::from(d.sample(rng)).normalize()
    }
}

impl Distrib for VectorsInUnitBall {
    type Sample = Vec3;

    /// Returns a vector uniformly distributed within the unit ball.
    fn sample(&self, rng: &mut DefaultRng) -> Vec3 {
        let d = Uniform([-1.0; 3]..[1.0; 3]);
        loop {
            let v = Vec3::from(d.sample(rng));
            if v.len_sqr() <= 1.0 {
                return v;
            }
        }
    }
}

impl Distrib for PointsOnUnitDisk {
    type Sample = Point2;

    /// Returns a 2-point uniformly sampled from the unit disk.
    ///
    /// See [`VectorsOnUnitDisk::sample`].
    fn sample(&self, rng: &mut DefaultRng) -> Point2 {
        VectorsOnUnitDisk.sample(rng).to_pt()
    }
}

impl Distrib for PointsInUnitBall {
    type Sample = Point3;

    /// Returns a 3-point uniformly sampled from the unit ball.
    ///
    /// See [`VectorsInUnitBall::sample`].
    fn sample(&self, rng: &mut DefaultRng) -> Point3 {
        VectorsInUnitBall.sample(rng).to_pt()
    }
}

impl Distrib for Bernoulli {
    type Sample = bool;

    /// Returns boolean values sampled from a Bernoulli distribution.
    fn sample(&self, rng: &mut DefaultRng) -> bool {
        Uniform(0.0f32..1.0).sample(rng) < self.0
    }
}

impl<D: Distrib, E: Distrib> Distrib for (D, E) {
    type Sample = (D::Sample, E::Sample);

    /// Returns a pair of samples, sampled from two separate distributions.
    fn sample(&self, rng: &mut DefaultRng) -> Self::Sample {
        (self.0.sample(rng), self.1.sample(rng))
    }
}

#[cfg(test)]
#[allow(clippy::manual_range_contains)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::vec::vec3;

    use super::*;

    const COUNT: usize = 1000;

    fn rng() -> DefaultRng {
        Default::default()
    }

    #[test]
    fn uniform_i32() {
        let dist = Uniform(-123..456);
        for r in dist.samples(rng()).take(COUNT) {
            assert!(-123 <= r && r < 456);
        }
    }

    #[test]
    fn uniform_f32() {
        let dist = Uniform(-1.23..4.56);
        for r in dist.samples(rng()).take(COUNT) {
            assert!(-1.23 <= r && r < 4.56);
        }
    }

    #[test]
    fn uniform_i32_array() {
        let dist = Uniform([0, -10]..[10, 15]);

        let sum = dist
            .samples(rng())
            .take(COUNT)
            .inspect(|&[x, y]| {
                assert!(0 <= x && x < 10);
                assert!(-10 <= y && x < 15);
            })
            .fold([0, 0], |[ax, ay], [x, y]| [ax + x, ay + y]);

        assert_eq!(sum, [4531, 1652]);
    }

    #[test]
    fn uniform_vec3() {
        let dist =
            Uniform(vec3::<f32, ()>(-2.0, 0.0, -1.0)..vec3(1.0, 2.0, 3.0));

        let mean = dist
            .samples(rng())
            .take(COUNT)
            .inspect(|v| {
                assert!(-2.0 <= v.x() && v.x() < 1.0);
                assert!(0.0 <= v.y() && v.y() < 2.0);
                assert!(-1.0 <= v.z() && v.z() < 3.0);
            })
            .sum::<Vec3>()
            / COUNT as f32;

        assert_eq!(mean, vec3(-0.46046025, 1.0209353, 0.9742225));
    }

    #[test]
    fn bernoulli() {
        let approx_100 = Bernoulli(0.1)
            .samples(rng())
            .take(COUNT)
            .filter(|&b| b)
            .count();
        assert_eq!(approx_100, 82);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn unit_circle() {
        for v in UnitCircle.samples(rng()).take(COUNT) {
            assert_approx_eq!(v.len_sqr(), 1.0, "non-unit vector: {v:?}");
        }
    }

    #[test]
    fn vectors_on_unit_disk() {
        for v in VectorsOnUnitDisk.samples(rng()).take(COUNT) {
            assert!(v.len_sqr() <= 1.0, "vector of len > 1.0: {v:?}");
        }
    }

    #[cfg(feature = "fp")]
    #[test]
    fn unit_sphere() {
        for v in UnitSphere.samples(rng()).take(COUNT) {
            assert_approx_eq!(v.len_sqr(), 1.0, "non-unit vector: {v:?}");
        }
    }

    #[test]
    fn unit_ball() {
        for v in VectorsInUnitBall.samples(rng()).take(COUNT) {
            assert!(v.len_sqr() <= 1.0, "vector of len > 1.0: {v:?}");
        }
    }

    #[test]
    fn zipped_pair() {
        let mut rng = rng();
        let dist = (Bernoulli(0.8), Uniform(0..4));
        assert_eq!(dist.sample(&mut rng), (true, 1));
        assert_eq!(dist.sample(&mut rng), (false, 3));
        assert_eq!(dist.sample(&mut rng), (true, 2));
    }
}
