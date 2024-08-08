use core::{array, fmt::Debug, ops::Range};

use crate::math::{Vec2, Vec3, Vector};

//
// Traits and types
//

type DefaultRng = Xorshift64;

/// Trait for generating values sampled from a probability distribution.
pub trait Distrib<R = DefaultRng>: Clone {
    /// The type of the elements of the sample space of `Self`, also called
    /// "outcomes".
    type Sample;

    /// Returns a pseudo-random value sampled from `self`.
    fn sample(&self, rng: &mut R) -> Self::Sample;

    /// Returns an iterator that yields samples from `self`.
    fn iter(&self, rng: R) -> Iter<Self, R> {
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

/// A uniform distribution of values in the given range.
#[derive(Clone, Debug)]
pub struct Uniform<T>(pub Range<T>);

/// A uniform distribution of 2-vectors on the (perimeter of) the unit circle.
#[derive(Copy, Clone, Debug)]
pub struct UnitCircle;

/// A uniform distribution of 2-vectors inside the (closed) unit disk.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitDisk;

/// A uniform distribution of 3-vectors on the (surface of) the unit sphere.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitSphere;

/// A uniform distribution of 3-vectors inside the (closed) unit ball.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitBall;

/// A Bernoulli distribution.
///
/// Generates boolean values such that:
/// * P(true) = p
/// * P(false) = 1 - p.
///
///  given a parameter p ∈ [0.0, 1.0].
#[derive(Copy, Clone, Debug)]
pub struct Bernoulli(pub f32);

/// Iterator returned by the [Distrib::iter()] method.
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

impl<D: Distrib> Iterator for Iter<D, DefaultRng> {
    type Item = D::Sample;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.sample(&mut self.1))
    }
}

impl Default for Xorshift64 {
    /// Returns a default `Xorshift64`.
    ///
    /// # Examples
    /// # use retrofire_core::math::rand::Xorshift64;
    /// let mut g = Xorshift64::default();
    /// assert_eq!(g.next_bits(), 133101616827);
    fn default() -> Self {
        // Random 64-bit prime
        Self::from_seed(Self::DEFAULT_SEED)
    }
}

//
// Local trait impls
//

impl Distrib for Uniform<i32> {
    type Sample = i32;

    /// Returns a uniformly distributed `i32` in the given range.
    fn sample(&self, rng: &mut DefaultRng) -> i32 {
        let bits = rng.next_bits() as i32;
        // TODO rem introduces slight bias
        bits.rem_euclid(self.0.end - self.0.start) + self.0.start
    }
}

impl Distrib for Uniform<f32> {
    type Sample = f32;

    /// Returns a uniformly distributed `f32` in the given range.
    fn sample(&self, rng: &mut DefaultRng) -> f32 {
        let Range { start, end } = self.0;
        // Bit repr of a random f32 in range 1.0..2.0
        let bits = 127 << 23 | rng.next_bits() >> 41;
        let unit = f32::from_bits(bits as u32) - 1.0;
        unit * (end - start) + start
    }
}

impl<T, O, const N: usize> Distrib for Uniform<[T; N]>
where
    T: Copy,
    Uniform<T>: Distrib<Sample = O>,
{
    type Sample = [O; N];

    /// Returns an array of values that represents a uniformly distributed point
    /// within the N-dimensional rectangular volume bounded by `self.1`.
    fn sample(&self, rng: &mut DefaultRng) -> [O; N] {
        array::from_fn(|i| Uniform(self.0.start[i]..self.0.end[i]).sample(rng))
    }
}

impl<Sc, Sp, const DIM: usize> Distrib for Uniform<Vector<[Sc; DIM], Sp>>
where
    Sc: Copy,
    Uniform<[Sc; DIM]>: Distrib<Sample = [Sc; DIM]>,
{
    type Sample = Vector<[Sc; DIM], Sp>;

    /// Returns a uniformly distributed vector within the rectangular volume
    /// bounded by the range `self.1`.
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

impl Distrib for UnitDisk {
    type Sample = Vec2;

    /// Returns a 2-vector uniformly distributed within the unit disk.
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

impl Distrib for UnitBall {
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

impl Distrib for Bernoulli {
    type Sample = bool;

    /// Returns boolean values sampled from a Bernoulli distribution.
    fn sample(&self, rng: &mut DefaultRng) -> bool {
        Uniform(0.0f32..1.0).sample(rng) < self.0
    }
}

#[cfg(test)]
#[allow(clippy::manual_range_contains)]
mod tests {
    use core::ops::Add;

    use crate::assert_approx_eq;
    use crate::math::vec::{splat, vec3};

    use super::*;

    const COUNT: usize = 1000;

    fn rng() -> DefaultRng {
        Default::default()
    }

    #[test]
    fn uniform_i32() {
        let dist = Uniform(-123..456);
        for r in dist.iter(rng()).take(COUNT) {
            assert!(-123 <= r && r < 456);
        }
    }

    #[test]
    fn uniform_f32() {
        let dist = Uniform(-1.23..4.56);
        for r in dist.iter(rng()).take(COUNT) {
            assert!(-1.23 <= r && r < 4.56);
        }
    }

    #[test]
    fn uniform_i32_array() {
        let dist = Uniform([0, -10]..[10, 15]);

        let sum = dist
            .iter(rng())
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
            .iter(rng())
            .take(COUNT)
            .inspect(|v| {
                assert!(-2.0 <= v.x() && v.x() < 1.0);
                assert!(0.0 <= v.y() && v.y() < 2.0);
                assert!(-1.0 <= v.z() && v.z() < 3.0);
            })
            .fold(splat(0.0), Add::add)
            / COUNT as f32;

        assert_eq!(mean, vec3(-0.46046025, 1.0209353, 0.9742225));
    }

    #[test]
    fn bernoulli() {
        let trues = Bernoulli(0.1)
            .iter(rng())
            .take(COUNT)
            .filter(|&b| b)
            .count();
        assert_eq!(trues, 82);
    }

    #[cfg(feature = "fp")]
    #[test]
    fn unit_circle() {
        for v in UnitCircle.iter(rng()).take(COUNT) {
            assert_approx_eq!(v.len_sqr(), 1.0, "non-unit vector: {v:?}");
        }
    }

    #[test]
    fn unit_disk() {
        for v in UnitDisk.iter(rng()).take(COUNT) {
            assert!(v.len_sqr() <= 1.0, "vector of len > 1.0: {v:?}");
        }
    }

    #[cfg(feature = "fp")]
    #[test]
    fn unit_sphere() {
        for v in UnitSphere.iter(rng()).take(COUNT) {
            assert_approx_eq!(v.len_sqr(), 1.0, "non-unit vector: {v:?}");
        }
    }

    #[test]
    fn unit_ball() {
        for v in UnitBall.iter(rng()).take(COUNT) {
            assert!(v.len_sqr() <= 1.0, "vector of len > 1.0: {v:?}");
        }
    }
}
