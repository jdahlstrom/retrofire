use core::{array::from_fn, fmt::Debug, ops::Range};

use crate::math::{Vec2, Vec3, Vector};

//
// Traits and types
//

/// Trait for generating values sampled from a probability distribution.
pub trait Distrib {
    /// The type of the elements of the sample space of `Self`, also called
    /// "outcomes".
    type Sample;
    /// Returns a pseudo-random value sampled from `self`.
    fn sample(&mut self) -> Self::Sample;
    /// Returns an iterator yielding samples from `self`.
    fn iter(&mut self) -> Iter<&mut Self> {
        Iter(self)
    }
}

/// A pseudo-random number generator (PRNG) that uses a Xorshift algorithm[^1]
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
pub struct Uniform<T>(pub Xorshift64, pub Range<T>);

/// A uniform distribution of 2-vectors on the (perimeter of) the unit circle.
#[derive(Copy, Clone, Debug)]
pub struct UnitCircle(pub Xorshift64);

/// A uniform distribution of 2-vectors inside the (closed) unit disk.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitDisk(pub Xorshift64);

/// A uniform distribution of 3-vectors on the (surface of) the unit sphere.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitSphere(pub Xorshift64);

/// A uniform distribution of 3-vectors inside the (closed) unit ball.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnitBall(pub Xorshift64);

/// A Bernoulli distribution.
///
/// Generates boolean values such that:
/// * P(true) = p
/// * P(false) = 1 - p.
///
///  given a parameter p ∈ [0.0, 1.0].
#[derive(Copy, Clone, Debug)]
pub struct Bernoulli(pub Xorshift64, pub f32);

/// Iterator returned by the [Distrib::iter()] method.
pub struct Iter<D>(D);

//
// Inherent impls
//

impl Xorshift64 {
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
    /// assert_eq!(g.gen(), 133101616827);
    /// assert_eq!(g.gen(), 12690785413091508870);
    /// assert_eq!(g.gen(), 7516749944291143043);
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
    /// assert_ne!(g.gen(), h.gen());
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
    pub fn gen(&mut self) -> u64 {
        let Self(x) = self;
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }
}

//
// Trait impls
//

impl<'a, D: Distrib> Iterator for Iter<&'a mut D> {
    type Item = D::Sample;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.sample())
    }
}

impl Default for Xorshift64 {
    fn default() -> Self {
        Self::from_seed(1)
    }
}

impl Distrib for Uniform<i32> {
    type Sample = i32;

    /// Returns a uniformly distributed `i32` in the given range.
    fn sample(&mut self) -> i32 {
        let Self(gen, Range { start, end }) = self;
        let bits = gen.gen() as i32;
        bits.rem_euclid(*end - *start) + *start
    }
}

impl Distrib for Uniform<f32> {
    type Sample = f32;

    /// Returns a uniformly distributed `f32` in the given range.
    fn sample(&mut self) -> f32 {
        let Self(gen, Range { start, end }) = self;
        // Bit repr of a random f32 in range 1.0..2.0
        let bits = 127 << 23 | gen.gen() >> 41;
        let unit = f32::from_bits(bits as u32) - 1.0;
        unit * (*end - *start) + *start
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
    fn sample(&mut self) -> [O; N] {
        from_fn(|i| {
            let Self(gen, rg) = self;
            let mut d = Uniform(*gen, rg.start[i]..rg.end[i]);
            let r = d.sample();
            self.0 = d.0;
            r
        })
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
    fn sample(&mut self) -> Self::Sample {
        let Self(gen, rg) = self;
        let mut d = Uniform(*gen, rg.start.0..rg.end.0);
        let r = d.sample();
        self.0 = d.0;
        r.into()
    }
}

impl Distrib for UnitCircle {
    type Sample = Vec2;

    /// Returns a 2-vector uniformly distributed on the unit circle.
    fn sample(&mut self) -> Vec2 {
        let mut d = Uniform(self.0, [-1.0; 2]..[1.0; 2]);
        let res = Vec2::from(d.sample()).normalize();
        self.0 = d.0;
        res
    }
}

impl Distrib for UnitDisk {
    type Sample = Vec2;

    /// Returns a 2-vector uniformly distributed within the unit disk.
    fn sample(&mut self) -> Vec2 {
        let mut d = Uniform(self.0, [-1.0f32; 2]..[1.0; 2]);
        loop {
            let v = Vec2::from(d.sample());
            if v.dot(&v) <= 1.0 {
                self.0 = d.0;
                return v;
            }
        }
    }
}

impl Distrib for UnitSphere {
    type Sample = Vec3;

    /// Returns a vector uniformly distributed on the unit sphere.
    fn sample(&mut self) -> Vec3 {
        let mut d = Uniform(self.0, [-1.0f32; 3]..[1.0; 3]);
        let res = Vec3::from(d.sample()).normalize();
        self.0 = d.0;
        res
    }
}

impl Distrib for UnitBall {
    type Sample = Vec3;

    /// Returns a vector uniformly distributed within the unit ball.
    fn sample(&mut self) -> Vec3 {
        let mut d = Uniform(self.0, [-1.0; 3]..[1.0; 3]);
        loop {
            let v = Vec3::from(d.sample());
            if v.dot(&v) <= 1.0 {
                self.0 = d.0;
                return v;
            }
        }
    }
}

impl Distrib for Bernoulli {
    type Sample = bool;

    /// Returns boolean values sampled from a Bernoulli distribution.
    fn sample(&mut self) -> bool {
        let mut d = Uniform(self.0, 0.0f32..1.0);
        let res = d.sample() < self.1;
        self.0 = d.0;
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::{vec3, Affine, Linear};

    use super::*;

    const COUNT: usize = 1000;

    #[test]
    fn uniform_i32() {
        let gen = Xorshift64::default();
        let mut d = Uniform(gen, -123..456);

        for r in d.iter().take(COUNT) {
            assert!(-123 <= r && r < 456);
        }
    }

    #[test]
    fn uniform_f32() {
        let gen = Xorshift64::default();
        let mut d = Uniform(gen, -1.23..4.56);

        for r in d.iter().take(COUNT) {
            assert!(-1.23 <= r && r < 4.56);
        }
    }

    #[test]
    fn uniform_i32_array() {
        let gen = Xorshift64::default();
        let mut d = Uniform(gen, [0, -10]..[10, 15]);

        for [x, y] in d.iter().take(COUNT) {
            assert!(0 <= x && x < 10);
            assert!(-10 <= y && x < 15);
        }
    }

    #[test]
    fn uniform_vec3() {
        let gen = Xorshift64::default();
        let mut d = Uniform(gen, vec3(-2.0, 0.0, -1.0)..vec3(1.0, 2.0, 3.0));
        let mut sum = vec3(0.0, 0.0, 0.0);
        for v in d.iter().take(COUNT) {
            assert!(-2.0 <= v.x() && v.x() < 1.0);
            assert!(0.0 <= v.y() && v.y() < 2.0);
            assert!(-1.0 <= v.z() && v.z() < 3.0);
            sum = sum.add(&v);
        }
        assert_approx_eq!(
            sum.mul(1.0 / COUNT as f32),
            vec3(-0.5251561, 1.0114789, 0.9670243)
        );
    }

    #[test]
    fn bernoulli() {
        let gen = Xorshift64::default();
        let mut d = Bernoulli(gen, 0.1);

        let mut trues = 0;
        for b in d.iter().take(COUNT) {
            trues += b as u32;
        }
        assert_eq!(trues, 93);
    }

    #[test]
    fn unit_circle() {
        let gen = Xorshift64::default();
        let mut d = UnitCircle(gen);
        for v in d.iter().take(COUNT) {
            assert_approx_eq!(v.len(), 1.0);
        }
    }

    #[test]
    fn unit_disk() {
        let gen = Xorshift64::default();
        let mut d = UnitDisk(gen);
        for v in d.iter().take(COUNT) {
            assert!(v.len() <= 1.0);
        }
    }

    #[test]
    fn unit_sphere() {
        let gen = Xorshift64::default();
        let mut d = UnitSphere(gen);
        for v in d.iter().take(COUNT) {
            assert_approx_eq!(v.len(), 1.0, eps = 1e-5);
        }
    }

    #[test]
    fn unit_ball() {
        let gen = Xorshift64::default();
        let mut d = UnitBall(gen);
        for v in d.iter().take(COUNT) {
            assert!(v.len() <= 1.0);
        }
    }
}
