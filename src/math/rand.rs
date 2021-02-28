use std::iter::from_fn;
use std::num::Wrapping;
use std::ops::{Range, RangeFull};
use std::time::SystemTime;

use crate::math::lerp;

pub struct Random { state: Wrapping<u64> }

pub trait Distrib<T> {
    //type Output;
    fn from(&self, r: &mut Random) -> T;
}

impl Random {
    // Value of A from P. L'Ecuyer, 1999. "Tables of linear congruential
    // generators of different sizes and good lattice structure".
    // Mathematics of Computation, 1999-01-01, Vol.68 (225), p.249-260.
    // DOI: 10.1090/S0025-5718-99-00996-5
    const A: Wrapping<u64> = Wrapping(2685821657736338717);
    const C: Wrapping<u64> = Wrapping(1);

    pub fn new() -> Random {
        let now = SystemTime::UNIX_EPOCH.elapsed().unwrap();
        Random { state: Wrapping(now.as_micros() as u64) }
    }
    pub fn seed(seed: u64) -> Random {
        Random { state: Wrapping(seed) }
    }

    pub fn next_bits(&mut self) -> u64 {
        self.state = self.state * Self::A + Self::C;
        self.state.0
    }

    pub fn next<T, D: Distrib<T>>(&mut self, distrib: &D) -> T {
        distrib.from(self)
    }

    pub fn iter<'a, T, D: Distrib<T> + 'a>(&'a mut self, distrib: D) -> impl Iterator<Item=T> + 'a {
        from_fn(move || Some(distrib.from(self)))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Uniform<R>(pub R);

impl Distrib<i32> for Uniform<RangeFull> {
    fn from(&self, r: &mut Random) -> i32 {
        (r.next_bits() >> 32) as i32
    }
}

impl Distrib<i32> for Uniform<Range<i32>> {
    fn from(&self, r: &mut Random) -> i32 {
        debug_assert!(!self.0.is_empty());
        let n = r.next::<i32, _>(&Uniform(..)) as i64;
        let start = self.0.start as i64;
        let end = self.0.end as i64;
        (start + n.rem_euclid(end - start)) as i32
    }
}

impl Distrib<u32> for Uniform<RangeFull> {
    fn from(&self, r: &mut Random) -> u32 {
        (r.next_bits() >> 32) as u32
    }
}

impl Distrib<u32> for Uniform<Range<u32>> {
    fn from(&self, r: &mut Random) -> u32 {
        debug_assert!(!self.0.is_empty());
        let n = r.next::<u32, _>(&Uniform(..)) as u64;
        let start = self.0.start as u64;
        let end = self.0.end as u64;
        (start + n.rem_euclid(end - start)) as u32
    }
}

impl Distrib<f32> for Uniform<Range<f32>> {
    fn from(&self, r: &mut Random) -> f32 {
        let bits = r.next_bits();
        let t = f32::from_bits((127 << 23 | bits >> 41) as u32) - 1.0;

        lerp(t, self.0.start, self.0.end)
    }
}

impl Distrib<f64> for Uniform<Range<f64>> {
    fn from(&self, r: &mut Random) -> f64 {
        let bits = r.next_bits();
        let t = f64::from_bits(1023 << 52 | bits >> 11) - 1.0;
        (1.0 - t) * self.0.start + t * self.0.end
    }
}

#[cfg(test)]
mod tests {
    use crate::math::ApproxEq;

    use super::*;

    const ROUNDS: usize = 1 << 20;

    #[test]
    fn bits_histogram_1d() {
        let mut rand = Random::new();

        let mut hist = [0u64; 256];

        for _ in 0..ROUNDS {
            hist[rand.next_bits() as usize >> 56] += 1;
        }
    }

    #[test]
    fn bits_histogram_2d() {
        let mut rand = Random::new();

        let mut hist = [[0u64; 32]; 32];

        for _ in 0..ROUNDS {
            let (a, b) = (rand.next_bits(), rand.next_bits());
            hist[a as usize >> 59][b as usize >> 59] += 1;
        }

        for row in &hist {
            println!("{}", row.iter().map(|&a| {
                let diff = (128 - (a as i64 >> 3)).abs() as u32;
                std::char::from_u32(diff + 'a' as u32).unwrap()
            }).collect::<String>());
        }
    }

    #[test]
    fn random_i32_in_range() {
        let (min, max) = (-123, 456);
        let mut r = Random::new();
        for i in r.iter(Uniform(min..max)).take(ROUNDS) {
            assert!(min <= i && i < max, "i={}", i);
        }
    }

    #[test]
    fn i32_in_max_range() {
        let mut r = Random::new();
        for _ in r.iter(Uniform(i32::MIN..i32::MAX)).take(ROUNDS) {}
    }

    #[test]
    fn random_u32_in_range() {
        let (min, max) = (1245, 987654);
        let mut r = Random::new();
        for u in r.iter(Uniform(min..max)).take(ROUNDS) {
            assert!(min <= u && u < max, "i={}", u);
        }
    }

    #[test]
    fn u32_in_max_range() {
        let mut r = Random::new();
        for _ in r.iter(Uniform(u32::MIN..u32::MAX)).take(ROUNDS) {}
    }

    #[test]
    fn random_f32_in_unit_interval() {
        let avg = Random::new().iter(Uniform(0.0..1.0_f32))
            .take(ROUNDS)
            .inspect(|&x| assert!(0.0 <= x && x < 1.0, "{} is not in [0, 1)", x))
            .sum::<f32>() / ROUNDS as f32;
        assert!(avg.abs_diff(0.5) < 0.001)
    }

    #[test]
    fn random_f64_in_unit_interval() {
        let avg = Random::new().iter(Uniform(0.0..1.0))
            .take(ROUNDS)
            .inspect(|&x| assert!(0.0 <= x && x < 1.0, "{} is not in [0, 1)", x))
            .sum::<f64>() / ROUNDS as f64;
        assert!(avg.abs_diff(0.5) < 0.001)
    }
}