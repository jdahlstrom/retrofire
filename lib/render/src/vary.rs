use math::Linear;

#[derive(Clone, Debug)]
pub struct Varying<T> {
    pub val: T,
    pub step: T,
}

impl<T> Varying<T>
where T: Linear<f32> + Copy
{
    pub fn from(val: T, step: T) -> Self {
        Self { val, step }
    }

    pub fn between(a: T, b: T, steps: f32) -> Self {
        // TODO Should think about this some more
        // debug_assert_ne!(steps, 0.0, "Steps cannot be zero");
        let step = b.sub(a).mul(1.0 / steps);
        Self { val: a, step }
    }
}

impl<T> Iterator for Varying<T>
where T: Linear<f32> + Copy
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        let res = self.val;
        self.val = self.val.add(self.step);
        Some(res)
    }
}

#[derive(Clone, Debug)]
pub struct Bresenham {
    val: usize,
    err: usize,
    ratio: (isize, usize),
}

impl Bresenham {
    pub fn between(a: usize, b: usize, steps: usize) -> Self {
        // TODO Should think about this some more
        // debug_assert_ne!(steps, 0.0, "Steps cannot be zero");
        Self {
            val: a,
            err: 0,
            ratio: (b.wrapping_sub(a) as isize, steps),
        }
    }
}

impl Iterator for Bresenham {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let Self { val, err, ratio: (num, den) } = self;
        let res = *val;
        *err += num.unsigned_abs();
        if *err >= *den {
            *val = val.wrapping_add(num.signum() as usize);
            *err -= *den;
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::vary::{Varying, Bresenham};
    use math::vec::*;

    #[test]
    fn varying_f32() {
        let v = Varying::between(0.0, 3.0, 3.0);
        assert_eq!(vec![0.0, 1.0, 2.0, 3.0], v.take(4).collect::<Vec<_>>());

        let v = Varying::between(100.0, -50.0, 3.0);
        assert_eq!(vec![100.0, 50.0, 0.0, -50.0], v.take(4).collect::<Vec<_>>());
    }


    #[test]
    fn varying_vec4() {
        let v = Varying::between(-2.0 * X, 4.0 * Y, 4.0);
        assert_eq!(vec![
            -2.0 * X,
            -1.5 * X + Y,
            -1.0 * X + 2.0 * Y,
            -0.5 * X + 3.0 * Y,
            4.0 * Y
        ], v.take(5).collect::<Vec<_>>());
    }

    #[test]
    fn bresenham() {
        let v = Varying::between(2.0, 6.0, 10.0);
        let b = Bresenham::between(2, 6, 10);

        for x in b.take(11).zip(v.take(11)) {
            println!("{} {}", x.0, x.1);
        }
    }
}
