use crate::Linear;

pub trait Vary: Copy {
    type Iter: Iterator<Item = Self>;

    fn vary(&self, other: &Self, steps: f32) -> Self::Iter;
}

impl Vary for usize {
    type Iter = Bresenham;

    #[inline]
    fn vary(&self, other: &Self, steps: f32) -> Bresenham {
        // TODO Should think about this some more
        // debug_assert_ne!(steps, 0.0, "Steps cannot be zero");
        Bresenham {
            val: *self,
            err: 0,
            ratio: (other.wrapping_sub(*self) as isize, steps as usize),
        }
    }
}

impl<T> Vary for T
where
    T: Linear<f32> + Copy
{
    type Iter = Varying<T>;

    #[inline]
    fn vary(&self, other: &Self, steps: f32) -> Varying<T> {
        let step = other.sub(*self).mul(steps.recip());
        Self::Iter { val: *self, step }
    }
}

#[derive(Debug)]
pub struct Varying<T> {
    pub val: T,
    pub step: T,
}

pub struct Bresenham {
    val: usize,
    err: usize,
    ratio: (isize, usize),
}

impl Iterator for Bresenham {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let res = self.val;
        let (num, den) = self.ratio;
        let step = num.signum() as usize;
        self.err += num.unsigned_abs();
        if self.err >= den {
            self.val = self.val.wrapping_add(step);
            self.err -= den;
        }
        Some(res)
    }
}

impl<T> Iterator for Varying<T>
where
    T: Linear<f32> + Copy
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        let res = self.val;
        self.val = self.val.add(self.step);
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::vary::Vary;
    use crate::vec::*;

    #[test]
    fn varying_f32() {
        let v = 0.0.vary(&3.0, 3.0);
        assert_eq!(vec![0.0, 1.0, 2.0, 3.0], v.take(4).collect::<Vec<_>>());

        let v = 100.0.vary(&-50.0, 3.0);
        assert_eq!(vec![100.0, 50.0, 0.0, -50.0], v.take(4).collect::<Vec<_>>());
    }


    #[test]
    fn varying_vec4() {
        let v = (-2.0 * X).vary(&(4.0 * Y), 4.0);
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
        let v = 2.0.vary(&6.0, 10.0);
        let b = 2.vary(&6, 10.0);

        for x in b.take(11).zip(v.take(11)) {
            println!("{} {}", x.0, x.1);
        }
    }
}
