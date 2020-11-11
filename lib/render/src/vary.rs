use math::Linear;

#[derive(Debug)]
pub struct Varying<T> {
    pub val: T,
    pub step: T,
}

impl<T: Linear<f32> + Copy> Varying<T> {
    pub fn from(val: T, step: T) -> Self {
        Self { val, step }
    }

    pub fn between(a: T, b: T, steps: f32) -> Self {
        let step = (b.add(a.mul(-1.0))).mul(1.0 / steps);
        Self { val: a, step }
    }

    pub fn step(&mut self) -> T {
        let res = self.val;
        self.val = self.val.add(self.step);
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::vary::Varying;
    #[test]
    fn test_varying_f32() {
        let mut v = Varying::between(0.0, 3.0, 3.0);
        assert_eq!([0.0, 1.0, 2.0, 3.0], [v.step(), v.step(), v.step(), v.step()]);

        let mut v = Varying::between(100.0, -50.0, 3.0);
        assert_eq!([100.0, 50.0, 0.0, -50.0], [v.step(), v.step(), v.step(), v.step()]);
    }
}
