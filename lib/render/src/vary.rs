use math::Linear;

#[derive(Debug)]
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
        let step = b.sub(a).mul(1.0 / steps);
        Self { val: a, step }
    }
}

impl<T> Iterator for Varying<T>
    where T: Linear<f32> + Copy
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let res = self.val;
        self.val = self.val.add(self.step);
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::vary::Varying;

    #[test]
    fn test_varying_f32() {
        let v = Varying::between(0.0, 3.0, 3.0);
        assert_eq!(vec![0.0, 1.0, 2.0, 3.0], v.take(4).collect::<Vec<_>>());

        let v = Varying::between(100.0, -50.0, 3.0);
        assert_eq!(vec![100.0, 50.0, 0.0, -50.0], v.take(4).collect::<Vec<_>>());
    }
}
