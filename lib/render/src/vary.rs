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
        //let step = (b.add(a.mul(-1.0))).mul(1.0 / steps);

        let step = b.mul(1.0 / steps)
            .add(a.mul(-1.0 / steps));

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
    use util::color::*;

    #[test]
    fn f32_varying() {
        let v = Varying::between(0.0, 3.0, 3.0);
        assert_eq!(vec![0.0, 1.0, 2.0, 3.0], v.take(4).collect::<Vec<_>>());

        let v = Varying::between(100.0, -50.0, 3.0);
        assert_eq!(vec![100.0, 50.0, 0.0, -50.0], v.take(4).collect::<Vec<_>>());
    }

    #[test]
    fn color_varying() {
        let from = rgb(255, 0, 64);
        let to = rgb(25, 192, 128);
        let steps = 100.0;

        let c = Varying::between(from, to, steps);

        println!("step={}", c.step);

        for x in c.take(101) {
            println!("{}", x);
        }

    }
}
