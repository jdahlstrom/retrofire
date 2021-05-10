use std::ops::*;

pub mod noise;

pub trait Signal<D> {
    type R;
    fn sample(&self, t: D) -> Self::R;

    fn quantize(self, dt: D) -> Quantize<Self, D>
    where
        Self: Sized,
        D: Default,
    {
        Quantize { sig: self, n: 0, dt }
    }

    fn scale<F>(self, factor: F) -> Scale<Self, F>
    where
        Self: Sized,
    {
        Scale { sig: self, factor }
    }
    fn translate<O>(self, offset: O) -> Translate<Self, O>
    where
        Self: Sized,
    {
        Translate { sig: self, offset }
    }
    fn repeat(self, range: Range<D>) -> Repeat<Self, D>
    where
        Self: Sized
    {
        Repeat { sig: self, range }
    }
    fn warp<F>(self, func: F) -> Warp<Self, F>
    where
        Self: Sized,
    {
        Warp { sig: self, func }
    }

    fn gain<G>(self, gain: G) -> Gain<Self, G>
    where
        Self: Sized,
    {
        Gain { sig: self, gain }
    }
    fn bias<B>(self, bias: B) -> Bias<Self, B>
    where
        Self: Sized,
    {
        Bias { sig: self, bias }
    }
    fn mix<S>(self, with: S) -> Mix<Self, S>
    where
        Self: Sized,
    {
        Mix { sig: self, with }
    }
    fn modulate<S>(self, with: S) -> Modulate<Self, S>
    where
        Self: Sized,
    {
        Modulate { sig: self, with }
    }
    fn map<F>(self, func: F) -> Map<Self, F>
    where
        Self: Sized,
    {
        Map { sig: self, func }
    }
}

pub struct Quantize<S, D> {
    sig: S,
    dt: D,
    n: usize,
}

impl<S, D> Iterator for Quantize<S, D>
where
    S: Signal<D>,
    D: AddAssign + Mul<f32, Output=D> + Copy,
{
    type Item = S::R;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.sig.sample(self.dt * self.n as f32);
        self.n += 1;
        Some(sample)
    }
}

pub trait Bounds<D> {
    fn bounds(&self) -> Option<Range<D>>;
}

pub fn from_fn<F>(f: F) -> FromFn<F> {
    FromFn(f)
}

#[derive(Copy, Clone)]
pub struct FromFn<F>(F);

impl<F, D, R> Signal<D> for FromFn<F>
where
    F: Fn(D) -> R,
{
    type R = R;

    fn sample(&self, t: D) -> Self::R {
        self.0(t)
    }
}

impl<S, D, T> Signal<D> for T
where
    S: Signal<D>,
    T: Deref<Target = S>,
{
    type R = S::R;

    fn sample(&self, t: D) -> Self::R {
        self.deref().sample(t)
    }
}

#[derive(Clone)]
pub struct Scale<S, F> {
    sig: S,
    factor: F,
}

impl<D, S, F> Signal<D> for Scale<S, F>
where
    D: Div<F, Output = D>,
    S: Signal<D>,
    F: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t / self.factor)
    }
}

#[derive(Clone)]
pub struct Translate<S, O> {
    sig: S,
    offset: O,
}

impl<D, S, O> Signal<D> for Translate<S, O>
where
    D: Sub<O, Output = D>,
    S: Signal<D>,
    O: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t - self.offset)
    }
}

#[derive(Clone)]
pub struct Repeat<S, D> {
    sig: S,
    range: Range<D>,
}

impl<D, S> Signal<D> for Repeat<S, D>
where
    D: Rem<Output = D> + Add<Output = D> + Sub<Output = D> + Copy,
    S: Signal<D>,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        let Range { start, end } = self.range;
        self.sig.sample(start + (t - start) % (end - start))
    }
}

#[derive(Clone)]
pub struct Warp<S, F> {
    sig: S,
    func: F,
}

impl<Dinn, Dout, S, F> Signal<Dout> for Warp<S, F>
where
    S: Signal<Dinn>,
    F: Fn(Dout) -> Dinn,
{
    type R = S::R;
    fn sample(&self, t: Dout) -> Self::R {
        self.sig.sample((self.func)(t))
    }
}

#[derive(Clone)]
pub struct Gain<S, G> {
    sig: S,
    gain: G,
}

impl<D, S, G> Signal<D> for Gain<S, G>
where
    S: Signal<D>,
    S::R: Mul<G, Output = S::R>,
    G: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) * self.gain
    }
}

#[derive(Clone)]
pub struct Bias<S, B> {
    sig: S,
    bias: B,
}

impl<D, S, B> Signal<D> for Bias<S, B>
where
    S: Signal<D>,
    S::R: Add<B, Output = S::R>,
    B: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) + self.bias
    }
}

#[derive(Clone)]
pub struct Map<S, F> {
    sig: S,
    func: F,
}

impl<D, S, F, R> Signal<D> for Map<S, F>
where
    S: Signal<D>,
    F: Fn(S::R) -> R,
{
    type R = R;
    fn sample(&self, t: D) -> Self::R {
        (self.func)(self.sig.sample(t))
    }
}

#[derive(Clone)]
pub struct Mix<S, W> {
    sig: S,
    with: W,
}

impl<D, S, W> Signal<D> for Mix<S, W>
where
    D: Copy,
    S: Signal<D>,
    W: Signal<D, R = S::R>,
    S::R: Add<W::R, Output = S::R>,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) + self.with.sample(t)
    }
}

#[derive(Clone)]
pub struct Modulate<S, W> {
    sig: S,
    with: W,
}

impl<D, S, W> Signal<D> for Modulate<S, W>
where
    D: Copy,
    S: Signal<D>,
    W: Signal<D, R = S::R>,
    S::R: Mul<W::R, Output = S::R>,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) * self.with.sample(t)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
