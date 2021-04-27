use std::ops;

pub mod noise;

pub trait Signal<D>: Sized {
    type R;
    fn sample(&self, t: D) -> Self::R;

    fn scale<F>(self, factor: F) -> Scale<Self, F> {
        Scale { sig: self, factor }
    }
    fn translate<O>(self, offset: O) -> Translate<Self, O> {
        Translate { sig: self, offset }
    }

    fn gain<G>(self, gain: G) -> Gain<Self, G> {
        Gain { sig: self, gain }
    }
    fn bias<B>(self, bias: B) -> Bias<Self, B> {
        Bias { sig: self, bias }
    }

    fn mix<S>(self, with: S) -> Mix<Self, S> {
        Mix { sig: self, with }
    }
    fn modulate<S>(self, with: S) -> Modulate<Self, S> {
        Modulate { sig: self, with }
    }
}

impl<F, D, R> Signal<D> for F
where
    F: Fn(D) -> R,
{
    type R = R;

    fn sample(&self, t: D) -> Self::R {
        self(t)
    }
}

pub struct Scale<S, F> {
    sig: S,
    factor: F,
}

impl<D, S, F> Signal<D> for Scale<S, F>
where
    D: ops::Div<F, Output = D>,
    S: Signal<D>,
    F: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t / self.factor)
    }
}

pub struct Translate<S, O> {
    sig: S,
    offset: O,
}

impl<D, S, O> Signal<D> for Translate<S, O>
where
    D: ops::Sub<O, Output = D>,
    S: Signal<D>,
    O: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t - self.offset)
    }
}

pub struct Gain<S, G> {
    sig: S,
    gain: G,
}

impl<D, S, G> Signal<D> for Gain<S, G>
where
    S: Signal<D>,
    S::R: ops::Mul<G, Output = S::R>,
    G: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) * self.gain
    }
}

pub struct Bias<S, B> {
    sig: S,
    bias: B,
}

impl<D, S, B> Signal<D> for Bias<S, B>
where
    S: Signal<D>,
    S::R: ops::Add<B, Output = S::R>,
    B: Copy,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) + self.bias
    }
}

pub struct Mix<S, W> {
    sig: S,
    with: W,
}

impl<D, S, W> Signal<D> for Mix<S, W>
where
    D: Copy,
    S: Signal<D>,
    W: Signal<D, R = S::R>,
    S::R: ops::Add<W::R, Output = S::R>,
{
    type R = S::R;
    fn sample(&self, t: D) -> Self::R {
        self.sig.sample(t) + self.with.sample(t)
    }
}

pub struct Modulate<S, W> {
    sig: S,
    with: W,
}

impl<D, S, W> Signal<D> for Modulate<S, W>
where
    D: Copy,
    S: Signal<D>,
    W: Signal<D, R = S::R>,
    S::R: ops::Mul<W::R, Output = S::R>,
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
