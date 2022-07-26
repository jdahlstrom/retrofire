use math::transform::translate;
use math::vec::Vec4;

use crate::{Rasterize, render::Render, State};
use crate::shade::Shader;

#[derive(Copy, Clone, Debug, Default)]
pub struct Particle {
    pub pos: Vec4,
    pub vel: Vec4,
    pub life: f32,
}

impl Particle {
    pub fn is_alive(&self) -> bool {
        self.life > 0.0
    }
}

#[derive(Debug)]
pub struct ParticleSys<G> {
    particles: Vec<Particle>,
    live_idx: usize,
    geom: G,
}

impl<G> ParticleSys<G> {
    pub fn new(n: usize, geom: G) -> Self {
        let particles = vec![Particle::default(); n];
        Self { particles, live_idx: n, geom, }
    }

    pub fn emit(&mut self, n: usize, mut create: impl FnMut() -> Particle) {
        self.assert_invariant();
        for _ in 0..n {
            if let Some(p) = self.particles.get_mut(self.live_idx.wrapping_sub(1)) {
                assert!(!p.is_alive());
                *p = create();
                if p.is_alive() {
                    self.live_idx -= 1;
                }
            }
        }
        self.assert_invariant();
    }

    pub fn update(&mut self, dt: f32, mut update: impl FnMut(&mut Particle)) {
        self.assert_invariant();
        for p in self.live_particles_mut() {
            update(p);
            p.pos = p.pos + dt * p.vel;
            p.life -= dt;
        }
        for i in self.live_idx..self.particles.len() {
            if !self.particles[i].is_alive() {
                self.particles.swap(i, self.live_idx);
                self.live_idx += 1;
            }
        }
        self.assert_invariant();
    }

    pub fn live_particles(&self) -> impl Iterator<Item=&Particle> {
        self.particles[self.live_idx..].iter()
    }

    pub fn live_particles_mut(&mut self) -> impl Iterator<Item=&mut Particle> {
        self.particles[self.live_idx..].iter_mut()
    }

    fn assert_invariant(&self) {
        let (dead, live) = self.particles.split_at(self.live_idx);
        debug_assert!(dead.iter().all(|p| !p.is_alive()));
        debug_assert!(live.iter().all(|p| p.is_alive()));
    }
}

impl<U, V, R> Render<U, V, V> for ParticleSys<R>
where
    U: Copy,
    V: Copy,
    R: Render<U, V, V> + Clone
{
    fn render<S, Ra>(&self, st: &mut State, shade: &mut S, raster: &mut Ra)
    where
        S: Shader<U, V, VtxOut = V>,
        Ra: Rasterize,
    {
        for p in self.live_particles() {
            let tf = translate(p.pos) * &st.modelview;

            let st = &mut State {
                modelview: tf,
                ..st.clone()
            };
            self.geom.render(st, shade, raster);
        }
    }
}

#[cfg(test)]
mod tests {
    use geom::Sprite;

    use super::*;

    #[test]
    fn test() {
        let mut ptx = ParticleSys::new(10, Sprite::<()>::default());
        for _ in 0..2 {
            ptx.emit(3, || Particle {
                life: 10.0,
                ..Particle::default()
            });
            ptx.update(1.0, |_|());
        }
        assert_eq!(10, ptx.particles.len());
        assert_eq!(4, ptx.live_idx);

        assert_eq!(
            vec![0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0],
            ptx.particles.iter().map(|p| p.life).collect::<Vec<_>>(),
        );
    }
}
