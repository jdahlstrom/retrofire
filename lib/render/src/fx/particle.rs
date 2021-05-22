use math::transform::translate;
use math::vec::Vec4;

use crate::{RasterOps, Render, Renderer};

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
pub struct ParticleSys<R> {
    particles: Vec<Particle>,
    live_idx: usize,
    visual: R,
}

impl<R> ParticleSys<R> {
    pub fn new(n: usize, visual: R) -> Self {
        let particles = vec![Particle::default(); n];
        Self { particles, live_idx: n, visual, }
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

impl<V: Copy, F: Copy, R: Render<V, F> + Clone> Render<V, F> for ParticleSys<R> {
    fn render<Rs>(&self, rdr: &mut Renderer, raster: &mut Rs)
    where
        Rs: RasterOps<V, F>
    {
        for p in self.live_particles() {
            let tf = translate(p.pos) * &rdr.modelview;

            let rdr = &mut Renderer {
                modelview: tf,
                ..rdr.clone()
            };
            self.visual.render(rdr, raster);
        }
    }
}

#[cfg(test)]
mod tests {
    use geom::Sprite;
    use math::vec::ORIGIN;

    use super::*;

    #[test]
    fn test() {
        let mut ptx = ParticleSys::new(10, Sprite {
            center: ORIGIN,
            width: 0.0,
            height: 0.0,
            vertex_attrs: [(), (), (), ()],
            face_attr: (),
        });
        for _ in 0..100 {
            ptx.emit(10, Particle::default);
            ptx.update(1.0, |_|());
        }
        dbg!(ptx);
    }
}
