
pub trait Animate {
    fn animate(&mut self, delta_t: f32);
}

impl<F> Animate for F where F: FnMut(f32) {
    fn animate(&mut self, delta_t: f32) {
        self(delta_t);
    }
}

pub struct Animation<'a> {
    pub speed: f32,
    anims: Vec<&'a mut dyn Animate>,
}

impl<'a> Animation<'a> {
    pub fn add(&mut self, anim: &'a mut dyn Animate) {
        self.anims.push(anim);
    }

    pub fn play(&mut self) {
        self.speed = 1.0;
    }
    pub fn pause(&mut self) {
        self.speed = 0.0;
    }
}

impl Default for Animation<'_> {
    fn default() -> Self {
        Animation { speed: 1.0, anims: Vec::new() }
    }
}

impl<'a> Animate for Animation<'a> {
    fn animate(&mut self, delta_t: f32) {
        if self.speed != 0.0 {
            for anim in self.anims.iter_mut() {
                anim.animate(self.speed * delta_t);
            }
        }
    }
}

pub fn repeat(interval: f32, anim: &mut dyn Animate) -> impl Animate + '_ {
    let mut elapsed = 0.0;
    move |delta_t| {
        elapsed += delta_t;
        while elapsed >= interval {
            anim.animate(interval);
            elapsed -= interval;
        }
    }
}
