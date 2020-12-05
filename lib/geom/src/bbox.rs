use std::f32::INFINITY;

use math::mat::Mat4;
use math::vec::{pt, Vec4};

#[derive(Copy, Clone, Debug, Default)]
pub struct BoundingBox {
    pub left_bot_front: Vec4,
    pub right_top_back: Vec4,
}

impl BoundingBox {
    pub fn of(vs: impl IntoIterator<Item=Vec4>) -> BoundingBox {
        let mut low = pt(INFINITY, INFINITY, INFINITY);
        let mut upp = pt(-INFINITY, -INFINITY, -INFINITY);

        for v in vs {
            for i in 0..3 {
                if v[i] < low[i] { low[i] = v[i]; }
                else if v[i] > upp[i] { upp[i] = v[i]; }
            }
        }

        BoundingBox {
            left_bot_front: low,
            right_top_back: upp,
        }
    }

    pub fn verts(&self) -> [Vec4; 8] {
        let lbf = self.left_bot_front;
        let rtb = self.right_top_back;
        [
            pt(lbf.x, lbf.y, lbf.z),
            pt(lbf.x, lbf.y, rtb.z),
            pt(lbf.x, rtb.y, lbf.z),
            pt(lbf.x, rtb.y, rtb.z),
            pt(rtb.x, lbf.y, lbf.z),
            pt(rtb.x, lbf.y, rtb.z),
            pt(rtb.x, rtb.y, lbf.z),
            pt(rtb.x, rtb.y, rtb.z),
        ]
    }

    pub fn transform(self, tf: &Mat4) -> BoundingBox {
        let a = tf * self.left_bot_front;
        let b = tf * self.right_top_back;

        BoundingBox {
            left_bot_front: a.zip_map(b, f32::min),
            right_top_back: a.zip_map(b, f32::max),
        }
    }
}