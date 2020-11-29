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
        let [mut l, mut b, mut f] = [INFINITY; 3];
        let [mut r, mut t, mut k] = [-INFINITY; 3];

        for Vec4 { x, y, z, .. } in vs {
            if x < l { l = x; } else if x > r { r = x; }
            if y < b { b = y; } else if y > t { t = y; }
            if z < f { f = x; } else if z > k { k = z; }
        }

        BoundingBox {
            left_bot_front: pt(l, b, f),
            right_top_back: pt(r, t, k)
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