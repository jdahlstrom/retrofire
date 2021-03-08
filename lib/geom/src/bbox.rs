use math::mat::Mat4;
use math::transform::Transform;
use math::vec::{Vec4, vec4};

#[derive(Copy, Clone, Debug, Default)]
pub struct BoundingBox {
    pub left_bot_front: Vec4,
    pub right_top_back: Vec4,
}

impl BoundingBox {
    /// Returns the smallest (axis-aligned) bounding box
    /// that contains every point in `vs`.
    pub fn of<'a>(vs: impl IntoIterator<Item=&'a Vec4>) -> Self {
        let mut lbf = Vec4::from([f32::INFINITY; 4]);
        let mut rtb = Vec4::from([f32::NEG_INFINITY; 4]);

        for &v in vs.into_iter() {
            lbf = lbf.zip_with(v, f32::min);
            rtb = rtb.zip_with(v, f32::max);
        }

        Self { left_bot_front: lbf, right_top_back: rtb }
    }

    /// Returns the vertex coordinates of `self`.
    pub fn verts(&self) -> [Vec4; 8] {
        let lbf = self.left_bot_front;
        let rtb = self.right_top_back;
        [
            vec4(lbf.x, lbf.y, lbf.z, lbf.w),
            vec4(lbf.x, lbf.y, rtb.z, rtb.w),
            vec4(lbf.x, rtb.y, lbf.z, lbf.w),
            vec4(lbf.x, rtb.y, rtb.z, rtb.w),
            vec4(rtb.x, lbf.y, lbf.z, lbf.w),
            vec4(rtb.x, lbf.y, rtb.z, rtb.w),
            vec4(rtb.x, rtb.y, lbf.z, lbf.w),
            vec4(rtb.x, rtb.y, rtb.z, rtb.w),
        ]
    }

    /// Returns the edges of `self`.
    pub fn edges(&self) -> Vec<[Vec4; 2]> {
        let [v0, v1, v2, v3, v4, v5, v6, v7] = self.verts();
        vec![
            [v0, v1], [v0, v2], [v1, v3], [v2, v3],
            [v4, v5], [v4, v6], [v5, v7], [v6, v7],
            [v0, v4], [v1, v5], [v2, v6], [v3, v7]
        ]
    }
}

impl Transform for BoundingBox {
    fn transform(&mut self, tf: &Mat4) {
        let mut verts = self.verts();
        verts.transform(tf);
        *self = BoundingBox::of(&verts)
    }
}
