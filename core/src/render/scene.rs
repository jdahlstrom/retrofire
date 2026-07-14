use core::fmt::{self, Debug, Formatter};

use crate::{
    geom::{Mesh, vertex},
    math::{Mat4, Point3, ProjMat3, Vec3, pt3},
};

use super::{
    Model, World,
    clip::{ClipVert, Status, view_frustum},
};

#[derive(Clone, Debug)]
pub struct Obj<A> {
    pub geom: Mesh<A>,
    pub bbox: BBox<Model>,
    pub tf: Mat4<Model, World>,
}

// TODO Decide whether upper bound is inclusive or exclusive
// TODO Needs to be more generic to work with clip points
#[derive(Copy, Clone, PartialEq)]
pub struct BBox<B>(pub Point3<B>, pub Point3<B>);

impl<A> Obj<A> {
    pub fn new(geom: Mesh<A>) -> Self {
        Self::with_transform(geom, Mat4::identity())
    }

    #[must_use]
    pub fn with_transform(geom: Mesh<A>, tf: Mat4<Model, World>) -> Self {
        let bbox = BBox::of(&geom);
        Self { geom, bbox, tf }
    }
}

impl<B> BBox<B> {
    pub fn empty() -> Self {
        Self([f32::INFINITY; 3].into(), [f32::NEG_INFINITY; 3].into())
    }

    pub fn of<A>(mesh: &Mesh<A, B>) -> Self {
        mesh.verts.iter().map(|v| &v.pos).collect()
    }

    /// If needed, enlarges `self` so that a point is just contained.
    pub fn extend(&mut self, pt: &Point3<B>) {
        let BBox(low, upp) = self;
        *low = low.zip_map(*pt, f32::min);
        *upp = upp.zip_map(*pt, f32::max);
    }

    /// Returns whether `self` contains at least one point.
    pub fn is_empty(&self) -> bool {
        let [x, y, z] = self.dims().0;
        x < 0.0 || y < 0.0 || z < 0.0
    }

    /// Returns the width, height, and depth of self.
    ///
    /// Note that if self is empty, at least one of the dimensions is negative.
    pub fn dims(&self) -> Vec3<B> {
        self.1 - self.0
    }

    /// Returns whether a point is within the bounds of `self`.
    pub fn contains(&self, pt: &Point3<B>) -> bool {
        let BBox(low, upp) = self;
        (0..3).all(|i| low[i] <= pt[i] && pt[i] <= upp[i])
    }

    /// Returns whether `self` contains any common points with another bbox.
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.intersect(other).is_empty()
    }

    /// Returns the smallest bounding box that contains both `self` and another
    /// bounding box.
    ///
    /// Bounding boxes constitute a partially ordered set (poset), with
    /// ordering imposed by containment. This function computes the least upper
    /// bound (lub), also called *join* or *supremum*, of two bboxes in that
    /// ordering.
    ///
    /// Note that this method is not called "union" because in general, the
    /// result contains points outside both of the input bboxes.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{math::pt3, render::BBox};
    ///
    /// let a = BBox::<()>(pt3(0.0, 0.0, 0.0), pt3(3.0, 1.0, 1.0));
    /// let b = BBox::<()>(pt3(1.0, 0.0, 0.0), pt3(2.0, 2.0, 2.0));
    ///
    /// assert_eq!(a.merge(&b), BBox(pt3(0.0, 0.0, 0.0), pt3(3.0, 2.0, 2.0)));
    /// ```
    pub fn merge(&self, other: &Self) -> Self {
        let low = self.0.zip_map(other.0, f32::min);
        let upp = self.1.zip_map(other.1, f32::max);
        BBox(low, upp)
    }

    /// Returns the largest bounding box contained by both `self` and another
    /// bounding box.
    ///
    /// Bounding boxes constitute a partially ordered set (poset), with
    /// ordering imposed by containment. This function computes the greatest
    /// lower bound (glb), also called *meet* or *infimum*, of two bboxes in
    /// that ordering.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{math::pt3, render::BBox};
    ///
    /// let a = BBox::<()>(pt3(0.0, 0.0, 0.0), pt3(3.0, 1.0, 1.0));
    /// let b = BBox::<()>(pt3(1.0, 0.0, 0.0), pt3(2.0, 2.0, 1.0));
    ///
    /// assert_eq!(a.intersect(&b), BBox(pt3(1.0, 0.0, 0.0), pt3(2.0, 1.0, 1.0)));
    ///
    /// // The intersection of disjoint bboxes is empty:
    ///
    /// let a = BBox::<()>(pt3(0.0, 0.0, 0.0), pt3(1.0, 1.0, 1.0));
    /// let b = BBox::<()>(pt3(2.0, 0.0, 0.0), pt3(3.0, 1.0, 1.0));
    ///
    /// assert!(a.intersect(&b).is_empty());
    ///
    /// // The intersection of bboxes sharing (a part of) a face, edge,
    /// // or an individual vertex is still nonempty:
    ///
    /// let a = BBox::<()>(pt3(0.0, 0.0, 0.0), pt3(1.0, 1.0, 1.0));
    /// let b = BBox::<()>(pt3(1.0, 0.0, 0.0), pt3(2.0, 1.0, 1.0));
    ///
    /// assert!(a.intersect(&b).contains(&pt3(1.0, 0.5, 0.5)));
    /// ```
    pub fn intersect(&self, other: &Self) -> Self {
        let low = self.0.zip_map(other.0, f32::max);
        let upp = self.1.zip_map(other.1, f32::min);
        BBox(low, upp)
    }

    #[rustfmt::skip]
    pub fn verts(&self) -> [Point3<B>; 8] {
        let [x0, y0, z0] = self.0.0;
        let [x1, y1, z1] = self.1.0;
        [
            pt3(x0, y0, z0), pt3(x0, y0, z1), pt3(x0, y1, z0), pt3(x0, y1, z1),
            pt3(x1, y0, z0), pt3(x1, y0, z1), pt3(x1, y1, z0), pt3(x1, y1, z1),
        ]
    }

    /// Returns whether `self` intersects the view frustum.
    ///
    /// Given a real-to-projection transform, tests this bounding box against
    /// the view frustum and returns whether the box (and thus any bounded
    /// geometry) is fully hidden, fully visible, or potentially partially
    /// visible.
    ///
    /// If this method returns `Hidden`, the box is definitely outside the
    /// frustum and any bounded geometry does not have to be drawn. If it
    /// returns `Visible`, it is fully inside the frustum, and contained
    /// geometry needs no clipping or culling.  If the return value is
    /// `Clipped`, the box and the geometry are *potentially* visible and
    /// more fine-grained culling is required.
    pub fn visibility(&self, tf: &ProjMat3<B>) -> Status {
        view_frustum::status(
            &self
                .verts()
                .map(|p| ClipVert::new(vertex(tf.apply(&p), ()))),
        )
    }
}

impl<B: Debug + Default> Debug for BBox<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("BBox<")?;
        Debug::fmt(&B::default(), f)?;
        f.write_str(">(")?;
        Debug::fmt(&self.0.0, f)?;
        Debug::fmt(&self.1.0, f)?;
        f.write_str(")")
    }
}

impl<A> Default for Obj<A> {
    /// Returns an empty `Obj`.
    fn default() -> Self {
        Self {
            geom: Mesh::default(),
            bbox: BBox::default(),
            tf: Mat4::default(),
        }
    }
}

impl<B> Default for BBox<B> {
    /// Returns an empty `BBox`.
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a, B> Extend<&'a Point3<B>> for BBox<B> {
    fn extend<I: IntoIterator<Item = &'a Point3<B>>>(&mut self, it: I) {
        it.into_iter().for_each(|pt| self.extend(pt));
    }
}

impl<'a, B> FromIterator<&'a Point3<B>> for BBox<B> {
    fn from_iter<I: IntoIterator<Item = &'a Point3<B>>>(it: I) -> Self {
        let mut bbox = BBox::default();
        Extend::extend(&mut bbox, it);
        bbox
    }
}

#[cfg(test)]
mod tests {
    use crate::math::pt3;

    use super::*;

    #[test]
    fn bbox_default() {
        assert!(BBox::<()>::default().is_empty());
        assert!(!BBox::<()>::default().contains(&Point3::origin()));
    }

    #[test]
    fn bbox_extend() {
        let mut bbox = BBox::<()>(pt3(-1.0, -2.0, -3.0), pt3(5.0, 3.0, 2.0));

        bbox.extend(&pt3(1.0, 1.0, 1.0));
        assert_eq!(bbox, BBox(pt3(-1.0, -2.0, -3.0), pt3(5.0, 3.0, 2.0)));

        bbox.extend(&pt3(-2.0, 3.0, 3.0));
        assert_eq!(bbox, BBox(pt3(-2.0, -2.0, -3.0), pt3(5.0, 3.0, 3.0)));
    }

    #[test]
    fn bbox_is_empty() {
        assert!(
            BBox::<()>(pt3(-1.0, 0.0, -1.0), pt3(1.0, 0.0, 1.0)).is_empty()
        );
        assert!(
            BBox::<()>(pt3(-1.0, -1.0, 1.0), pt3(1.0, 1.0, -1.0)).is_empty()
        );
        assert!(
            !BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0)).is_empty()
        );
        assert!(
            !BBox::<()>(pt3(-1.0, 10.0, -1.0), pt3(1.0, f32::INFINITY, 1.0))
                .is_empty()
        );
    }

    #[test]
    fn bbox_contains() {
        assert!(
            !BBox::<()>(pt3(-1.0, 0.0, -1.0), pt3(1.0, 0.0, 1.0))
                .contains(&pt3(0.0, 1.0, 0.0))
        );
        assert!(
            BBox::<()>(pt3(-1.0, 0.0, -1.0), pt3(1.0, 0.0, 1.0))
                .contains(&pt3(0.0, 0.0, 0.0))
        );
        assert!(
            BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0))
                .contains(&pt3(-1.0, 0.0, 0.0))
        );
        assert!(
            BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0))
                .contains(&pt3(0.0, 0.0, 0.0))
        );
    }
}
